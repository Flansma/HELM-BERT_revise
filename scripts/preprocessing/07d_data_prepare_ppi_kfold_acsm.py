"""K-fold data preparation for Propedia v2 with clustering-based splits.

This script implements clustering-based k-fold splits using complex-level signatures:
1. Cluster peptide-protein complexes using K-Means on aCSM-ALL signatures
2. For each fold: test≈20% of clusters, remaining≈80% split as train:val=4:1
3. Generate negative pairs (1:4 ratio) within each split separately
4. Ensure no negative pair duplication within each fold (cross-fold negatives allowed)

Key features:
- Complex-level clustering captures interaction patterns
- Each fold: test≈20%, train≈64%, val≈16% (from 80% non-test split 4:1)
- Negative pairs generated within each split (train/val/test)
- No cluster overlap between splits within a fold; across folds clusters reused as per standard CV

Output format:
- Columns: Peptide_Sequence, Receptor_Sequence, Label, fold, split
- Label: 1 for positive pairs, 0 for negative pairs
- fold: 0-4 for 5-fold CV
- split: train/val/test
"""

# Set thread limits to prevent OpenBLAS errors
import os
thread_count = str(min(8, os.cpu_count() or 8))
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, thread_count)

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime
import sys
import argparse
import pickle
import lightning as L

# Safe tqdm import
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # noqa: kwargs unused
        return iterable


# Configuration (repo-relative defaults; CLI-overridable)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_FILE = REPO_ROOT / 'local_data/intermediate_product/Propedia_v2_unique_ppi_HELM_SMILES.csv'
DEFAULT_OUTPUT_FILE = REPO_ROOT / 'data/downstream/Propedia_v2_ppi_5fold_acsm.csv'
DEFAULT_LOG_DIR = REPO_ROOT / 'outputs/preprocessing'
LOG_FILE_NAME = 'prepare_kfold_ppi_acsm.log'

N_FOLDS = 5
SEED = 42
NEGATIVE_RATIO = 4  # Ratio of negative to positive pairs (1:4)

# This script now always ensures both:
# (A) No intra-fold duplicates (train/val/test within same fold)
# (B) No inter-fold test duplicates (test sets across different folds)

# Split ratio for train/val within non-test data
# In 5-fold CV: test=20% (1 fold), train+val=80% (4 folds)
# Within the 80%, split 7:1:2 → train=70%, val=10% (of total)
VAL_RATIO_OF_TRAINVAL = 0.125  # 10% of total / 80% non-test = 12.5% of non-test

# Feature extraction configuration
FEATURE_TYPE = 'acsm-all'
SIGNATURE_DIR = str(REPO_ROOT / 'local_data/intermediate_product/signatures_acsm_all')
COMPLEX_SIGNATURE_FILE = 'complex_signatures_acsm_all.csv'

# Clustering configuration
N_COMPLEX_CLUSTERS = 100  # Fine-grained clustering, then hierarchically group into folds
DIMENSIONALITY_REDUCTION = 'pca'  # 'pca', 'tsne', or None (t-SNE only supports up to 3 components)
DIM_REDUCTION_COMPONENTS = 50
OPTIMIZE_CLUSTERS = False
USE_HIERARCHICAL_FOLD_ASSIGNMENT = True  # Group similar clusters using hierarchical clustering

# Column names
PEPTIDE_SEQ_COL = 'Peptide_Sequence'
PROTEIN_SEQ_COL = 'Receptor_Sequence'
PDB_COL = 'PDB'
PEPTIDE_CHAIN_COL = 'Peptide_Chain'
PROTEIN_CHAIN_COL = 'Receptor_Chain'

PEPTIDE_HELM_COL = 'Peptide_HELM'
PEPTIDE_SMILES_COL = 'Peptide_SMILES'
PEPTIDE_LENGTH_COL = 'Peptide_Length'
RECEPTOR_LENGTH_COL = 'Receptor_Length'
COMPLEX_FILE_COL = 'Complex_File'
WEIGHT_COL = 'weight'  # Weight column from oligomeric state processing

OUTPUT_PEPTIDE_COL = 'Peptide_Sequence'
OUTPUT_PROTEIN_COL = 'Receptor_Sequence'
OUTPUT_LABEL_COL = 'Label'
OUTPUT_FOLD_COL = 'fold'
OUTPUT_SPLIT_COL = 'split'

SPLIT_TRAIN = 'train'
SPLIT_VAL = 'val'
SPLIT_TEST = 'test'

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Initialize module-level logger
logger = logging.getLogger(__name__)
log_dir = None

# Helper functions

def get_fold_seed(base_seed: int, fold_idx: int, split_offset: int = 0) -> int:
    """Generate unique seed for each fold and split."""
    return base_seed + fold_idx * 1000 + split_offset


def _pair_key(pep: str, prot: str) -> Tuple[str, str]:
    """Normalize pair key for consistent comparison."""
    return (str(pep).strip().upper(), str(prot).strip().upper())


def setup_logging() -> Tuple[logging.Logger, Path]:
    """Set up logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(DEFAULT_LOG_DIR)
    # Standardize: <dataset_name>_<method>_<timestamp>
    log_dir = base / f"propedia_v2_ppi_acsm_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / LOG_FILE_NAME
    
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Log file: {log_file.absolute()}")
    
    return logger, log_dir


# Feature extraction

class ComplexSignatureFeaturizer:
    """Load and use pre-computed aCSM-ALL signatures for complexes."""
    
    def __init__(self, signature_dir: str = SIGNATURE_DIR, logger: Optional[logging.Logger] = None):
        """Initialize with signature directory."""
        self.signature_dir = Path(signature_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Load complex signatures
        complex_sig_path = self.signature_dir / COMPLEX_SIGNATURE_FILE
        if complex_sig_path.exists():
            self.logger.info(f"Loading complex signatures from: {complex_sig_path}")
            # Specify dtypes for columns
            self.complex_signatures = pd.read_csv(complex_sig_path, 
                                                 dtype={'pdb_id': 'str', 'peptide_chain': 'str', 
                                                       'protein_chain': 'str', 'complex_file': 'str'})
            sig_cols = [col for col in self.complex_signatures.columns if col.startswith('sig_')]
            
            # Vectorized loading - much faster than iterrows
            # Ensure uniqueness by pdb_id
            cs = self.complex_signatures.drop_duplicates('pdb_id')
            self.complex_lookup = dict(zip(
                cs['pdb_id'].astype(str), 
                cs[sig_cols].to_numpy(dtype=np.float32)
            ))
            
            self.logger.info(f"Loaded {len(self.complex_lookup)} complex signatures")
            self.n_features = len(sig_cols)
            self.logger.info(f"Signature dimensions: {self.n_features}")
        else:
            raise ValueError(f"Complex signature file not found: {complex_sig_path}. "
                           f"Please run 06_data_generate_acsm_signatures.py first.")
    
    def get_complex_signatures(self, pdb_ids: List[str]) -> np.ndarray:
        """Get pre-computed signatures for complexes."""
        self.logger.info(f"Getting signatures for {len(pdb_ids)} complexes")
        
        features_array = np.zeros((len(pdb_ids), self.n_features), dtype=np.float32)
        found_count = 0
        
        for i, pdb_id in enumerate(pdb_ids):
            if pdb_id in self.complex_lookup:
                features_array[i] = self.complex_lookup[pdb_id]
                found_count += 1
            else:
                self.logger.warning(f"Signature not found for complex {pdb_id}")
        
        self.logger.info(f"Found signatures for {found_count}/{len(pdb_ids)} complexes")
        return features_array


def find_optimal_clusters(embeddings_scaled: np.ndarray, 
                         min_clusters: int = 50, 
                         max_clusters: int = 200, 
                         step: int = 50,
                         seed: int = 42,
                         sample_size: Optional[int] = None,
                         logger: Optional[logging.Logger] = None) -> int:
    """Find optimal number of clusters using silhouette analysis."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    best_score = -1
    best_k = min_clusters
    
    max_possible = min(len(embeddings_scaled) - 1, max_clusters)
    min_clusters = min(min_clusters, max_possible)
    
    # Guard against too few samples
    if min_clusters < 2:
        logger.warning(f"Too few samples for clustering optimization, returning 2")
        return 2
    
    if sample_size is None:
        sample_size = min(5000, len(embeddings_scaled))
    
    if sample_size < len(embeddings_scaled):
        logger.info(f"Using {sample_size} samples for silhouette analysis")
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(embeddings_scaled), sample_size, replace=False)
        sample_embeddings = embeddings_scaled[sample_indices]
    else:
        sample_embeddings = embeddings_scaled
    
    cluster_range = list(range(min_clusters, max_possible + 1, step))
    logger.info(f"Testing cluster counts: {cluster_range}")
    
    for k in tqdm(cluster_range, desc="Testing cluster counts"):
        try:
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                random_state=seed,
                n_init=10
            )
            
            if sample_size < len(embeddings_scaled):
                labels = kmeans.fit_predict(sample_embeddings)
                score = silhouette_score(sample_embeddings, labels)
            else:
                labels = kmeans.fit_predict(embeddings_scaled)
                score = silhouette_score(embeddings_scaled, labels)
                
            logger.info(f"  k={k}: silhouette score = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            logger.warning(f"  k={k}: failed - {str(e)}")
    
    logger.info(f"Best cluster count: {best_k} (score: {best_score:.3f})")
    return best_k


def cluster_complexes(pdb_ids: List[str], n_clusters: int, embedder,
                     seed: int = 42, optimize_clusters: bool = True,
                     logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    """Cluster complexes using K-Means on aCSM-ALL signatures.

    Returns:
        Tuple of (pdb_to_cluster, embeddings_scaled, cluster_labels)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Clustering {len(pdb_ids)} complexes into {n_clusters} clusters")

    # Only use PDBs with signatures
    available = set(embedder.complex_lookup.keys())
    pdb_ids_with_sig = [p for p in pdb_ids if p in available]
    missing = len(pdb_ids) - len(pdb_ids_with_sig)
    if missing:
        logger.warning(f"{missing} PDBs have no aCSM signatures and will be excluded from clustering")

    if not pdb_ids_with_sig:
        raise ValueError("No PDBs with signatures available for clustering")

    embeddings = embedder.get_complex_signatures(pdb_ids_with_sig)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Apply dimensionality reduction if configured
    if DIMENSIONALITY_REDUCTION and embeddings.shape[1] > DIM_REDUCTION_COMPONENTS:
        if DIMENSIONALITY_REDUCTION.lower() == 'pca':
            logger.info(f"Applying PCA to reduce dimensions to {DIM_REDUCTION_COMPONENTS}")
            reducer = PCA(n_components=DIM_REDUCTION_COMPONENTS, random_state=seed)
            embeddings_scaled = reducer.fit_transform(embeddings_scaled)
            explained_variance = reducer.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance ratio: {explained_variance:.3f}")
        elif DIMENSIONALITY_REDUCTION.lower() == 'tsne':
            logger.info(f"Applying t-SNE to reduce dimensions to {DIM_REDUCTION_COMPONENTS}")
            reducer = TSNE(
                n_components=DIM_REDUCTION_COMPONENTS,
                random_state=seed,
                perplexity=min(30, len(embeddings_scaled) - 1),
                max_iter=1000,
                verbose=1
            )
            embeddings_scaled = reducer.fit_transform(embeddings_scaled)
            logger.info(f"t-SNE dimensionality reduction completed")
        else:
            logger.warning(f"Unknown dimensionality reduction method: {DIMENSIONALITY_REDUCTION}, skipping")

    if optimize_clusters and len(pdb_ids) > 50:
        logger.info("Finding optimal number of clusters...")
        optimal_k = find_optimal_clusters(
            embeddings_scaled,
            min_clusters=50,
            max_clusters=min(n_clusters, len(pdb_ids) - 1),
            step=50,
            seed=seed,
            logger=logger
        )
        actual_clusters = optimal_k
    else:
        actual_clusters = min(n_clusters, len(pdb_ids))
        if actual_clusters < n_clusters:
            logger.warning(f"Reduced clusters from {n_clusters} to {actual_clusters}")

    try:
        kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=seed,
            n_init=10,
            init='k-means++'
        )
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
    except Exception as e:
        logger.error(f"K-Means clustering failed: {str(e)}")
        raise

    # Map back to PDBs with signatures only
    pdb_to_cluster = {pdb: int(label) for pdb, label in zip(pdb_ids_with_sig, cluster_labels)}

    _, counts = np.unique(cluster_labels, return_counts=True)
    logger.info(f"Complex clustering results:")
    logger.info(f"  - Clusters used: {actual_clusters}")
    logger.info(f"  - Cluster sizes: Min={counts.min()}, Max={counts.max()}, "
               f"Mean={counts.mean():.1f}, Std={counts.std():.1f}")

    return pdb_to_cluster, embeddings_scaled, cluster_labels


def assign_clusters_to_folds_constrained(
    cluster_ids: List[int],
    pairs_per_cluster: Dict[int, int],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    n_folds: int,
    seed: int,
    logger: logging.Logger,
    max_deviation: float = 0.15
) -> Dict[int, int]:
    """Assign clusters to folds using Constrained K-Means (Iterative Refinement).

    Algorithm:
    1. Initialize centroids using standard K-Means on cluster centroids.
    2. Iteratively refine assignments to satisfy size constraints (weighted by pairs).
    3. Move clusters from over-full folds to under-full folds minimizing cost increase.

    Args:
        cluster_ids: List of unique cluster IDs
        pairs_per_cluster: Mapping of cluster_id -> number of pairs
        embeddings: PCA-reduced embeddings (N_samples x D_features)
        cluster_labels: Cluster assignment for each sample (N_samples,)
        n_folds: Number of folds
        seed: Random seed
        logger: Logger instance
        max_deviation: Maximum allowed deviation from target size (default: 0.15)

    Returns:
        Dictionary mapping cluster_id -> fold_idx
    """
    if not cluster_ids:
        raise ValueError("cluster_ids cannot be empty")

    logger.info(
        f"Assigning {len(cluster_ids)} clusters to {n_folds} folds "
        f"(Constrained K-Means, max_deviation={max_deviation:.1%})..."
    )

    # 1. Compute Cluster Centroids & Weights
    cluster_centroids = []
    cluster_weights = []
    cid_map = []  # Index to Cluster ID

    for cid in cluster_ids:
        mask = cluster_labels == cid
        if mask.sum() == 0:
            continue
        centroid = embeddings[mask].mean(axis=0)
        weight = pairs_per_cluster.get(cid, 0)
        
        cluster_centroids.append(centroid)
        cluster_weights.append(weight)
        cid_map.append(cid)

    X = np.array(cluster_centroids)
    weights = np.array(cluster_weights)
    n_clusters = len(X)
    
    total_weight = weights.sum()
    target_weight = total_weight / n_folds
    min_weight = target_weight * (1 - max_deviation)
    max_weight = target_weight * (1 + max_deviation)

    logger.info(f"Total pairs: {total_weight}, Target per fold: {target_weight:.0f}")
    logger.info(f"Allowed range: [{min_weight:.0f}, {max_weight:.0f}]")

    # 2. Initial K-Means Initialization (Unconstrained)
    kmeans = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    initial_labels = kmeans.fit_predict(X)
    fold_centroids = kmeans.cluster_centers_

    # Current assignments
    assignments = initial_labels.copy()
    
    # 3. Iterative Refinement (Balancing)
    # We treat this as a flow problem: Move mass from Source (Over) to Sink (Under)
    # Cost = Distance to new centroid - Distance to old centroid
    
    max_iter = 100
    rng = np.random.default_rng(seed)

    for iteration in range(max_iter):
        # Calculate current fold weights
        fold_weights = np.zeros(n_folds)
        for i in range(n_clusters):
            fold_weights[assignments[i]] += weights[i]
            
        # Identify Source (Over-full) and Sink (Under-full) folds
        sources = [f for f in range(n_folds) if fold_weights[f] > max_weight]
        sinks = [f for f in range(n_folds) if fold_weights[f] < min_weight]
        
        if not sources and not sinks:
            logger.info(f"Converged at iteration {iteration}: All folds within constraints.")
            break
            
        # If we have sources but no sinks (or vice versa), we just need to rebalance
        # Treat any non-source as a potential sink if sources exist
        if sources and not sinks:
            sinks = [f for f in range(n_folds) if f not in sources]
        elif sinks and not sources:
            sources = [f for f in range(n_folds) if f not in sinks]

        # Find best move: Cluster c from Source S -> Sink T
        best_move = None
        best_cost_reduction = float('inf')

        # Shuffle to avoid cycles/bias
        rng.shuffle(sources)
        rng.shuffle(sinks)

        moved = False
        
        for s in sources:
            # Clusters currently in source s
            candidates = [i for i in range(n_clusters) if assignments[i] == s]
            
            for i in candidates:
                # Try moving to any sink
                for t in sinks:
                    # Check if moving helps (or at least is valid)
                    # We want to reduce the overflow of S or reduce underflow of T
                    
                    # Cost: (Dist to T) - (Dist to S)
                    dist_s = np.linalg.norm(X[i] - fold_centroids[s])
                    dist_t = np.linalg.norm(X[i] - fold_centroids[t])
                    cost = dist_t - dist_s
                    
                    # Heuristic: Prioritize moves that fix constraints
                    # If S is huge, we desperately want to move out.
                    # If T is tiny, we desperately want to move in.
                    
                    # Check if move is valid (doesn't make T explode)
                    if fold_weights[t] + weights[i] > max_weight * 1.05: # 5% buffer during moves
                        continue
                        
                    if cost < best_cost_reduction:
                        best_cost_reduction = cost
                        best_move = (i, s, t)

        if best_move:
            idx, s, t = best_move
            assignments[idx] = t
            moved = True
            # Update centroids (simple running average update or recompute)
            # Recomputing is safer
            # (Skipping recompute for speed in loop, will update periodically)
        else:
            logger.warning(f"No valid moves found at iteration {iteration}. Stopping.")
            break

    # 4. Final Centroid Update & Assignment
    cluster_to_fold = {}
    for i in range(n_clusters):
        cluster_to_fold[cid_map[i]] = assignments[i]

    # Validate
    logger.info("Final Constrained Assignments:")
    final_weights = np.zeros(n_folds)
    for i in range(n_clusters):
        final_weights[assignments[i]] += weights[i]

    for f in range(n_folds):
        deviation = abs(final_weights[f] - target_weight) / target_weight
        logger.info(f"  Fold {f}: {final_weights[f]:.0f} pairs (deviation: {deviation:.1%})")
        if deviation > max_deviation:
            logger.warning(f"  Fold {f} exceeds max_deviation ({deviation:.1%} > {max_deviation:.1%})")

    return cluster_to_fold


def split_clusters_by_pairs(
    cluster_ids: List[int],
    pairs_per_cluster: Dict[int, int],
    n_folds: int,
    seed: int,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[int, Dict[str, Set[int]]]:
    """Split clusters into train/val/test sets for k-fold cross-validation.

    Uses hierarchical clustering on cluster centroids to assign clusters to folds.
    Similar clusters are grouped into the same fold to create homogeneous test sets.
    Within each fold, clusters are further split into train/val based on pair counts.

    Args:
        cluster_ids: List of unique cluster IDs to split
        pairs_per_cluster: Mapping of cluster_id -> number of unique pairs in cluster
        n_folds: Number of folds for cross-validation
        seed: Random seed for reproducibility
        embeddings: PCA-reduced embeddings for all samples (N_samples x D_features)
        cluster_labels: Cluster assignment for each sample (N_samples,)

    Returns:
        Dictionary mapping fold_idx -> {'train': set, 'val': set, 'test': set}
        where each set contains cluster IDs

    Raises:
        ValueError: If inputs are invalid or insufficient for k-fold splitting
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not cluster_ids:
        raise ValueError("cluster_ids cannot be empty")
    if len(cluster_ids) < n_folds:
        raise ValueError(
            f"Insufficient clusters ({len(cluster_ids)}) for {n_folds}-fold CV"
        )
    if embeddings is None or cluster_labels is None:
        raise ValueError("embeddings and cluster_labels are required for hierarchical grouping")

    # Assign clusters to folds balancing similarity and size
    cluster_to_fold = assign_clusters_to_folds_constrained(
        cluster_ids=cluster_ids,
        pairs_per_cluster=pairs_per_cluster,
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        n_folds=n_folds,
        seed=seed,
        logger=logger,
        max_deviation=0.15  # 15% tolerance as requested
    )

    # Group clusters by fold
    fold_test_clusters = defaultdict(set)
    for cid, fold_idx in cluster_to_fold.items():
        fold_test_clusters[fold_idx].add(cid)

    test_bins = [fold_test_clusters[i] for i in range(n_folds)]

    # Log and validate pair distribution
    logger.info("Test set pair distribution across folds:")
    total_pairs = sum(pairs_per_cluster.values())
    target_per_fold = total_pairs / n_folds

    for fold_idx in range(n_folds):
        fold_pairs = sum(pairs_per_cluster.get(cid, 0) for cid in test_bins[fold_idx])
        deviation = abs(fold_pairs - target_per_fold) / target_per_fold * 100
        logger.info(
            f"  Fold {fold_idx}: {fold_pairs:5d} pairs "
            f"({len(test_bins[fold_idx]):3d} clusters, "
            f"deviation: {deviation:5.1f}%)"
        )
    
    fold_splits = {}
    
    for fold_idx in range(n_folds):
        # Test clusters for this fold
        test_clusters = set(test_bins[fold_idx])
        
        # Train+val clusters (all others)
        train_val_clusters = [c for c in cluster_ids if c not in test_clusters]
        
# Calculate target val pairs based on pair count (12.5% of train+val for 7:1:2 ratio)
        total_train_val_pairs = sum(pairs_per_cluster.get(c, 0) for c in train_val_clusters)
        target_val_pairs = int(total_train_val_pairs * VAL_RATIO_OF_TRAINVAL)
        
        # Sort by pair count (descending) with random tiebreaker
        fold_rng = np.random.default_rng(get_fold_seed(seed, fold_idx))
        # Randomized order for validation selection (pair-count target)
        train_val_sorted = list(train_val_clusters)
        fold_rng = np.random.default_rng(get_fold_seed(seed, fold_idx))
        fold_rng.shuffle(train_val_sorted)

        # Greedy accumulation to meet validation pair target
        val_clusters = set()
        current_val_pairs = 0
        
        for cluster_id in train_val_sorted:
            cluster_pairs = pairs_per_cluster.get(cluster_id, 0)
            if current_val_pairs >= target_val_pairs:
                break
            val_clusters.add(cluster_id)
            current_val_pairs += cluster_pairs
        
        # Ensure at least one cluster in val if possible
        if not val_clusters and train_val_sorted:
            val_clusters.add(train_val_sorted[0])
        
        # Remaining clusters go to train
        train_clusters = set(train_val_sorted) - val_clusters
        
        fold_splits[fold_idx] = {
            SPLIT_TRAIN: train_clusters,
            SPLIT_VAL: val_clusters,
            SPLIT_TEST: test_clusters
        }
        
        # Log the balance (by pairs)
        train_pairs = sum(pairs_per_cluster.get(c, 0) for c in train_clusters)
        val_pairs = sum(pairs_per_cluster.get(c, 0) for c in val_clusters)
        test_pairs = sum(pairs_per_cluster.get(c, 0) for c in test_clusters)
        logger.debug(f"Fold {fold_idx} pair distribution: "
                    f"train={train_pairs}, val={val_pairs}, test={test_pairs}")
    
    return fold_splits


# Negative pair generation

def create_negative_pairs_dataframe(peptide_sequences: List[str], protein_sequences: List[str],
                                  peptide_records: dict,
                                  protein_records: dict) -> pd.DataFrame:
    """Create negative pairs DataFrame preserving all molecular properties."""
    negative_rows = []
    
    for pep_seq, prot_seq in zip(peptide_sequences, protein_sequences):
        pep_record = peptide_records.get(pep_seq)
        prot_record = protein_records.get(prot_seq)
        
        if pep_record is None or prot_record is None:
            raise KeyError(f"Missing record for negative pair: "
                          f"peptide='{pep_seq[:20]}...', protein='{prot_seq[:20]}...'")
        
        row = {
            PEPTIDE_SEQ_COL: pep_seq,
            PEPTIDE_HELM_COL: pep_record[PEPTIDE_HELM_COL],
            PEPTIDE_SMILES_COL: pep_record[PEPTIDE_SMILES_COL],
            PEPTIDE_CHAIN_COL: pep_record[PEPTIDE_CHAIN_COL],
            PEPTIDE_LENGTH_COL: pep_record[PEPTIDE_LENGTH_COL],
            
            PROTEIN_SEQ_COL: prot_seq,
            PROTEIN_CHAIN_COL: prot_record[PROTEIN_CHAIN_COL],
            RECEPTOR_LENGTH_COL: prot_record[RECEPTOR_LENGTH_COL],
            
            PDB_COL: 'NEGATIVE',
            COMPLEX_FILE_COL: 'NEGATIVE.pdb',
            OUTPUT_LABEL_COL: 0,
            WEIGHT_COL: 1.0  # Negative pairs always have weight 1.0
        }
        
        negative_rows.append(row)
    
    return pd.DataFrame(negative_rows)


def generate_negative_pairs_for_split(n_negative: int,
                                    all_peptides: Set[str], 
                                    all_proteins: Set[str], 
                                    seed: int,
                                    global_positive_pairs: Set[Tuple[str, str]],
                                    existing_negatives: Set[Tuple[str, str]],
                                    peptide_records: dict,
                                    protein_records: dict,
                                    logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Generate negative pairs for a single split."""
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.debug(f"Generating {n_negative} negative pairs with seed {seed}")
    
    # Calculate maximum available negatives (PREVENTING DATA LEAKAGE)
    # Must exclude ALL global positives to prevent data leakage between folds
    global_pos_in_split = {(p, r) for (p, r) in global_positive_pairs 
                          if p in all_peptides and r in all_proteins}
    neg_blocked_in_split = {(p, r) for (p, r) in existing_negatives 
                           if p in all_peptides and r in all_proteins}
    total_pairs = len(all_peptides) * len(all_proteins)
    max_available = total_pairs - len(global_pos_in_split) - len(neg_blocked_in_split)
    
    logger.info(f"    Split pool: {len(all_peptides)} peptides × {len(all_proteins)} proteins = {total_pairs} total pairs")
    logger.info(f"    Excluding: {len(global_pos_in_split)} global positives + {len(neg_blocked_in_split)} used negatives")
    logger.info(f"    Available for negative generation: {max_available} pairs")
    if n_negative > max_available:
        logger.info(f"    SHORTAGE: Requested {n_negative} but can only generate {max_available} (ratio will be 1:{max_available/n_negative*4:.1f} instead of 1:4)")
    
    if n_negative > max_available:
        logger.warning(f"Requested {n_negative} negatives but only {max_available} available; capping")
        n_negative = min(n_negative, max_available)
        if max_available < n_negative * 0.5:  # Less than 50% of requested
            logger.warning(f"WARNING: Severe negative sample shortage! Consider reducing NEGATIVE_RATIO from {NEGATIVE_RATIO}")
    
    if n_negative <= 0:
        logger.warning("No negative pairs can be generated for this split")
        return create_negative_pairs_dataframe([], [], peptide_records, protein_records)
    
    excluded_pairs = global_positive_pairs | existing_negatives
    
    peptides_array = np.array(sorted(all_peptides))
    proteins_array = np.array(sorted(all_proteins))
    
    rng = np.random.default_rng(seed)
    
    negative_pairs = []
    negative_pairs_set = set()  # For fast lookup within this generation
    # Dynamic batch size for better performance
    batch_size = max(10000, min(n_negative * 10, 1000000))
    
    attempts = 0
    max_attempts = 100
    
    while len(negative_pairs) < n_negative and attempts < max_attempts:
        peptide_indices = rng.integers(0, len(peptides_array), batch_size)
        protein_indices = rng.integers(0, len(proteins_array), batch_size)
        
        candidate_peptides = peptides_array[peptide_indices]
        candidate_proteins = proteins_array[protein_indices]
        
        for pep, prot in zip(candidate_peptides, candidate_proteins):
            pair = (pep, prot)
            if pair not in excluded_pairs and pair not in negative_pairs_set:
                negative_pairs.append(pair)
                negative_pairs_set.add(pair)  # Prevent duplicates within this batch
                if len(negative_pairs) >= n_negative:
                    break
        
        attempts += 1
    
    # Fallback: if still not enough, use generator pattern for memory efficiency
    if len(negative_pairs) < n_negative:
        logger.warning(f"Random generation only produced {len(negative_pairs)}/{n_negative} pairs, using generator")
        
        def yield_valid_pairs(peps, prots, excluded):
            """Generator for valid negative pairs."""
            for pep in sorted(peps):  # Sort for determinism
                for prot in sorted(prots):
                    pair = (pep, prot)
                    if pair not in excluded:
                        yield pair
        
        needed = n_negative - len(negative_pairs)
        produced = 0
        
        for pair in yield_valid_pairs(all_peptides, all_proteins, 
                                     excluded_pairs | negative_pairs_set):
            negative_pairs.append(pair)
            negative_pairs_set.add(pair)
            produced += 1
            if produced >= needed:
                break
        
        if produced < needed:
            logger.warning(f"Could only generate {len(negative_pairs)}/{n_negative} negative pairs")
    
    peptide_sequences = [p[0] for p in negative_pairs[:n_negative]]
    protein_sequences = [p[1] for p in negative_pairs[:n_negative]]
    
    negative_df = create_negative_pairs_dataframe(
        peptide_sequences, protein_sequences,
        peptide_records, protein_records
    )
    
    return negative_df


# Main processing function

def prepare_propedia_v2_kfold(
    source_file: str,
    output_file: str,
    n_folds: int = 5,
    seed: int = 42,
    n_complex_clusters: int = 600,
    negative_ratio: int = 4
) -> pd.DataFrame:
    """Create k-fold splits for Propedia v2 with clustering."""
    
    source_path = Path(source_file)
    output_path = Path(output_file)
    
    logger.info(f"Loading data from: {source_path}")
    df_all = pd.read_csv(source_path)
    
    required_cols = [PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL, PDB_COL]
    if not all(col in df_all.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {df_all.columns.tolist()}")
    
    logger.info(f"Loaded {len(df_all)} total entries")
    
    # Use all positive pairs without removing duplicates
    # Weight system already handles oligomeric state duplications
    df_positive = df_all.copy()
    
    if OUTPUT_LABEL_COL not in df_positive.columns:
        df_positive[OUTPUT_LABEL_COL] = 1
    
    logger.info(f"Total positive rows (with oligomeric variations): {len(df_positive)}")
    n_unique_pairs = df_positive[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates().shape[0]
    logger.info(f"Unique positive pairs: {n_unique_pairs}")
    logger.info(f"Unique peptides: {df_positive[PEPTIDE_SEQ_COL].nunique()}")
    logger.info(f"Unique proteins: {df_positive[PROTEIN_SEQ_COL].nunique()}")
    
    global_positive_pairs = set(zip(df_positive[PEPTIDE_SEQ_COL], df_positive[PROTEIN_SEQ_COL]))
    
    # Create featurizer
    embedder = ComplexSignatureFeaturizer(signature_dir=SIGNATURE_DIR)
    
    # Cluster complexes
    logger.info("=" * 60)
    logger.info("Step 1: Clustering complexes")
    logger.info("=" * 60)
    
    # Get unique PDB IDs
    unique_pdb_ids = df_positive[PDB_COL].unique().tolist()
    logger.info(f"Unique complexes (PDB IDs): {len(unique_pdb_ids)}")
    
    # Cluster complexes
    pdb_to_cluster, embeddings_scaled, cluster_labels = cluster_complexes(
        unique_pdb_ids, n_complex_clusters, embedder, seed,
        optimize_clusters=OPTIMIZE_CLUSTERS
    )

    # Add cluster information
    df_positive['complex_cluster'] = df_positive[PDB_COL].map(pdb_to_cluster)

    # Handle missing signatures
    missing = df_positive['complex_cluster'].isna().sum()
    if missing:
        logger.warning(f"Complex signatures missing for {missing} rows; dropping them from splits")
        df_positive = df_positive.dropna(subset=['complex_cluster']).copy()

    # Get unique cluster IDs
    unique_cluster_ids = sorted(set(pdb_to_cluster.values()))
    logger.info(f"Total clusters created: {len(unique_cluster_ids)}")

    # Guard: Ensure we have enough clusters for k-fold splitting
    if len(unique_cluster_ids) < n_folds:
        raise ValueError(f"Not enough clusters ({len(unique_cluster_ids)}) for {n_folds}-fold CV. "
                        f"Consider reducing N_COMPLEX_CLUSTERS or n_folds.")

    # Calculate unique pairs per cluster for balanced splitting
    pairs_per_cluster = (df_positive
                        .drop_duplicates([PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL, 'complex_cluster'])
                        .groupby('complex_cluster')
                        .size()
                        .astype(int)
                        .to_dict())

    # Split clusters into folds using hierarchical clustering
    logger.info("=" * 60)
    logger.info("Step 2: Grouping clusters into folds via hierarchical clustering")
    logger.info("=" * 60)

    if not USE_HIERARCHICAL_FOLD_ASSIGNMENT:
        raise ValueError(
            "USE_HIERARCHICAL_FOLD_ASSIGNMENT must be True. "
            "Greedy fallback has been removed for code quality."
        )

    fold_cluster_splits = split_clusters_by_pairs(
        cluster_ids=unique_cluster_ids,
        pairs_per_cluster=pairs_per_cluster,
        n_folds=n_folds,
        seed=seed,
        embeddings=embeddings_scaled,
        cluster_labels=cluster_labels,
    )
    
    # ================================================================
    # STEP 1: Assign test pair ownership (always applied)
    # ================================================================
    
    # Always assign test pair ownership for correct evaluation
    logger.info("Assigning test pair ownership to ensure no cross-fold test duplicates...")
    
    pair_candidates = defaultdict(set)  # (pep,prot) -> {folds that would put it in test}
    fold_test_clusters = {f: set(fold_cluster_splits[f][SPLIT_TEST]) for f in range(n_folds)}
    
    # Find which pairs appear in which fold's test clusters
    ppc = (df_positive[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL, 'complex_cluster']]
           .drop_duplicates())
    
    for f in range(n_folds):
        test_c = fold_test_clusters[f]
        # All pairs in this fold's test clusters
        pairs_f = ppc[ppc['complex_cluster'].isin(test_c)][[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates()
        for pep, prot in pairs_f.itertuples(index=False):
            pair_candidates[_pair_key(pep, prot)].add(f)
    
    # Greedy load-balancing: assign to fold with smallest current load
    pair_owner = {}
    test_load = [0] * n_folds
    # Process pairs with fewer candidate folds first (more constrained)
    for pair, cand_folds in sorted(pair_candidates.items(), key=lambda kv: len(kv[1])):
        if len(cand_folds) > 1:
            # deterministic tie-breaker: by load, then by fold index
            best = min(cand_folds, key=lambda f: (test_load[f], f))
            pair_owner[pair] = best
            test_load[best] += 1
        else:
            # Pair only appears in one fold's test
            pair_owner[pair] = list(cand_folds)[0]
            test_load[list(cand_folds)[0]] += 1
    
    # Log statistics
    multi_fold_pairs = sum(1 for cands in pair_candidates.values() if len(cands) > 1)
    logger.info(f"  Found {multi_fold_pairs} pairs appearing in multiple fold's test clusters")
    logger.info(f"  Test load distribution after assignment: {test_load}")
    
    # Build property mappings (vectorized for speed)
    logger.info("Building complete property mappings...")
    
    # Vectorized peptide records build
    pep_cols = [PEPTIDE_SEQ_COL, PEPTIDE_HELM_COL, PEPTIDE_SMILES_COL, PEPTIDE_CHAIN_COL, PEPTIDE_LENGTH_COL]
    peptide_records = dict(df_positive[pep_cols]
                          .drop_duplicates(PEPTIDE_SEQ_COL)
                          .set_index(PEPTIDE_SEQ_COL)
                          .to_dict('index'))
    
    # Vectorized protein records build
    prot_cols = [PROTEIN_SEQ_COL, PROTEIN_CHAIN_COL, RECEPTOR_LENGTH_COL]
    protein_records = dict(df_positive[prot_cols]
                          .drop_duplicates(PROTEIN_SEQ_COL)
                          .set_index(PROTEIN_SEQ_COL)
                          .to_dict('index'))
    
    # ================================================================
    # STEP 2: Process each fold with test→val→train order
    # ================================================================
    all_data = []
    # negatives: only fold-level uniqueness to keep sampling flexible
    # (no global negative dedup)
    
    # Process order: test first to ensure it gets its pairs
    PROCESS_ORDER = [SPLIT_TEST, SPLIT_VAL, SPLIT_TRAIN]
    
    for fold_idx in range(n_folds):
        logger.info("")
        logger.info(f"Processing fold {fold_idx}...")
        
        # ================================================================
        # Determine protein winner splits based on score (majority assignment)
        # ================================================================
        SPLIT_PRIORITY = {SPLIT_TEST: 0, SPLIT_VAL: 1, SPLIT_TRAIN: 2}
        RESOLVE_BY = 'weighted'  # 'rows' or 'weighted'
        
        # 1) Get split candidates
        split_candidates = {}
        for sn in [SPLIT_TEST, SPLIT_VAL, SPLIT_TRAIN]:
            clusters = fold_cluster_splits[fold_idx][sn]
            pos = df_positive[df_positive['complex_cluster'].isin(clusters)].copy()
            # Apply test ownership first
            if sn == SPLIT_TEST and not pos.empty:
                key_df = pos[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].to_numpy()
                keep_mask = np.fromiter(
                    (pair_owner.get(_pair_key(pep, prot), fold_idx) == fold_idx for pep, prot in key_df),
                    dtype=bool, count=len(pos)
                )
                pos = pos[keep_mask]
            split_candidates[sn] = pos
        
        # 2) Calculate protein scores per split
        protein2score = defaultdict(lambda: {SPLIT_TRAIN: 0.0, SPLIT_VAL: 0.0, SPLIT_TEST: 0.0})
        for sn, pos in split_candidates.items():
            if pos.empty:
                continue
            if RESOLVE_BY == 'weighted' and WEIGHT_COL in pos.columns:
                scores = pos.groupby(PROTEIN_SEQ_COL)[WEIGHT_COL].sum()
            else:
                scores = pos.groupby(PROTEIN_SEQ_COL).size()
            for prot, sc in scores.items():
                key = str(prot).strip().upper()
                protein2score[key][sn] = float(sc)
        
        # 3) Determine winner split for each protein
        protein2winner = {}
        for prot, scdict in protein2score.items():
            # Winner = highest score, tiebreaker = test>val>train
            best_sn = min(scdict.keys(), key=lambda sn: (-scdict[sn], SPLIT_PRIORITY[sn]))
            protein2winner[prot] = best_sn
        
        # Log protein assignment summary
        winner_counts = defaultdict(int)
        for winner in protein2winner.values():
            winner_counts[winner] += 1
        logger.info(f"Protein winner assignment: test={winner_counts[SPLIT_TEST]}, "
                   f"val={winner_counts[SPLIT_VAL]}, train={winner_counts[SPLIT_TRAIN]}")
        
        fold_data_list = []
        # Single registry for ALL pairs (positive and negative) used in this fold
        fold_pair_registry = set()
        
        for split_name in PROCESS_ORDER:
            # Get cluster pairs for this split
            clusters_for_split = fold_cluster_splits[fold_idx][split_name]
            
            # Get positive pairs for this split (vectorized)
            split_pos = df_positive[df_positive['complex_cluster'].isin(clusters_for_split)].copy()
            
            # Test split: enforce ownership to prevent cross-fold duplicates
            if split_name == SPLIT_TEST and not split_pos.empty:
                before = len(split_pos)
                # Only keep pairs this fold owns
                key_df = split_pos[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].to_numpy()
                keep_mask = np.fromiter(
                    (pair_owner.get(_pair_key(pep, prot), fold_idx) == fold_idx for pep, prot in key_df),
                    dtype=bool, count=len(split_pos)
                )
                split_pos = split_pos[keep_mask].copy()
                removed = before - len(split_pos)
                if removed > 0:
                    logger.info(f"    [test ownership] removed {removed} rows for cross-fold deduplication "
                                f"(test positives now {len(split_pos)})")
            
            # Protein collision pruning based on majority assignment
            if not split_pos.empty:
                prot_series_norm = split_pos[PROTEIN_SEQ_COL].astype(str).str.strip().str.upper()
                before = len(split_pos)
                keep_mask = prot_series_norm.map(lambda p: protein2winner.get(p, split_name) == split_name)
                split_pos = split_pos[keep_mask.values].copy()
                pruned = before - len(split_pos)
                if pruned > 0:
                    logger.info(f"    [{split_name}] pruned {pruned} rows by protein majority assignment")
            
            # Intra-fold deduplication: remove pairs already used in this fold
            if not split_pos.empty:
                before_within = len(split_pos)
                # Check against fold registry
                pair_arr = [_pair_key(p, r) for p, r in zip(split_pos[PEPTIDE_SEQ_COL], split_pos[PROTEIN_SEQ_COL])]
                keep_mask_within = np.array([pk not in fold_pair_registry for pk in pair_arr])
                split_pos = split_pos[keep_mask_within].copy()
                
                rem_within = before_within - len(split_pos)
                if rem_within > 0:
                    logger.info(f"    [{split_name}] removed {rem_within} rows for within-fold deduplication "
                                f"(positives now {len(split_pos)})")
                
                # Register positive pairs immediately
                for p, r in zip(split_pos[PEPTIDE_SEQ_COL], split_pos[PROTEIN_SEQ_COL]):
                    fold_pair_registry.add(_pair_key(p, r))
            
            # Get unique peptides and proteins for this split
            split_peptides = set(split_pos[PEPTIDE_SEQ_COL]) if not split_pos.empty else set()
            split_proteins = set(split_pos[PROTEIN_SEQ_COL]) if not split_pos.empty else set()
            
            # Generate negative pairs for this split
            if not split_pos.empty:
                split_seed = get_fold_seed(seed, fold_idx, {'train': 0, 'val': 1, 'test': 2}[split_name])
                
                # Calculate negative count based on unique positive pairs (not row count)
                n_pos_pairs = split_pos[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates().shape[0]
                logger.info(f"    {split_name}: {n_pos_pairs} unique positive pairs → requesting {n_pos_pairs * negative_ratio} negatives")
                
                # Existing exclude = fold registry (fold-level only)
                existing_exclude = fold_pair_registry
                
                target_neg = n_pos_pairs * negative_ratio
                split_neg = generate_negative_pairs_for_split(
                    n_negative=target_neg,
                    all_peptides=split_peptides,
                    all_proteins=split_proteins,
                    seed=split_seed,
                    global_positive_pairs=global_positive_pairs,
                    existing_negatives=existing_exclude,
                    peptide_records=peptide_records,
                    protein_records=protein_records,
                    logger=logger
                )
                if len(split_neg) < target_neg:
                    raise AssertionError(f"[{split_name}] insufficient negatives in split domain: got {len(split_neg)}/" 
                                         f"{target_neg}. Cannot meet 1:{negative_ratio} pair ratio.")
                
                # Register negative pairs immediately (fold-level)
                neg_keys = [_pair_key(p, r) for p, r in zip(split_neg[PEPTIDE_SEQ_COL], split_neg[PROTEIN_SEQ_COL])]
                fold_pair_registry.update(neg_keys)
                logger.info(f"    {split_name}: Generated {len(split_neg)} negatives (ratio 1:{len(split_neg)/n_pos_pairs:.1f})")
                
                # Combine positive and negative
                split_df = pd.concat([split_pos, split_neg], ignore_index=True)
                
                # Shuffle
                split_df = split_df.sample(frac=1, random_state=split_seed).reset_index(drop=True)
            else:
                split_df = split_pos
            
            # Add fold and split info
            split_df[OUTPUT_FOLD_COL] = fold_idx
            split_df[OUTPUT_SPLIT_COL] = split_name
            
            # Remove temporary columns
            if 'complex_cluster' in split_df.columns:
                split_df = split_df.drop(columns=['complex_cluster'])
            
            fold_data_list.append(split_df)
            
            logger.info(f"  {split_name}: {len(split_df)} total "
                       f"({len(split_pos)} pos, {len(split_df)-len(split_pos)} neg)")
        
        # Add to all data
        all_data.extend(fold_data_list)
        
        # Verify no protein overlap between splits within this fold (should be 0 after pruning)
        fold_df = pd.concat(fold_data_list, ignore_index=True)
        train_df = fold_df[(fold_df[OUTPUT_SPLIT_COL] == SPLIT_TRAIN) & (fold_df[OUTPUT_LABEL_COL] == 1)]
        val_df = fold_df[(fold_df[OUTPUT_SPLIT_COL] == SPLIT_VAL) & (fold_df[OUTPUT_LABEL_COL] == 1)]
        test_df = fold_df[(fold_df[OUTPUT_SPLIT_COL] == SPLIT_TEST) & (fold_df[OUTPUT_LABEL_COL] == 1)]
        
        if not train_df.empty and not val_df.empty:
            train_proteins = set(train_df[OUTPUT_PROTEIN_COL].str.strip().str.upper())
            val_proteins = set(val_df[OUTPUT_PROTEIN_COL].str.strip().str.upper())
            protein_overlap = train_proteins & val_proteins
            assert len(protein_overlap) == 0, f"Fold {fold_idx}: Unexpected protein overlap between train and val after pruning"
        
        if not train_df.empty and not test_df.empty:
            train_proteins = set(train_df[OUTPUT_PROTEIN_COL].str.strip().str.upper())
            test_proteins = set(test_df[OUTPUT_PROTEIN_COL].str.strip().str.upper())
            protein_overlap = train_proteins & test_proteins
            assert len(protein_overlap) == 0, f"Fold {fold_idx}: Unexpected protein overlap between train and test after pruning"
        
        if not val_df.empty and not test_df.empty:
            val_proteins = set(val_df[OUTPUT_PROTEIN_COL].str.strip().str.upper())
            test_proteins = set(test_df[OUTPUT_PROTEIN_COL].str.strip().str.upper())
            protein_overlap = val_proteins & test_proteins
            assert len(protein_overlap) == 0, f"Fold {fold_idx}: Unexpected protein overlap between val and test after pruning"
        
        # Log cluster distribution
        logger.info(f"Fold {fold_idx} cluster distribution:")
        logger.info(f"  Train clusters: {len(fold_cluster_splits[fold_idx][SPLIT_TRAIN])}")
        logger.info(f"  Val clusters: {len(fold_cluster_splits[fold_idx][SPLIT_VAL])}")
        logger.info(f"  Test clusters: {len(fold_cluster_splits[fold_idx][SPLIT_TEST])}")

        # 7:1:2 ratio validation (unique positive pairs)
        fold_pos = df_positive[df_positive['complex_cluster'].isin(
            fold_cluster_splits[fold_idx][SPLIT_TRAIN] | fold_cluster_splits[fold_idx][SPLIT_VAL] | fold_cluster_splits[fold_idx][SPLIT_TEST]
        )][[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL, 'complex_cluster']].drop_duplicates()
        t_cnt = len(fold_pos[fold_pos['complex_cluster'].isin(fold_cluster_splits[fold_idx][SPLIT_TRAIN])][[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates())
        v_cnt = len(fold_pos[fold_pos['complex_cluster'].isin(fold_cluster_splits[fold_idx][SPLIT_VAL])][[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates())
        e_cnt = len(fold_pos[fold_pos['complex_cluster'].isin(fold_cluster_splits[fold_idx][SPLIT_TEST])][[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates())
        total_cnt = max(1, t_cnt + v_cnt + e_cnt)
        logger.info(f"  Split ratio (pairs): train={t_cnt/total_cnt*100:.1f}%, val={v_cnt/total_cnt*100:.1f}%, test={e_cnt/total_cnt*100:.1f}% (target ≈70/10/20)")
    
    # Combine all folds
    logger.info("=" * 60)
    logger.info("Step 3: Combining and saving results")
    logger.info("=" * 60)
    
    result_df = pd.concat(all_data, ignore_index=True)
    
    # Optimize dtypes for memory and disk efficiency
    result_df[OUTPUT_LABEL_COL] = result_df[OUTPUT_LABEL_COL].astype('int8')
    result_df[OUTPUT_FOLD_COL] = result_df[OUTPUT_FOLD_COL].astype('int8')
    if WEIGHT_COL in result_df.columns:
        result_df[WEIGHT_COL] = result_df[WEIGHT_COL].astype('float32')
    for col in [PEPTIDE_LENGTH_COL, RECEPTOR_LENGTH_COL]:
        if col in result_df.columns:
            result_df[col] = result_df[col].astype('int16')
    
    # Reorder columns - ensure weight column is preserved
    original_columns = list(df_all.columns)
    final_columns = original_columns + [OUTPUT_FOLD_COL, OUTPUT_SPLIT_COL]
    result_df = result_df[[col for col in final_columns if col in result_df.columns]]
    
    # Final assertions: verify no duplicate negative pairs within each fold
    for fold_idx in range(n_folds):
        fold_neg = result_df[(result_df[OUTPUT_FOLD_COL] == fold_idx) & 
                             (result_df[OUTPUT_LABEL_COL] == 0)]
        if not fold_neg.empty:
            fold_neg_pairs = fold_neg[[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]]
            assert not fold_neg_pairs.duplicated().any(), \
                f"Duplicate negative pairs within fold {fold_idx}"
    
    # ================================================================
    # STEP 3: Enhanced assertions for both conditions
    # ================================================================
    
    # (A) Verify no intra-fold duplicates (all pairs: positive + negative)
    logger.info("Verifying intra-fold pair uniqueness...")
    for fold_idx in range(n_folds):
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        T = set(zip(fold_data[fold_data[OUTPUT_SPLIT_COL]==SPLIT_TRAIN][OUTPUT_PEPTIDE_COL],
                   fold_data[fold_data[OUTPUT_SPLIT_COL]==SPLIT_TRAIN][OUTPUT_PROTEIN_COL]))
        V = set(zip(fold_data[fold_data[OUTPUT_SPLIT_COL]==SPLIT_VAL][OUTPUT_PEPTIDE_COL],
                   fold_data[fold_data[OUTPUT_SPLIT_COL]==SPLIT_VAL][OUTPUT_PROTEIN_COL]))
        E = set(zip(fold_data[fold_data[OUTPUT_SPLIT_COL]==SPLIT_TEST][OUTPUT_PEPTIDE_COL],
                   fold_data[fold_data[OUTPUT_SPLIT_COL]==SPLIT_TEST][OUTPUT_PROTEIN_COL]))
        
        assert not (T & V), f"Fold {fold_idx}: Train-Val overlap detected!"
        assert not (T & E), f"Fold {fold_idx}: Train-Test overlap detected!"
        assert not (V & E), f"Fold {fold_idx}: Val-Test overlap detected!"
    logger.info("  No intra-fold pair duplicates")
    
    # (B) Verify no inter-fold test duplicates (positive pairs)
    logger.info("Verifying inter-fold test uniqueness...")
    test_overlaps = []
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            Ti = set(zip(
                result_df[(result_df[OUTPUT_FOLD_COL]==i)&(result_df[OUTPUT_SPLIT_COL]==SPLIT_TEST)&(result_df[OUTPUT_LABEL_COL]==1)][OUTPUT_PEPTIDE_COL],
                result_df[(result_df[OUTPUT_FOLD_COL]==i)&(result_df[OUTPUT_SPLIT_COL]==SPLIT_TEST)&(result_df[OUTPUT_LABEL_COL]==1)][OUTPUT_PROTEIN_COL]
            ))
            Tj = set(zip(
                result_df[(result_df[OUTPUT_FOLD_COL]==j)&(result_df[OUTPUT_SPLIT_COL]==SPLIT_TEST)&(result_df[OUTPUT_LABEL_COL]==1)][OUTPUT_PEPTIDE_COL],
                result_df[(result_df[OUTPUT_FOLD_COL]==j)&(result_df[OUTPUT_SPLIT_COL]==SPLIT_TEST)&(result_df[OUTPUT_LABEL_COL]==1)][OUTPUT_PROTEIN_COL]
            ))
            overlap = Ti & Tj
            if overlap:
                test_overlaps.append((i, j, len(overlap)))
    
    if test_overlaps:
        raise AssertionError(f"Test sets overlap across folds: {test_overlaps}")
    logger.info("  No inter-fold test duplicates")
    
    # Safety assertion: Check no positive-negative collision within each fold
    logger.info("Checking for positive-negative collisions within folds...")
    for f in range(n_folds):
        fold_df = result_df[result_df[OUTPUT_FOLD_COL] == f]
        pos = set(zip(
            fold_df[fold_df[OUTPUT_LABEL_COL]==1][OUTPUT_PEPTIDE_COL],
            fold_df[fold_df[OUTPUT_LABEL_COL]==1][OUTPUT_PROTEIN_COL]
        ))
        neg = set(zip(
            fold_df[fold_df[OUTPUT_LABEL_COL]==0][OUTPUT_PEPTIDE_COL],
            fold_df[fold_df[OUTPUT_LABEL_COL]==0][OUTPUT_PROTEIN_COL]
        ))
        assert len(pos & neg) == 0, f"Fold {f}: positive/negative collision detected"
    logger.info("  No positive-negative collisions within any fold")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved to: {output_path}")
    
    # Save clustering info (to log directory for consistency)
    cluster_info_file = Path(log_dir) / f"propedia_v2_ppi_clustering_info_{n_folds}fold_acsm.pkl"
    cluster_info = {
        'pdb_to_cluster': pdb_to_cluster,
        'fold_cluster_splits': fold_cluster_splits,
        'n_complex_clusters': len(unique_cluster_ids),
        'n_folds': n_folds,
        'seed': seed
    }
    with open(cluster_info_file, 'wb') as f:
        pickle.dump(cluster_info, f)
    logger.info(f"Saved clustering info to: {cluster_info_file}")
    
    # Also save human-readable split info in JSON format
    import json
    split_info_file = Path(log_dir) / f"propedia_v2_ppi_split_info_{n_folds}fold_acsm.json"
    split_info = {
        'method': 'acsm_clustering',
        'n_folds': n_folds,
        'seed': seed,
        'negative_ratio': negative_ratio,
        'n_complex_clusters': len(unique_cluster_ids),
        'unique_pdbs': len(unique_pdb_ids),
        'dimensionality_reduction': DIMENSIONALITY_REDUCTION,
        'dim_reduction_components': DIM_REDUCTION_COMPONENTS,
        'optimize_clusters': OPTIMIZE_CLUSTERS,
        'folds': {}
    }
    
    # Store cluster assignments for each fold
    for fold_idx in range(n_folds):
        split_info['folds'][fold_idx] = {
            'train_clusters': sorted(list(fold_cluster_splits[fold_idx][SPLIT_TRAIN])),
            'val_clusters': sorted(list(fold_cluster_splits[fold_idx][SPLIT_VAL])),
            'test_clusters': sorted(list(fold_cluster_splits[fold_idx][SPLIT_TEST])),
            'fold_seed': get_fold_seed(seed, fold_idx)
        }
    
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Saved split info to: {split_info_file}")
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("Final Summary:")
    logger.info("=" * 60)
    logger.info(f"Total entries: {len(result_df)}")
    logger.info(f"Unique peptides: {result_df[OUTPUT_PEPTIDE_COL].nunique()}")
    logger.info(f"Unique proteins: {result_df[OUTPUT_PROTEIN_COL].nunique()}")
    
    # Calculate and log overlap statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("Overlap Analysis between Train/Val/Test splits:")
    logger.info("=" * 60)
    
    for fold_idx in range(n_folds):
        logger.info(f"\nFold {fold_idx} overlap analysis:")
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        
        # Get unique peptides and proteins for each split
        train_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TRAIN]
        val_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_VAL]
        test_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TEST]
        
        train_peptides = set(train_data[OUTPUT_PEPTIDE_COL])
        val_peptides = set(val_data[OUTPUT_PEPTIDE_COL])
        test_peptides = set(test_data[OUTPUT_PEPTIDE_COL])
        
        train_proteins = set(train_data[OUTPUT_PROTEIN_COL])
        val_proteins = set(val_data[OUTPUT_PROTEIN_COL])
        test_proteins = set(test_data[OUTPUT_PROTEIN_COL])
        
        # Calculate peptide overlaps
        train_val_pep_overlap = len(train_peptides & val_peptides)
        train_test_pep_overlap = len(train_peptides & test_peptides)
        val_test_pep_overlap = len(val_peptides & test_peptides)
        
        # Calculate protein overlaps
        train_val_prot_overlap = len(train_proteins & val_proteins)
        train_test_prot_overlap = len(train_proteins & test_proteins)
        val_test_prot_overlap = len(val_proteins & test_proteins)
        
        # Calculate percentages
        train_val_pep_pct = (train_val_pep_overlap / len(val_peptides) * 100) if len(val_peptides) > 0 else 0
        train_test_pep_pct = (train_test_pep_overlap / len(test_peptides) * 100) if len(test_peptides) > 0 else 0
        val_test_pep_pct = (val_test_pep_overlap / len(test_peptides) * 100) if len(test_peptides) > 0 else 0
        
        train_val_prot_pct = (train_val_prot_overlap / len(val_proteins) * 100) if len(val_proteins) > 0 else 0
        train_test_prot_pct = (train_test_prot_overlap / len(test_proteins) * 100) if len(test_proteins) > 0 else 0
        val_test_prot_pct = (val_test_prot_overlap / len(test_proteins) * 100) if len(test_proteins) > 0 else 0
        
        logger.info(f"  Peptide overlaps:")
        logger.info(f"    Train-Val: {train_val_pep_overlap}/{len(val_peptides)} ({train_val_pep_pct:.1f}% of val)")
        logger.info(f"    Train-Test: {train_test_pep_overlap}/{len(test_peptides)} ({train_test_pep_pct:.1f}% of test)")
        logger.info(f"    Val-Test: {val_test_pep_overlap}/{len(test_peptides)} ({val_test_pep_pct:.1f}% of test)")
        
        logger.info(f"  Protein overlaps:")
        logger.info(f"    Train-Val: {train_val_prot_overlap}/{len(val_proteins)} ({train_val_prot_pct:.1f}% of val)")
        logger.info(f"    Train-Test: {train_test_prot_overlap}/{len(test_proteins)} ({train_test_prot_pct:.1f}% of test)")
        logger.info(f"    Val-Test: {val_test_prot_overlap}/{len(test_proteins)} ({val_test_prot_pct:.1f}% of test)")
    
    # Calculate and log average overlaps across all folds
    logger.info("")
    logger.info("=" * 60)
    logger.info("Average overlap across all folds:")
    logger.info("=" * 60)
    
    # Store overlap percentages for averaging
    train_val_pep_pcts = []
    train_test_pep_pcts = []
    val_test_pep_pcts = []
    train_val_prot_pcts = []
    train_test_prot_pcts = []
    val_test_prot_pcts = []
    
    for fold_idx in range(n_folds):
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        
        train_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TRAIN]
        val_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_VAL]
        test_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TEST]
        
        train_peptides = set(train_data[OUTPUT_PEPTIDE_COL])
        val_peptides = set(val_data[OUTPUT_PEPTIDE_COL])
        test_peptides = set(test_data[OUTPUT_PEPTIDE_COL])
        
        train_proteins = set(train_data[OUTPUT_PROTEIN_COL])
        val_proteins = set(val_data[OUTPUT_PROTEIN_COL])
        test_proteins = set(test_data[OUTPUT_PROTEIN_COL])
        
        # Calculate overlaps
        if len(val_peptides) > 0:
            train_val_pep_pcts.append(len(train_peptides & val_peptides) / len(val_peptides) * 100)
        if len(test_peptides) > 0:
            train_test_pep_pcts.append(len(train_peptides & test_peptides) / len(test_peptides) * 100)
            val_test_pep_pcts.append(len(val_peptides & test_peptides) / len(test_peptides) * 100)
        
        if len(val_proteins) > 0:
            train_val_prot_pcts.append(len(train_proteins & val_proteins) / len(val_proteins) * 100)
        if len(test_proteins) > 0:
            train_test_prot_pcts.append(len(train_proteins & test_proteins) / len(test_proteins) * 100)
            val_test_prot_pcts.append(len(val_proteins & test_proteins) / len(test_proteins) * 100)
    
    # Helper function to safely compute mean and std
    def safe_mean_std(arr):
        if arr:
            return float(np.mean(arr)), float(np.std(arr))
        return 0.0, 0.0
    
    # Calculate and display averages
    logger.info("  Average peptide overlaps:")
    m, s = safe_mean_std(train_val_pep_pcts)
    logger.info(f"    Train-Val: {m:.1f}% ± {s:.1f}%")
    m, s = safe_mean_std(train_test_pep_pcts)
    logger.info(f"    Train-Test: {m:.1f}% ± {s:.1f}%")
    m, s = safe_mean_std(val_test_pep_pcts)
    logger.info(f"    Val-Test: {m:.1f}% ± {s:.1f}%")
    
    logger.info("  Average protein overlaps:")
    m, s = safe_mean_std(train_val_prot_pcts)
    logger.info(f"    Train-Val: {m:.1f}% ± {s:.1f}%")
    m, s = safe_mean_std(train_test_prot_pcts)
    logger.info(f"    Train-Test: {m:.1f}% ± {s:.1f}%")
    m, s = safe_mean_std(val_test_prot_pcts)
    logger.info(f"    Val-Test: {m:.1f}% ± {s:.1f}%")
    
    # Pair-level distribution and ratio
    logger.info("")
    logger.info("Pair-level distribution (unique pairs only):")
    pos_pairs = result_df[result_df[OUTPUT_LABEL_COL] == 1][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
    neg_pairs = result_df[result_df[OUTPUT_LABEL_COL] == 0][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
    pos_pair_cnt = len(pos_pairs)
    neg_pair_cnt = len(neg_pairs)
    logger.info(f"  Positive pairs: {pos_pair_cnt}")
    logger.info(f"  Negative pairs: {neg_pair_cnt}")
    if pos_pair_cnt > 0:
        logger.info(f"  Positive:Negative (pairs) = 1:{neg_pair_cnt/pos_pair_cnt:.1f}")
    
    # Also show fold-by-fold breakdown
    logger.info("\nFold-by-fold unique pair ratios:")
    for fold_idx in range(n_folds):
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        fold_pos_pairs = fold_data[fold_data[OUTPUT_LABEL_COL] == 1][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
        fold_neg_pairs = fold_data[fold_data[OUTPUT_LABEL_COL] == 0][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
        fold_pos_cnt = len(fold_pos_pairs)
        fold_neg_cnt = len(fold_neg_pairs)
        fold_ratio = (fold_neg_cnt / fold_pos_cnt) if fold_pos_cnt > 0 else 0
        logger.info(f"  Fold {fold_idx}: {fold_pos_cnt} pos, {fold_neg_cnt} neg → 1:{fold_ratio:.1f}")

    # Per-split pair ratios per fold
    logger.info("\nFold-by-fold per-split unique pair ratios:")
    for fold_idx in range(n_folds):
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        msg_parts = []
        for split_name in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
            split_pos_pairs = fold_data[(fold_data[OUTPUT_SPLIT_COL] == split_name) & (fold_data[OUTPUT_LABEL_COL] == 1)][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
            split_neg_pairs = fold_data[(fold_data[OUTPUT_SPLIT_COL] == split_name) & (fold_data[OUTPUT_LABEL_COL] == 0)][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
            sp = len(split_pos_pairs); sn = len(split_neg_pairs)
            sr = (sn / sp) if sp > 0 else 0
            msg_parts.append(f"{split_name}: {sp} pos, {sn} neg → 1:{sr:.1f}")
        logger.info(f"  Fold {fold_idx}: " + "; ".join(msg_parts))

    # Row-level label distribution (canonical target 1:4)
    logger.info("")
    logger.info("Label distribution (all rows):")
    label_dist = result_df[OUTPUT_LABEL_COL].value_counts()
    pos_rows = int(label_dist.get(1, 0))
    neg_rows = int(label_dist.get(0, 0))
    logger.info(f"  Positive (1): {pos_rows}")
    logger.info(f"  Negative (0): {neg_rows}")
    if pos_rows > 0:
        logger.info(f"  Positive:Negative (rows) = 1:{neg_rows/pos_rows:.1f}")
    
    # Verify cluster and PDB splits
    logger.info("")
    logger.info("Verifying cluster-based and PDB-based splits...")
    
    for fold_idx in range(n_folds):
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        
        # Check for cluster overlap between splits
        splits = {}
        pdb_splits = {}
        for split_name in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
            split_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == split_name]
            if not split_data.empty and OUTPUT_LABEL_COL in split_data.columns:
                positive_data = split_data[split_data[OUTPUT_LABEL_COL] == 1]
                
                # Get unique PDB IDs and their clusters
                clusters_in_split = set()
                pdbs_in_split = set()
                for _, row in positive_data.iterrows():
                    pdb_id = row[PDB_COL]
                    if pdb_id != 'NEGATIVE':
                        pdbs_in_split.add(pdb_id)
                        if pdb_id in pdb_to_cluster:
                            clusters_in_split.add(pdb_to_cluster[pdb_id])
                
                splits[split_name] = clusters_in_split
                pdb_splits[split_name] = pdbs_in_split
        
        # Check cluster overlaps
        if len(splits) == 3:
            train_val_overlap = splits[SPLIT_TRAIN] & splits[SPLIT_VAL]
            train_test_overlap = splits[SPLIT_TRAIN] & splits[SPLIT_TEST]
            val_test_overlap = splits[SPLIT_VAL] & splits[SPLIT_TEST]
            
            if train_val_overlap or train_test_overlap or val_test_overlap:
                logger.warning(f"Fold {fold_idx}: Cluster overlap detected!")
                if train_val_overlap:
                    logger.warning(f"  Train-Val: {len(train_val_overlap)} clusters")
                if train_test_overlap:
                    logger.warning(f"  Train-Test: {len(train_test_overlap)} clusters")
                if val_test_overlap:
                    logger.warning(f"  Val-Test: {len(val_test_overlap)} clusters")
            else:
                logger.info(f"Fold {fold_idx}: No cluster overlap between splits")
        
        # Check PDB overlaps
        if len(pdb_splits) == 3:
            pdb_train_val = pdb_splits[SPLIT_TRAIN] & pdb_splits[SPLIT_VAL]
            pdb_train_test = pdb_splits[SPLIT_TRAIN] & pdb_splits[SPLIT_TEST]
            pdb_val_test = pdb_splits[SPLIT_VAL] & pdb_splits[SPLIT_TEST]
            
            if pdb_train_val or pdb_train_test or pdb_val_test:
                logger.error(f"Fold {fold_idx}: PDB overlap detected!")
                if pdb_train_val:
                    logger.error(f"  Train-Val: {len(pdb_train_val)} PDBs")
                if pdb_train_test:
                    logger.error(f"  Train-Test: {len(pdb_train_test)} PDBs")
                if pdb_val_test:
                    logger.error(f"  Val-Test: {len(pdb_val_test)} PDBs")
            else:
                logger.info(f"Fold {fold_idx}: No PDB overlap between splits ")
        
        # Check pair overlaps (including all data - positives and negatives)
        train_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TRAIN]
        val_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_VAL]
        test_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TEST]
        
        train_pairs = set(zip(train_data[OUTPUT_PEPTIDE_COL], train_data[OUTPUT_PROTEIN_COL]))
        val_pairs = set(zip(val_data[OUTPUT_PEPTIDE_COL], val_data[OUTPUT_PROTEIN_COL]))
        test_pairs = set(zip(test_data[OUTPUT_PEPTIDE_COL], test_data[OUTPUT_PROTEIN_COL]))
        
        # Check overlaps
        train_val_overlap = train_pairs & val_pairs
        train_test_overlap = train_pairs & test_pairs
        val_test_overlap = val_pairs & test_pairs
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            error_msg = f"Fold {fold_idx}: Pair overlap detected within fold!\n"
            if train_val_overlap:
                error_msg += f"  Train-Val overlap: {len(train_val_overlap)} pairs\n"
            if train_test_overlap:
                error_msg += f"  Train-Test overlap: {len(train_test_overlap)} pairs\n"
            if val_test_overlap:
                error_msg += f"  Val-Test overlap: {len(val_test_overlap)} pairs"
            raise AssertionError(error_msg)
        else:
            logger.info(f"Fold {fold_idx}: No pair overlap within fold")
    
    # Split distribution by fold (counts)
    logger.info("")
    logger.info("Split distribution by fold:")
    split_stats = result_df.groupby([OUTPUT_FOLD_COL, OUTPUT_SPLIT_COL, OUTPUT_LABEL_COL]).size().unstack()
    logger.info("\n" + split_stats.to_string())

    logger.info("=" * 60)
    logger.info("RESULT: aCSM clustering k-fold split completed successfully")
    logger.info("=" * 60)
    
    return result_df


def main() -> None:
    global SIGNATURE_DIR
    global DEFAULT_LOG_DIR
    global logger, log_dir

    parser = argparse.ArgumentParser(description='Prepare Propedia v2 PPI k-fold (aCSM clustering)')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument('--signatures', type=str, default=SIGNATURE_DIR,
                        help='Directory containing aCSM signatures (complex_signatures_acsm_all.csv)')
    parser.add_argument('--log-dir', type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument('--folds', type=int, default=N_FOLDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--negative-ratio', type=int, default=NEGATIVE_RATIO)
    parser.add_argument('--clusters', type=int, default=N_COMPLEX_CLUSTERS)
    args = parser.parse_args()

    # Override paths
    SIGNATURE_DIR = args.signatures
    DEFAULT_LOG_DIR = Path(args.log_dir)
    # Setup logging inside main to avoid side effects on import
    logger, log_dir = setup_logging()
    
    # Seed everything for reproducibility
    L.seed_everything(args.seed, workers=True)
    
    logger.info(f"Log directory: {log_dir}")
    logger.info("==================================================")
    logger.info("Propedia v2 PPI K-fold Preparation")
    logger.info("==================================================")
    logger.info(f"Source file: {args.source}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Number of folds: {args.folds}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Method: aCSM clustering")
    logger.info(f"Evaluation mode: Strict (no intra/inter-fold test duplicates)")
    test_ratio = 1.0 / args.folds
    val_ratio = (1 - test_ratio) * VAL_RATIO_OF_TRAINVAL
    train_ratio = (1 - test_ratio) * (1 - VAL_RATIO_OF_TRAINVAL)
    logger.info(f"{args.folds}-fold CV split (7:1:2 ratio): Each fold uses test≈{test_ratio:.1%}, train≈{train_ratio:.1%}, val≈{val_ratio:.1%}")
    logger.info(f"Negative ratio: 1:{args.negative_ratio} (positive:negative)")
    logger.info("==================================================")
    
    prepare_propedia_v2_kfold(
        source_file=args.source,
        output_file=args.output,
        n_folds=args.folds,
        seed=args.seed,
        n_complex_clusters=args.clusters,
        negative_ratio=args.negative_ratio
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error(f"Script failed with error: {str(e)}")
        logging.getLogger(__name__).error("Full traceback:", exc_info=True)
        sys.exit(1)
