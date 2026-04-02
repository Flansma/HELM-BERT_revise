"""K-fold data preparation for Propedia v2 with pair-based grouping.

This script creates k-fold splits ensuring no pair duplication:
1. Group all PDBs that share the same (peptide, protein) pair
2. Randomly distribute these pair-groups into 5 folds
3. For each fold: test=groups in fold; train/val=groups in other folds (4:1 ratio)
4. Generate negative pairs within each split using only that split's sequences
5. Ensure no pair duplication within each fold (train/val/test), and no TEST-TEST duplication across folds

Key features:
- Pair-based grouping ensures complete pair isolation
- All PDBs with same pair stay together (handles oligomeric states)
- Random distribution of pair-groups for balanced folds
- Negatives generated within each split independently
- No data leakage between folds

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
from typing import Set, Tuple, List
from datetime import datetime
import sys
import argparse
import lightning as L


# Configuration (repo-relative defaults; CLI-overridable)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_FILE = REPO_ROOT / 'local_data/intermediate_product/Propedia_v2_unique_ppi_HELM_SMILES.csv'
DEFAULT_OUTPUT_FILE = REPO_ROOT / 'data/downstream/Propedia_v2_ppi_5fold_random.csv'
DEFAULT_LOG_DIR = REPO_ROOT / 'outputs/preprocessing'
LOG_FILE_NAME = 'prepare_kfold_ppi_random.log'

N_FOLDS = 5
SEED = 42
NEGATIVE_RATIO = 4  # Ratio of negative to positive pairs (1:4)

# Split ratio for train/val within non-test data
# In 5-fold CV: test=20% (1 fold), train+val=80% (4 folds)
# Within the 80%, split 7:1:2 → train=70%, val=10% (of total)
VAL_RATIO_OF_TRAINVAL = 0.125  # 10% of total / 80% non-test = 12.5% of non-test

# Column names in source file
PEPTIDE_SEQ_COL = 'Peptide_Sequence'
PROTEIN_SEQ_COL = 'Receptor_Sequence'
PDB_COL = 'PDB'
PEPTIDE_CHAIN_COL = 'Peptide_Chain'
PROTEIN_CHAIN_COL = 'Receptor_Chain'

# Additional column names from source
PEPTIDE_HELM_COL = 'Peptide_HELM'
PEPTIDE_SMILES_COL = 'Peptide_SMILES'
PEPTIDE_LENGTH_COL = 'Peptide_Length'
RECEPTOR_LENGTH_COL = 'Receptor_Length'
COMPLEX_FILE_COL = 'Complex_File'
WEIGHT_COL = 'weight'  # Weight column from oligomeric state processing

# Output column names
OUTPUT_PEPTIDE_COL = 'Peptide_Sequence'
OUTPUT_PROTEIN_COL = 'Receptor_Sequence'
OUTPUT_LABEL_COL = 'Label'
OUTPUT_FOLD_COL = 'fold'
OUTPUT_SPLIT_COL = 'split'

# Split types
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


def setup_logging() -> Tuple[logging.Logger, Path]:
    """Set up logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(DEFAULT_LOG_DIR)
    # Standardize: <dataset_name>_<method>_<timestamp>
    log_dir = base / f"propedia_v2_ppi_random_{timestamp}"
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
                                    protein_records: dict) -> pd.DataFrame:
    """Generate negative pairs for a single split with global tracking."""
    logger.debug(f"Generating {n_negative} negative pairs with seed {seed}")
    
    # Calculate maximum available negatives
    pos_in_split = {(p, r) for (p, r) in global_positive_pairs 
                    if p in all_peptides and r in all_proteins}
    neg_blocked_in_split = {(p, r) for (p, r) in existing_negatives 
                           if p in all_peptides and r in all_proteins}
    total_pairs = len(all_peptides) * len(all_proteins)
    max_available = total_pairs - len(pos_in_split) - len(neg_blocked_in_split)
    
    if n_negative > max_available:
        logger.warning(f"Requested {n_negative} negatives but only {max_available} available; capping")
        n_negative = min(n_negative, max_available)
    
    if n_negative <= 0:
        logger.warning("No negative pairs can be generated for this split")
        return create_negative_pairs_dataframe([], [], peptide_records, protein_records)
    
    # All excluded pairs
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
        # Generate random indices
        peptide_indices = rng.integers(0, len(peptides_array), batch_size)
        protein_indices = rng.integers(0, len(proteins_array), batch_size)
        
        # Create candidate pairs
        candidate_peptides = peptides_array[peptide_indices]
        candidate_proteins = proteins_array[protein_indices]
        
        # Filter valid pairs
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
    
    # Extract sequences
    peptide_sequences = [p[0] for p in negative_pairs[:n_negative]]
    protein_sequences = [p[1] for p in negative_pairs[:n_negative]]
    
    # Create DataFrame
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
    negative_ratio: int = 4,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """Create k-fold splits with pre-generated negatives."""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
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
    
    # Create global positive pairs set
    global_positive_pairs = set(zip(df_positive[PEPTIDE_SEQ_COL], df_positive[PROTEIN_SEQ_COL]))
    
    # Get all unique sequences
    all_peptides = set(df_positive[PEPTIDE_SEQ_COL].unique())
    all_proteins = set(df_positive[PROTEIN_SEQ_COL].unique())
    
    # Build complete property mappings (vectorized for speed)
    logger.info("Building complete property mappings...")
    
    # Vectorized peptide records build
    pep_cols = [PEPTIDE_SEQ_COL, PEPTIDE_HELM_COL, PEPTIDE_SMILES_COL, PEPTIDE_CHAIN_COL, PEPTIDE_LENGTH_COL]
    peptide_records = (df_positive[pep_cols]
                      .drop_duplicates(PEPTIDE_SEQ_COL)
                      .set_index(PEPTIDE_SEQ_COL)
                      .to_dict('index'))
    
    # Vectorized protein records build
    prot_cols = [PROTEIN_SEQ_COL, PROTEIN_CHAIN_COL, RECEPTOR_LENGTH_COL]
    protein_records = (df_positive[prot_cols]
                      .drop_duplicates(PROTEIN_SEQ_COL)
                      .set_index(PROTEIN_SEQ_COL)
                      .to_dict('index'))
    
    logger.info(f"Built complete records for {len(peptide_records)} peptides and {len(protein_records)} proteins")
    
    # Step 1: Group PDBs by pairs
    logger.info("=" * 60)
    logger.info("Step 1: Grouping PDBs by (peptide, protein) pairs")
    logger.info("=" * 60)
    
    from collections import defaultdict
    
    # Build pair to PDBs mapping
    pair_to_pdbs = defaultdict(set)
    pair_to_rows = defaultdict(int)  # Count rows per pair for balancing
    
    for _, row in df_positive.iterrows():
        pair = (row[PEPTIDE_SEQ_COL], row[PROTEIN_SEQ_COL])
        pdb = row[PDB_COL]
        pair_to_pdbs[pair].add(pdb)
        pair_to_rows[pair] += 1
    
    unique_pairs = list(pair_to_pdbs.keys())
    logger.info(f"Unique pairs: {len(unique_pairs)}")
    logger.info(f"Unique PDBs: {df_positive[PDB_COL].nunique()}")
    
    # Step 2: Distribute pair-groups into folds
    logger.info("=" * 60)
    logger.info("Step 2: Distributing pair-groups into k folds")
    logger.info("=" * 60)
    
    # Shuffle pairs randomly
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pairs)
    
    # Balance folds by unique pair counts (not row counts)
    fold_pair_sets = [set() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds  # track number of pairs per fold
    
    for pair in unique_pairs:
        min_fold_idx = int(np.argmin(fold_sizes))
        fold_pair_sets[min_fold_idx].add(pair)
        fold_sizes[min_fold_idx] += 1
    
    # Verify no pair overlaps between folds
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            overlap = fold_pair_sets[i] & fold_pair_sets[j]
            if overlap:
                raise AssertionError(f"Pair overlap between fold {i} and {j}: {len(overlap)} pairs")
    
    # Log fold distribution
    for fold_idx in range(n_folds):
        pairs_in_fold = fold_pair_sets[fold_idx]
        pdbs_in_fold = set()
        rows_in_fold = 0
        for pair in pairs_in_fold:
            pdbs_in_fold.update(pair_to_pdbs[pair])
            rows_in_fold += pair_to_rows[pair]
        logger.info(f"Fold {fold_idx}: {len(pairs_in_fold)} pairs, {len(pdbs_in_fold)} PDBs, {rows_in_fold} rows")
    
    # Build output dataframe
    all_data = []
    
    # no global negative dedup – fold-level only
    
    total_possible_pairs = len(all_peptides) * len(all_proteins)
    total_possible_negatives = total_possible_pairs - len(global_positive_pairs)
    
    # Report totals clearly
    logger.info(f"Total positive rows: {len(df_positive)}")
    logger.info(f"Unique positive pairs: {len(unique_pairs)}")
    logger.info(f"Total possible negative pairs: {total_possible_negatives}")
    
    # Step 3: Process each fold
    logger.info("=" * 60)
    logger.info("Step 3: Processing each fold")
    logger.info("=" * 60)
    
    # Process each fold using pair-based assignments
    for fold_idx in range(n_folds):
        logger.info("")
        logger.info(f"Processing fold {fold_idx}...")
        
        # Get pairs for test fold
        test_pairs = fold_pair_sets[fold_idx]
        
        # Get all test PDBs
        test_pdbs = set()
        for pair in test_pairs:
            test_pdbs.update(pair_to_pdbs[pair])
        
        # Get train/val pairs (all other folds)
        train_val_pairs = []
        for other_fold in range(n_folds):
            if other_fold != fold_idx:
                train_val_pairs.extend(fold_pair_sets[other_fold])
        
        # Split train/val pairs based on pair count (not row count)
        rng_fold = np.random.default_rng(get_fold_seed(seed, fold_idx))
        train_val_pairs = list(train_val_pairs)
        rng_fold.shuffle(train_val_pairs)
        
        target_val = int(len(train_val_pairs) * VAL_RATIO_OF_TRAINVAL)
        val_pairs = set(train_val_pairs[:target_val])
        train_pairs = set(train_val_pairs[target_val:])
        
        # Ensure at least one pair in val if possible
        if not val_pairs and train_val_pairs:
            first_pair = train_val_pairs[0]
            val_pairs.add(first_pair)
            train_pairs.discard(first_pair)
        
        # Get PDBs for train/val
        train_pdbs = set()
        val_pdbs = set()
        for pair in train_pairs:
            train_pdbs.update(pair_to_pdbs[pair])
        for pair in val_pairs:
            val_pdbs.update(pair_to_pdbs[pair])
        
        logger.info(f"Fold {fold_idx} distribution:")
        logger.info(f"  Test: {len(test_pairs)} pairs, {len(test_pdbs)} PDBs")
        logger.info(f"  Train: {len(train_pairs)} pairs, {len(train_pdbs)} PDBs")
        logger.info(f"  Val: {len(val_pairs)} pairs, {len(val_pdbs)} PDBs")
        total_pairs_fold = max(1, len(test_pairs) + len(train_pairs) + len(val_pairs))
        logger.info(f"  Split ratio (pairs): train={len(train_pairs)/total_pairs_fold*100:.1f}%, val={len(val_pairs)/total_pairs_fold*100:.1f}%, test={len(test_pairs)/total_pairs_fold*100:.1f}% (target ≈70/10/20)")
        
        
        # Get positive pairs based on pair assignment (not PDB!)
        # This ensures exact pair isolation between folds
        test_pair_df = pd.DataFrame(list(test_pairs), columns=[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL])
        test_pos = df_positive.merge(test_pair_df, on=[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL], how="inner")
        
        train_pair_df = pd.DataFrame(list(train_pairs), columns=[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL])
        train_pos = df_positive.merge(train_pair_df, on=[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL], how="inner")
        
        val_pair_df = pd.DataFrame(list(val_pairs), columns=[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL])
        val_pos = df_positive.merge(val_pair_df, on=[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL], how="inner")
        
        logger.info(f"Fold {fold_idx} positive distribution:")
        logger.info(f"  Train: {len(train_pos)} positives")
        logger.info(f"  Val:   {len(val_pos)} positives")
        logger.info(f"  Test:  {len(test_pos)} positives")
        
        # Get unique peptides and proteins for each split
        train_peptides = set(train_pos[PEPTIDE_SEQ_COL])
        train_proteins = set(train_pos[PROTEIN_SEQ_COL])
        val_peptides = set(val_pos[PEPTIDE_SEQ_COL])
        val_proteins = set(val_pos[PROTEIN_SEQ_COL])
        test_peptides = set(test_pos[PEPTIDE_SEQ_COL])
        test_proteins = set(test_pos[PROTEIN_SEQ_COL])
        
        # Generate negatives for each split using SPLIT-SPECIFIC peptide/protein pools
        logger.info(f"Generating negative pairs for fold {fold_idx}...")
        
        # Calculate negative counts based on unique positive pairs (1:4 ratio per split)
        n_train_pairs = train_pos[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates().shape[0] if not train_pos.empty else 0
        n_val_pairs = val_pos[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates().shape[0] if not val_pos.empty else 0
        n_test_pairs = test_pos[[PEPTIDE_SEQ_COL, PROTEIN_SEQ_COL]].drop_duplicates().shape[0] if not test_pos.empty else 0
        
        logger.info(
            f"  Negative generation pools: "
            f"train={len(train_peptides)}x{len(train_proteins)}, "
            f"val={len(val_peptides)}x{len(val_proteins)}, "
            f"test={len(test_peptides)}x{len(test_proteins)}"
        )
        logger.info(f"  Target negative pairs: train={n_train_pairs * negative_ratio}, val={n_val_pairs * negative_ratio}, test={n_test_pairs * negative_ratio}")
        
        # Build fold-level pools for fallback
        fold_all_peptides = train_peptides | val_peptides | test_peptides
        fold_all_proteins = train_proteins | val_proteins | test_proteins

        # Generate fold-level negatives once, then split by target counts
        fold_negatives = set()
        fold_seed = get_fold_seed(seed, fold_idx, 99)
        test_target = n_test_pairs * negative_ratio
        val_target = n_val_pairs * negative_ratio
        train_target = n_train_pairs * negative_ratio
        total_target = test_target + val_target + train_target

        fold_neg_df = generate_negative_pairs_for_split(
            n_negative=total_target,
            all_peptides=fold_all_peptides,
            all_proteins=fold_all_proteins,
            seed=fold_seed,
            global_positive_pairs=global_positive_pairs,
            existing_negatives=fold_negatives,
            peptide_records=peptide_records,
            protein_records=protein_records
        )
        if len(fold_neg_df) < total_target:
            raise AssertionError(
                f"Fold {fold_idx}: insufficient negatives in fold domain: got {len(fold_neg_df)}/{total_target}."
            )

        # Partition deterministically: test -> val -> train
        fold_neg_df = fold_neg_df.sample(frac=1, random_state=fold_seed).reset_index(drop=True)
        test_neg = fold_neg_df.iloc[:test_target].copy()
        val_neg = fold_neg_df.iloc[test_target:test_target+val_target].copy()
        train_neg = fold_neg_df.iloc[test_target+val_target:test_target+val_target+train_target].copy()

        test_neg_pairs = set(zip(test_neg[PEPTIDE_SEQ_COL], test_neg[PROTEIN_SEQ_COL]))
        val_neg_pairs = set(zip(val_neg[PEPTIDE_SEQ_COL], val_neg[PROTEIN_SEQ_COL]))
        train_neg_pairs = set(zip(train_neg[PEPTIDE_SEQ_COL], train_neg[PROTEIN_SEQ_COL]))
        assert (
            not (test_neg_pairs & val_neg_pairs)
            and not (test_neg_pairs & train_neg_pairs)
            and not (val_neg_pairs & train_neg_pairs)
        ), f"Fold {fold_idx}: negative pair overlap after partitioning"
        logger.info(f"  Generated fold negatives: test={len(test_neg)}, val={len(val_neg)}, train={len(train_neg)}")

        # Define per-split seeds for shuffling consistency
        train_seed = get_fold_seed(seed, fold_idx, 0)
        val_seed = get_fold_seed(seed, fold_idx, 1)
        test_seed = get_fold_seed(seed, fold_idx, 2)
        
        # === VERIFICATION ===
        # Verify no negative pair overlap between splits within this fold
        train_val_neg_overlap = train_neg_pairs & val_neg_pairs
        train_test_neg_overlap = train_neg_pairs & test_neg_pairs
        val_test_neg_overlap = val_neg_pairs & test_neg_pairs
        
        if train_val_neg_overlap or train_test_neg_overlap or val_test_neg_overlap:
            error_msg = f"Fold {fold_idx}: Negative pair overlap detected between splits!"
            if train_val_neg_overlap:
                error_msg += f"\n  Train-Val negative overlap: {len(train_val_neg_overlap)} pairs"
            if train_test_neg_overlap:
                error_msg += f"\n  Train-Test negative overlap: {len(train_test_neg_overlap)} pairs"
            if val_test_neg_overlap:
                error_msg += f"\n  Val-Test negative overlap: {len(val_test_neg_overlap)} pairs"
            raise AssertionError(error_msg)
        
        logger.info(f"  ✓ No negative pair overlap between splits in fold {fold_idx}")
        
        # === RATIO VERIFICATION ===
        actual_train_ratio = len(train_neg_pairs) / n_train_pairs if n_train_pairs > 0 else 0
        actual_val_ratio = len(val_neg_pairs) / n_val_pairs if n_val_pairs > 0 else 0
        actual_test_ratio = len(test_neg_pairs) / n_test_pairs if n_test_pairs > 0 else 0
        
        logger.info(f"  Actual negative ratios - train: 1:{actual_train_ratio:.1f}, val: 1:{actual_val_ratio:.1f}, test: 1:{actual_test_ratio:.1f}")
        
        # fold-level only; do not update any global set
        
        # Combine positive and negative for each split
        train_data = pd.concat([train_pos, train_neg], ignore_index=True)
        val_data = pd.concat([val_pos, val_neg], ignore_index=True)
        test_data = pd.concat([test_pos, test_neg], ignore_index=True)
        
        # Shuffle each split
        train_data = train_data.sample(frac=1, random_state=train_seed).reset_index(drop=True)
        val_data = val_data.sample(frac=1, random_state=val_seed).reset_index(drop=True)
        test_data = test_data.sample(frac=1, random_state=test_seed).reset_index(drop=True)
        
        # Add fold and split information
        train_data[OUTPUT_FOLD_COL] = fold_idx
        train_data[OUTPUT_SPLIT_COL] = SPLIT_TRAIN
        val_data[OUTPUT_FOLD_COL] = fold_idx
        val_data[OUTPUT_SPLIT_COL] = SPLIT_VAL
        test_data[OUTPUT_FOLD_COL] = fold_idx
        test_data[OUTPUT_SPLIT_COL] = SPLIT_TEST
        
        # Add all splits for this fold
        all_data.extend([train_data, val_data, test_data])
        
        logger.info(f"Fold {fold_idx} summary:")
        logger.info(f"  Train: {len(train_data)} total ({len(train_pos)} pos, {len(train_neg)} neg)")
        logger.info(f"  Val:   {len(val_data)} total ({len(val_pos)} pos, {len(val_neg)} neg)")
        logger.info(f"  Test:  {len(test_data)} total ({len(test_pos)} pos, {len(test_neg)} neg)")
    
    # Step 3: Combine and save
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
    
    # Reorder columns - simplified (removed duplicate branch)
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
    
    # Verify test sets don't overlap across folds
    test_overlaps = []
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            test_i = result_df[(result_df[OUTPUT_FOLD_COL] == i) & 
                              (result_df[OUTPUT_SPLIT_COL] == SPLIT_TEST) & 
                              (result_df[OUTPUT_LABEL_COL] == 1)][
                [OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]
            ]
            test_j = result_df[(result_df[OUTPUT_FOLD_COL] == j) & 
                              (result_df[OUTPUT_SPLIT_COL] == SPLIT_TEST) & 
                              (result_df[OUTPUT_LABEL_COL] == 1)][
                [OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]
            ]
            
            # Check for overlapping pairs
            test_i_pairs = set(zip(test_i[OUTPUT_PEPTIDE_COL], test_i[OUTPUT_PROTEIN_COL]))
            test_j_pairs = set(zip(test_j[OUTPUT_PEPTIDE_COL], test_j[OUTPUT_PROTEIN_COL]))
            overlap = test_i_pairs & test_j_pairs
            
            if overlap:
                test_overlaps.append((i, j, len(overlap)))
                logger.error(f"Test sets for fold {i} and fold {j} overlap: {len(overlap)} pairs")
    
    if test_overlaps:
        raise AssertionError(f"Test sets overlap across folds: {test_overlaps}")
    
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
    
    # Save split information for reproducibility
    import json
    split_info_file = Path(log_dir) / f"propedia_v2_ppi_split_info_{n_folds}fold_random.json"
    split_info = {
        'method': 'pair_grouped_random',
        'n_folds': n_folds,
        'seed': seed,
        'negative_ratio': negative_ratio,
        'unique_pdbs': df_positive[PDB_COL].nunique(),
        'unique_pairs': len(unique_pairs),
        'folds': {}
    }
    
    # Store PDB assignments for each fold
    for fold_idx in range(n_folds):
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        positive_data = fold_data[fold_data[OUTPUT_LABEL_COL] == 1]
        
        train_pdbs = set(positive_data[positive_data[OUTPUT_SPLIT_COL] == SPLIT_TRAIN][PDB_COL].unique())
        val_pdbs = set(positive_data[positive_data[OUTPUT_SPLIT_COL] == SPLIT_VAL][PDB_COL].unique())
        test_pdbs = set(positive_data[positive_data[OUTPUT_SPLIT_COL] == SPLIT_TEST][PDB_COL].unique())
        
        train_pdbs.discard('NEGATIVE')
        val_pdbs.discard('NEGATIVE')
        test_pdbs.discard('NEGATIVE')
        
        split_info['folds'][fold_idx] = {
            'train_pdbs': sorted(list(train_pdbs)),
            'val_pdbs': sorted(list(val_pdbs)),
            'test_pdbs': sorted(list(test_pdbs)),
            'fold_seed': get_fold_seed(seed, fold_idx)
        }
    
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Saved split info to: {split_info_file}")
    
    # Final summary
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
    
    # Label distribution
    label_dist = result_df[OUTPUT_LABEL_COL].value_counts()
    logger.info("")
    logger.info("Label distribution:")
    logger.info(f"  Positive (1): {label_dist.get(1, 0)}")
    logger.info(f"  Negative (0): {label_dist.get(0, 0)}")
    if label_dist.get(1, 0) > 0:
        logger.info(f"  Ratio: 1:{label_dist.get(0, 0)/label_dist.get(1, 0):.1f}")
    
    # Split distribution by fold
    logger.info("")
    logger.info("Split distribution by fold:")
    split_stats = result_df.groupby([OUTPUT_FOLD_COL, OUTPUT_SPLIT_COL, OUTPUT_LABEL_COL]).size().unstack()
    logger.info("\n" + split_stats.to_string())
    
    # Verify data integrity
    logger.info("")
    logger.info("Verifying split integrity...")
    
    # Verification: no pair overlap between splits within a fold
    # PDB overlap is expected with pair-based splitting and is only logged (no assertion)
    for fold_idx in range(n_folds):
        fold_data = result_df[result_df[OUTPUT_FOLD_COL] == fold_idx]
        
        # Get pairs for each split
        train_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TRAIN]
        val_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_VAL]
        test_data = fold_data[fold_data[OUTPUT_SPLIT_COL] == SPLIT_TEST]
        
        # Get positive data only for PDB checking
        train_pos_data = train_data[train_data[OUTPUT_LABEL_COL] == 1]
        val_pos_data = val_data[val_data[OUTPUT_LABEL_COL] == 1]
        test_pos_data = test_data[test_data[OUTPUT_LABEL_COL] == 1]
        
        # Check PDB overlaps (only for positive pairs)
        train_pdbs = set(train_pos_data[PDB_COL]) if not train_pos_data.empty else set()
        val_pdbs = set(val_pos_data[PDB_COL]) if not val_pos_data.empty else set()
        test_pdbs = set(test_pos_data[PDB_COL]) if not test_pos_data.empty else set()
        
        # Remove 'NEGATIVE' PDB from checks
        train_pdbs.discard('NEGATIVE')
        val_pdbs.discard('NEGATIVE')
        test_pdbs.discard('NEGATIVE')
        
        pdb_train_val_overlap = train_pdbs & val_pdbs
        pdb_train_test_overlap = train_pdbs & test_pdbs
        pdb_val_test_overlap = val_pdbs & test_pdbs
        
        # Note: PDB overlap is expected in pair-based splitting
        if pdb_train_val_overlap or pdb_train_test_overlap or pdb_val_test_overlap:
            logger.debug(f"Fold {fold_idx}: PDB overlap between splits (expected in pair-based splitting):")
            if pdb_train_val_overlap:
                logger.debug(f"  Train-Val PDB overlap: {len(pdb_train_val_overlap)} PDBs")
            if pdb_train_test_overlap:
                logger.debug(f"  Train-Test PDB overlap: {len(pdb_train_test_overlap)} PDBs")
            if pdb_val_test_overlap:
                logger.debug(f"  Val-Test PDB overlap: {len(pdb_val_test_overlap)} PDBs")
        else:
            logger.info(f"Fold {fold_idx}: No PDB overlap between splits")
        
        # Check pair overlaps
        train_pairs = set(zip(train_data[OUTPUT_PEPTIDE_COL], train_data[OUTPUT_PROTEIN_COL]))
        val_pairs = set(zip(val_data[OUTPUT_PEPTIDE_COL], val_data[OUTPUT_PROTEIN_COL]))
        test_pairs = set(zip(test_data[OUTPUT_PEPTIDE_COL], test_data[OUTPUT_PROTEIN_COL]))
        
        # Check overlaps
        train_val_overlap = train_pairs & val_pairs
        train_test_overlap = train_pairs & test_pairs
        val_test_overlap = val_pairs & test_pairs
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            error_msg = f"Fold {fold_idx}: Pair overlap detected within fold!"
            if train_val_overlap:
                error_msg += f"\n  Train-Val overlap: {len(train_val_overlap)} pairs"
            if train_test_overlap:
                error_msg += f"\n  Train-Test overlap: {len(train_test_overlap)} pairs"
            if val_test_overlap:
                error_msg += f"\n  Val-Test overlap: {len(val_test_overlap)} pairs"
            raise AssertionError(error_msg)
        else:
            logger.info(f"Fold {fold_idx}: No pair overlap within fold ")
    
    # Verify positive/negative balance (unique pairs only)
    logger.info("\nPair-level distribution (unique pairs only):")
    pos_pairs = result_df[result_df[OUTPUT_LABEL_COL] == 1][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
    neg_pairs = result_df[result_df[OUTPUT_LABEL_COL] == 0][[OUTPUT_PEPTIDE_COL, OUTPUT_PROTEIN_COL]].drop_duplicates()
    pos_pair_cnt = len(pos_pairs)
    neg_pair_cnt = len(neg_pairs)
    pair_ratio = (neg_pair_cnt / pos_pair_cnt) if pos_pair_cnt > 0 else 0
    logger.info(f"  Positive pairs: {pos_pair_cnt}")
    logger.info(f"  Negative pairs: {neg_pair_cnt}")
    logger.info(f"  Positive:Negative (pairs) = 1:{pair_ratio:.1f}")
    
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
    
    logger.info("=" * 60)
    logger.info("RESULT: Pair-grouped random k-fold split completed successfully")
    logger.info("=" * 60)
    
    return result_df


def main() -> None:
    global DEFAULT_LOG_DIR
    # Parse CLI
    parser = argparse.ArgumentParser(description='Prepare Propedia v2 PPI k-fold (pair-grouped random)')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument('--log-dir', type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument('--folds', type=int, default=N_FOLDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--negative-ratio', type=int, default=NEGATIVE_RATIO)
    args = parser.parse_args()

    # Seed everything for reproducibility
    L.seed_everything(args.seed, workers=True)

    # Configure logging with selected log dir
    DEFAULT_LOG_DIR = Path(args.log_dir)
    global logger, log_dir
    logger, log_dir = setup_logging()

    logger.info(f"Log directory: {log_dir}")
    logger.info("==================================================")
    logger.info("Propedia v2 PPI K-fold Preparation")
    logger.info("==================================================")
    logger.info(f"Source file: {args.source}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Number of folds: {args.folds}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Method: Pair-grouped random distribution")
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
        negative_ratio=args.negative_ratio
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error(f"Script failed with error: {str(e)}")
        logging.getLogger(__name__).error("Full traceback:", exc_info=True)
        sys.exit(1)
