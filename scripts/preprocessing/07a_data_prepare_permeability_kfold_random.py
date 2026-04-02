"""Universal k-fold data preparation script for all models.

This script creates k-fold splits with fold numbers using common_kfold logic.
The output can be used by any model (PeptideCLM, MoLFormer, Uni-Mol, etc.).
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import KFold, train_test_split
from datetime import datetime
import sys
import argparse
import lightning as L

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR YOUR USE CASE
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_FILE = REPO_ROOT / 'data/mlm/cycpeptmpdb_deduplicated.csv'
DEFAULT_OUTPUT_FILE = REPO_ROOT / 'data/downstream/cycpeptmpdb_permeability_random_10fold.csv'
DEFAULT_LOG_DIR = REPO_ROOT / 'outputs/preprocessing'  # Log directory
LOG_FILE_NAME = 'prepare_kfold_permeability.log'  # Log file name

N_FOLDS = 10
SEED = 42
SMILES_COL = 'SMILES'
HELM_COL = 'HELM'
TARGET_COL = 'Permeability'
INVALID_THRESHOLD = -10  # Filter samples with Permeability <= -10

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Output column names
OUTPUT_SMILES_COL = 'SMILES'
OUTPUT_HELM_COL = 'HELM'
OUTPUT_TARGET_COL = 'Permeability'
OUTPUT_FOLD_COL = 'fold'
OUTPUT_SPLIT_COL = 'split'

# Split types
SPLIT_TRAIN = 'train'
SPLIT_VAL = 'val'
SPLIT_TEST = 'test'

# ============================================================================

def setup_logging():
    """Set up logging to both console and file."""
    # Create timestamped subdirectory for this preprocessing task
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(DEFAULT_LOG_DIR)
    log_dir = base / f"kfold_reg_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file in the timestamped directory
    log_file = log_dir / LOG_FILE_NAME
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Print log file location
    print(f"Log file: {log_file.absolute()}")
    
    return logger, log_dir

logger = logging.getLogger(__name__)


def create_kfold_splits(
    df: pd.DataFrame,
    n_folds: int,
    seed: int
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create k-fold cross-validation splits where each fold serves as test once.
    
    For each fold:
    - One fold becomes the test set (1/k of data)
    - From remaining data, randomly select same amount for validation (1/k of data)
    - Rest becomes training set (~(k-2)/k of data)
    
    This ensures:
    - Each sample appears exactly once as test
    - Validation sets are different for each fold (no overlap between folds)
    - Consistent split sizes across all folds
    
    Args:
        df: DataFrame to split (must have at least n_folds samples)
        n_folds: Number of folds for cross-validation (typically 5 or 10)
        seed: Random seed for reproducibility
        
    Returns:
        List of (train_df, val_df, test_df) tuples, one per fold
    """
    # Validate inputs
    if df.empty:
        raise ValueError("Cannot create k-fold splits from empty DataFrame")
        
    if len(df) < n_folds:
        raise ValueError(f"DataFrame must have at least {n_folds} samples for {n_folds}-fold CV, got {len(df)}")
        
    if n_folds < 2:
        raise ValueError(f"Number of folds must be at least 2, got {n_folds}")
        
    logger.info(f"Creating {n_folds}-fold CV with equal test/val sizes")
    
    # Use KFold with shuffle (standard scikit-learn approach)
    n_total = len(df)
    
    # Create K equal folds with shuffling
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_folds = list(kfold.split(df))
    
    fold_results = []
    for fold_idx, (train_val_indices, test_indices) in enumerate(all_folds):
        # Current fold becomes test set
        test_df = df.iloc[test_indices].reset_index(drop=True)
        test_size = len(test_df)
        
        # Use train_val_indices directly from KFold
        remaining_df = df.iloc[train_val_indices].reset_index(drop=True)
        
        # From remaining data, randomly select same amount as test for validation
        val_size = test_size  # Equal to test size
        
        # Use fold-specific seed to ensure different validation sets for each fold
        fold_seed = seed + fold_idx
        
        # Use sklearn's train_test_split for clean, standard splitting
        train_df, val_df = train_test_split(
            remaining_df,
            test_size=val_size,
            random_state=fold_seed,
            shuffle=True
        )
        
        # Verify fold integrity
        assert len(train_df) + len(val_df) + len(test_df) == n_total, f"Data loss in fold {fold_idx}"
        
        fold_results.append((train_df, val_df, test_df))
    
    # Log the actual sizes for first fold
    train_size = len(fold_results[0][0])
    val_size = len(fold_results[0][1])
    test_size = len(fold_results[0][2])
    
    logger.info(f"K-fold CV created: {n_folds} folds with Train≈{train_size}, Val={val_size}, Test={test_size} each")
    logger.info(f"Ratios per fold - Train: {train_size/n_total:.1%}, Val: {val_size/n_total:.1%}, Test: {test_size/n_total:.1%}")
    
    # Final validation: ensure all samples are used exactly n_folds times
    total_appearances = sum(len(fold[0]) + len(fold[1]) + len(fold[2]) for fold in fold_results)
    assert total_appearances == n_total * n_folds, "Sample count mismatch in k-fold splits"
    
    return fold_results


def prepare_kfold_data(
    source_file: str,
    output_file: str,
    n_folds: int = 10,
    seed: int = 42,
    smiles_col: str = 'SMILES',
    helm_col: str = 'HELM',
    target_col: str = 'Permeability',
    invalid_threshold: float = -10
):
    """Create k-fold splits with fold and split information.
    
    Args:
        source_file: Path to source CSV file
        output_file: Path to output CSV file
        n_folds: Number of folds (default: 10)
        seed: Random seed (default: 42)
        smiles_col: Name of SMILES column (default: 'SMILES')
        helm_col: Name of HELM column (default: 'HELM')
        target_col: Name of target column (default: 'Permeability')
        invalid_threshold: Filter samples with target <= this value (default: -10)
    
    Returns:
        DataFrame with columns: SMILES, HELM, target, fold, split
    """
    
    # Use absolute paths directly
    source_path = Path(source_file)
    output_path = Path(output_file)
    
    logger.info(f"Loading data from: {source_path}")
    try:
        df = pd.read_csv(source_path)
    except Exception as e:
        logger.error(f"Failed to load data from {source_path}: {str(e)}")
        raise
    
    # Keep only relevant columns and remove NaN
    try:
        data = df[[smiles_col, helm_col, target_col]].dropna()
    except KeyError as e:
        logger.error(f"Missing required columns in {source_path}")
        logger.error(f"Expected columns: {smiles_col}, {helm_col}, {target_col}")
        logger.error(f"Found columns: {list(df.columns)}")
        raise
    # Rename columns to standard names
    data = data.rename(columns={
        smiles_col: OUTPUT_SMILES_COL, 
        helm_col: OUTPUT_HELM_COL,
        target_col: OUTPUT_TARGET_COL
    })
    
    # Filter invalid samples
    initial_count = len(data)
    data = data[data[OUTPUT_TARGET_COL] > invalid_threshold]
    filtered_count = initial_count - len(data)
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} samples with {OUTPUT_TARGET_COL} <= {invalid_threshold} from {source_file}")
    
    logger.info(f"Dataset size: {len(data)} samples")
    
    # Create k-fold splits using embedded logic
    folds = create_kfold_splits(data, n_folds=n_folds, seed=seed)
    
    # Build output dataframe
    all_data = []
    for fold_idx, (train_df, val_df, test_df) in enumerate(folds):
        for df, split in [(train_df, SPLIT_TRAIN), (val_df, SPLIT_VAL), (test_df, SPLIT_TEST)]:
            df = df.copy()
            df[OUTPUT_FOLD_COL] = fold_idx
            df[OUTPUT_SPLIT_COL] = split
            all_data.append(df)
        
        logger.info(f"Fold {fold_idx}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Combine and save
    result_df = pd.concat(all_data, ignore_index=True)
    result_df = result_df[[OUTPUT_SMILES_COL, OUTPUT_HELM_COL, OUTPUT_TARGET_COL, OUTPUT_FOLD_COL, OUTPUT_SPLIT_COL]]
    
    # Ensure output directory exists (though in this case, it's the same directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {str(e)}")
        raise
    
    # Log summary
    logger.info(f"\nSummary:")
    logger.info(f"Total entries: {len(result_df)} ({len(data)} samples × {n_folds} folds)")
    logger.info(f"Split distribution:\n{result_df.groupby([OUTPUT_FOLD_COL, OUTPUT_SPLIT_COL]).size().unstack()}")
    
    return result_df


def main():
    global DEFAULT_LOG_DIR
    parser = argparse.ArgumentParser(description='Prepare k-fold splits for permeability')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument('--log-dir', type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument('--folds', type=int, default=N_FOLDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--invalid-threshold', type=float, default=INVALID_THRESHOLD)
    args = parser.parse_args()

    # Set all random seeds for reproducibility
    L.seed_everything(args.seed, workers=True)

    # Configure logging now that paths are known
    DEFAULT_LOG_DIR = Path(args.log_dir)
    global logger, log_dir
    logger, log_dir = setup_logging()

    logger.info(f"Log directory: {log_dir}")
    logger.info("==================================================")
    logger.info("K-fold Data Preparation")
    logger.info("==================================================")
    logger.info(f"Source file: {args.source}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Number of folds: {args.folds}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Invalid threshold: {args.invalid_threshold}")
    logger.info("==================================================")
    
    prepare_kfold_data(
        source_file=args.source,
        output_file=args.output,
        n_folds=args.folds,
        seed=args.seed,
        smiles_col=SMILES_COL,
        helm_col=HELM_COL,
        target_col=TARGET_COL,
        invalid_threshold=args.invalid_threshold
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        sys.exit(1)
