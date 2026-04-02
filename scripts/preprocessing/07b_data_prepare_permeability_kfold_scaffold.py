"""Scaffold-based k-fold data preparation for permeability.

Uses Murcko scaffolds to cluster molecules, then distributes scaffold groups
into k folds using a zigzag pattern for balanced sizes. This ensures that
molecules sharing the same scaffold never appear in both train and test sets.

Output format matches 05_data_prepare_kfold_permeability.py:
  SMILES, HELM, Permeability, fold, split
"""

import pandas as pd
import numpy as np
import logging
import sys
import argparse
from pathlib import Path
from datetime import datetime
from rdkit.Chem.Scaffolds import MurckoScaffold
import lightning as L

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_FILE = REPO_ROOT / 'data/mlm/cycpeptmpdb_deduplicated.csv'
DEFAULT_OUTPUT_FILE = REPO_ROOT / 'data/downstream/cycpeptmpdb_permeability_scaffold_10fold.csv'
DEFAULT_LOG_DIR = REPO_ROOT / 'outputs/preprocessing'
LOG_FILE_NAME = 'prepare_scaffold_kfold_permeability.log'

N_FOLDS = 10
SEED = 42
SMILES_COL = 'SMILES'
HELM_COL = 'HELM'
TARGET_COL = 'Permeability'
INVALID_THRESHOLD = -10

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

OUTPUT_SMILES_COL = 'SMILES'
OUTPUT_HELM_COL = 'HELM'
OUTPUT_TARGET_COL = 'Permeability'
OUTPUT_FOLD_COL = 'fold'
OUTPUT_SPLIT_COL = 'split'

SPLIT_TRAIN = 'train'
SPLIT_VAL = 'val'
SPLIT_TEST = 'test'

# ============================================================================


def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(DEFAULT_LOG_DIR) / f"scaffold_kfold_reg_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME

    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    print(f"Log file: {log_file.absolute()}")
    return logger, log_dir


logger = logging.getLogger(__name__)


def generate_scaffold(smiles: str) -> str:
    """Generate Murcko scaffold from SMILES. Returns empty string on failure."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=False
        )
    except Exception:
        return ''


def scaffold_to_folds(df: pd.DataFrame, n_folds: int) -> list:
    """Cluster by Murcko scaffold and distribute into n_folds using zigzag.

    Returns:
        List of n_folds lists, each containing row indices for that fold.
    """
    scaffolds = {}
    no_scaffold = []
    for i, smiles in enumerate(df[OUTPUT_SMILES_COL]):
        scaffold = generate_scaffold(smiles)
        if scaffold == '':
            no_scaffold.append(i)
            continue
        scaffolds.setdefault(scaffold, []).append(i)

    if no_scaffold:
        logger.warning(f"{len(no_scaffold)} molecules failed scaffold generation, grouped together")
        scaffolds['__no_scaffold__'] = no_scaffold

    logger.info(f"Found {len(scaffolds)} unique scaffolds from {len(df)} molecules")

    # Sort scaffold groups: largest first, break ties by first index
    sorted_groups = sorted(
        scaffolds.values(),
        key=lambda x: (len(x), x[0]),
        reverse=True,
    )

    # Zigzag distribution into folds (balanced sizes)
    folds = [[] for _ in range(n_folds)]
    for i, group in enumerate(sorted_groups):
        r = i % (2 * n_folds)
        idx = min(r, 2 * n_folds - r - 1)
        folds[idx].extend(group)

    sizes = [len(f) for f in folds]
    logger.info(f"Fold sizes: {sizes} (min={min(sizes)}, max={max(sizes)})")

    return folds


def prepare_scaffold_kfold(
    source_file: str,
    output_file: str,
    n_folds: int = 10,
    seed: int = 42,
    smiles_col: str = 'SMILES',
    helm_col: str = 'HELM',
    target_col: str = 'Permeability',
    invalid_threshold: float = -10,
):
    source_path = Path(source_file)
    output_path = Path(output_file)

    logger.info(f"Loading data from: {source_path}")
    df = pd.read_csv(source_path)

    # Keep relevant columns
    try:
        data = df[[smiles_col, helm_col, target_col]].dropna()
    except KeyError:
        logger.error(f"Expected columns: {smiles_col}, {helm_col}, {target_col}")
        logger.error(f"Found columns: {list(df.columns)}")
        raise

    data = data.rename(columns={
        smiles_col: OUTPUT_SMILES_COL,
        helm_col: OUTPUT_HELM_COL,
        target_col: OUTPUT_TARGET_COL,
    })

    # Filter invalid
    initial = len(data)
    data = data[data[OUTPUT_TARGET_COL] > invalid_threshold].reset_index(drop=True)
    filtered = initial - len(data)
    if filtered > 0:
        logger.info(f"Filtered {filtered} samples with {OUTPUT_TARGET_COL} <= {invalid_threshold}")
    logger.info(f"Dataset size: {len(data)} samples")

    # Scaffold-based fold assignment
    fold_indices = scaffold_to_folds(data, n_folds)

    # For each fold: fold_i = test, fold_(i+1)%n = val, rest = train
    all_rows = []
    for fold_i in range(n_folds):
        test_idx = fold_indices[fold_i]
        val_idx = fold_indices[(fold_i + 1) % n_folds]
        train_idx = []
        for j in range(n_folds):
            if j != fold_i and j != (fold_i + 1) % n_folds:
                train_idx.extend(fold_indices[j])

        for idx_list, split in [
            (train_idx, SPLIT_TRAIN),
            (val_idx, SPLIT_VAL),
            (test_idx, SPLIT_TEST),
        ]:
            chunk = data.iloc[idx_list].copy()
            chunk[OUTPUT_FOLD_COL] = fold_i
            chunk[OUTPUT_SPLIT_COL] = split
            all_rows.append(chunk)

        logger.info(
            f"Fold {fold_i}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    result = pd.concat(all_rows, ignore_index=True)
    result = result[[OUTPUT_SMILES_COL, OUTPUT_HELM_COL, OUTPUT_TARGET_COL, OUTPUT_FOLD_COL, OUTPUT_SPLIT_COL]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Saved to: {output_path}")

    logger.info(f"Total entries: {len(result)} ({len(data)} samples x {n_folds} folds)")
    logger.info(f"Split distribution:\n{result.groupby([OUTPUT_FOLD_COL, OUTPUT_SPLIT_COL]).size().unstack()}")

    return result


def main():
    global DEFAULT_LOG_DIR
    parser = argparse.ArgumentParser(description='Scaffold-based k-fold splits for permeability')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument('--log-dir', type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument('--folds', type=int, default=N_FOLDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--invalid-threshold', type=float, default=INVALID_THRESHOLD)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    DEFAULT_LOG_DIR = Path(args.log_dir)
    global logger
    logger, log_dir = setup_logging()

    logger.info("==================================================")
    logger.info("Scaffold-based K-fold Data Preparation")
    logger.info("==================================================")
    logger.info(f"Source file: {args.source}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Number of folds: {args.folds}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Invalid threshold: {args.invalid_threshold}")
    logger.info("==================================================")

    prepare_scaffold_kfold(
        source_file=args.source,
        output_file=args.output,
        n_folds=args.folds,
        seed=args.seed,
        smiles_col=SMILES_COL,
        helm_col=HELM_COL,
        target_col=TARGET_COL,
        invalid_threshold=args.invalid_threshold,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.error("Full traceback:", exc_info=True)
        sys.exit(1)
