"""Strip connection information from HELM notation.

Creates new CSV files with connection section removed for controlled experiment.

HELM format: sequence$connection$group$extra$
Stripping:   sequence$$$$

Output files are saved with '_no_conn' suffix alongside originals.
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime

# Logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("outputs") / "preprocessing" / f"strip_helm_connections_{timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "strip_connections.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")

MLM_DIR = Path("data/mlm")
DOWNSTREAM_DIR = Path("data/downstream")

# (input_path, helm_column, output_path)
FILES = [
    # MLM training data
    (MLM_DIR / "chembl_v2_deduplicated.csv",
     "helm_notation",
     MLM_DIR / "chembl_v2_deduplicated_no_conn.csv"),

    (MLM_DIR / "cycpeptmpdb_deduplicated.csv",
     "HELM",
     MLM_DIR / "cycpeptmpdb_deduplicated_no_conn.csv"),

    (MLM_DIR / "propedia_v2_deduplicated.csv",
     "Peptide_HELM",
     MLM_DIR / "propedia_v2_deduplicated_no_conn.csv"),

    # Downstream: permeability (single-split)
    (DOWNSTREAM_DIR / "cycpeptmpdb_permeability_train.csv",
     "HELM",
     DOWNSTREAM_DIR / "cycpeptmpdb_permeability_train_no_conn.csv"),

    (DOWNSTREAM_DIR / "cycpeptmpdb_permeability_test.csv",
     "HELM",
     DOWNSTREAM_DIR / "cycpeptmpdb_permeability_test_no_conn.csv"),

    # Downstream: permeability (k-fold)
    (DOWNSTREAM_DIR / "cycpeptmpdb_permeability_random_10fold.csv",
     "HELM",
     DOWNSTREAM_DIR / "cycpeptmpdb_permeability_random_10fold_no_conn.csv"),

    (DOWNSTREAM_DIR / "cycpeptmpdb_permeability_scaffold_10fold.csv",
     "HELM",
     DOWNSTREAM_DIR / "cycpeptmpdb_permeability_scaffold_10fold_no_conn.csv"),

    # Downstream: PPI (single-split)
    (DOWNSTREAM_DIR / "propedia_ppi_random_train.csv",
     "Peptide_HELM",
     DOWNSTREAM_DIR / "propedia_ppi_random_train_no_conn.csv"),

    (DOWNSTREAM_DIR / "propedia_ppi_random_test.csv",
     "Peptide_HELM",
     DOWNSTREAM_DIR / "propedia_ppi_random_test_no_conn.csv"),

    (DOWNSTREAM_DIR / "propedia_ppi_acsm_train.csv",
     "Peptide_HELM",
     DOWNSTREAM_DIR / "propedia_ppi_acsm_train_no_conn.csv"),

    (DOWNSTREAM_DIR / "propedia_ppi_acsm_test.csv",
     "Peptide_HELM",
     DOWNSTREAM_DIR / "propedia_ppi_acsm_test_no_conn.csv"),

    # Downstream: PPI (k-fold)
    (DOWNSTREAM_DIR / "Propedia_v2_ppi_5fold_acsm.csv",
     "Peptide_HELM",
     DOWNSTREAM_DIR / "Propedia_v2_ppi_5fold_acsm_no_conn.csv"),

    (DOWNSTREAM_DIR / "Propedia_v2_ppi_5fold_random.csv",
     "Peptide_HELM",
     DOWNSTREAM_DIR / "Propedia_v2_ppi_5fold_random_no_conn.csv"),
]


def strip_connection(helm: str) -> str:
    """Remove connection section from HELM notation.

    PEPTIDE1{A.C.D}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$ -> PEPTIDE1{A.C.D}$$$$
    """
    if not isinstance(helm, str):
        return helm
    parts = helm.split("$")
    if len(parts) >= 5:
        return parts[0] + "$$$$"
    return helm


def main():
    logger.info("=" * 60)
    logger.info("Strip HELM Connection Information")
    logger.info("=" * 60)

    total_files = 0
    total_modified = 0

    for input_path, helm_col, output_path in FILES:
        if not input_path.exists():
            logger.warning(f"SKIP: {input_path} (not found)")
            continue

        df = pd.read_csv(input_path)
        if helm_col not in df.columns:
            logger.warning(f"SKIP: {input_path} (no column '{helm_col}')")
            continue

        orig_col = df[helm_col].copy()
        df[helm_col] = df[helm_col].apply(strip_connection)

        changed = (orig_col != df[helm_col]).sum()
        total = len(df)
        total_files += 1
        total_modified += changed

        logger.info(f"{input_path.name} -> {output_path.name}: {total} rows, {changed} modified ({changed/total*100:.1f}%)")

        if changed > 0:
            idx = (orig_col != df[helm_col]).idxmax()
            logger.info(f"  before: {str(orig_col[idx])[:80]}")
            logger.info(f"  after:  {str(df[helm_col][idx])[:80]}")

        df.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info(f"Done. {total_files} files processed, {total_modified} total rows modified.")
    logger.info("Original files are unchanged.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
