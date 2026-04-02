"""Permeability DataModule with DataCollator for dynamic padding.

Supports two modes:
  1. Single-split: train_file + test_file (legacy)
  2. K-fold CV: kfold_file with fold + split columns
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from .data_collators import DataCollatorForRegression
from .datasets import HELMDataset

logger = logging.getLogger(__name__)

EXPECTED_KFOLD_SPLITS = {"train", "val", "test"}


@dataclass
class PermeabilityDataConfig:
    """Configuration for permeability DataModule.

    For single-split mode, provide train_file and test_file.
    For k-fold mode, provide kfold_file instead.
    """

    # Single-split mode
    train_file: str = ""
    test_file: str = ""

    # K-fold mode
    kfold_file: str = ""
    n_folds: int = 10
    fold_column: str = "fold"
    split_column: str = "split"

    # Common
    helm_column: str = "HELM"
    target_column: str = "Permeability"
    val_ratio: float = 0.1
    batch_size: int = 32
    max_seq_length: int = 512
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42


class PermeabilityDataModule(L.LightningDataModule):
    """DataModule for permeability regression.

    Supports single-split and k-fold CV modes.
    Uses DataCollator for dynamic padding at batch level.
    """

    def __init__(
        self,
        config: PermeabilityDataConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.data_stats: Dict[str, Any] = {}

        self._collate_fn = DataCollatorForRegression(tokenizer=self.tokenizer)

        # K-fold state
        self._kfold_df: Optional[pd.DataFrame] = None
        self._current_fold: Optional[int] = None

    @property
    def is_kfold(self) -> bool:
        return bool(self.config.kfold_file)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. For single-split mode, creates datasets immediately.
        For k-fold mode, loads the CSV; call setup_fold() to select a fold.
        """
        if self.is_kfold:
            self._setup_kfold()
        else:
            self._setup_single_split()

    def _setup_kfold(self) -> None:
        """Load k-fold CSV (once). Datasets are created per-fold via setup_fold()."""
        if self._kfold_df is not None:
            return

        kfold_path = Path(self.config.kfold_file)
        if not kfold_path.exists():
            raise FileNotFoundError(f"K-fold file not found: {kfold_path}")

        self._kfold_df = pd.read_csv(kfold_path)

        fold_col = self.config.fold_column
        split_col = self.config.split_column
        for col in (fold_col, split_col, self.config.helm_column, self.config.target_column):
            if col not in self._kfold_df.columns:
                raise ValueError(f"Column '{col}' not found in {kfold_path}")

        self._validate_kfold_layout()

        n_folds_actual = self._kfold_df[fold_col].nunique()
        logger.info(
            "Loaded k-fold file: %d rows, %d folds from %s",
            len(self._kfold_df), n_folds_actual, kfold_path,
        )

    def _validate_kfold_layout(self) -> None:
        """Validate fold IDs and required train/val/test split coverage."""
        if self._kfold_df is None:
            raise RuntimeError("K-fold CSV must be loaded before validation")

        fold_col = self.config.fold_column
        split_col = self.config.split_column
        fold_ids = sorted(self._kfold_df[fold_col].unique().tolist())
        actual_folds = len(fold_ids)

        if actual_folds != self.config.n_folds:
            raise ValueError(
                f"Expected {self.config.n_folds} folds from config, found {actual_folds} "
                f"in {self.config.kfold_file}"
            )

        split_values = set(self._kfold_df[split_col].unique())
        unexpected_splits = split_values - EXPECTED_KFOLD_SPLITS
        if unexpected_splits:
            raise ValueError(
                f"Unexpected split values in {self.config.kfold_file}: "
                f"{sorted(unexpected_splits)}"
            )

        for fold in fold_ids:
            fold_splits = set(
                self._kfold_df.loc[self._kfold_df[fold_col] == fold, split_col].unique()
            )
            missing_splits = EXPECTED_KFOLD_SPLITS - fold_splits
            if missing_splits:
                raise ValueError(
                    f"Fold {fold} in {self.config.kfold_file} is missing required splits: "
                    f"{sorted(missing_splits)}"
                )

    def get_fold_ids(self) -> list[int]:
        """Return the sorted fold labels defined in the k-fold CSV."""
        if not self.is_kfold:
            raise RuntimeError("get_fold_ids() requires kfold_file in config")

        if self._kfold_df is None:
            self._setup_kfold()

        return sorted(self._kfold_df[self.config.fold_column].unique().tolist())

    def setup_fold(self, fold: int) -> None:
        """Configure datasets for a specific fold. Requires k-fold mode."""
        if not self.is_kfold:
            raise RuntimeError("setup_fold() requires kfold_file in config")

        # Ensure CSV is loaded
        if self._kfold_df is None:
            self._setup_kfold()

        available_folds = self.get_fold_ids()
        if fold not in available_folds:
            raise ValueError(f"Unknown fold {fold}. Available folds: {available_folds}")

        fold_col = self.config.fold_column
        split_col = self.config.split_column

        train_df = self._kfold_df[
            (self._kfold_df[fold_col] == fold) & (self._kfold_df[split_col] == "train")
        ]
        val_df = self._kfold_df[
            (self._kfold_df[fold_col] == fold) & (self._kfold_df[split_col] == "val")
        ]
        test_df = self._kfold_df[
            (self._kfold_df[fold_col] == fold) & (self._kfold_df[split_col] == "test")
        ]

        if len(train_df) == 0:
            raise ValueError(f"No training samples for fold {fold}")

        self.train_dataset = self._create_dataset(train_df)
        self.val_dataset = self._create_dataset(val_df) if len(val_df) > 0 else None
        self.test_dataset = self._create_dataset(test_df) if len(test_df) > 0 else None
        self._current_fold = fold

        self._compute_statistics(train_df, val_df, test_df if len(test_df) > 0 else None)
        logger.info(
            "Fold %d: %d train, %d val, %d test",
            fold, len(train_df), len(val_df), len(test_df),
        )

    def _setup_single_split(self) -> None:
        """Legacy single-split mode using train_file + test_file."""
        if self.train_dataset is not None:
            return

        train_file = Path(self.config.train_file)
        test_file = Path(self.config.test_file)

        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        train_df = pd.read_csv(train_file)
        logger.info(f"Loaded train: {len(train_df)} samples from {train_file}")

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.config.val_ratio,
            random_state=self.config.seed,
            shuffle=True,
        )
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")

        test_df = None
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded test: {len(test_df)} samples from {test_file}")

        self.train_dataset = self._create_dataset(train_df)
        self.val_dataset = self._create_dataset(val_df)
        if test_df is not None:
            self.test_dataset = self._create_dataset(test_df)

        self._compute_statistics(train_df, val_df, test_df)

    def _create_dataset(self, df: pd.DataFrame) -> HELMDataset:
        return HELMDataset(
            sequences=df[self.config.helm_column].tolist(),
            labels=df[self.config.target_column].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )

    def _compute_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
    ) -> None:
        target_col = self.config.target_column
        self.data_stats = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df) if test_df is not None else 0,
        }
        all_targets = pd.concat([train_df[target_col], val_df[target_col]])
        if test_df is not None:
            all_targets = pd.concat([all_targets, test_df[target_col]])
        self.data_stats["mean_target"] = all_targets.mean()
        self.data_stats["std_target"] = all_targets.std()
        logger.info(
            f"Target stats: mean={self.data_stats['mean_target']:.3f}, "
            f"std={self.data_stats['std_target']:.3f}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )
