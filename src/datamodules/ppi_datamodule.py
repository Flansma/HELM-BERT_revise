"""PPI DataModule for peptide-protein interaction tasks.

Supports dual-encoder setup with HELM-BERT for peptides and ESM-2 for proteins.
Uses DataCollator for dynamic padding at batch level.

Supports two modes:
  1. Single-split: train_file + test_file (legacy)
  2. K-fold CV: kfold_file with fold + split columns
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.utils.embedding_cache import EmbeddingCache
from src.utils.embedding_generator import generate_drug_embeddings, generate_target_embeddings

from .data_collators import DataCollatorForPPI, DataCollatorForPPIEmbedding
from .dual_sequence_dataset import DualSequenceDataset
from .embedding_only_dataset import EmbeddingOnlyDataset

logger = logging.getLogger(__name__)

EXPECTED_KFOLD_SPLITS = {"train", "val", "test"}


@dataclass
class PPIDataConfig:
    """Configuration for PPI DataModule.

    For single-split mode, provide train_file and test_file.
    For k-fold mode, provide kfold_file instead.
    """

    # Single-split mode
    train_file: str = ""
    test_file: str = ""

    # K-fold mode
    kfold_file: str = ""
    n_folds: int = 5
    fold_column: str = "fold"
    split_column: str = "split"

    # Column names
    drug_column: str = "Peptide_HELM"
    target_column: str = "Receptor_Sequence"
    label_column: str = "Label"

    # Target encoder (ESM-2)
    target_encoder: str = "facebook/esm2_t33_650M_UR50D"

    # Data loading
    val_ratio: float = 0.1
    batch_size: int = 32
    max_drug_length: int = 512
    max_target_length: int = 1024
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42

    # Cached embeddings
    use_cached_embeddings: bool = True
    cache_dir: str = "./data/embeddings"

    # Encoder paths (for auto-generation of cache)
    drug_encoder: str = "./checkpoints/helmbert-base"
    trust_remote_code: bool = True

    # Cache naming (for EmbeddingCache lookup)
    cache_drug_encoder_name: str = "helmbert"
    cache_target_encoder_name: str = "esm2"
    cache_dataset_type: str = "ppi"


class PPIDataModule(L.LightningDataModule):
    """DataModule for peptide-protein interaction prediction.

    Supports single-split and k-fold CV modes.
    Uses DataCollator for dynamic padding at batch level.
    """

    def __init__(
        self,
        config: PPIDataConfig,
        drug_tokenizer: PreTrainedTokenizer,
        target_tokenizer: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config

        self._drug_tokenizer = drug_tokenizer
        self._target_tokenizer = target_tokenizer

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.data_stats: Dict[str, Any] = {}

        self.drug_embeddings: Optional[Dict[str, torch.Tensor]] = None
        self.target_embeddings: Optional[Dict[str, torch.Tensor]] = None

        self._collate_fn: Optional[Any] = None

        # K-fold state
        self._kfold_df: Optional[pd.DataFrame] = None
        self._current_fold: Optional[int] = None

    @property
    def is_kfold(self) -> bool:
        return bool(self.config.kfold_file)

    @property
    def drug_tokenizer(self) -> PreTrainedTokenizer:
        return self._drug_tokenizer

    @property
    def target_tokenizer(self):
        if self._target_tokenizer is None:
            self._target_tokenizer = AutoTokenizer.from_pretrained(
                self.config.target_encoder
            )
        return self._target_tokenizer

    def _use_cached_embeddings(self) -> bool:
        return self.config.use_cached_embeddings

    def _use_embeddings_only(self) -> bool:
        return (
            self._use_cached_embeddings()
            and self.drug_embeddings is not None
            and self.target_embeddings is not None
        )

    def _get_all_unique_sequences(self) -> tuple[List[str], List[str]]:
        """Return unique drug/target sequences from all data (kfold or single-split)."""
        drug_col = self.config.drug_column
        target_col = self.config.target_column

        if self.is_kfold:
            if self._kfold_df is None:
                self._load_kfold_csv()
            all_drugs = list(self._kfold_df[drug_col].unique())
            all_targets = list(self._kfold_df[target_col].unique())
            return all_drugs, all_targets

        train_df = pd.read_csv(self.config.train_file)
        all_drugs = set(train_df[drug_col].unique())
        all_targets = set(train_df[target_col].unique())

        test_file = Path(self.config.test_file)
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            all_drugs.update(test_df[drug_col].unique())
            all_targets.update(test_df[target_col].unique())

        return list(all_drugs), list(all_targets)

    def prepare_data(self) -> None:
        """Generate embedding cache if missing. Runs once on rank 0."""
        if not self._use_cached_embeddings():
            return

        unique_drugs, unique_targets = self._get_all_unique_sequences()
        cache = EmbeddingCache(Path(self.config.cache_dir))

        generate_drug_embeddings(
            cache=cache,
            sequences=unique_drugs,
            encoder_name=self.config.cache_drug_encoder_name,
            dataset_type=self.config.cache_dataset_type,
            pretrained_path=self.config.drug_encoder,
            max_length=self.config.max_drug_length,
            batch_size=self.config.batch_size,
            trust_remote_code=self.config.trust_remote_code,
        )

        generate_target_embeddings(
            cache=cache,
            sequences=unique_targets,
            encoder_name=self.config.cache_target_encoder_name,
            dataset_type=self.config.cache_dataset_type,
            pretrained_path=self.config.target_encoder,
            max_length=self.config.max_target_length,
            batch_size=self.config.batch_size,
        )

        logger.info("Embedding cache ready at: %s", self.config.cache_dir)

    def _load_kfold_csv(self) -> None:
        """Load k-fold CSV once."""
        if self._kfold_df is not None:
            return
        kfold_path = Path(self.config.kfold_file)
        if not kfold_path.exists():
            raise FileNotFoundError(f"K-fold file not found: {kfold_path}")

        self._kfold_df = pd.read_csv(kfold_path)

        required = [
            self.config.fold_column, self.config.split_column,
            self.config.drug_column, self.config.target_column, self.config.label_column,
        ]
        for col in required:
            if col not in self._kfold_df.columns:
                raise ValueError(f"Column '{col}' not found in {kfold_path}")

        self._validate_kfold_layout()

        n_folds_actual = self._kfold_df[self.config.fold_column].nunique()
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
            self._load_kfold_csv()

        return sorted(self._kfold_df[self.config.fold_column].unique().tolist())

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. For kfold mode, loads CSV; call setup_fold() to select fold."""
        if self.is_kfold:
            self._load_kfold_csv()
        else:
            self._setup_single_split()

    def setup_fold(self, fold: int) -> None:
        """Configure datasets for a specific fold. Requires k-fold mode."""
        if not self.is_kfold:
            raise RuntimeError("setup_fold() requires kfold_file in config")

        if self._kfold_df is None:
            self._load_kfold_csv()

        available_folds = self.get_fold_ids()
        if fold not in available_folds:
            raise ValueError(f"Unknown fold {fold}. Available folds: {available_folds}")

        fold_col = self.config.fold_column
        split_col = self.config.split_column
        drug_col = self.config.drug_column
        target_col = self.config.target_column
        label_col = self.config.label_column

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

        # Load embeddings for this fold's sequences
        all_fold_df = pd.concat([train_df, val_df, test_df])
        self._load_embeddings(all_fold_df, None)

        # Initialize DataCollator
        if self._use_embeddings_only():
            self._collate_fn = DataCollatorForPPIEmbedding()
        else:
            self._collate_fn = DataCollatorForPPI(
                drug_tokenizer=self.drug_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

        self.train_dataset = self._create_dataset(train_df, drug_col, target_col, label_col)
        self.val_dataset = (
            self._create_dataset(val_df, drug_col, target_col, label_col)
            if len(val_df) > 0 else None
        )
        self.test_dataset = (
            self._create_dataset(test_df, drug_col, target_col, label_col)
            if len(test_df) > 0 else None
        )
        self._current_fold = fold

        self._compute_statistics(train_df, val_df, test_df if len(test_df) > 0 else None, label_col)
        self._check_overlaps(train_df, val_df, test_df if len(test_df) > 0 else None, drug_col, target_col)
        logger.info(
            "Fold %d: %d train, %d val, %d test",
            fold, len(train_df), len(val_df), len(test_df),
        )

    def _setup_single_split(self) -> None:
        """Legacy single-split mode."""
        if self.train_dataset is not None:
            return

        train_file = Path(self.config.train_file)
        test_file = Path(self.config.test_file)
        drug_col = self.config.drug_column
        target_col = self.config.target_column
        label_col = self.config.label_column

        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        full_train_df = pd.read_csv(train_file)
        logger.info(f"Loaded train: {len(full_train_df)} samples from {train_file}")

        train_df, val_df = train_test_split(
            full_train_df,
            test_size=self.config.val_ratio,
            random_state=self.config.seed,
            stratify=full_train_df[label_col],
        )
        logger.info(
            f"Split: {len(train_df)} train, {len(val_df)} val "
            f"(val_ratio={self.config.val_ratio})"
        )

        test_df = None
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded test: {len(test_df)} samples from {test_file}")
        else:
            logger.warning(f"Test file not found: {test_file}")

        self._load_embeddings(full_train_df, test_df)

        if self._use_embeddings_only():
            self._collate_fn = DataCollatorForPPIEmbedding()
        else:
            self._collate_fn = DataCollatorForPPI(
                drug_tokenizer=self.drug_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

        self.train_dataset = self._create_dataset(train_df, drug_col, target_col, label_col)
        self.val_dataset = self._create_dataset(val_df, drug_col, target_col, label_col)
        if test_df is not None:
            self.test_dataset = self._create_dataset(test_df, drug_col, target_col, label_col)

        self._compute_statistics(train_df, val_df, test_df, label_col)
        self._check_overlaps(train_df, val_df, test_df, drug_col, target_col)

    def _load_embeddings(
        self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame]
    ) -> None:
        """Load pre-computed embeddings from cache."""
        if not self._use_cached_embeddings():
            return

        drug_col = self.config.drug_column
        target_col = self.config.target_column

        all_drugs = set(train_df[drug_col].unique())
        all_targets = set(train_df[target_col].unique())
        if test_df is not None:
            all_drugs.update(test_df[drug_col].unique())
            all_targets.update(test_df[target_col].unique())

        embedding_cache = EmbeddingCache(Path(self.config.cache_dir))

        logger.info(f"Loading drug embeddings ({self.config.cache_drug_encoder_name})...")
        self.drug_embeddings = embedding_cache.load_embeddings(
            encoder_name=self.config.cache_drug_encoder_name,
            dataset_type=self.config.cache_dataset_type,
            sequences=list(all_drugs),
            role="drug",
        )
        logger.info(f"Loaded {len(self.drug_embeddings)} drug embeddings")

        logger.info(f"Loading target embeddings ({self.config.cache_target_encoder_name})...")
        self.target_embeddings = embedding_cache.load_embeddings(
            encoder_name=self.config.cache_target_encoder_name,
            dataset_type=self.config.cache_dataset_type,
            sequences=list(all_targets),
            role="target",
        )
        logger.info(f"Loaded {len(self.target_embeddings)} target embeddings")

    def _create_dataset(
        self,
        df: pd.DataFrame,
        drug_col: str,
        target_col: str,
        label_col: str,
    ) -> Dataset:
        drug_seqs = df[drug_col].tolist()
        target_seqs = df[target_col].tolist()
        labels = [float(x) for x in df[label_col].tolist()]

        if self._use_embeddings_only():
            return EmbeddingOnlyDataset(
                sequences_a=drug_seqs,
                sequences_b=target_seqs,
                labels=labels,
                embeddings_a=self.drug_embeddings,
                embeddings_b=self.target_embeddings,
            )
        else:
            return DualSequenceDataset(
                sequences_a=drug_seqs,
                sequences_b=target_seqs,
                labels=labels,
                tokenizer_a=self.drug_tokenizer,
                tokenizer_b=self.target_tokenizer,
                max_length_a=self.config.max_drug_length,
                max_length_b=self.config.max_target_length,
                embeddings_a=self.drug_embeddings,
                embeddings_b=self.target_embeddings,
            )

    def _compute_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        label_col: str,
    ) -> None:
        all_labels = pd.concat([train_df[label_col], val_df[label_col]])
        if test_df is not None:
            all_labels = pd.concat([all_labels, test_df[label_col]])

        unique_labels, counts = np.unique(all_labels, return_counts=True)

        self.data_stats = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df) if test_df is not None else 0,
            "num_classes": len(unique_labels),
            "class_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
        }
        logger.info(f"Class distribution: {self.data_stats['class_distribution']}")

    def _check_overlaps(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        drug_col: str,
        target_col: str,
    ) -> None:
        train_pairs = set(zip(train_df[drug_col], train_df[target_col]))
        val_pairs = set(zip(val_df[drug_col], val_df[target_col]))

        train_val_overlap = train_pairs & val_pairs
        if train_val_overlap:
            logger.warning(f"Train/Val overlap: {len(train_val_overlap)} pairs")

        if test_df is not None:
            test_pairs = set(zip(test_df[drug_col], test_df[target_col]))
            train_test_overlap = train_pairs & test_pairs
            val_test_overlap = val_pairs & test_pairs

            if train_test_overlap:
                logger.error(f"Train/Test overlap: {len(train_test_overlap)} pairs!")
            if val_test_overlap:
                logger.error(f"Val/Test overlap: {len(val_test_overlap)} pairs!")

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

    def predict_dataloader(self) -> Optional[DataLoader]:
        return self.test_dataloader()
