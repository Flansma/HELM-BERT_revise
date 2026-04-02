"""K-fold result aggregation utilities.

Saves per-fold metrics and computes summary statistics (mean ± std).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Metric ordering for CSV output
REGRESSION_METRICS = ["r2", "rmse", "mae", "pearson"]
CLASSIFICATION_METRICS = [
    "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
    "f1", "accuracy", "precision", "recall",
]


class KFoldResultAggregator:
    """Collects per-fold metrics and produces summary statistics.

    Usage:
        agg = KFoldResultAggregator(results_dir, run_name, task_type="regression")
        for fold in range(n_folds):
            metrics = train_and_evaluate(fold)
            agg.add_fold(fold, metrics)
        agg.save_summary()
    """

    def __init__(
        self,
        results_dir: Path,
        run_name: str,
        task_type: str = "regression",
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.task_type = task_type
        self.fold_results: List[Dict[str, float]] = []

        self._metric_keys = (
            REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS
        )

    def add_fold(
        self,
        fold: int,
        metrics: Dict[str, float],
        duration: Optional[float] = None,
    ) -> None:
        """Record metrics for a single fold."""
        record = {"fold": fold}
        for key in self._metric_keys:
            record[key] = metrics.get(key, metrics.get(f"test_{key}", float("nan")))
        if duration is not None:
            record["duration_seconds"] = duration
        self.fold_results.append(record)

        logger.info(
            "Fold %d: %s",
            fold,
            ", ".join(f"{k}={record[k]:.4f}" for k in self._metric_keys if k in record),
        )

    def save_fold_results(self) -> Path:
        """Save per-fold results CSV."""
        df = pd.DataFrame(self.fold_results)
        path = self.results_dir / f"{self.run_name}_fold_results.csv"
        df.to_csv(path, index=False)
        logger.info("Fold results saved to %s", path)
        return path

    def save_summary(self) -> Path:
        """Compute mean ± std and save summary CSV."""
        # Save fold results first
        self.save_fold_results()

        # Compute summary
        df = pd.DataFrame(self.fold_results)
        rows = []
        for metric in self._metric_keys:
            if metric in df.columns:
                values = df[metric].dropna()
                rows.append({
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "n_folds": len(values),
                })

        summary_df = pd.DataFrame(rows)
        path = self.results_dir / f"{self.run_name}_summary.csv"
        summary_df.to_csv(path, index=False)

        # Log summary
        logger.info("=" * 60)
        logger.info("K-Fold Summary (%d folds)", len(self.fold_results))
        logger.info("=" * 60)
        for _, row in summary_df.iterrows():
            logger.info("  %s: %.4f ± %.4f", row["metric"], row["mean"], row["std"])

        logger.info("Summary saved to %s", path)
        return path
