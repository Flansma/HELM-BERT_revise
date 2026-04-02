#!/usr/bin/env python
"""PPI (Peptide-Protein Interaction) training script.

Supports both single-split and k-fold cross-validation modes.

Usage:
    # Single-split
    python scripts/train_ppi.py --config configs/ppi_random.yaml
    python scripts/train_ppi.py --config configs/ppi_acsm.yaml

    # K-fold cross-validation
    python scripts/train_ppi.py --config configs/ppi_random_kfold.yaml
    python scripts/train_ppi.py --config configs/ppi_acsm_kfold.yaml
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.append(str(Path(__file__).parent.parent))

import lightning as L
import numpy as np
import pandas as pd
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scripts.training_utils import (
    SEPARATOR_LINE,
    config_to_checkpoint_config,
    config_to_display_config,
    create_callbacks,
    create_output_dirs,
    load_best_checkpoint,
    load_config,
    log_completion,
    log_header,
    log_summary,
    log_training_start,
    mark_completion,
    setup_logging,
    setup_training_env,
    to_dict,
    build_tags,
)
from src.datamodules import PPIDataModule, PPIDataConfig
from src.models.ppi_lightning import HELMGLaMLightning, PPITrainingConfig


def _build_training_config(config):
    esm_hidden_sizes = OmegaConf.to_container(config.esm_hidden_sizes, resolve=True)
    return PPITrainingConfig(
        encoder_lr=config.training.encoder_lr,
        head_lr=config.training.head_lr,
        weight_decay=config.training.weight_decay,
        max_epochs=config.training.max_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        mlp_dropout=config.model.head.dropout,
        num_classes=config.model.head.num_classes,
        pos_weight=OmegaConf.select(config, "model.head.pos_weight", default=None),
        freeze_drug_encoder=config.model.drug_encoder.freeze,
        freeze_target_encoder=config.model.target_encoder.freeze,
        use_cached_embeddings=config.training.use_cached_embeddings,
        target_encoder=config.model.target_encoder.pretrained_path,
        esm_hidden_sizes=esm_hidden_sizes,
        prediction_threshold=config.classification.prediction_threshold,
    )


def _build_data_config(config):
    return PPIDataConfig(
        train_file=OmegaConf.select(config, "data.train_file", default=""),
        test_file=OmegaConf.select(config, "data.test_file", default=""),
        kfold_file=OmegaConf.select(config, "data.kfold_file", default=""),
        n_folds=OmegaConf.select(config, "data.n_folds", default=5),
        fold_column=OmegaConf.select(config, "data.fold_column", default="fold"),
        split_column=OmegaConf.select(config, "data.split_column", default="split"),
        drug_column=config.data.drug_column,
        target_column=config.data.target_column,
        label_column=config.data.label_column,
        target_encoder=config.model.target_encoder.pretrained_path,
        val_ratio=config.data.val_ratio,
        batch_size=config.training.batch_size,
        max_drug_length=config.data.max_drug_length,
        max_target_length=config.data.max_target_length,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.training.seed,
        use_cached_embeddings=config.training.use_cached_embeddings,
        cache_dir=config.paths.cache_dir,
        drug_encoder=config.model.drug_encoder.pretrained_path,
        trust_remote_code=config.model.trust_remote_code,
        cache_drug_encoder_name=config.cache.drug_encoder_name,
        cache_target_encoder_name=config.cache.target_encoder_name,
        cache_dataset_type=config.cache.dataset_type,
    )


def _evaluate_and_save(trainer, model, datamodule, results_dir, run_name, config, logger):
    """Run test evaluation, save predictions and metrics. Returns metrics dict."""
    if datamodule.test_dataset is None:
        logger.warning("No test dataset found, skipping evaluation")
        return {}

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_output = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    logits = np.concatenate([b["predictions"].cpu().numpy() for b in predictions_output]).flatten()
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    targets = np.concatenate([b["targets"].cpu().numpy() for b in predictions_output]).flatten()

    threshold = config.classification.prediction_threshold
    pred_labels = (probs >= threshold).astype(int)

    pred_df = pd.DataFrame({
        "pred_prob": probs,
        "pred_label": pred_labels,
        "actual": targets.astype(int),
    })
    pred_file = results_dir / f"predictions_{run_name}.csv"
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"Saved predictions to {pred_file}")

    metrics = trainer.test(model, datamodule)[0]
    logger.info("Test Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    metrics_file = results_dir / f"metrics_{run_name}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    return metrics


def train_single_fold(fold, config, datamodule, base_output_dir, config_dict, logger):
    """Train a single fold. Returns metrics dict and duration."""
    fold_start = time.time()

    fold_name = f"fold_{fold}"
    output_dir, checkpoint_dir = create_output_dirs(base_output_dir, fold_name)

    datamodule.setup_fold(fold)

    training_config = _build_training_config(config)

    model = HELMGLaMLightning(
        drug_model_path=config.model.drug_encoder.pretrained_path,
        training_config=training_config,
        trust_remote_code=config.model.trust_remote_code,
    )

    checkpoint_config = config_to_checkpoint_config(config)
    display_config = config_to_display_config(config)
    callbacks = create_callbacks(
        checkpoint_dir,
        config.training.early_stopping_patience,
        checkpoint_config,
        display_config,
    )

    run_name = f"{config.experiment_name}_fold_{fold}"
    wandb_logger = None if config.logging.disable_wandb else WandbLogger(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=run_name,
        save_dir=output_dir,
        config=config_dict,
        tags=build_tags(config, ["ppi", "kfold", f"fold{fold}"]),
    )

    trainer = L.Trainer(
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=config.training.gradient_clip_val,
        deterministic=config.trainer.deterministic,
        default_root_dir=output_dir,
        log_every_n_steps=config.trainer.log_every_n_steps,
    )

    logger.info(f"--- Fold {fold} ---")
    trainer.fit(model, datamodule)

    model = load_best_checkpoint(trainer, HELMGLaMLightning, strict=False)
    metrics = _evaluate_and_save(
        trainer, model, datamodule, output_dir, run_name, config, logger,
    )

    mark_completion(output_dir)

    # Finalize wandb run for this fold before starting next
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

    fold_duration = time.time() - fold_start
    logger.info(f"Fold {fold} completed in {fold_duration / 60:.1f} min")

    return metrics, fold_duration


def train_kfold(config, logger):
    """Run k-fold cross-validation for PPI."""
    from scripts.utils.result_aggregator import KFoldResultAggregator

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_config = _build_data_config(config)
    config_dict = to_dict(config)

    run_name = f"{config.experiment_name}_{timestamp}"
    base_output_dir = Path(config.paths.output_dir) / run_name
    base_output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    drug_tokenizer = AutoTokenizer.from_pretrained(
        config.model.drug_encoder.pretrained_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    datamodule = PPIDataModule(config=data_config, drug_tokenizer=drug_tokenizer)
    datamodule.prepare_data()
    datamodule.setup()
    fold_ids = datamodule.get_fold_ids()

    results_dir = Path(config.paths.results_dir)
    aggregator = KFoldResultAggregator(results_dir, run_name, task_type="classification")

    logger.info(
        "Starting %d-fold cross-validation (%s, folds=%s)",
        len(fold_ids),
        config.experiment_name,
        fold_ids,
    )
    logger.info(SEPARATOR_LINE)

    for fold in fold_ids:
        metrics, duration = train_single_fold(
            fold, config, datamodule, base_output_dir, config_dict, logger,
        )
        aggregator.add_fold(fold, metrics, duration)

    aggregator.save_summary()
    mark_completion(base_output_dir)

    total_duration = time.time() - start_time
    log_summary(logger, total_duration, base_output_dir)
    log_completion(logger, "K-fold PPI training")


def train_single_split(config, logger):
    """Legacy single-split training."""
    start_time = time.time()

    data_config = _build_data_config(config)
    config_dict = to_dict(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{timestamp}"
    output_dir, checkpoint_dir = create_output_dirs(Path(config.paths.output_dir), run_name)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Drug encoder: {config.model.drug_encoder.pretrained_path}")
    logger.info(f"Target encoder: {config.model.target_encoder.pretrained_path}")
    logger.info(f"Use cached embeddings: {config.training.use_cached_embeddings}")
    logger.info(f"Max epochs: {config.training.max_epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(SEPARATOR_LINE)

    drug_tokenizer = AutoTokenizer.from_pretrained(
        config.model.drug_encoder.pretrained_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    datamodule = PPIDataModule(config=data_config, drug_tokenizer=drug_tokenizer)
    datamodule.prepare_data()
    training_config = _build_training_config(config)

    model = HELMGLaMLightning(
        drug_model_path=config.model.drug_encoder.pretrained_path,
        training_config=training_config,
        trust_remote_code=config.model.trust_remote_code,
    )

    checkpoint_config = config_to_checkpoint_config(config)
    display_config = config_to_display_config(config)
    callbacks = create_callbacks(
        checkpoint_dir,
        config.training.early_stopping_patience,
        checkpoint_config,
        display_config,
    )
    wandb_logger = None if config.logging.disable_wandb else WandbLogger(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=run_name,
        save_dir=output_dir,
        config=config_dict,
        tags=build_tags(config, ["ppi", "downstream", "classification"]),
    )

    trainer = L.Trainer(
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=config.training.gradient_clip_val,
        deterministic=config.trainer.deterministic,
        default_root_dir=output_dir,
        log_every_n_steps=config.trainer.log_every_n_steps,
    )

    log_training_start(logger, "PPI training")
    trainer.fit(model, datamodule)

    model = load_best_checkpoint(trainer, HELMGLaMLightning, strict=False)
    _evaluate_and_save(trainer, model, datamodule, config.paths.results_dir, run_name, config, logger)

    training_duration = time.time() - start_time
    log_summary(logger, training_duration, output_dir)
    mark_completion(output_dir)
    log_completion(logger, "PPI training")


def main():
    if "--config" not in sys.argv:
        print("Error: --config is required for PPI training.")
        print("  python scripts/train_ppi.py --config configs/ppi_random.yaml")
        print("  python scripts/train_ppi.py --config configs/ppi_acsm.yaml")
        print("  python scripts/train_ppi.py --config configs/ppi_random_kfold.yaml")
        sys.exit(1)

    config = load_config(task="ppi")
    setup_training_env(config.training.seed, config.trainer.float32_matmul_precision)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_log_dir = Path(config.paths.output_dir)
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(tmp_log_dir, timestamp, "train_ppi")
    log_header(logger, "PPI Classification Training")

    is_kfold = bool(OmegaConf.select(config, "data.kfold_file", default=""))

    if is_kfold:
        logger.info("Mode: K-fold cross-validation")
        train_kfold(config, logger)
    else:
        logger.info("Mode: Single-split")
        train_single_split(config, logger)


if __name__ == "__main__":
    main()
