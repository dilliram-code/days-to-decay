#!/usr/bin/env python3
"""Training Execution Script for Days to Decay.

Loads training configuration, partitions dataset via group-based sample_id splitting,
executes transfer learning regression training with early stopping, saves best model,
and calibrates conformal prediction intervals.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PerishableFoodDataset, group_split_dataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.model_factory import build_model
from src.training.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_pipeline(config_path: Path, epochs_override: int = None) -> Dict[str, Any]:
    """Runs the complete training workflow based on configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if epochs_override is not None:
        config.setdefault("training", {})["epochs"] = epochs_override

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    metadata_path = PROJECT_ROOT / config.get("data", {}).get("metadata_path", "data/metadata/dataset.csv")
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}. Please collect data or run synthetic generation.")

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise ValueError(f"Dataset metadata at {metadata_path} is empty.")

    logger.info(f"Loaded {len(df)} records from {metadata_path}")

    # Group-based splitting by sample_id
    splits_cfg = config.get("data", {}).get("splits", {})
    train_ratio = float(splits_cfg.get("train_ratio", 0.70))
    val_ratio = float(splits_cfg.get("val_ratio", 0.15))
    test_ratio = float(splits_cfg.get("test_ratio", 0.15))
    group_col = splits_cfg.get("group_column", "sample_id")

    df_train, df_val, df_test = group_split_dataset(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_col=group_col,
        random_state=seed,
    )

    # Save splits for exact reproducibility during independent evaluation
    splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(splits_dir / "train_split.csv", index=False)
    df_val.to_csv(splits_dir / "val_split.csv", index=False)
    df_test.to_csv(splits_dir / "test_split.csv", index=False)
    logger.info(f"Split artifacts saved to {splits_dir}")

    # Dataset & Dataloaders
    img_size = int(config.get("data", {}).get("image_size", 224))
    train_transform = get_train_transforms(image_size=img_size)
    eval_transform = get_eval_transforms(image_size=img_size)
    target_transform = config.get("data", {}).get("target_transform", "none")

    train_dataset = PerishableFoodDataset(
        df_train,
        root_dir=PROJECT_ROOT,
        transform=train_transform,
        target_transform=target_transform,
    )
    val_dataset = PerishableFoodDataset(
        df_val,
        root_dir=PROJECT_ROOT,
        transform=eval_transform,
        target_transform=target_transform,
    )

    batch_size = int(config.get("training", {}).get("batch_size", 16))
    num_workers = int(config.get("training", {}).get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Model
    model_cfg = config.get("model", {})
    model = build_model(
        backbone=model_cfg.get("backbone", "resnet50"),
        pretrained=model_cfg.get("pretrained", True),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        activation=model_cfg.get("activation", "relu"),
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    results = trainer.fit()
    logger.info(f"Training run completed: Best Val Loss = {results['best_val_loss']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Days to Decay shelf-life regressor")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epochs override",
    )
    args = parser.parse_args()
    train_pipeline(args.config, epochs_override=args.epochs)


if __name__ == "__main__":
    main()

