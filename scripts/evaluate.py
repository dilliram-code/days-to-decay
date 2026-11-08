#!/usr/bin/env python3
"""Independent Model Evaluation Script for Days to Decay.

Evaluates trained checkpoint on held-out test split, benchmarks against
naive baseline mean prediction, and generates comprehensive figures and reports.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PerishableFoodDataset
from src.data.transforms import get_eval_transforms
from src.evaluation.evaluator import ModelEvaluator
from src.models.model_factory import build_model, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


def evaluate_pipeline(
    checkpoint_path: Path,
    test_csv: Path,
    config_path: Path,
    reports_dir: Path,
) -> Dict[str, Any]:
    """Runs test evaluation suite and generates reports."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}")

    df_test = pd.read_csv(test_csv)
    logger.info(f"Loaded {len(df_test)} test records from {test_csv}")

    # Baseline calculation from training split if available
    train_split_csv = test_csv.parent / "train_split.csv"
    baseline_mean = None
    if train_split_csv.exists():
        df_train = pd.read_csv(train_split_csv)
        baseline_mean = float(df_train["remaining_days"].mean())
        logger.info(f"Baseline mean training shelf-life: {baseline_mean:.2f} days")

    # Dataloader
    img_size = int(config.get("data", {}).get("image_size", 224))
    eval_transform = get_eval_transforms(image_size=img_size)
    target_transform = config.get("data", {}).get("target_transform", "none")

    test_dataset = PerishableFoodDataset(
        df_test,
        root_dir=PROJECT_ROOT,
        transform=eval_transform,
        target_transform=target_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model
    model_cfg = config.get("model", {})
    model = build_model(
        backbone=model_cfg.get("backbone", "resnet50"),
        pretrained=False,
        dropout_rate=float(model_cfg.get("dropout_rate", 0.3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        activation=model_cfg.get("activation", "relu"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(checkpoint_path, model, map_location=str(device))

    evaluator = ModelEvaluator(
        model=model,
        device=device,
        target_transform=target_transform,
        baseline_mean_days=baseline_mean,
    )

    eval_output = evaluator.evaluate_test_set(test_loader)
    history_path = PROJECT_ROOT / config.get("artifacts", {}).get("history_path", "artifacts/training_history.csv")
    
    evaluator.generate_full_evaluation_report(
        eval_output=eval_output,
        reports_dir=reports_dir,
        history_csv=history_path if history_path.exists() else None,
        model_name=f"FoodAgingRegressor ({model_cfg.get('backbone', 'resnet50')})",
    )

    logger.info("Evaluation pipeline completed successfully.")
    return eval_output["metrics"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained food aging regressor")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "best_model.pth",
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits" / "test_split.csv",
        help="Path to test_split.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_config.yaml",
        help="Path to train_config.yaml",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Reports output directory",
    )
    args = parser.parse_args()

    evaluate_pipeline(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        config_path=args.config,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
