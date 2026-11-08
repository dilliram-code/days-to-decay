#!/usr/bin/env python3
"""CLI Prediction Utility for Days to Decay.

Takes an input image file and food category, runs inference using the
trained regression model, applies conformal prediction uncertainty intervals,
and prints structured prediction output.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import invert_target_transform
from src.data.schema import SUPPORTED_CATEGORIES, determine_aging_stage
from src.data.transforms import get_eval_transforms
from src.models.model_factory import build_model, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict")


def predict_shelf_life(
    image_path: Path,
    food_category: str,
    checkpoint_path: Path = PROJECT_ROOT / "artifacts" / "best_model.pth",
    calibration_path: Path = PROJECT_ROOT / "artifacts" / "calibration_residuals.json",
    image_size: int = 224,
    device_str: str = "auto",
) -> dict:
    """Runs single-image shelf-life inference."""
    cat = food_category.strip().lower()
    if cat not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Category '{cat}' not supported. Choose from {SUPPORTED_CATEGORIES}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Preprocess
    transform = get_eval_transforms(image_size=image_size)
    with Image.open(image_path) as raw_img:
        img_rgb = raw_img.convert("RGB")
    tensor_img = transform(img_rgb).unsqueeze(0).to(device)

    # Load Model
    model = build_model(backbone="resnet50", pretrained=False)
    load_checkpoint(checkpoint_path, model, map_location=str(device))
    model.eval()

    with torch.no_grad():
        pred_raw = model(tensor_img).cpu().item()

    pred_days = round(float(max(0.0, pred_raw)), 2)
    stage, condition = determine_aging_stage(cat, pred_days)

    # Conformal uncertainty calibration
    lower_bound = None
    upper_bound = None
    uncertainty_method = "unavailable"
    
    if calibration_path.exists():
        try:
            with open(calibration_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            q_val = float(calib.get("quantile_residual_days", 0.0))
            if q_val > 0:
                lower_bound = round(max(0.0, pred_days - q_val), 2)
                upper_bound = round(pred_days + q_val, 2)
                uncertainty_method = f"inductive_conformal_prediction ({int(calib.get('confidence_level', 0.9)*100)}% coverage)"
        except Exception as e:
            logger.warning(f"Could not load conformal calibration: {e}")

    result = {
        "success": True,
        "food_category": cat,
        "image_path": str(image_path),
        "predicted_remaining_days": pred_days,
        "lower_bound_days": lower_bound,
        "upper_bound_days": upper_bound,
        "uncertainty_method": uncertainty_method,
        "aging_stage": stage,
        "visual_condition": condition,
        "model_version": "1.0.0",
        "prediction_type": "experimental_estimate",
        "disclaimer": "This prediction is an empirical visual estimate and does not establish microbiological food safety.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Estimate food shelf life from photo")
    parser.add_argument("--image", type=Path, required=True, help="Path to photograph")
    parser.add_argument("--category", choices=SUPPORTED_CATEGORIES, required=True, help="Food category")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts" / "best_model.pth")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    args = parser.parse_args()

    res = predict_shelf_life(args.image, args.category, checkpoint_path=args.checkpoint)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n=== DAYS TO DECAY: Shelf-Life Estimation ===")
        print(f"Category:                 {res['food_category'].upper()}")
        print(f"Predicted Remaining Life: {res['predicted_remaining_days']} days")
        if res['lower_bound_days'] is not None:
            print(f"90% Prediction Interval:  {res['lower_bound_days']} - {res['upper_bound_days']} days ({res['uncertainty_method']})")
        print(f"Aging Stage:              {res['aging_stage']}")
        print(f"Visual Condition:         {res['visual_condition']}")
        print(f"Scientific Disclaimer:    {res['disclaimer']}\n")


if __name__ == "__main__":
    main()
