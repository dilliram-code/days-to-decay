"""Model service: single-instance model loading, state caching, and inference."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from backend.app.config import settings
from src.data.dataset import invert_target_transform
from src.models.model_factory import build_model, load_checkpoint

logger = logging.getLogger("model_service")


class ModelService:
    """Manages loaded neural network weights, device placement, and inference execution."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device = self._select_device()
        self.checkpoint_loaded = False
        self.architecture = "resnet50"
        self.target_transform = "none"
        self.conformal_quantile: Optional[float] = None
        self.conformal_confidence: float = 0.90
        self.calibration_metadata: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}

    def _select_device(self) -> torch.device:
        cfg_dev = settings.DEVICE.lower()
        if cfg_dev == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(cfg_dev)

    def load_model(self) -> None:
        """Loads model weights and calibration once at application startup."""
        logger.info(f"Initializing ModelService on {self.device}")
        ckpt_path = Path(settings.MODEL_CHECKPOINT_PATH)
        meta_path = Path(settings.MODEL_METADATA_PATH)
        calib_path = Path(settings.CALIBRATION_FILE)

        # Build baseline architecture
        self.model = build_model(
            backbone=self.architecture,
            pretrained=False,
            dropout_rate=0.3,
            hidden_dim=256,
            activation="relu",
        ).to(self.device)

        if ckpt_path.exists():
            try:
                load_checkpoint(ckpt_path, self.model, map_location=str(self.device))
                self.checkpoint_loaded = True
                logger.info(f"Loaded trained checkpoint: {ckpt_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {ckpt_path}: {e}")
                self.checkpoint_loaded = False
        else:
            logger.warning(
                f"No checkpoint found at {ckpt_path}. Model running with initialized weights. "
                "Train the model or run demo mode."
            )
            self.checkpoint_loaded = False

        self.model.eval()

        # Load metadata if present
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.model_metadata = json.load(f)
                    self.target_transform = self.model_metadata.get("target_transform", "none")
            except Exception as e:
                logger.warning(f"Could not load metadata from {meta_path}: {e}")

        # Load conformal calibration if present
        if calib_path.exists():
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    self.calibration_metadata = json.load(f)
                self.conformal_quantile = float(self.calibration_metadata.get("quantile_residual_days", 0.0))
                self.conformal_confidence = float(self.calibration_metadata.get("confidence_level", 0.90))
                logger.info(
                    f"Loaded conformal calibration ({self.conformal_confidence*100:.0f}% coverage: +- {self.conformal_quantile:.2f}d)"
                )
            except Exception as e:
                logger.warning(f"Failed loading conformal calibration from {calib_path}: {e}")

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> float:
        """Executes forward pass and inverts any target transforms."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        tensor = tensor.to(self.device)
        raw_pred = self.model(tensor).cpu().item()
        natural_pred = invert_target_transform(raw_pred, self.target_transform)
        # Ensure physical non-negativity
        return max(0.0, float(natural_pred))

    def get_uncertainty_interval(self, pred_days: float) -> Tuple[Optional[float], Optional[float], str]:
        """Calculates prediction intervals using calibrated conformal residuals."""
        if self.conformal_quantile is not None and self.conformal_quantile > 0:
            lower = max(0.0, round(pred_days - self.conformal_quantile, 2))
            upper = round(pred_days + self.conformal_quantile, 2)
            method = f"conformal_prediction ({int(self.conformal_confidence * 100)}% coverage)"
            return lower, upper, method
        return None, None, "unavailable"


# Singleton instance
model_service = ModelService()
