"""Factory utilities for instantiating, checkpointing, and exporting regression models."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.regression_model import FoodAgingRegressor

logger = logging.getLogger("model_factory")


def build_model(
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    hidden_dim: int = 256,
    activation: str = "relu",
) -> FoodAgingRegressor:
    """Builds and returns a FoodAgingRegressor instance."""
    logger.info(f"Instantiating FoodAgingRegressor with backbone={backbone}, pretrained={pretrained}, activation={activation}")
    return FoodAgingRegressor(
        backbone_name=backbone,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        activation=activation,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_path: Union[str, Path],
) -> Path:
    """Saves a training checkpoint containing weights, optimizer state, metrics, and hyperparams."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_loss": val_loss,
        "metrics": metrics,
        "config": config,
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Saved model checkpoint to {save_path} (epoch {epoch}, val_loss {val_loss:.4f})")
    return save_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Loads weights and state from a checkpoint into a model instance."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    # Support both pure state_dict and full checkpoint dicts
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Successfully loaded model weights from {checkpoint_path}")
    return checkpoint


def export_model_metadata(metadata: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Exports model card and operational metadata to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata exported to {output_path}")
    return output_path
