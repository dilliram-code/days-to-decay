"""Unit tests for neural network regression architecture and metrics."""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.model_factory import build_model, load_checkpoint, save_checkpoint
from src.models.regression_model import FoodAgingRegressor
from src.training.metrics import compute_regression_metrics


def test_resnet50_forward_shape():
    """Verifies that ResNet50 produces a continuous single-value scalar per image."""
    model = build_model(backbone="resnet50", pretrained=False, activation="relu")
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 1)
    assert output.dtype == torch.float32
    # ReLU ensures non-negativity
    assert torch.all(output >= 0.0)


def test_backbone_freezing():
    """Verifies that freeze_backbone disables gradients for feature extractor."""
    model = build_model(backbone="resnet50", pretrained=False)
    model.freeze_backbone(True)

    for param in model.backbone.parameters():
        assert param.requires_grad is False

    # Regression head must remain trainable
    for param in model.regression_head.parameters():
        assert param.requires_grad is True

    model.freeze_backbone(False)
    for param in model.backbone.parameters():
        assert param.requires_grad is True


def test_checkpoint_roundtrip():
    """Verifies checkpoint saving and weight reloading."""
    model = build_model(backbone="resnet18", pretrained=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pth"
        metrics = {"mae": 0.45, "rmse": 0.62}
        config = {"model": {"backbone": "resnet18"}}

        save_checkpoint(model, optimizer=None, epoch=1, val_loss=0.35, metrics=metrics, config=config, save_path=ckpt_path)
        assert ckpt_path.exists()

        loaded_model = build_model(backbone="resnet18", pretrained=False)
        checkpoint = load_checkpoint(ckpt_path, loaded_model)

        assert checkpoint["epoch"] == 1
        assert checkpoint["val_loss"] == 0.35
        assert checkpoint["metrics"]["mae"] == 0.45


def test_regression_metrics_calculation():
    """Verifies mathematical correctness of evaluation metrics."""
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([2.5, 4.0, 5.0, 8.5])
    # Absolute errors: [0.5, 0.0, 1.0, 0.5]
    # MAE = 2.0 / 4 = 0.5
    # MedAE = median([0, 0.5, 0.5, 1.0]) = 0.5

    metrics = compute_regression_metrics(y_true, y_pred, tolerances=(0.5, 1.0))

    assert metrics["mae"] == 0.5
    assert metrics["medae"] == 0.5
    assert metrics["within_0.5d_pct"] == 75.0  # 3 of 4 <= 0.5
    assert metrics["within_1.0d_pct"] == 100.0 # 4 of 4 <= 1.0

