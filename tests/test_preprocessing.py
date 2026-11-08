"""Unit tests for image preprocessing and normalization."""
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_eval_transforms,
    get_train_transforms,
    preprocess_pil_image,
)


def test_preprocess_pil_mode_and_size():
    """Verifies that arbitrary modes (RGBA, L) are converted to 224x224 RGB."""
    # RGBA image
    rgba_img = Image.new("RGBA", (400, 300), color=(255, 128, 0, 200))
    proc = preprocess_pil_image(rgba_img, image_size=224)
    assert proc.mode == "RGB"
    assert proc.size == (224, 224)

    # Grayscale image
    gray_img = Image.new("L", (150, 150), color=120)
    proc_gray = preprocess_pil_image(gray_img, image_size=224)
    assert proc_gray.mode == "RGB"
    assert proc_gray.size == (224, 224)


def test_eval_transforms_normalization():
    """Verifies deterministic output tensor shape and type."""
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    eval_tf = get_eval_transforms(image_size=224)
    tensor = eval_tf(img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # Verify tensor is not all zeros
    assert not torch.all(tensor == 0)


def test_train_transforms_shape():
    """Verifies train augmentation returns (3, 224, 224)."""
    img = Image.new("RGB", (256, 256), color=(50, 180, 80))
    train_tf = get_train_transforms(image_size=224)
    tensor = train_tf(img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
