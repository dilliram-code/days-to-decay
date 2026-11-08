"""Image transformation and preprocessing pipelines for Days to Decay."""
from typing import Tuple
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns data augmentation pipeline for training.
    
    Augmentation is specifically calibrated for perishable food aging:
    - Moderate geometric rotations and flips
    - Subtle brightness/contrast jitter (avoids destroying yellow/brown degradation cues)
    - Resized crop ensuring context preservation
    """
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Crucial: Mild jitter only. Aggressive hue shift destroys banana yellow / browning cues!
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.08,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns deterministic preprocessing pipeline for validation, testing, and production inference.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_pil_image(image: Image.Image, image_size: int = 224) -> Image.Image:
    """
    Ensures RGB mode and resizes cleanly to target square dimensions.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

