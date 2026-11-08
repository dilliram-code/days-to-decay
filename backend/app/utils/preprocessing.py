"""Inference image preprocessing utility."""
import io
from typing import Tuple
from PIL import Image
import torch

from src.data.transforms import get_eval_transforms


def bytes_to_tensor(image_bytes: bytes, image_size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """
    Decodes image bytes into a standardized RGB PIL image and normalized torch tensor.
    """
    with Image.open(io.BytesIO(image_bytes)) as raw_img:
        img_rgb = raw_img.convert("RGB")
    
    transform = get_eval_transforms(image_size=image_size)
    tensor = transform(img_rgb).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor, img_rgb
