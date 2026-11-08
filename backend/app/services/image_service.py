"""Image handling and preprocessing service."""
import logging
from typing import Tuple
from fastapi import UploadFile
from PIL import Image
import torch

from backend.app.utils.preprocessing import bytes_to_tensor
from backend.app.utils.validation import validate_uploaded_image

logger = logging.getLogger("image_service")


class ImageService:
    """Coordinates file extraction and conversion to tensor format."""

    @staticmethod
    async def process_upload(file: UploadFile, image_size: int = 224) -> Tuple[torch.Tensor, str]:
        """Validates upload file and yields processed torch tensor and file format."""
        raw_bytes, ext = await validate_uploaded_image(file)
        tensor, _ = bytes_to_tensor(raw_bytes, image_size=image_size)
        return tensor, ext


image_service = ImageService()
