"""Upload file validation and security checks."""
import io
from pathlib import Path
from typing import Tuple
from fastapi import HTTPException, UploadFile, status
from PIL import Image

from backend.app.config import settings
from src.data.schema import SUPPORTED_CATEGORIES


def validate_category(category: str) -> str:
    """Validates that requested category is supported."""
    cat_clean = category.strip().lower()
    if cat_clean not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported food category '{category}'. Must be one of: {list(SUPPORTED_CATEGORIES)}",
        )
    return cat_clean


async def validate_uploaded_image(file: UploadFile) -> Tuple[bytes, str]:
    """
    Validates uploaded image file size, extension, MIME type, and structural integrity.
    """
    # 1. Filename & Extension check
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename.",
        )

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '.{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # 2. Content-Type check if provided
    if file.content_type and file.content_type not in settings.ALLOWED_MIME_TYPES:
        # Don't hard-fail on generic application/octet-stream if extension is valid, but reject non-images
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid MIME type '{file.content_type}'. Must be an image format.",
            )

    # 3. Read bytes & check size limit
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed upload size ({settings.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.1f} MB).",
        )

    # 4. Deep image verification with PIL
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image data: {e}",
        )

    return content, ext
