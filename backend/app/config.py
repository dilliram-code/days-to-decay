"""Application configuration and environment settings."""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    APP_NAME: str = "DAYS TO DECAY"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Model and artifacts
    MODEL_CHECKPOINT_PATH: str = str(PROJECT_ROOT / "artifacts" / "best_model.pth")
    MODEL_METADATA_PATH: str = str(PROJECT_ROOT / "artifacts" / "model_metadata.json")
    CALIBRATION_FILE: str = str(PROJECT_ROOT / "artifacts" / "calibration_residuals.json")
    DEVICE: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Upload and security
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
    ALLOWED_MIME_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
