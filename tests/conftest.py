"""Shared pytest fixtures for Days to Decay test suite."""
import io
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from src.data.schema import DATASET_COLUMNS


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """TestClient instance for FastAPI integration tests."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Generates a synthetic 224x224 RGB image for software testing."""
    # Create simple colored circle on light background
    arr = np.full((224, 224, 3), 240, dtype=np.uint8)
    # Add a yellowish-green patch
    arr[50:180, 50:180] = [210, 200, 40]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_image_bytes(sample_pil_image: Image.Image) -> bytes:
    """Yields JPEG byte stream of sample image."""
    buffer = io.BytesIO()
    sample_pil_image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def temp_dataset_dir(sample_pil_image: Image.Image) -> Generator[Path, None, None]:
    """Creates a temporary workspace with raw and processed images and dataset.csv."""
    temp_dir = Path(tempfile.mkdtemp(prefix="days_to_decay_test_"))
    raw_dir = temp_dir / "data" / "raw"
    proc_dir = temp_dir / "data" / "processed"
    meta_dir = temp_dir / "data" / "metadata"

    for cat in ["banana", "avocado", "bread"]:
        (raw_dir / cat).mkdir(parents=True, exist_ok=True)
        (proc_dir / cat).mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # Create 3 unique samples with 3 days each = 9 images total (enables group splitting)
    for sample_idx in range(1, 4):
        sample_id = f"BANANA_{sample_idx:03d}"
        for day in range(3):
            rel_path = f"data/processed/banana/{sample_id}_day{day}.jpg"
            full_img_path = temp_dir / rel_path
            sample_pil_image.save(full_img_path, format="JPEG")

            records.append({
                "image_path": rel_path,
                "food_category": "banana",
                "sample_id": sample_id,
                "capture_date": f"2026-11-0{day+1}",
                "capture_day": day,
                "initial_weight": 140.0,
                "temperature": 21.0,
                "humidity": 55.0,
                "initial_freshness": "fresh",
                "observed_spoilage_date": "2026-11-08",
                "remaining_days": float(7 - day),
                "storage_condition": "room_temperature",
                "collection_notes": "Test fixture",
            })

    df = pd.DataFrame(records, columns=DATASET_COLUMNS)
    df.to_csv(meta_dir / "dataset.csv", index=False)

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)
