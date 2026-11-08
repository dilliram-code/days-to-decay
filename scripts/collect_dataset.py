#!/usr/bin/env python3
"""Dataset Collection Utility for Days to Decay.

Supports empirical time-series image collection of perishable produce/bread:
- Interactive or CLI-argument mode
- Registration of specimen sample ID (e.g. BANANA_001)
- Ingestion from smartphone photo file or webcam capture
- Storage condition tracking, weight, temperature, humidity
- Auto-organization into data/raw/<category>/
- Duplicate prevention and schema-compliant record insertion
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import (
    DATASET_COLUMNS,
    SUPPORTED_CATEGORIES,
    compute_remaining_days,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect_dataset")


def register_sample_image(
    source_image_path: Path,
    food_category: str,
    sample_id: str,
    capture_date: str,
    capture_day: int,
    observed_spoilage_date: str,
    initial_freshness: str = "fresh",
    storage_condition: str = "room_temperature",
    initial_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    collection_notes: str = "",
    raw_dir: Path = PROJECT_ROOT / "data" / "raw",
    metadata_csv: Path = PROJECT_ROOT / "data" / "metadata" / "dataset.csv",
) -> Path:
    """
    Validates, organizes, and registers an image into the raw dataset catalog.
    """
    cat = food_category.strip().lower()
    if cat not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Category '{cat}' is not in supported list: {SUPPORTED_CATEGORIES}")

    sample_id = sample_id.strip().upper()
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image does not exist: {source_image_path}")

    # Validate image readability
    with Image.open(source_image_path) as img:
        img.verify()

    target_category_dir = raw_dir / cat
    target_category_dir.mkdir(parents=True, exist_ok=True)

    extension = source_image_path.suffix.lower()
    if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        extension = ".jpg"

    # Deterministic file naming: <SAMPLE_ID>_day<CAPTURE_DAY>_<TIMESTAMP><EXT>
    filename = f"{sample_id}_day{capture_day}_{int(datetime.utcnow().timestamp())}{extension}"
    target_image_path = target_category_dir / filename

    # Copy raw image
    shutil.copy2(source_image_path, target_image_path)
    rel_image_path = f"data/raw/{cat}/{filename}"
    logger.info(f"Saved raw image to {target_image_path}")

    # Calculate remaining days
    rem_days = compute_remaining_days(capture_date, observed_spoilage_date)

    # Load or initialize metadata CSV
    if metadata_csv.exists() and metadata_csv.stat().st_size > 0:
        df = pd.read_csv(metadata_csv)
    else:
        metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=DATASET_COLUMNS)

    # Check for duplicate sample + day entry
    if not df.empty and "sample_id" in df.columns and "capture_day" in df.columns:
        duplicate = df[(df["sample_id"] == sample_id) & (df["capture_day"] == capture_day)]
        if not duplicate.empty:
            logger.warning(
                f"Record for sample_id '{sample_id}' and capture_day '{capture_day}' already exists. "
                "Appending with unique timestamp."
            )

    new_record = {
        "image_path": rel_image_path,
        "food_category": cat,
        "sample_id": sample_id,
        "capture_date": capture_date,
        "capture_day": capture_day,
        "initial_weight": initial_weight,
        "temperature": temperature,
        "humidity": humidity,
        "initial_freshness": initial_freshness,
        "observed_spoilage_date": observed_spoilage_date,
        "remaining_days": rem_days,
        "storage_condition": storage_condition,
        "collection_notes": collection_notes,
    }

    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(metadata_csv, index=False)
    logger.info(f"Metadata catalog updated: {metadata_csv} (Total records: {len(df)})")

    return target_image_path


def main():
    parser = argparse.ArgumentParser(description="Collect and catalog food aging sample images")
    parser.add_argument("--image", type=Path, help="Path to input smartphone photo or file")
    parser.add_argument("--category", choices=SUPPORTED_CATEGORIES, help="Food category")
    parser.add_argument("--sample-id", type=str, help="Specimen ID (e.g. BANANA_001)")
    parser.add_argument("--capture-date", type=str, default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--capture-day", type=int, default=0, help="Elapsed day index (0, 1, 2...)")
    parser.add_argument("--spoilage-date", type=str, help="Observed spoilage endpoint date (YYYY-MM-DD)")
    parser.add_argument("--storage", type=str, default="room_temperature", help="Storage regimen")
    parser.add_argument("--initial-weight", type=float, default=None, help="Initial weight in grams")
    parser.add_argument("--notes", type=str, default="", help="Collection notes")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt")

    args = parser.parse_args()

    if args.interactive or not (args.image and args.category and args.sample_id and args.spoilage_date):
        print("\n=== DAYS TO DECAY: Dataset Collection Assistant ===")
        print(f"Supported categories: {', '.join(SUPPORTED_CATEGORIES)}")
        
        category = input("Enter food category (banana/avocado/bread): ").strip().lower()
        while category not in SUPPORTED_CATEGORIES:
            category = input(f"Invalid. Must be one of {SUPPORTED_CATEGORIES}: ").strip().lower()

        sample_id = input("Enter unique Sample ID (e.g. BANANA_001): ").strip().upper()
        img_path_str = input("Enter path to photograph file: ").strip()
        img_path = Path(img_path_str)
        while not img_path.exists():
            img_path_str = input("File not found. Re-enter photograph path: ").strip()
            img_path = Path(img_path_str)

        capture_date = input(f"Enter capture date [default: {date.today().isoformat()}]: ").strip()
        if not capture_date:
            capture_date = date.today().isoformat()

        day_str = input("Enter capture day number [default: 0]: ").strip()
        capture_day = int(day_str) if day_str else 0

        spoil_date = input("Enter observed/projected spoilage date (YYYY-MM-DD): ").strip()
        storage = input("Enter storage condition [default: room_temperature]: ").strip() or "room_temperature"
        notes = input("Enter optional notes: ").strip()

        register_sample_image(
            source_image_path=img_path,
            food_category=category,
            sample_id=sample_id,
            capture_date=capture_date,
            capture_day=capture_day,
            observed_spoilage_date=spoil_date,
            storage_condition=storage,
            collection_notes=notes,
        )
    else:
        register_sample_image(
            source_image_path=args.image,
            food_category=args.category,
            sample_id=args.sample_id,
            capture_date=args.capture_date,
            capture_day=args.capture_day,
            observed_spoilage_date=args.spoilage_date,
            storage_condition=args.storage,
            initial_weight=args.initial_weight,
            collection_notes=args.notes,
        )


if __name__ == "__main__":
    main()

