#!/usr/bin/env python3
"""Dataset Preprocessing Pipeline for Days to Decay.

Processes raw photographs into standardized 224x224 RGB images:
- Validates file integrity
- Resizes cleanly to target square dimensions
- Preserves aspect ratio if requested or standardizes via center-crop/bilinear resize
- Outputs standardized images into data/processed/<category>/
- Updates data/metadata/dataset.csv with processed image paths
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import DATASET_COLUMNS, SUPPORTED_CATEGORIES
from src.data.transforms import preprocess_pil_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess_dataset")


def process_dataset(
    metadata_csv: Path = PROJECT_ROOT / "data" / "metadata" / "dataset.csv",
    output_dir: Path = PROJECT_ROOT / "data" / "processed",
    image_size: int = 224,
    force: bool = False,
) -> int:
    """
    Standardizes all images listed in dataset.csv into 224x224 RGB format.
    """
    if not metadata_csv.exists() or metadata_csv.stat().st_size == 0:
        logger.warning(f"No metadata found at {metadata_csv}. Nothing to preprocess.")
        return 0

    df = pd.read_csv(metadata_csv)
    if df.empty:
        logger.warning("Metadata CSV contains 0 rows.")
        return 0

    processed_count = 0
    updated_rows = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        raw_path_str = str(row["image_path"])
        cat = str(row.get("food_category", "unknown")).lower()
        sample_id = str(row.get("sample_id", f"SAMPLE_{idx}"))
        
        raw_path = PROJECT_ROOT / raw_path_str if not Path(raw_path_str).is_absolute() else Path(raw_path_str)
        if not raw_path.exists():
            logger.error(f"Row {idx}: Raw image does not exist: {raw_path}")
            continue

        category_out_dir = output_dir / cat
        category_out_dir.mkdir(parents=True, exist_ok=True)

        target_filename = f"{raw_path.stem}.jpg"
        target_path = category_out_dir / target_filename
        rel_target_path = f"data/processed/{cat}/{target_filename}"

        if target_path.exists() and not force:
            logger.debug(f"Row {idx}: Target already exists: {rel_target_path}, skipping conversion.")
        else:
            try:
                with Image.open(raw_path) as raw_img:
                    proc_img = preprocess_pil_image(raw_img, image_size=image_size)
                    proc_img.save(target_path, "JPEG", quality=95)
                logger.debug(f"Processed row {idx} -> {rel_target_path}")
            except Exception as e:
                logger.error(f"Row {idx}: Failed to process {raw_path}: {e}")
                continue

        row_copy = row.to_dict()
        row_copy["image_path"] = rel_target_path
        updated_rows.append(row_copy)
        processed_count += 1

    if updated_rows:
        new_df = pd.DataFrame(updated_rows)
        # Update metadata CSV with processed paths
        new_df.to_csv(metadata_csv, index=False)
        logger.info(f"Preprocessing completed. {processed_count} images processed and saved to {output_dir}")
        logger.info(f"Updated metadata saved to {metadata_csv}")

    return processed_count


def main():
    parser = argparse.ArgumentParser(description="Preprocess food aging photographs to 224x224 RGB")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata" / "dataset.csv",
        help="Path to dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Processed images directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target square image resolution",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed images",
    )
    args = parser.parse_args()

    process_dataset(
        metadata_csv=args.metadata,
        output_dir=args.output_dir,
        image_size=args.image_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()

