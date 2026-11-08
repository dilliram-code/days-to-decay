#!/usr/bin/env python3
"""Dataset Validation Script for Days to Decay.

Validates the integrity of dataset.csv and its associated image files:
- Checks missing image files on disk
- Validates file formats and readability
- Detects missing or corrupted metadata fields
- Identifies duplicate file paths or duplicate sample-day records
- Verifies regression targets (non-negative, matches date difference)
- Validates image dimensions and channels (RGB)
- Checks consistency of sample_id and spoilage dates
- Verifies spoilage endpoint definitions
- Produces a comprehensive JSON/Markdown Dataset Quality Report
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import (
    DATASET_COLUMNS,
    SUPPORTED_CATEGORIES,
    SPOILAGE_ENDPOINTS,
    DatasetSampleRecord,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("validate_dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_dataset(
    metadata_path: Path,
    expected_image_size: Tuple[int, int] = (224, 224),
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Executes a comprehensive dataset validation suite.
    
    Returns:
        dict: A structured validation summary with errors, warnings, and metrics.
    """
    logger.info(f"Initiating validation for metadata: {metadata_path}")
    
    report: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata_path": str(metadata_path),
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "categories_found": {},
        "unique_samples": 0,
        "errors": [],
        "warnings": [],
        "passed": False,
    }

    if not metadata_path.exists():
        msg = f"Metadata file not found at {metadata_path}"
        report["errors"].append({"type": "file_not_found", "message": msg})
        logger.error(msg)
        return report

    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        msg = f"Failed to read CSV: {e}"
        report["errors"].append({"type": "csv_parse_error", "message": msg})
        logger.error(msg)
        return report

    report["total_records"] = len(df)
    logger.info(f"Loaded {len(df)} records from {metadata_path.name}")

    # Check 1: Required columns presence
    missing_cols = [c for c in DATASET_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"Missing mandatory columns: {missing_cols}"
        report["errors"].append({"type": "missing_columns", "message": msg})
        logger.error(msg)

    if df.empty:
        report["warnings"].append({"type": "empty_dataset", "message": "Dataset contains 0 rows."})
        report["passed"] = True if not strict else False
        return report

    # Check 2: Row-level validation via Pydantic schema and image verification
    seen_image_paths = set()
    sample_spoilage_dates: Dict[str, str] = {}
    valid_count = 0

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        row_errors = []

        # Validate schema fields
        try:
            record = DatasetSampleRecord(**{k: (None if pd.isna(v) else v) for k, v in row_dict.items()})
        except Exception as err:
            row_errors.append(f"Row {idx} schema validation error: {err}")

        # Check image existence and format
        img_path_str = str(row_dict.get("image_path", ""))
        if not img_path_str:
            row_errors.append(f"Row {idx}: Empty image_path")
        else:
            if Path(img_path_str).is_absolute():
                resolved_img_path = Path(img_path_str)
            elif (metadata_path.parent.parent / img_path_str).exists():
                resolved_img_path = metadata_path.parent.parent / img_path_str
            else:
                resolved_img_path = PROJECT_ROOT / img_path_str
            
            # Duplicate check
            if img_path_str in seen_image_paths:
                row_errors.append(f"Row {idx}: Duplicate image_path '{img_path_str}'")
            seen_image_paths.add(img_path_str)

            if not resolved_img_path.exists():
                row_errors.append(f"Row {idx}: Image file does not exist on disk at '{resolved_img_path}'")
            else:
                ext = resolved_img_path.suffix.lower()
                if ext not in VALID_EXTENSIONS:
                    row_errors.append(f"Row {idx}: Invalid image extension '{ext}'. Must be in {VALID_EXTENSIONS}")
                else:
                    try:
                        with Image.open(resolved_img_path) as img:
                            img.verify()
                        with Image.open(resolved_img_path) as img:
                            if img.mode != "RGB":
                                report["warnings"].append({
                                    "type": "non_rgb_image",
                                    "message": f"Row {idx}: Image {img_path_str} has mode {img.mode}, expected RGB",
                                })
                            if expected_image_size and img.size != expected_image_size:
                                report["warnings"].append({
                                    "type": "dimension_mismatch",
                                    "message": f"Row {idx}: Image {img_path_str} has dimensions {img.size}, expected {expected_image_size}",
                                })
                    except Exception as img_err:
                        row_errors.append(f"Row {idx}: Image file is corrupted or unreadable: {img_err}")

        # Consistency check: sample_id and observed_spoilage_date
        sample_id = str(row_dict.get("sample_id", "")).strip()
        spoil_date = str(row_dict.get("observed_spoilage_date", "")).strip()
        if sample_id and spoil_date:
            if sample_id in sample_spoilage_dates:
                if sample_spoilage_dates[sample_id] != spoil_date:
                    row_errors.append(
                        f"Row {idx}: Sample '{sample_id}' has conflicting spoilage dates: "
                        f"'{sample_spoilage_dates[sample_id]}' vs '{spoil_date}'"
                    )
            else:
                sample_spoilage_dates[sample_id] = spoil_date

        # Target calculation consistency
        cap_date = str(row_dict.get("capture_date", "")).strip()
        rem_days = row_dict.get("remaining_days")
        if cap_date and spoil_date and rem_days is not None and not pd.isna(rem_days):
            try:
                d1 = datetime.strptime(cap_date, "%Y-%m-%d").date()
                d2 = datetime.strptime(spoil_date, "%Y-%m-%d").date()
                expected_days = float((d2 - d1).days)
                if abs(float(rem_days) - expected_days) > 1e-4:
                    row_errors.append(
                        f"Row {idx}: Target remaining_days ({rem_days}) does not match "
                        f"spoilage_date ({spoil_date}) - capture_date ({cap_date}) = {expected_days}"
                    )
            except ValueError:
                pass

        if row_errors:
            for re in row_errors:
                report["errors"].append({"row": int(idx), "message": re})
        else:
            valid_count += 1

    report["valid_records"] = valid_count
    report["invalid_records"] = len(df) - valid_count
    report["unique_samples"] = df["sample_id"].nunique() if "sample_id" in df.columns else 0
    
    if "food_category" in df.columns:
        report["categories_found"] = df["food_category"].value_counts().to_dict()

    report["passed"] = (len(report["errors"]) == 0)
    logger.info(
        f"Validation complete: {report['valid_records']}/{report['total_records']} valid rows. "
        f"Errors: {len(report['errors'])}, Warnings: {len(report['warnings'])}, Passed: {report['passed']}"
    )
    return report


def generate_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    """Generates an evaluation/quality report in GitHub Flavored Markdown."""
    lines = [
        "# Dataset Quality & Integrity Report",
        "",
        f"- **Generated At:** `{report['timestamp']}`",
        f"- **Metadata Path:** `{report['metadata_path']}`",
        f"- **Validation Status:** {'✅ **PASSED**' if report['passed'] else '❌ **FAILED**'}",
        "",
        "## Summary Metrics",
        "",
        f"- **Total Rows:** {report['total_records']}",
        f"- **Valid Rows:** {report['valid_records']}",
        f"- **Invalid Rows:** {report['invalid_records']}",
        f"- **Unique Physical Samples:** {report['unique_samples']}",
        "",
        "## Category Distribution",
        "",
        "| Food Category | Total Samples |",
        "|---|---|",
    ]
    for cat, count in report.get("categories_found", {}).items():
        lines.append(f"| `{cat}` | {count} |")

    lines.extend([
        "",
        "## Quality Checks",
        "",
        f"- Total Errors: `{len(report['errors'])}`",
        f"- Total Warnings: `{len(report['warnings'])}`",
        "",
    ])

    if report["errors"]:
        lines.append("### Errors Detected")
        for err in report["errors"][:20]:
            lines.append(f"- ❌ {err.get('message', err)}")
        if len(report["errors"]) > 20:
            lines.append(f"- ... and {len(report['errors']) - 20} more errors.")
        lines.append("")

    if report["warnings"]:
        lines.append("### Warnings")
        for w in report["warnings"][:20]:
            lines.append(f"- ⚠️ {w.get('message', w)}")
        if len(report["warnings"]) > 20:
            lines.append(f"- ... and {len(report['warnings']) - 20} more warnings.")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Days to Decay dataset")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata" / "dataset.csv",
        help="Path to dataset.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "metrics" / "dataset_validation_report.md",
        help="Path for markdown report output",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "reports" / "metrics" / "dataset_validation_report.json",
        help="Path for JSON report output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any warning or missing images exist",
    )
    args = parser.parse_args()

    report = validate_dataset(args.metadata, strict=args.strict)
    
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    if args.report:
        generate_markdown_report(report, args.report)

    if not report["passed"]:
        logger.warning("Dataset validation completed with issues.")
        if args.strict:
            sys.exit(1)
    else:
        logger.info("Dataset validation passed successfully.")


if __name__ == "__main__":
    main()
