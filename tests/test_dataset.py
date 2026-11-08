"""Unit tests for dataset loading, transforms, and zero-leakage group splitting."""
from pathlib import Path
import pandas as pd
import pytest
import torch

from scripts.validate_dataset import validate_dataset
from src.data.dataset import (
    PerishableFoodDataset,
    group_split_dataset,
    invert_target_transform,
    transform_target,
)
from src.data.transforms import get_eval_transforms, get_train_transforms


def test_dataset_loading_and_tensors(temp_dataset_dir: Path):
    """Verifies that PerishableFoodDataset correctly loads and formats tensors."""
    csv_path = temp_dataset_dir / "data" / "metadata" / "dataset.csv"
    df = pd.read_csv(csv_path)

    transform = get_eval_transforms(image_size=224)
    dataset = PerishableFoodDataset(df, root_dir=temp_dataset_dir, transform=transform)

    assert len(dataset) == 9
    sample = dataset[0]

    assert "image" in sample
    assert "target" in sample
    assert "raw_target" in sample
    assert "category" in sample
    assert "sample_id" in sample

    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert isinstance(sample["target"].item(), float)
    assert sample["target"].item() >= 0.0


def test_zero_leakage_group_split(temp_dataset_dir: Path):
    """Verifies that photographs of the same physical specimen never span multiple splits."""
    csv_path = temp_dataset_dir / "data" / "metadata" / "dataset.csv"
    df = pd.read_csv(csv_path)

    # 3 samples: BANANA_001, BANANA_002, BANANA_003
    train_df, val_df, test_df = group_split_dataset(
        df,
        train_ratio=0.34,
        val_ratio=0.33,
        test_ratio=0.33,
        group_col="sample_id",
        random_state=42,
    )

    train_groups = set(train_df["sample_id"].unique())
    val_groups = set(val_df["sample_id"].unique())
    test_groups = set(test_df["sample_id"].unique())

    # Assert strictly disjoint partitions
    assert len(train_groups.intersection(val_groups)) == 0, "Train and Val share sample_id!"
    assert len(train_groups.intersection(test_groups)) == 0, "Train and Test share sample_id!"
    assert len(val_groups.intersection(test_groups)) == 0, "Val and Test share sample_id!"

    # Assert all original samples are accounted for
    all_assigned = train_groups.union(val_groups).union(test_groups)
    assert all_assigned == set(df["sample_id"].unique())


def test_target_transforms():
    """Verifies forward and inverse target transformations."""
    y = np.array([0.0, 1.5, 5.0, 10.0])
    y_log = transform_target(y, "log1p")
    y_inv = invert_target_transform(y_log, "log1p")

    np.testing.assert_allclose(y, y_inv, atol=1e-5)

    # Negative value clipping
    y_neg = np.array([-2.5])
    y_clipped = invert_target_transform(y_neg, "none")
    assert y_clipped[0] >= 0.0


def test_validate_dataset_script(temp_dataset_dir: Path):
    """Tests the dataset validation engine on a compliant metadata structure."""
    csv_path = temp_dataset_dir / "data" / "metadata" / "dataset.csv"
    report = validate_dataset(csv_path, expected_image_size=(224, 224))

    assert report["passed"] is True
    assert report["total_records"] == 9
    assert report["valid_records"] == 9
    assert report["unique_samples"] == 3
    assert len(report["errors"]) == 0
