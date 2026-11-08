"""PyTorch Dataset implementation and Group-Aware Splitting for Days to Decay."""
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

logger = logging.getLogger("dataset")


def transform_target(y: Union[float, np.ndarray, torch.Tensor], transform_type: str = "none"):
    """Applies monotonic forward target transformation (e.g. log1p)."""
    if transform_type == "log1p":
        if isinstance(y, torch.Tensor):
            return torch.log1p(torch.clamp(y, min=0.0))
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    return y


def invert_target_transform(y: Union[float, np.ndarray, torch.Tensor], transform_type: str = "none"):
    """Inverts target transformation back to natural remaining days."""
    if transform_type == "log1p":
        if isinstance(y, torch.Tensor):
            return torch.clamp(torch.expm1(y), min=0.0)
        return np.clip(np.expm1(y), a_min=0.0, a_max=None)
    if isinstance(y, torch.Tensor):
        return torch.clamp(y, min=0.0)
    return np.clip(y, a_min=0.0, a_max=None)


class PerishableFoodDataset(Dataset):
    """
    PyTorch Dataset representing multi-category perishable food images
    with continuous remaining shelf-life targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_column: str = "remaining_days",
        target_transform: str = "none",
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.transform = transform
        self.target_column = target_column
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_rel_path = str(row["image_path"])
        full_path = self.root_dir / img_rel_path if not Path(img_rel_path).is_absolute() else Path(img_rel_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found at path: {full_path}")

        try:
            with Image.open(full_path) as raw_img:
                image = raw_img.convert("RGB")
        except Exception as e:
            raise IOError(f"Corrupted or unreadable image at {full_path}: {e}")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        raw_target = float(row[self.target_column])
        transformed_target = transform_target(raw_target, self.target_transform)

        return {
            "image": image_tensor,
            "target": torch.tensor([transformed_target], dtype=torch.float32),
            "raw_target": torch.tensor([raw_target], dtype=torch.float32),
            "category": str(row.get("food_category", "unknown")),
            "sample_id": str(row.get("sample_id", "unknown")),
            "image_path": img_rel_path,
        }


def group_split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_col: str = "sample_id",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train, validation, and test splits strictly grouped by `sample_id`.
    
    Prevents data leakage: All images captured over time for a single physical specimen
    (e.g., BANANA_001 on Day 0, Day 1, Day 2...) reside strictly inside one split.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-4, f"Splits must sum to 1.0, got {total_ratio}"
    
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    unique_groups = df[group_col].unique()
    num_groups = len(unique_groups)

    if num_groups < 3:
        logger.warning(
            f"Only {num_groups} unique samples found. Group splitting requires at least 3 samples "
            "to form distinct train, val, and test partitions. Falling back to non-overlapping allocation."
        )

    # First split: (train + val) vs test
    test_size_ratio = test_ratio / 1.0
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size_ratio, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df[group_col]))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Second split: train vs val from train_val pool
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_ratio, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val[group_col]))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Verify zero group leakage
    train_groups = set(df_train[group_col].unique())
    val_groups = set(df_val[group_col].unique())
    test_groups = set(df_test[group_col].unique())

    assert len(train_groups.intersection(val_groups)) == 0, "Data leakage detected: train and val share sample_id!"
    assert len(train_groups.intersection(test_groups)) == 0, "Data leakage detected: train and test share sample_id!"
    assert len(val_groups.intersection(test_groups)) == 0, "Data leakage detected: val and test share sample_id!"

    logger.info(
        f"Group split complete ({group_col}): "
        f"Train={len(df_train)} rows ({len(train_groups)} samples), "
        f"Val={len(df_val)} rows ({len(val_groups)} samples), "
        f"Test={len(df_test)} rows ({len(test_groups)} samples)"
    )

    return df_train, df_val, df_test
