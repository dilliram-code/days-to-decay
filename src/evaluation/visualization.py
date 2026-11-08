"""Visualization utilities for Days to Decay model evaluation."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("visualization")

# Set research aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": True,
})


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Actual vs. Predicted Remaining Shelf-Life (Days)",
) -> Path:
    """Creates a scatter plot with perfect agreement line and residual shading."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Scatter points
    ax.scatter(y_true, y_pred, alpha=0.6, color="#0f766e", edgecolors="w", s=45, label="Test Samples")

    # Identity line y = x
    max_val = max(float(np.max(y_true)) if len(y_true) else 10.0, float(np.max(y_pred)) if len(y_pred) else 10.0) + 1.0
    ax.plot([0, max_val], [0, max_val], color="#ef4444", linestyle="--", linewidth=1.8, label="Ideal (y = x)")

    # +- 1.0 day error band
    ax.fill_between([0, max_val], [-1, max_val - 1], [1, max_val + 1], color="#14b8a6", alpha=0.12, label="±1.0 Day Tolerance")

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Ground Truth Remaining Days")
    ax.set_ylabel("Predicted Remaining Days")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="upper left")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved actual vs predicted plot to {output_path}")
    return output_path


def plot_residual_distribution(
    residuals: np.ndarray,
    output_path: Path,
    title: str = "Residual Distribution (Actual - Predicted)",
) -> Path:
    """Plots a histogram and KDE of prediction residuals."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sns.histplot(residuals, kde=True, color="#3b82f6", bins=20, ax=ax, edgecolor="black")

    ax.axvline(0.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.axvline(float(np.mean(residuals)), color="#f59e0b", linestyle="-", linewidth=1.5, label=f"Mean Error: {np.mean(residuals):.2f}d")

    ax.set_xlabel("Error (Actual - Predicted Days)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(frameon=True)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved residual distribution plot to {output_path}")
    return output_path


def plot_training_history(
    history_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plots training and validation loss & MAE curves across epochs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Loss curve
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#0284c7", linewidth=2)
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#ea580c", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend(frameon=True)

    # MAE curve
    if "val_mae" in history_df.columns:
        ax2.plot(history_df["epoch"], history_df["val_mae"], label="Validation MAE", color="#16a34a", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE (Days)")
        ax2.set_title("Validation MAE Progression", fontweight="bold")
        ax2.legend(frameon=True)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training history curves to {output_path}")
    return output_path


def plot_error_by_category(
    category_errors: Dict[str, List[float]],
    output_path: Path,
) -> Path:
    """Plots boxplot of absolute prediction errors broken down by food category."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for cat, errs in category_errors.items():
        for e in errs:
            records.append({"Category": cat.capitalize(), "Absolute Error (Days)": e})

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    if not df.empty:
        sns.boxplot(data=df, x="Category", y="Absolute Error (Days)", palette="Set2", ax=ax)
        sns.stripplot(data=df, x="Category", y="Absolute Error (Days)", color="black", alpha=0.3, jitter=0.2, ax=ax)

    ax.set_title("Absolute Error Distribution by Category", fontweight="bold", pad=12)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved category error plot to {output_path}")
    return output_path
