"""Loss functions for shelf-life regression."""
from typing import Callable
import torch
import torch.nn as nn


def get_loss_function(name: str = "huber", delta: float = 1.0) -> Callable:
    """
    Returns appropriate PyTorch loss function for continuous food shelf-life regression.
    
    Options:
      - 'huber' (SmoothL1Loss): Robust to outliers while quadratic near zero.
      - 'mse': Penalizes large estimation errors quadratically.
      - 'mae': Penalizes errors linearly, robust to extreme outliers.
    """
    name_clean = name.lower().strip()
    if name_clean == "huber" or name_clean == "smooth_l1":
        return nn.SmoothL1Loss(beta=delta)
    elif name_clean == "mse":
        return nn.MSELoss()
    elif name_clean == "mae" or name_clean == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss function '{name}'. Choose from 'huber', 'mse', 'mae'.")
