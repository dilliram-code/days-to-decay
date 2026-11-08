"""Regression evaluation metrics for Days to Decay."""
from typing import Dict, Sequence, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def compute_regression_metrics(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
    tolerances: Sequence[float] = (0.5, 1.0, 2.0),
) -> Dict[str, float]:
    """
    Computes comprehensive regression metrics for food aging shelf-life estimation:
      - MAE (Mean Absolute Error in days)
      - RMSE (Root Mean Squared Error in days)
      - R2 (Coefficient of Determination)
      - MedAE (Median Absolute Error in days)
      - Accuracy within tolerances (e.g. within +-0.5d, +-1.0d, +-2.0d)
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "medae": 0.0,
            "sample_count": 0,
        }

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    medae = float(median_absolute_error(y_t, y_p))
    
    # R2 can be undefined if variance of y_true is 0
    if len(y_t) > 1 and np.var(y_t) > 1e-8:
        r2 = float(r2_score(y_t, y_p))
    else:
        r2 = 0.0

    residuals = np.abs(y_t - y_p)
    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "medae": round(medae, 4),
        "sample_count": int(len(y_t)),
    }

    for tol in tolerances:
        within_tol = float(np.mean(residuals <= tol) * 100.0)
        metrics[f"within_{tol}d_pct"] = round(within_tol, 2)

    return metrics

