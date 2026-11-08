"""Model evaluation engine and report generation for Days to Decay."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import invert_target_transform
from src.data.schema import determine_aging_stage
from src.evaluation.visualization import (
    plot_actual_vs_predicted,
    plot_error_by_category,
    plot_residual_distribution,
    plot_training_history,
)
from src.training.metrics import compute_regression_metrics

logger = logging.getLogger("evaluator")


class ModelEvaluator:
    """
    Evaluates trained food aging regression models against held-out test sets
    and statistical baseline benchmarks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        target_transform: str = "none",
        baseline_mean_days: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.target_transform = target_transform
        self.baseline_mean_days = baseline_mean_days

    @torch.no_grad()
    def evaluate_test_set(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Runs test inference, aggregates per-sample predictions, and calculates metrics."""
        self.model.eval()
        records = []

        for batch in test_loader:
            images = batch["image"].to(self.device)
            raw_targets = batch["raw_target"].numpy().ravel()
            categories = batch["category"]
            sample_ids = batch["sample_id"]
            img_paths = batch["image_path"]

            preds = self.model(images)
            preds_np = preds.cpu().numpy().ravel()
            preds_natural = invert_target_transform(preds_np, self.target_transform)

            for i in range(len(raw_targets)):
                y_true = float(raw_targets[i])
                y_pred = float(preds_natural[i])
                cat = str(categories[i])
                stage, condition = determine_aging_stage(cat, y_true)

                records.append({
                    "sample_id": str(sample_ids[i]),
                    "food_category": cat,
                    "image_path": str(img_paths[i]),
                    "ground_truth_days": y_true,
                    "predicted_days": y_pred,
                    "residual": round(y_true - y_pred, 4),
                    "absolute_error": round(abs(y_true - y_pred), 4),
                    "aging_stage": stage,
                    "visual_condition": condition,
                })

        df_results = pd.DataFrame(records)
        if df_results.empty:
            logger.warning("Test dataset was empty. No evaluations generated.")
            return {"results_df": df_results, "metrics": {}, "baseline_metrics": {}}

        y_t = df_results["ground_truth_days"].values
        y_p = df_results["predicted_days"].values
        model_metrics = compute_regression_metrics(y_t, y_p)

        # Baseline: Predict constant mean of training set or ground truth mean
        baseline_pred_val = self.baseline_mean_days if self.baseline_mean_days is not None else float(np.mean(y_t))
        baseline_preds = np.full_like(y_t, fill_value=baseline_pred_val)
        baseline_metrics = compute_regression_metrics(y_t, baseline_preds)

        # Category breakdowns
        cat_metrics = {}
        for cat, group in df_results.groupby("food_category"):
            cat_metrics[cat] = compute_regression_metrics(
                group["ground_truth_days"].values,
                group["predicted_days"].values,
            )

        return {
            "results_df": df_results,
            "metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "category_metrics": cat_metrics,
            "baseline_value": baseline_pred_val,
        }

    def generate_full_evaluation_report(
        self,
        eval_output: Dict[str, Any],
        reports_dir: Path,
        history_csv: Optional[Path] = None,
        model_name: str = "ResNet50 Regressor",
    ) -> Path:
        """Generates figures, metrics JSON, and GitHub-style evaluation report."""
        reports_dir = Path(reports_dir)
        fig_dir = reports_dir / "figures"
        metrics_dir = reports_dir / "metrics"
        fig_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        df_results = eval_output["results_df"]
        metrics = eval_output["metrics"]
        base_metrics = eval_output["baseline_metrics"]
        cat_metrics = eval_output["category_metrics"]

        # 1. Generate Figures
        if not df_results.empty:
            plot_actual_vs_predicted(
                df_results["ground_truth_days"].values,
                df_results["predicted_days"].values,
                output_path=fig_dir / "actual_vs_predicted.png",
            )
            plot_residual_distribution(
                df_results["residual"].values,
                output_path=fig_dir / "residual_distribution.png",
            )
            cat_errors = {
                cat: group["absolute_error"].tolist()
                for cat, group in df_results.groupby("food_category")
            }
            plot_error_by_category(
                cat_errors,
                output_path=fig_dir / "error_by_category.png",
            )

        if history_csv and history_csv.exists():
            try:
                hist_df = pd.read_csv(history_csv)
                plot_training_history(hist_df, output_path=fig_dir / "training_curves.png")
            except Exception as e:
                logger.warning(f"Failed to plot history from {history_csv}: {e}")

        # 2. Save JSON Metrics
        metrics_json_path = metrics_dir / "evaluation_metrics.json"
        summary_payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": model_name,
            "test_sample_count": len(df_results),
            "deep_learning_metrics": metrics,
            "baseline_metrics": base_metrics,
            "category_breakdown": cat_metrics,
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        # 3. Generate Markdown Report
        report_md_path = reports_dir / "evaluation_report.md"
        lines = [
            "# DAYS TO DECAY — Model Evaluation Report",
            "",
            f"- **Evaluated Model:** `{model_name}`",
            f"- **Timestamp:** `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`",
            f"- **Test Samples Evaluated:** {len(df_results)}",
            "",
            "## 1. Executive Summary & Baseline Comparison",
            "",
            "To verify scientific merit, the deep learning model is benchmarked against a naive mean-predictor baseline.",
            "",
            "| Metric | Deep Learning Model | Naive Baseline | Improvement |",
            "|---|---|---|---|",
            f"| **MAE (Days)** | **{metrics.get('mae', 'N/A')}** | {base_metrics.get('mae', 'N/A')} | {'+' if metrics.get('mae', 0) < base_metrics.get('mae', 0) else ''}{round(base_metrics.get('mae', 0) - metrics.get('mae', 0), 2)}d |",
            f"| **RMSE (Days)** | **{metrics.get('rmse', 'N/A')}** | {base_metrics.get('rmse', 'N/A')} | {'+' if metrics.get('rmse', 0) < base_metrics.get('rmse', 0) else ''}{round(base_metrics.get('rmse', 0) - metrics.get('rmse', 0), 2)}d |",
            f"| **R² Score** | **{metrics.get('r2', 'N/A')}** | {base_metrics.get('r2', 'N/A')} | — |",
            f"| **Median AE (Days)** | **{metrics.get('medae', 'N/A')}** | {base_metrics.get('medae', 'N/A')} | — |",
            f"| **Within ±1.0 Day (%)** | **{metrics.get('within_1.0d_pct', 'N/A')}%** | {base_metrics.get('within_1.0d_pct', 'N/A')}% | — |",
            "",
            "## 2. Performance Breakdown by Category",
            "",
            "| Category | Samples | MAE (Days) | RMSE (Days) | Within ±1.0d (%) |",
            "|---|---|---|---|---|",
        ]

        for cat, cm in cat_metrics.items():
            lines.append(
                f"| `{cat.capitalize()}` | {cm.get('sample_count', 0)} | {cm.get('mae', 'N/A')}d | {cm.get('rmse', 'N/A')}d | {cm.get('within_1.0d_pct', 'N/A')}% |"
            )

        lines.extend([
            "",
            "## 3. Evaluation Visualizations",
            "",
            "- **Actual vs Predicted Scatter:** `reports/figures/actual_vs_predicted.png`",
            "- **Residual Distribution:** `reports/figures/residual_distribution.png`",
            "- **Error Breakdown by Category:** `reports/figures/error_by_category.png`",
            "- **Training Loss & MAE Curves:** `reports/figures/training_curves.png`",
            "",
            "## 4. Limitations & Research Boundary",
            "",
            "1. **Environmental Variance:** Visual aging rates accelerate with temperature and humidity. Images captured under differing conditions may exhibit variance not fully captured by RGB pixels alone.",
            "2. **Microbiological Safety:** This model estimates visible shelf life. Visual appearance does not substitute for pathogen detection (e.g. Salmonella, Listeria).",
            "3. **Group Integrity:** Splitting strictly segregated `sample_id` across splits to prevent data leakage.",
            "",
        ])

        report_md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Evaluation report written to {report_md_path}")
        return report_md_path
