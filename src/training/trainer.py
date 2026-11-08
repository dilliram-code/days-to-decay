"""Training pipeline for continuous food shelf-life regression."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import invert_target_transform
from src.models.model_factory import export_model_metadata, save_checkpoint
from src.training.losses import get_loss_function
from src.training.metrics import compute_regression_metrics

logger = logging.getLogger("trainer")


class ModelTrainer:
    """
    Manages end-to-end model training, validation, early stopping,
    checkpointing, and conformal calibration.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Auto device detection: MPS (Apple Silicon) -> CUDA -> CPU
        if device is not None:
            self.device = device
        else:
            cfg_device = config.get("training", {}).get("device", "auto")
            if cfg_device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(cfg_device)

        logger.info(f"Using execution device: {self.device}")
        self.model = model.to(self.device)

        # Loss function
        loss_cfg = config.get("training", {}).get("loss", {})
        self.criterion = get_loss_function(
            name=loss_cfg.get("name", "huber"),
            delta=float(loss_cfg.get("delta", 1.0)),
        )

        # Differential learning rates for backbone vs regression head
        opt_cfg = config.get("training", {}).get("optimizer", {})
        backbone_lr = float(opt_cfg.get("backbone_lr", 5e-5))
        head_lr = float(opt_cfg.get("head_lr", 5e-4))
        weight_decay = float(opt_cfg.get("weight_decay", 0.01))

        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if "regression_head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )

        # Scheduler
        epochs = int(config.get("training", {}).get("epochs", 25))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=float(opt_cfg.get("eta_min", 1e-6)),
        )

        # Early stopping & checkpoints
        es_cfg = config.get("training", {}).get("early_stopping", {})
        self.patience = int(es_cfg.get("patience", 5))
        self.min_delta = float(es_cfg.get("min_delta", 0.005))
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Artifact paths
        art_cfg = config.get("artifacts", {})
        self.best_model_path = Path(art_cfg.get("best_model_path", "artifacts/best_model.pth"))
        self.history_path = Path(art_cfg.get("history_path", "artifacts/training_history.csv"))
        self.metadata_path = Path(art_cfg.get("metadata_path", "artifacts/model_metadata.json"))
        self.calibration_path = Path(art_cfg.get("calibration_path", "artifacts/calibration_residuals.json"))

        self.history: List[Dict[str, Any]] = []
        self.target_transform = config.get("data", {}).get("target_transform", "none")
        self.freeze_epochs = int(config.get("model", {}).get("freeze_backbone_epochs", 3))

    def train_epoch(self, epoch: int) -> float:
        """Trains one full epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Backbone freeze schedule
        if epoch < self.freeze_epochs:
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone(True)
        else:
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone(False)

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluates model, returning loss, metrics dictionary, and raw true/pred arrays."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_y_true = []
        all_y_pred = []

        for batch in data_loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            raw_targets = batch["raw_target"].numpy().ravel()

            preds = self.model(images)
            loss = self.criterion(preds, targets)

            total_loss += loss.item()
            num_batches += 1

            preds_np = preds.cpu().numpy().ravel()
            # Invert target transform if active
            preds_natural = invert_target_transform(preds_np, self.target_transform)

            all_y_true.extend(raw_targets)
            all_y_pred.extend(preds_natural)

        avg_loss = total_loss / max(1, num_batches)
        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)

        metrics = compute_regression_metrics(y_true_arr, y_pred_arr)
        return avg_loss, metrics, y_true_arr, y_pred_arr

    def fit(self) -> Dict[str, Any]:
        """Runs the complete training and validation cycle."""
        epochs = int(self.config.get("training", {}).get("epochs", 25))
        logger.info(f"Starting training run: {epochs} epochs on {self.device}")

        best_metrics = {}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics, _, _ = self.evaluate(self.val_loader)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[-1]["lr"]
            epoch_log = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "learning_rate": round(current_lr, 7),
            }
            self.history.append(epoch_log)

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_metrics['mae']:.2f}d | "
                f"Val R2: {val_metrics['r2']:.3f}"
            )

            # Checkpoint selection based on validation loss
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                best_metrics = val_metrics
                self.patience_counter = 0

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    metrics=val_metrics,
                    config=self.config,
                    save_path=self.best_model_path,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={self.patience})")
                    break

        # Save training history
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)
        logger.info(f"Training history saved to {self.history_path}")

        # Calibrate conformal prediction on validation set using best weights
        self.calibrate_conformal_intervals()

        # Export metadata
        self.export_metadata(best_metrics)

        return {"best_val_loss": self.best_val_loss, "best_metrics": best_metrics}

    def calibrate_conformal_intervals(self) -> Optional[float]:
        """
        Calibrates finite-sample conformal prediction bounds using validation residuals.
        
        Given miscoverage rate alpha = 1 - confidence_level (e.g. 0.10 for 90% confidence),
        the conformal score is the absolute prediction error s_i = |y_i - \hat{y}_i|.
        The adjusted quantile guarantees:
          P(y \in [\hat{y} - \hat{q}, \hat{y} + \hat{q}]) >= 1 - alpha.
        """
        conf_cfg = self.config.get("conformal", {})
        if not conf_cfg.get("enabled", True):
            return None

        # Load best weights if available
        if self.best_model_path.exists():
            from src.models.model_factory import load_checkpoint
            load_checkpoint(self.best_model_path, self.model, map_location=str(self.device))

        _, _, y_true, y_pred = self.evaluate(self.val_loader)
        if len(y_true) < 2:
            logger.warning("Insufficient validation samples to calibrate conformal prediction intervals.")
            return None

        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        confidence = float(conf_cfg.get("confidence_level", 0.90))
        alpha = 1.0 - confidence

        # Finite-sample correction level: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = min(1.0, max(0.0, level))
        q_hat = float(np.quantile(residuals, level, method="higher"))

        calib_data = {
            "method": "inductive_conformal_prediction",
            "confidence_level": confidence,
            "quantile_residual_days": round(q_hat, 4),
            "num_calibration_samples": int(n),
            "calibrated_at": datetime.utcnow().isoformat() + "Z",
            "status": "calibrated",
        }

        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_path, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)

        logger.info(f"Conformal calibration saved ({confidence*100:.0f}% coverage: +- {q_hat:.2f} days)")
        return q_hat

    def export_metadata(self, best_metrics: Dict[str, float]) -> None:
        """Exports comprehensive model metadata for API and reproducibility."""
        metadata = {
            "project_name": "DAYS TO DECAY",
            "model_version": "1.0.0",
            "architecture": self.config.get("model", {}).get("backbone", "resnet50"),
            "input_dimensions": [3, 224, 224],
            "target_unit": "days",
            "target_transform": self.target_transform,
            "supported_categories": self.config.get("data", {}).get("categories", ["banana", "avocado", "bread"]),
            "training_completed_at": datetime.utcnow().isoformat() + "Z",
            "validation_metrics": best_metrics,
            "disclaimer": (
                "Shelf-life estimates represent visual degradation trends under training conditions. "
                "Visual appearance alone does not establish microbiological safety or pathogen absence."
            ),
        }
        export_model_metadata(metadata, self.metadata_path)
