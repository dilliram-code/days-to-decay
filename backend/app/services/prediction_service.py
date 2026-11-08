"""Prediction service coordinating image processing, inference, and response generation."""
import logging
from typing import Optional
from fastapi import UploadFile

from backend.app.schemas import PredictionResponse
from backend.app.services.image_service import image_service
from backend.app.services.model_service import model_service
from backend.app.utils.validation import validate_category
from src.data.schema import determine_aging_stage

logger = logging.getLogger("prediction_service")


class PredictionService:
    """Manages the full prediction pipeline for single and batch image inputs."""

    @staticmethod
    async def predict_single_image(
        file: UploadFile,
        food_category: str,
    ) -> PredictionResponse:
        """Executes complete prediction workflow on a single uploaded food image."""
        category = validate_category(food_category)
        tensor, _ = await image_service.process_upload(file)

        # Forward pass
        pred_days_raw = model_service.predict_tensor(tensor)
        pred_days = round(pred_days_raw, 2)

        # Aging stage & visual condition
        stage, condition = determine_aging_stage(category, pred_days)

        # Uncertainty intervals (conformal prediction)
        lower, upper, method = model_service.get_uncertainty_interval(pred_days)

        prediction_type = (
            "trained_model_estimate"
            if model_service.checkpoint_loaded
            else "uncalibrated_experimental_estimate"
        )

        return PredictionResponse(
            success=True,
            food_category=category,
            predicted_remaining_days=pred_days,
            lower_bound_days=lower,
            upper_bound_days=upper,
            uncertainty_method=method,
            aging_stage=stage,
            condition=condition,
            model_version=model_service.model_metadata.get("model_version", "1.0.0"),
            prediction_type=prediction_type,
            disclaimer=(
                "This prediction is an empirical visual estimate and does not establish microbiological food safety."
            ),
        )


prediction_service = PredictionService()

