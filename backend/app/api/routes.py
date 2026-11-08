"""API routes for Days to Decay."""
import logging
from typing import List
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.config import settings
from backend.app.dependencies import get_model_service
from backend.app.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    CategoriesResponse,
    CategoryDetail,
    CategoryStageInfo,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from backend.app.services.model_service import ModelService
from backend.app.services.prediction_service import prediction_service
from src.data.schema import AGING_STAGES_CONFIG, SPOILAGE_ENDPOINTS, SUPPORTED_CATEGORIES

logger = logging.getLogger("routes")
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and system diagnostics",
)
async def health_check(ms: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Returns application health, device status, and model readiness."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_loaded=ms.checkpoint_loaded,
        model_architecture=ms.architecture,
        device=str(ms.device),
        supported_categories=list(SUPPORTED_CATEGORIES),
        conformal_calibrated=(ms.conformal_quantile is not None),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict remaining shelf life from a single photograph",
)
async def predict_single(
    file: UploadFile = File(..., description="Food item photograph (JPG, PNG, WEBP)"),
    food_category: str = Form(..., description="Category: banana, avocado, or bread"),
) -> PredictionResponse:
    """
    Analyzes an uploaded photograph of perishable produce or bread and estimates:
    - Predicted remaining shelf life in days
    - 90% prediction uncertainty interval (conformal prediction)
    - Aging stage
    - Visual condition
    """
    return await prediction_service.predict_single_image(file=file, food_category=food_category)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict shelf life for multiple photographs",
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="List of food item photographs"),
    food_category: str = Form(..., description="Food category for the batch"),
) -> BatchPredictionResponse:
    """Processes multiple photographs sequentially and returns a prediction per image."""
    items: List[BatchPredictionItem] = []

    for f in files:
        try:
            pred = await prediction_service.predict_single_image(file=f, food_category=food_category)
            items.append(BatchPredictionItem(filename=f.filename or "unknown", prediction=pred))
        except Exception as e:
            logger.error(f"Error predicting {f.filename}: {e}")
            items.append(BatchPredictionItem(filename=f.filename or "unknown", error=str(e)))

    return BatchPredictionResponse(
        success=True,
        total_processed=len(items),
        predictions=items,
    )


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    summary="List supported categories and deterioration stages",
)
async def get_categories() -> CategoriesResponse:
    """Returns supported perishable categories, nominal life, endpoints, and aging stages."""
    details = []
    for cat in SUPPORTED_CATEGORIES:
        ep_info = SPOILAGE_ENDPOINTS.get(cat, {})
        stage_tuples = AGING_STAGES_CONFIG.get(cat, [])
        stages = [
            CategoryStageInfo(
                threshold_remaining_days=t,
                stage_name=s,
                condition=c,
            )
            for t, s, c in stage_tuples
        ]

        details.append(
            CategoryDetail(
                category=cat,
                nominal_shelf_life_days=ep_info.get("nominal_shelf_life_days", 7.0),
                spoilage_endpoint=ep_info.get("endpoint_name", "Visible spoilage"),
                stages=stages,
            )
        )
    return CategoriesResponse(categories=details)


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Retrieve model architecture and evaluation metrics",
)
async def get_model_info(ms: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    """Returns architecture specs, checkpoint status, and validation metrics."""
    return ModelInfoResponse(
        model_version=ms.model_metadata.get("model_version", "1.0.0"),
        architecture=ms.architecture,
        input_dimensions=[3, 224, 224],
        device=str(ms.device),
        checkpoint_loaded=ms.checkpoint_loaded,
        checkpoint_path=settings.MODEL_CHECKPOINT_PATH,
        conformal_calibration=ms.calibration_metadata,
        evaluation_metrics=ms.model_metadata.get("validation_metrics"),
        disclaimer=(
            "Estimates represent visual deterioration trends under representative conditions. "
            "They do not substitute for food microbiology testing."
        ),
    )
