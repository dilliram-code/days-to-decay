"""Pydantic schemas for API request and response validation."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Application status")
    app_name: str
    version: str
    model_loaded: bool
    model_architecture: str
    device: str
    supported_categories: List[str]
    conformal_calibrated: bool


class PredictionResponse(BaseModel):
    success: bool = True
    food_category: str
    predicted_remaining_days: float
    lower_bound_days: Optional[float] = None
    upper_bound_days: Optional[float] = None
    uncertainty_method: str = "unavailable"
    aging_stage: str
    condition: str
    model_version: str = "1.0.0"
    prediction_type: str = "experimental_estimate"
    disclaimer: str = (
        "This prediction is an estimate and does not establish microbiological food safety."
    )


class BatchPredictionItem(BaseModel):
    filename: str
    prediction: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    success: bool = True
    total_processed: int
    predictions: List[BatchPredictionItem]


class CategoryStageInfo(BaseModel):
    stage_name: str
    threshold_remaining_days: float
    condition: str


class CategoryDetail(BaseModel):
    category: str
    nominal_shelf_life_days: float
    spoilage_endpoint: str
    stages: List[CategoryStageInfo]


class CategoriesResponse(BaseModel):
    categories: List[CategoryDetail]


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    input_dimensions: List[int]
    device: str
    checkpoint_loaded: bool
    checkpoint_path: str
    conformal_calibration: Dict[str, Any]
    evaluation_metrics: Optional[Dict[str, Any]] = None
    disclaimer: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None

