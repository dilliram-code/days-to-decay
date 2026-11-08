"""Dataset schema, column specifications, and category definitions."""
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

SUPPORTED_CATEGORIES = ("banana", "avocado", "bread")

DATASET_COLUMNS = [
    "image_path",
    "food_category",
    "sample_id",
    "capture_date",
    "capture_day",
    "initial_weight",
    "temperature",
    "humidity",
    "initial_freshness",
    "observed_spoilage_date",
    "remaining_days",
    "storage_condition",
    "collection_notes",
]

# Category-specific spoilage endpoints and criteria
SPOILAGE_ENDPOINTS = {
    "banana": {
        "endpoint_name": "Overripe / Senescent Stage 7+",
        "description": "Total pulp softening with brown coverage exceeding 70% or mold at stem/pulp, rendering fruit unpalatable.",
        "nominal_shelf_life_days": 10.0,
    },
    "avocado": {
        "endpoint_name": "Internal / External Rancidity and Rancid Browning",
        "description": "Severe skin collapse, rancid odor, internal flesh darkening/stringiness, or external mycelium growth.",
        "nominal_shelf_life_days": 8.0,
    },
    "bread": {
        "endpoint_name": "Visible Mycelial Mold Colony or Severe Desiccation",
        "description": "First macroscopic appearance of fungal sporulation (Rhizopus, Penicillium, Aspergillus) or severe petrification.",
        "nominal_shelf_life_days": 6.0,
    },
}

# Empirical aging stage thresholds (in remaining days)
AGING_STAGES_CONFIG: Dict[str, List[Tuple[float, str, str]]] = {
    "banana": [
        (7.0, "Unripe / Green", "All green or green-yellow peel, firm starch pulp"),
        (4.0, "Prime Ripe / Yellow", "Bright yellow with green tips, optimal sweetness"),
        (2.0, "Mid Ripening / Sugar Spots", "Yellow with early brown speckles, high sugar content"),
        (0.5, "Late Ripening / Moderate Browning", "Heavy brown spotting, softened flesh, nearing culinary limit"),
        (0.0, "Senescent / Spoiled", "Extensive browning (>70%), pulp collapse or surface mold"),
    ],
    "avocado": [
        (5.0, "Firm Green / Unripe", "Bright green skin, hard flesh, unyielding to gentle pressure"),
        (3.0, "Breaking / Turning Dark", "Dark green to olive skin, slight give under firm thumb pressure"),
        (1.5, "Ripe / Ready to Eat", "Dark purple/black skin, yields gently to soft palm pressure"),
        (0.5, "Overripe / Past Prime", "Indentation persists, soft hollow pockets, beginning internal browning"),
        (0.0, "Spoiled / Decayed", "Skin collapse, visible fungal colonies, rancid fermentation"),
    ],
    "bread": [
        (4.0, "Fresh Baked / Soft", "Springy crumb, fragrant crust, optimal moisture retention"),
        (2.0, "Staling / Loss of Softness", "Crumb retrogradation, firmer texture, dry surface"),
        (1.0, "Firm / Pre-Mold Stage", "Noticeably dry and crumbly, loss of aroma, high microbial risk"),
        (0.2, "Early Mold Colonies", "Microscopic or pinpoint mold spores emerging, discarded immediately"),
        (0.0, "Advanced Spoilage", "Widespread fungal sporulation, hazardous for consumption"),
    ],
}


class DatasetSampleRecord(BaseModel):
    """Pydantic model validating an individual row in dataset.csv."""

    image_path: str = Field(..., description="Relative or absolute path to the processed image file")
    food_category: str = Field(..., description="Category: banana, avocado, or bread")
    sample_id: str = Field(..., description="Unique physical fruit/loaf identifier e.g. BANANA_001")
    capture_date: str = Field(..., description="Date of photograph (YYYY-MM-DD)")
    capture_day: int = Field(..., ge=0, description="Elapsed day counter relative to sample initiation")
    initial_weight: Optional[float] = Field(None, ge=0.0, description="Initial mass in grams")
    temperature: Optional[float] = Field(None, description="Ambient temperature in degrees Celsius")
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Ambient relative humidity %")
    initial_freshness: str = Field("fresh", description="Initial visual assessment")
    observed_spoilage_date: str = Field(..., description="Verified spoilage date (YYYY-MM-DD)")
    remaining_days: float = Field(..., ge=0.0, description="Target: observed_spoilage_date - capture_date in days")
    storage_condition: str = Field("room_temperature", description="Storage regimen e.g. room_temperature, refrigerated")
    collection_notes: Optional[str] = Field("", description="Optional contextual notes")

    @field_validator("food_category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        val = value.strip().lower()
        if val not in SUPPORTED_CATEGORIES:
            raise ValueError(f"food_category must be one of {SUPPORTED_CATEGORIES}, got '{val}'")
        return val

    @field_validator("capture_date", "observed_spoilage_date")
    @classmethod
    def validate_iso_date(cls, value: str) -> str:
        try:
            datetime.strptime(value.strip(), "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date '{value}' must be in YYYY-MM-DD format")
        return value.strip()


def compute_remaining_days(capture_date_str: str, spoilage_date_str: str) -> float:
    """Calculates remaining days = observed_spoilage_date - capture_date."""
    d_cap = datetime.strptime(capture_date_str.strip(), "%Y-%m-%d").date()
    d_spoil = datetime.strptime(spoilage_date_str.strip(), "%Y-%m-%d").date()
    delta = (d_spoil - d_cap).days
    return float(max(0.0, delta))


def determine_aging_stage(food_category: str, remaining_days: float) -> Tuple[str, str]:
    """
    Returns (aging_stage, visual_condition) based on category and remaining days.
    """
    cat = food_category.strip().lower()
    stages = AGING_STAGES_CONFIG.get(cat, AGING_STAGES_CONFIG["banana"])
    
    # Stages are ordered descending by threshold
    for threshold, stage_name, condition in stages:
        if remaining_days >= threshold:
            return stage_name, condition
    
    # Fallback to the lowest stage
    return stages[-1][1], stages[-1][2]
