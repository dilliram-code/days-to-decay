"""FastAPI dependency injection utilities."""
from backend.app.services.model_service import ModelService, model_service


def get_model_service() -> ModelService:
    """Returns the singleton ModelService instance."""
    return model_service
