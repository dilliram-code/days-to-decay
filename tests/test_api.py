"""API endpoint integration tests."""
import io
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Verifies GET /health returns operational diagnostic metrics."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "model_loaded" in data
    assert "supported_categories" in data
    assert "banana" in data["supported_categories"]


def test_categories_endpoint(client: TestClient):
    """Verifies GET /categories enumerates supported categories and degradation stages."""
    res = client.get("/categories")
    assert res.status_code == 200
    data = res.json()

    assert "categories" in data
    cat_names = [c["category"] for c in data["categories"]]
    assert "banana" in cat_names
    assert "avocado" in cat_names
    assert "bread" in cat_names

    for cat in data["categories"]:
        assert len(cat["stages"]) > 0
        assert "nominal_shelf_life_days" in cat


def test_model_info_endpoint(client: TestClient):
    """Verifies GET /model-info provides architecture specs and scientific disclaimer."""
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()

    assert "architecture" in data
    assert "input_dimensions" in data
    assert "disclaimer" in data


def test_predict_endpoint_valid(client: TestClient, sample_image_bytes: bytes):
    """Verifies POST /predict successfully generates a valid continuous regression response."""
    files = {"file": ("test_banana.jpg", sample_image_bytes, "image/jpeg")}
    data = {"food_category": "banana"}

    res = client.post("/predict", files=files, data=data)
    assert res.status_code == 200
    body = res.json()

    assert body["success"] is True
    assert body["food_category"] == "banana"
    assert isinstance(body["predicted_remaining_days"], (int, float))
    assert body["predicted_remaining_days"] >= 0.0
    assert "aging_stage" in body
    assert "condition" in body
    assert "disclaimer" in body


def test_predict_endpoint_invalid_category(client: TestClient, sample_image_bytes: bytes):
    """Verifies POST /predict rejects unsupported food categories with HTTP 400."""
    files = {"file": ("test_pizza.jpg", sample_image_bytes, "image/jpeg")}
    data = {"food_category": "pizza"}

    res = client.post("/predict", files=files, data=data)
    assert res.status_code == 400
    assert "Unsupported food category" in res.json()["detail"]


def test_predict_endpoint_invalid_file_type(client: TestClient):
    """Verifies POST /predict rejects non-image text uploads with HTTP 400."""
    files = {"file": ("document.txt", b"Hello, not an image", "text/plain")}
    data = {"food_category": "banana"}

    res = client.post("/predict", files=files, data=data)
    assert res.status_code == 400


def test_predict_batch_endpoint(client: TestClient, sample_image_bytes: bytes):
    """Verifies POST /predict/batch processes multiple images."""
    files = [
        ("files", ("img1.jpg", sample_image_bytes, "image/jpeg")),
        ("files", ("img2.jpg", sample_image_bytes, "image/jpeg")),
    ]
    data = {"food_category": "avocado"}

    res = client.post("/predict/batch", files=files, data=data)
    assert res.status_code == 200
    body = res.json()

    assert body["success"] is True
    assert body["total_processed"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["prediction"]["food_category"] == "avocado"
