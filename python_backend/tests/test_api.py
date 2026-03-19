import pytest
import io
from fastapi.testclient import TestClient
from main import app
from PIL import Image

client = TestClient(app)

def test_analyze_image():
    # Create a dummy image
    image = Image.new("RGB", (224, 224), "white")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Send request to orchestration endpoint
    response = client.post(
        "/api/v1/analyze/image",
        files={"full_image": ("test.jpg", img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data

    nutrition_data = data["data"]
    assert "foodName" in nutrition_data
    assert "calories" in nutrition_data
    assert "macros" in nutrition_data
    assert "protein" in nutrition_data["macros"]
    assert "confidenceScore" in nutrition_data
    assert 0.0 <= nutrition_data["confidenceScore"] <= 1.0
    assert "warnings" in nutrition_data
    assert isinstance(nutrition_data["warnings"], list)
