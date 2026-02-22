from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_corrupt_image():
    corrupt_bytes = b"This is not a valid image file, it is just text."
    files = {'file': ('corrupt.jpg', corrupt_bytes, 'image/jpeg')}
    response = client.post("/api/v1/analyze/image", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")

if __name__ == "__main__":
    test_corrupt_image()
