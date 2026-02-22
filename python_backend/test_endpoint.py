import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("--- Test 1: Empty ---")
response = client.post("/api/v1/analyze/image")
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")

print("\n--- Test 2: Text file instead of image ---")
response = client.post("/api/v1/analyze/image", files={"file": ("test.txt", b"hello world", "text/plain")})
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")

print("\n--- Test 3: Corrupt image bytes ---")
response = client.post("/api/v1/analyze/image", files={"file": ("test.jpg", b"fake jpeg bytes that will fail PIL", "image/jpeg")})
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")

print("\n--- Test 4: Pattern D Context Mode ---")
response = client.post("/api/v1/analyze/image", files={"context_image": ("c.jpg", b"fake", "image/jpeg"), "crops": ("cr.jpg", b"fake2", "image/jpeg")})
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")
