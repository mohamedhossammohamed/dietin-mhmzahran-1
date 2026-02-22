from fastapi.testclient import TestClient
import traceback
import sys

# Mock VisionPipeline before importing app
import vision_engine
class MockVision:
    def estimate_volume(self, xb): return 100.0
    def run_mobile_sam(self, xb): return [0,0,10,10]
vision_engine.VisionPipeline = lambda: MockVision()

from main import app

client = TestClient(app, raise_server_exceptions=False)

print("Test 1:")
response = client.post("/api/v1/analyze/image")
print(response.status_code, response.text)

print("\nTest 2:")
with open("test.jpg", "wb") as f: f.write(b"fake image data")
response = client.post("/api/v1/analyze/image", files={"file": ("test.jpg", open("test.jpg", "rb"), "image/jpeg")})
print(response.status_code, response.text)

