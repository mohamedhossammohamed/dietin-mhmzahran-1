from fastapi.testclient import TestClient
import traceback

from main import app

client = TestClient(app, raise_server_exceptions=True)

try:
    print("Test 1:")
    response = client.post("/api/v1/analyze/image")
    print(response.status_code, response.text)
except Exception as e:
    traceback.print_exc()

try:
    print("\nTest 2:")
    with open("dummy.jpg", "wb") as f: f.write(b"fake image")
    response = client.post("/api/v1/analyze/image", files={"file": ("dummy.jpg", open("dummy.jpg", "rb"), "image/jpeg")})
    print(response.status_code, response.text)
except Exception as e:
    traceback.print_exc()
