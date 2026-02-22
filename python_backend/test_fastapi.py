from fastapi.testclient import TestClient
import traceback
from main import app

client = TestClient(app)

try:
    response = client.post("/api/v1/analyze/image")
    print(response.status_code, response.text)
except Exception as e:
    traceback.print_exc()
