import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_mode2():
    # Send request without file or full_image, but with context_image and crops
    response = client.post(
        "/api/v1/analyze/image",
        files={
            "context_image": ("context.jpg", b"fake context", "image/jpeg"),
            "crops": ("crop.jpg", b"fake crop", "image/jpeg")
        }
    )
    print("STATUS:", response.status_code)
    try:
        print("JSON REPY:", response.json())
    except:
        print("RAW REPLY:", response.text)

if __name__ == "__main__":
    test_mode2()
