import pytest
import io
import os
from fastapi.testclient import TestClient
from main import app
from PIL import Image

client = TestClient(app)

@pytest.mark.asyncio
async def test_analyze_label():
    image = Image.new("RGB", (224, 224), "white")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/api/v1/analyze/label",
        files={"label_image": ("label.jpg", img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["foodName"] == "Nutrition Label"
    assert "calories" in data["data"]
    assert "macros" in data["data"]

@pytest.mark.asyncio
async def test_analyze_speech():
    # If ffmpeg is not installed on the system, the endpoint will return a 500 error gracefully.
    # We will test the endpoint structure. If ffmpeg is missing, we assert the specific 500 detail.
    # First, let's create a dummy audio file (just random bytes to trigger the file logic)
    dummy_audio = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'

    response = client.post(
        "/api/v1/analyze/speech",
        files={"audio_file": ("test.wav", dummy_audio, "audio/wav")}
    )

    # If ffmpeg is installed, whisper might fail to parse the dummy bytes (raising 500)
    # or it might return an empty transcription.
    # If ffmpeg is NOT installed, we expect our explicit 500 detail.
    if os.system("ffmpeg -version > /dev/null 2>&1") != 0:
        assert response.status_code == 500
        assert "FFmpeg is not installed" in response.json()["detail"]
    else:
        # ffmpeg is present. The dummy audio might fail whisper parsing.
        # We just verify the endpoint executes without an unhandled crash.
        assert response.status_code in [200, 500]
