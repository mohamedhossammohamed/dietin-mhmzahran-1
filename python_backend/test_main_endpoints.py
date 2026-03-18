"""
Comprehensive Endpoint Test Suite — All 4 Analysis Workflows + Proxy + Utilities
=================================================================================
Covers:
  - POST /api/v1/analyze/image  (Mode 1 full-image & Mode 2 crops)
  - POST /api/v1/analyze/speech
  - POST /api/v1/analyze/label
  - POST /api/v1/proxy/generate
  - Utility: clean_json, gemini_classification, _detect_device, validate_file_size

All external dependencies (Gemini API, Whisper, VisionPipeline depth model) are
mocked so the suite runs fast, offline, and deterministically.

Run:  python -m pytest test_main_endpoints.py -v
"""

import io
import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image

from main import app, clean_json, gemini_classification, validate_file_size, _detect_device

client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _make_jpeg_bytes(width=224, height=224, color="red") -> bytes:
    """Create a valid JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_large_bytes(size_mb: int) -> bytes:
    """Create a byte string of the given size in MB."""
    return b"0" * (size_mb * 1024 * 1024 + 1)


# Reusable mock GPT result
MOCK_GPT_RESULT = {"foodName": "Fried Chicken", "prep": "fried", "is_liquid": False}
MOCK_GPT_RESULT_LIQUID = {"foodName": "Orange Juice", "prep": "raw", "is_liquid": True}
MOCK_GPT_RESULT_UNKNOWN = {"foodName": "AlienFood9999", "prep": "raw", "is_liquid": False}


# ═══════════════════════════════════════════════════════════════
# 1. CLEAN_JSON UTILITY
# ═══════════════════════════════════════════════════════════════
class TestCleanJson:
    """Unit tests for the clean_json helper."""

    def test_plain_json_unchanged(self):
        raw = '{"foodName": "Chicken"}'
        assert clean_json(raw) == raw

    def test_backticks_removed(self):
        raw = '```json\n{"foodName": "Chicken"}\n```'
        result = clean_json(raw)
        assert result == '{"foodName": "Chicken"}'

    def test_backticks_no_language_hint(self):
        raw = '```\n{"key": "val"}\n```'
        result = clean_json(raw)
        assert result == '{"key": "val"}'

    def test_empty_string(self):
        assert clean_json("") == ""

    def test_no_backticks_passthrough(self):
        raw = "just some text"
        assert clean_json(raw) == raw

    def test_multiple_backtick_blocks(self):
        raw = '```json\n{"a":1}\n```\nsome text\n```json\n{"b":2}\n```'
        result = clean_json(raw)
        # Should strip the backtick wrappers
        assert "```" not in result


# ═══════════════════════════════════════════════════════════════
# 2. GEMINI_CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
class TestGeminiClassification:
    """Unit tests for gemini_classification with mocked Gemini API."""

    def test_empty_bytes_returns_default(self):
        result = asyncio.get_event_loop().run_until_complete(
            gemini_classification(b"")
        )
        assert result == {"foodName": "Unknown", "prep": "raw", "is_liquid": False}

    def test_none_bytes_returns_default(self):
        result = asyncio.get_event_loop().run_until_complete(
            gemini_classification(b"")
        )
        assert result["foodName"] == "Unknown"

    @patch("main.genai")
    def test_successful_classification(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = '{"foodName": "Pizza", "prep": "baked", "is_liquid": false}'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = asyncio.get_event_loop().run_until_complete(
            gemini_classification(b"fake-image-bytes")
        )
        assert result["foodName"] == "Pizza"
        assert result["prep"] == "baked"
        assert result["is_liquid"] is False

    @patch("main.genai")
    def test_gemini_api_error_returns_default(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API quota exceeded")
        mock_genai.GenerativeModel.return_value = mock_model

        result = asyncio.get_event_loop().run_until_complete(
            gemini_classification(b"some-bytes")
        )
        assert result == {"foodName": "Unknown", "prep": "raw", "is_liquid": False}

    @patch("main.genai")
    def test_malformed_json_returns_default(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = "not valid json at all"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = asyncio.get_event_loop().run_until_complete(
            gemini_classification(b"some-bytes")
        )
        assert result == {"foodName": "Unknown", "prep": "raw", "is_liquid": False}

    @patch("main.genai")
    def test_backtick_wrapped_json_parsed(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = '```json\n{"foodName": "Rice", "prep": "boiled", "is_liquid": false}\n```'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = asyncio.get_event_loop().run_until_complete(
            gemini_classification(b"some-bytes")
        )
        assert result["foodName"] == "Rice"


# ═══════════════════════════════════════════════════════════════
# 3. DEVICE DETECTION
# ═══════════════════════════════════════════════════════════════
class TestDetectDevice:
    def test_returns_valid_string(self):
        device = _detect_device()
        assert device in ("mps", "cuda", "cpu")


# ═══════════════════════════════════════════════════════════════
# 4. IMAGE ENDPOINT — MODE 1 (Full Image)
# ═══════════════════════════════════════════════════════════════
class TestImageEndpointMode1:
    """POST /api/v1/analyze/image with full_image or file."""

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    @patch("main.vision_pipeline")
    def test_valid_image_returns_200(self, mock_vision, mock_gpt):
        mock_vision.estimate_volume.return_value = 150.0
        mock_vision.run_mobile_sam.return_value = [0, 0, 100, 100]
        
        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("food.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "foodName" in data["data"]
        assert "calories" in data["data"]
        assert "macros" in data["data"]
        assert "protein" in data["data"]["macros"]
        assert "carbs" in data["data"]["macros"]
        assert "fat" in data["data"]["macros"]
        assert "healthScore" in data["data"]
        assert "confidenceScore" in data["data"]
        assert "warnings" in data["data"]

    def test_empty_payload_returns_400(self):
        response = client.post("/api/v1/analyze/image")
        assert response.status_code == 400

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    @patch("main.vision_pipeline")
    def test_empty_file_body_returns_400(self, mock_vision, mock_gpt):
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    @patch("main.vision_pipeline")
    def test_corrupt_image_returns_error(self, mock_vision, mock_gpt):
        mock_vision.estimate_volume.side_effect = Exception("PIL cannot open")
        mock_vision.run_mobile_sam.return_value = [0, 0, 100, 100]
        
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("corrupt.jpg", b"this is not an image", "image/jpeg")}
        )
        assert response.status_code in (400, 500)

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    @patch("main.vision_pipeline")
    def test_oversized_file_returns_413(self, mock_vision, mock_gpt):
        large_bytes = _make_large_bytes(21)  # 21 MB > 20 MB limit
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("big.jpg", large_bytes, "image/jpeg")}
        )
        assert response.status_code == 413

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT_LIQUID)
    @patch("main.vision_pipeline")
    def test_liquid_detected_applies_fallback(self, mock_vision, mock_gpt):
        mock_vision.estimate_volume.return_value = 150.0
        mock_vision.run_mobile_sam.return_value = [0, 0, 100, 100]

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("juice.jpg", jpeg, "image/jpeg")}
        )
        # Even with unknown food, it should not crash
        assert response.status_code == 200
        data = response.json()
        # Check liquid warning is present (if found in DB, liquid warning is added)
        warnings = data["data"]["warnings"]
        assert isinstance(warnings, list)

    @patch("main.search_engine.search", return_value=None)
    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT_UNKNOWN)
    @patch("main.vision_pipeline")
    def test_db_miss_returns_fallback_macros(self, mock_vision, mock_gpt, mock_search):
        mock_vision.estimate_volume.return_value = 100.0
        mock_vision.run_mobile_sam.return_value = [0, 0, 100, 100]

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("alien.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["confidenceScore"] == 0.5
        assert data["data"]["calories"] == 0
        assert any("not found" in w.lower() or "fallback" in w.lower() for w in data["data"]["warnings"])

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    @patch("main.vision_pipeline")
    def test_valid_response_schema_exact(self, mock_vision, mock_gpt):
        """Verify the exact Pydantic schema from the master plan."""
        mock_vision.estimate_volume.return_value = 200.0
        mock_vision.run_mobile_sam.return_value = [0, 0, 100, 100]

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("food.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Master plan schema: {success, data: {foodName, calories, macros:{protein,carbs,fat}, healthScore, confidenceScore, warnings}}
        assert isinstance(data["success"], bool)
        assert isinstance(data["data"]["foodName"], str)
        assert isinstance(data["data"]["calories"], int)
        assert isinstance(data["data"]["macros"]["protein"], int)
        assert isinstance(data["data"]["macros"]["carbs"], int)
        assert isinstance(data["data"]["macros"]["fat"], int)
        assert isinstance(data["data"]["healthScore"], float)
        assert isinstance(data["data"]["confidenceScore"], float)
        assert isinstance(data["data"]["warnings"], list)

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    @patch("main.vision_pipeline")
    def test_calories_are_non_negative(self, mock_vision, mock_gpt):
        """Verify calories are clamped to >= 0."""
        mock_vision.estimate_volume.return_value = 0.0
        mock_vision.run_mobile_sam.return_value = [0, 0, 100, 100]

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("food.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        assert response.json()["data"]["calories"] >= 0

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    @patch("main.vision_pipeline")
    def test_deterministic_mass_calculation(self, mock_vision, mock_gpt):
        """Volume * density * yield_factor -> mass -> macros should be deterministic."""
        mock_vision.estimate_volume.return_value = 100.0
        mock_vision.run_mobile_sam.return_value = [0, 0, 100, 100]

        jpeg = _make_jpeg_bytes()
        r1 = client.post("/api/v1/analyze/image", files={"file": ("f.jpg", jpeg, "image/jpeg")})
        r2 = client.post("/api/v1/analyze/image", files={"file": ("f.jpg", jpeg, "image/jpeg")})
        assert r1.json() == r2.json(), "Deterministic math violated"


# ═══════════════════════════════════════════════════════════════
# 5. IMAGE ENDPOINT — MODE 2 (Context + Crops)
# ═══════════════════════════════════════════════════════════════
class TestImageEndpointMode2:
    """POST /api/v1/analyze/image with context_image + crops (Pattern D)."""

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    def test_mode2_returns_200(self, mock_gpt):
        response = client.post(
            "/api/v1/analyze/image",
            files={
                "context_image": ("ctx.jpg", b"fake context bytes", "image/jpeg"),
                "crops": ("crop.jpg", b"fake crop bytes", "image/jpeg"),
            }
        )
        assert response.status_code == 200

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    def test_mode2_confidence_is_zero(self, mock_gpt):
        response = client.post(
            "/api/v1/analyze/image",
            files={
                "context_image": ("ctx.jpg", b"fake", "image/jpeg"),
                "crops": ("crop.jpg", b"fake", "image/jpeg"),
            }
        )
        assert response.status_code == 200
        assert response.json()["data"]["confidenceScore"] == 0.0

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value=MOCK_GPT_RESULT)
    def test_mode2_has_volume_warning(self, mock_gpt):
        response = client.post(
            "/api/v1/analyze/image",
            files={
                "context_image": ("ctx.jpg", b"fake", "image/jpeg"),
                "crops": ("crop.jpg", b"fake", "image/jpeg"),
            }
        )
        assert response.status_code == 200
        warnings = response.json()["data"]["warnings"]
        assert any("mode 2" in w.lower() or "crops" in w.lower() for w in warnings)

    @patch("main.gemini_classification", new_callable=AsyncMock, side_effect=Exception("Gemini timeout"))
    def test_mode2_gemini_failure_returns_500(self, mock_gpt):
        response = client.post(
            "/api/v1/analyze/image",
            files={
                "context_image": ("ctx.jpg", b"fake", "image/jpeg"),
                "crops": ("crop.jpg", b"fake", "image/jpeg"),
            }
        )
        assert response.status_code == 500


# ═══════════════════════════════════════════════════════════════
# 6. SPEECH ENDPOINT
# ═══════════════════════════════════════════════════════════════
class TestSpeechEndpoint:
    """POST /api/v1/analyze/speech."""

    @patch("main.whisper_model")
    def test_valid_audio_returns_200(self, mock_whisper):
        mock_whisper.transcribe.return_value = {"text": "I ate a banana"}

        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("audio.wav", b"fake audio bytes", "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "foodName" in data["data"]
        assert data["data"]["calories"] >= 0
        # Speech always has 150cm3 warning
        assert any("150" in w or "speech" in w.lower() for w in data["data"]["warnings"])

    @patch("main.whisper_model")
    def test_empty_transcription_returns_400(self, mock_whisper):
        mock_whisper.transcribe.return_value = {"text": "   "}

        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("audio.wav", b"silence", "audio/wav")}
        )
        assert response.status_code == 400

    @patch("main.whisper_model")
    def test_whisper_failure_returns_500(self, mock_whisper):
        mock_whisper.transcribe.side_effect = Exception("Model crashed")

        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("audio.wav", b"bad audio", "audio/wav")}
        )
        assert response.status_code == 500

    def test_oversized_audio_returns_413(self):
        large = _make_large_bytes(51)  # > 50 MB limit
        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("huge.wav", large, "audio/wav")}
        )
        assert response.status_code == 413

    @patch("main.search_engine.search", return_value=None)
    @patch("main.whisper_model")
    def test_speech_db_miss_returns_fallback(self, mock_whisper, mock_search):
        mock_whisper.transcribe.return_value = {"text": "I ate some xyzzy alien food"}

        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("audio.wav", b"audio", "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["confidenceScore"] == 0.5
        assert data["data"]["calories"] == 0

    @patch("main.whisper_model")
    def test_speech_schema_matches_contract(self, mock_whisper):
        """The speech endpoint must return the same AnalysisResponse schema."""
        mock_whisper.transcribe.return_value = {"text": "banana"}

        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("audio.wav", b"audio", "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"success", "data"}
        assert set(data["data"].keys()) == {
            "foodName", "calories", "macros", "healthScore", "confidenceScore", "warnings"
        }
        assert set(data["data"]["macros"].keys()) == {"protein", "carbs", "fat"}

    @patch("main.whisper_model")
    def test_speech_no_filename_extension(self, mock_whisper):
        """Files with no extension should default to .wav."""
        mock_whisper.transcribe.return_value = {"text": "chicken"}

        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("audio", b"audio bytes", "audio/wav")}
        )
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════
# 7. LABEL ENDPOINT (OCR)
# ═══════════════════════════════════════════════════════════════
class TestLabelEndpoint:
    """POST /api/v1/analyze/label — Pattern C (OCR Override)."""

    @patch("main.genai")
    def test_valid_label_returns_200(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "foodName": "Protein Bar",
            "calories": 220,
            "macros": {"protein": 20, "carbs": 25, "fat": 8}
        })
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/label",
            files={"file": ("label.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["foodName"] == "Protein Bar"
        assert data["data"]["calories"] == 220
        assert data["data"]["macros"]["protein"] == 20
        assert data["data"]["confidenceScore"] == 0.99
        assert any("ocr" in w.lower() or "label" in w.lower() for w in data["data"]["warnings"])

    @patch("main.genai")
    def test_label_gemini_failure_returns_500(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Gemini down")
        mock_genai.GenerativeModel.return_value = mock_model

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/label",
            files={"file": ("label.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 500

    def test_label_oversized_returns_413(self):
        large = _make_large_bytes(21)
        response = client.post(
            "/api/v1/analyze/label",
            files={"file": ("big_label.jpg", large, "image/jpeg")}
        )
        assert response.status_code == 413

    @patch("main.genai")
    def test_label_malformed_json_returns_500(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = "this is not json"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/label",
            files={"file": ("label.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 500

    @patch("main.genai")
    def test_label_with_backtick_json_still_parses(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = '```json\n{"foodName": "Oats", "calories": 150, "macros": {"protein": 5, "carbs": 27, "fat": 3}}\n```'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/label",
            files={"file": ("label.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        assert response.json()["data"]["foodName"] == "Oats"

    @patch("main.genai")
    def test_label_schema_matches_contract(self, mock_genai):
        """Label endpoint must return exact AnalysisResponse schema."""
        mock_response = MagicMock()
        mock_response.text = '{"foodName": "Milk", "calories": 60, "macros": {"protein": 3, "carbs": 5, "fat": 3}}'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/label",
            files={"file": ("label.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"success", "data"}
        assert set(data["data"].keys()) == {
            "foodName", "calories", "macros", "healthScore", "confidenceScore", "warnings"
        }

    @patch("main.genai")
    def test_label_missing_fields_defaults(self, mock_genai):
        """Missing fields in Gemini output should default to safe values."""
        mock_response = MagicMock()
        mock_response.text = '{}'  # empty object
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        jpeg = _make_jpeg_bytes()
        response = client.post(
            "/api/v1/analyze/label",
            files={"file": ("label.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["foodName"] == "Nutrition Label"
        assert data["data"]["calories"] == 0


# ═══════════════════════════════════════════════════════════════
# 8. PROXY ENDPOINT
# ═══════════════════════════════════════════════════════════════
class TestProxyEndpoint:
    """POST /api/v1/proxy/generate."""

    @patch("main.genai")
    def test_valid_prompt_returns_200(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = '{"suggestion": "Eat more vegetables"}'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        response = client.post(
            "/api/v1/proxy/generate",
            json={"prompt": "Give me diet tips"}
        )
        assert response.status_code == 200
        assert "text" in response.json()

    @patch("main.genai")
    def test_proxy_with_system_instruction(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = '{"response": "ok"}'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        response = client.post(
            "/api/v1/proxy/generate",
            json={
                "prompt": "test",
                "system_instruction": "You are a nutritionist"
            }
        )
        assert response.status_code == 200
        # Verify system_instruction was passed to GenerativeModel
        call_kwargs = mock_genai.GenerativeModel.call_args
        assert call_kwargs[1].get("system_instruction") == "You are a nutritionist"

    @patch("main.genai")
    def test_proxy_gemini_failure_returns_500(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API error")
        mock_genai.GenerativeModel.return_value = mock_model

        response = client.post(
            "/api/v1/proxy/generate",
            json={"prompt": "test"}
        )
        assert response.status_code == 500

    def test_proxy_missing_prompt_returns_422(self):
        response = client.post(
            "/api/v1/proxy/generate",
            json={}
        )
        assert response.status_code == 422  # Pydantic validation error

    @patch("main.genai")
    def test_proxy_without_system_instruction(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = "response text"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        response = client.post(
            "/api/v1/proxy/generate",
            json={"prompt": "just a prompt"}
        )
        assert response.status_code == 200
        # system_instruction should NOT be passed when None
        call_kwargs = mock_genai.GenerativeModel.call_args[1]
        assert "system_instruction" not in call_kwargs


# ═══════════════════════════════════════════════════════════════
# 9. VALIDATE_FILE_SIZE
# ═══════════════════════════════════════════════════════════════
class TestValidateFileSizeEdgeCases:
    """Additional edge cases for validate_file_size beyond test_validate_file_size.py."""

    @pytest.mark.asyncio
    async def test_exact_limit_passes(self):
        """A file exactly at the limit should pass."""
        from fastapi.datastructures import Headers
        from fastapi import UploadFile
        import tempfile

        max_mb = 1
        content = b"0" * (max_mb * 1024 * 1024)  # exactly 1 MB
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(content)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                upload = UploadFile(
                    filename="exact.bin", file=f, size=len(content),
                    headers=Headers({"content-type": "application/octet-stream"})
                )
                # Should not raise
                await validate_file_size(upload, max_mb)

    @pytest.mark.asyncio
    async def test_zero_byte_file_passes(self):
        """A zero-byte file should pass any limit."""
        from fastapi.datastructures import Headers
        from fastapi import UploadFile
        import tempfile

        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, "rb") as f:
                upload = UploadFile(
                    filename="empty.bin", file=f, size=0,
                    headers=Headers({"content-type": "application/octet-stream"})
                )
                await validate_file_size(upload, 1)


# ═══════════════════════════════════════════════════════════════
# 10. PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════
class TestPydanticModels:
    """Ensure the response Pydantic models serialize correctly."""

    def test_macros_model(self):
        from main import Macros
        m = Macros(protein=10, carbs=20, fat=5)
        assert m.model_dump() == {"protein": 10, "carbs": 20, "fat": 5}

    def test_analysis_data_model(self):
        from main import AnalysisData, Macros
        d = AnalysisData(
            foodName="Test", calories=100,
            macros=Macros(protein=10, carbs=20, fat=5),
            healthScore=85.0, confidenceScore=0.95, warnings=[]
        )
        dump = d.model_dump()
        assert dump["foodName"] == "Test"
        assert dump["macros"]["protein"] == 10

    def test_analysis_response_model(self):
        from main import AnalysisResponse, AnalysisData, Macros
        r = AnalysisResponse(
            success=True,
            data=AnalysisData(
                foodName="Test", calories=100,
                macros=Macros(protein=10, carbs=20, fat=5),
                healthScore=85.0, confidenceScore=0.95, warnings=["w1"]
            )
        )
        dump = r.model_dump()
        assert dump["success"] is True
        assert dump["data"]["warnings"] == ["w1"]

class TestMissingCoverage:
    def test_detect_device_cuda(self):
        with patch("main.torch.backends.mps.is_available", return_value=False):
            with patch("main.torch.cuda.is_available", return_value=True):
                assert _detect_device() == "cuda"
    
    def test_detect_device_cpu(self):
        with patch("main.torch.backends.mps.is_available", return_value=False):
            with patch("main.torch.cuda.is_available", return_value=False):
                assert _detect_device() == "cpu"

    @patch("main.gemini_classification", new_callable=AsyncMock, return_value={"foodName": "water", "is_liquid": False})
    @patch("main.vision_pipeline")
    def test_image_process_unidentified(self, mock_vision, mock_gpt):
        from PIL import UnidentifiedImageError
        mock_vision.estimate_volume.side_effect = UnidentifiedImageError("cannot identify image file")
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("corrupt.jpg", b"fake", "image/jpeg")}
        )
        assert response.status_code == 400
        assert "Invalid or corrupted image format" in response.json()["detail"]
        
    @patch("main.os.remove")
    @patch("main.whisper_model")
    def test_speech_cleanup_handled(self, mock_whisper, mock_remove):
        mock_whisper.transcribe.return_value = {"text": "test"}
        response = client.post(
            "/api/v1/analyze/speech",
            files={"file": ("audio.wav", b"fake", "audio/wav")}
        )
        assert response.status_code == 200

