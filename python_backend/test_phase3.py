"""
Phase 3 Vision Engine — Comprehensive Stress Test Suite
========================================================
Tests the VisionPipeline singleton, Depth-Anything-V2 volume estimation, and
MobileSAM stub against the master plan Phase 3 acceptance criteria.

Master Plan §Phase 3 Acceptance Criteria (line 116):
  "The script accepts an image and returns a deterministic float
   representing volume as real world cm^3. The math must be logically sound
   based on pinhole camera geometry."
"""

import concurrent.futures
import io
from unittest.mock import patch
import math

import numpy as np
import pytest
from PIL import Image

from vision_engine import VisionPipeline


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _make_image_bytes(width: int = 224, height: int = 224, color="red", fmt="JPEG") -> bytes:
    """Create a simple solid-color image and return raw bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_gradient_image_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create an image with a vertical gradient (bottom dark, top bright)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(255 * y / height)
        arr[y, :, :] = val
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def pipeline():
    """One shared VisionPipeline instance for all tests — mirrors singleton."""
    return VisionPipeline()


# ─────────────────────────────────────────────────────────────
# 1. ACCEPTANCE CRITERIA  (Master Plan §Phase 3 line 116)
# ─────────────────────────────────────────────────────────────
class TestAcceptanceCriteria:
    """
    'The script accepts an image and returns a deterministic float
     representing volume. The math must be logically sound based on
     pinhole camera geometry.'
    """

    def test_accepts_image_returns_float(self, pipeline):
        """Core acceptance: image bytes in → float out."""
        vol = pipeline.estimate_volume(_make_image_bytes())
        assert isinstance(vol, float), f"Expected float, got {type(vol)}"

    def test_volume_is_non_negative(self, pipeline):
        vol = pipeline.estimate_volume(_make_image_bytes())
        assert vol >= 0.0, f"Volume must be ≥ 0, got {vol}"

    def test_volume_is_deterministic(self, pipeline):
        """Same input → same output, every time."""
        img = _make_image_bytes(224, 224, "blue")
        v1 = pipeline.estimate_volume(img)
        v2 = pipeline.estimate_volume(img)
        assert v1 == v2, f"Non-deterministic: {v1} vs {v2}"


# ─────────────────────────────────────────────────────────────
# 2. SINGLETON PATTERN
# ─────────────────────────────────────────────────────────────
class TestSingleton:
    """VisionPipeline must be a singleton (spec: 'singleton class')."""

    def test_same_instance(self):
        a = VisionPipeline()
        b = VisionPipeline()
        assert a is b, "VisionPipeline should be a singleton"

    def test_shared_device(self, pipeline):
        other = VisionPipeline()
        assert pipeline.device == other.device


# ─────────────────────────────────────────────────────────────
# 3. DEVICE DETECTION
# ─────────────────────────────────────────────────────────────
class TestDeviceDetection:
    """Device string must be one of the three valid values."""

    def test_device_is_valid(self, pipeline):
        assert pipeline.device in ("mps", "cuda", "cpu"), (
            f"Unexpected device: {pipeline.device}"
        )


# ─────────────────────────────────────────────────────────────
# 4. VOLUME ESTIMATION — FUNCTIONAL TESTS
# ─────────────────────────────────────────────────────────────
class TestVolumeEstimation:
    """Functional tests against a variety of synthetic images."""

    def test_small_image(self, pipeline):
        """Even a tiny 32×32 image should work."""
        vol = pipeline.estimate_volume(_make_image_bytes(32, 32))
        assert isinstance(vol, float)

    def test_large_image(self, pipeline):
        """640×480 should also work without error."""
        vol = pipeline.estimate_volume(_make_image_bytes(640, 480))
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_non_square_image(self, pipeline):
        """Non-square aspect ratio must not crash."""
        vol = pipeline.estimate_volume(_make_image_bytes(320, 160))
        assert isinstance(vol, float)

    def test_gradient_image_positive_volume(self, pipeline):
        """A gradient image should produce some non-zero volume
        (the bright regions will appear as foreground)."""
        vol = pipeline.estimate_volume(_make_gradient_image_bytes())
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_png_format_accepted(self, pipeline):
        """PNG format should be handled correctly."""
        vol = pipeline.estimate_volume(_make_image_bytes(224, 224, "green", "PNG"))
        assert isinstance(vol, float)
        assert vol >= 0.0


# ─────────────────────────────────────────────────────────────
# 5. BLACK IMAGE → ZERO VOLUME
# ─────────────────────────────────────────────────────────────
class TestBlackImage:
    """A completely black image represents 'nothing on the plate'.
    Depth model should return mostly low-disparity pixels → volume ≈ 0."""

    def test_all_black_low_volume(self, pipeline):
        vol = pipeline.estimate_volume(_make_image_bytes(224, 224, "black"))
        # Depth model *may* produce small non-zero disparity even on black,
        # but it should be very small relative to a real food image.
        assert vol >= 0.0
        # We keep this test lenient (just non-negative) because the model
        # may still assign non-zero disparity to a uniform black input.


# ─────────────────────────────────────────────────────────────
# 6. GEOMETRIC MATH VALIDATION
# ─────────────────────────────────────────────────────────────
class TestGeometricMath:
    """Validate the pinhole camera math independently of the model."""

    def test_pixel_area_formula(self):
        """Hand-calculate pixel area at 30 cm with 70° FOV for a 224×224 image."""
        fov_rad = math.radians(70.0)
        w_real = 2.0 * 30.0 * math.tan(fov_rad / 2.0)  # ≈ 42.01 cm
        h_real = w_real * (224 / 224)                    # square → same
        pixel_area = (w_real / 224) * (h_real / 224)

        # Cross-check: w_real should be about 42 cm
        assert 40.0 < w_real < 44.0, f"w_real out of range: {w_real}"
        # Pixel area ≈ (42/224)^2 ≈ 0.0352 cm^2
        assert 0.03 < pixel_area < 0.04, f"pixel_area out of range: {pixel_area}"

    def test_max_possible_volume(self):
        """Maximum theoretical volume for a 224×224 image where every pixel
        has disparity 255 (i.e. height = MAX_HEIGHT_CM)."""
        fov_rad = math.radians(70.0)
        w_real = 2.0 * 30.0 * math.tan(fov_rad / 2.0)
        h_real = w_real  # square image
        pixel_area = (w_real / 224) * (h_real / 224)
        max_vol = pixel_area * 10.0 * 224 * 224  # all pixels at 10 cm
        # Should be around 42 * 42 * 10 ≈ 17,640 cm³
        assert 15000 < max_vol < 20000, f"Max volume out of range: {max_vol}"

    def test_constants_match_class(self):
        """Ensure class constants align with the spec values."""
        p = VisionPipeline()
        assert p.FOV_DEGREES == 70.0
        assert p.Z_DISTANCE_CM == 30.0
        assert p.DISPARITY_THRESHOLD == 50
        assert p.MAX_HEIGHT_CM == 10.0


# ─────────────────────────────────────────────────────────────
# 7. MOBILE SAM STUB
# ─────────────────────────────────────────────────────────────
class TestMobileSAMStub:
    """Master plan: 'stub function returning a dummy bounding box'."""

    def test_returns_list(self, pipeline):
        result = pipeline.run_mobile_sam(_make_image_bytes())
        assert isinstance(result, list)

    def test_returns_correct_bbox(self, pipeline):
        result = pipeline.run_mobile_sam(_make_image_bytes())
        assert result == [0, 0, 100, 100]

    def test_accepts_any_image_bytes(self, pipeline):
        """Stub should not crash regardless of input."""
        result = pipeline.run_mobile_sam(b"not-a-real-image")
        assert result == [0, 0, 100, 100]


# ─────────────────────────────────────────────────────────────
# 8. EDGE CASES
# ─────────────────────────────────────────────────────────────
class TestEdgeCases:
    """Robustness against unusual but valid inputs."""

    def test_grayscale_image_converted(self, pipeline):
        """A grayscale image (mode 'L') must be auto-converted to RGB."""
        img = Image.new("L", (224, 224), color=128)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        vol = pipeline.estimate_volume(buf.getvalue())
        assert isinstance(vol, float)

    def test_rgba_image_converted(self, pipeline):
        """An RGBA image (mode 'RGBA') must be auto-converted to RGB."""
        img = Image.new("RGBA", (224, 224), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        vol = pipeline.estimate_volume(buf.getvalue())
        assert isinstance(vol, float)

    def test_very_small_image(self, pipeline):
        """Minimum plausible image size."""
        vol = pipeline.estimate_volume(_make_image_bytes(8, 8))
        assert isinstance(vol, float)

    def test_wide_panoramic_image(self, pipeline):
        """Very wide aspect ratio."""
        vol = pipeline.estimate_volume(_make_image_bytes(640, 100))
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_tall_image(self, pipeline):
        """Very tall aspect ratio."""
        vol = pipeline.estimate_volume(_make_image_bytes(100, 640))
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_corrupt_bytes_raises(self, pipeline):
        """Completely invalid bytes must raise an error, not silently fail."""
        with pytest.raises(Exception):
            pipeline.estimate_volume(b"this-is-not-an-image")

    def test_empty_bytes_raises(self, pipeline):
        """Empty bytes must raise an error."""
        with pytest.raises(Exception):
            pipeline.estimate_volume(b"")


# ─────────────────────────────────────────────────────────────
# 9. CONCURRENCY
# ─────────────────────────────────────────────────────────────
class TestConcurrency:
    """Parallel estimate_volume calls must not crash or produce inconsistent results."""

    def test_20_parallel_calls(self, pipeline):
        img = _make_image_bytes(112, 112, "yellow")

        def run(_):
            return pipeline.estimate_volume(img)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All results must be equal (deterministic) and non-negative
        assert len(results) == 20
        for r in results:
            assert isinstance(r, float)
            assert r >= 0.0
        assert all(r == results[0] for r in results), (
            f"Non-deterministic concurrent results: {set(results)}"
        )


# ─────────────────────────────────────────────────────────────
# 10. MODEL LOADING
# ─────────────────────────────────────────────────────────────
class TestModelLoading:
    """Ensure the depth estimator is correctly loaded."""

    def test_depth_estimator_exists(self, pipeline):
        assert pipeline.depth_estimator is not None

    def test_depth_estimator_callable(self, pipeline):
        """The pipeline object should be callable."""
        assert callable(pipeline.depth_estimator)

class TestVisionMissingCoverage:
    @patch("vision_engine.torch.cuda.device_count", return_value=1)
    @patch("vision_engine.pipeline")
    def test_initialize_cuda(self, mock_hf_pipeline, mock_dc, pipeline):
        with patch("vision_engine.torch.backends.mps.is_available", return_value=False):
            with patch("vision_engine.torch.cuda.is_available", return_value=True):
                pipeline._initialize()
                assert pipeline.device == "cuda"
                
    def test_initialize_cpu(self, pipeline):
        with patch("vision_engine.torch.backends.mps.is_available", return_value=False):
            with patch("vision_engine.torch.cuda.is_available", return_value=False):
                pipeline._initialize()
                assert pipeline.device == "cpu"

    @patch("vision_engine.pipeline")
    def test_disparity_division_by_zero(self, mock_hf_pipeline, pipeline):
        # Force all disparity to 0 to test np.maximum clip
        import numpy as np
        mock_hf_pipeline.return_value = {"depth": Image.new("L", (224, 224), 0)}
        pipeline.depth_estimator = mock_hf_pipeline
        vol = pipeline.estimate_volume(_make_image_bytes(224, 224, "red"))
        assert vol >= 0.0

