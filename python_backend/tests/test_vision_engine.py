import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import io
import numpy as np
from vision_engine import VisionPipeline

def test_run_mobile_sam():
    """Test the MobileSAM stub without loading any ML model."""
    pipeline = VisionPipeline()
    image = Image.new("RGB", (224, 224), "white")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    masks = pipeline.run_mobile_sam(img_bytes)
    assert isinstance(masks, dict)
    assert "masks" in masks
    assert "bbox" in masks["masks"][0]

def test_estimate_volume_with_mock():
    """Test estimate_volume with the depth model mocked to avoid network/download."""
    pipeline = VisionPipeline()

    # Provide a mock initialized state so _initialize() is skipped
    pipeline._initialized = True
    pipeline.device = "cpu"

    # Mock processor and model
    mock_processor = MagicMock()
    mock_model = MagicMock()

    # Simulate processor output: dict-like with .to() returning itself
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    # Simulate model output: predicted_depth as a torch tensor (flat)
    import torch
    depth_tensor = torch.ones(1, 224, 224)
    mock_outputs = MagicMock()
    mock_outputs.predicted_depth = depth_tensor
    mock_model.return_value = mock_outputs

    pipeline.processor = mock_processor
    pipeline.model = mock_model

    image = Image.new("RGB", (224, 224), "white")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    volume = pipeline.estimate_volume(img_bytes)
    assert isinstance(volume, float)
    assert volume > 0.0

def test_estimate_volume_invalid_image():
    """Test that invalid image bytes raise a ValueError."""
    pipeline = VisionPipeline()
    pipeline._initialized = True
    pipeline.device = "cpu"
    pipeline.processor = MagicMock()
    pipeline.model = MagicMock()

    with pytest.raises(ValueError, match="Failed to decode image"):
        pipeline.estimate_volume(b"not-an-image")
