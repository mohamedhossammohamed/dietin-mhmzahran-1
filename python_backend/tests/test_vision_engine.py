from PIL import Image
import io
from vision_engine import vision_engine

def test_vision_pipeline():
    # Create a dummy image (e.g., 224x224 white image)
    image = Image.new("RGB", (224, 224), "white")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Test MobileSAM stub
    masks = vision_engine.run_mobile_sam(img_byte_arr)
    assert isinstance(masks, dict)
    assert "masks" in masks
    assert "bbox" in masks["masks"][0]

    # Test estimate_volume (returns fallback arbitrary volume for empty white image since protrusion < 1cm)
    volume = vision_engine.estimate_volume(img_byte_arr)
    assert isinstance(volume, float)
    assert volume > 0.0
