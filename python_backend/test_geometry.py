import sys
import numpy as np
sys.path.append('/Users/mohammedhossam/Desktop/dietin-mhmzahran-1/python_backend')
from vision_engine import VisionPipeline
import io
import os
from PIL import Image

def test():
    pipeline = VisionPipeline()
    
    # Create sample dummy image bytes
    test_image = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # Manually run the steps to print intermediates
    result = pipeline.depth_estimator(test_image)
    depth_array = np.array(result["depth"])
    
    print(f"Max Disparity: {np.max(depth_array)}, Min Disparity: {np.min(depth_array)}")
    
    vol = pipeline.estimate_volume(img_bytes)
    print(f"Inverse Geometric Execution Successful. Estimated Volume: {vol:.2f} cm^3")

if __name__ == "__main__":
    test()
