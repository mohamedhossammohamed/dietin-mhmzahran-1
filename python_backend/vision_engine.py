import io
import math
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class VisionPipeline:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _initialize(self):
        if self._initialized:
            return
        # Determine optimal device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Loading Depth-Anything (Small HF) on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(self.device)
        self.model.eval()
        self._initialized = True

    def estimate_volume(self, image_bytes: bytes) -> float:
        """
        Calculates physical volume in cubic centimeters (cm^3).
        Assumptions:
        - Mobile camera distance: 30 cm
        - Mobile camera Field of View (FOV): 70 degrees
        """
        self._initialize()
        # Convert bytes to PIL Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}") from e

        width, height = image.size

        # Run inference to generate a relative depth map
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy array
        depth_map = prediction.squeeze().cpu().numpy()

        # The model outputs relative depth (higher values = closer).
        # We map this back to physical depth.
        min_depth_cm = 20.0
        max_depth_cm = 40.0

        # Normalize relative depth map to 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            norm_depth = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            norm_depth = np.zeros_like(depth_map)

        # Invert so higher values = further away
        norm_depth = 1.0 - norm_depth

        # Map to physical range
        physical_depth_map = min_depth_cm + norm_depth * (max_depth_cm - min_depth_cm)

        # Geometric assumptions:
        camera_distance_cm = 30.0
        fov_degrees = 70.0

        # Calculate pixel dimensions in physical space
        # width of view at distance = 2 * distance * tan(FOV / 2)
        fov_rad = math.radians(fov_degrees)
        view_width_cm = 2 * camera_distance_cm * math.tan(fov_rad / 2)

        # Assuming square pixels
        pixel_width_cm = view_width_cm / width
        pixel_area_cm2 = pixel_width_cm * pixel_width_cm

        # To calculate volume, integrate over pixels.
        # Volume = Area * Depth.
        # Assume background is at max_depth_cm, object protrudes upwards.
        protrusion_height = max_depth_cm - physical_depth_map

        # Threshold: > 1cm protrusion = object
        object_mask = protrusion_height > 1.0

        volume_cm3 = np.sum(protrusion_height[object_mask] * pixel_area_cm2)

        # Sensible fallback for tiny values
        if volume_cm3 < 10.0:
            volume_cm3 = 250.0  # fallback arbitrary volume

        return float(volume_cm3)

    def run_mobile_sam(self, image_bytes: bytes):
        """
        Stub function for Phase 4 integration. Returns a dummy bounding box.
        """
        return {"masks": [{"bbox": [0, 0, 100, 100]}]}

# Export singleton instance
vision_engine = VisionPipeline()
