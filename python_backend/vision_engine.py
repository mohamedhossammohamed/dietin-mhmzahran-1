import io
import math
import numpy as np
from PIL import Image
import torch
import time
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from logger import logger

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
        
        start_time = time.time()
        # Determine optimal device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            logger.info(f"Vision Engine: Loading Depth-Anything-Small-HF on {self.device}...")
            self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
            self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(self.device)
            self.model.eval()
            self._initialized = True
            duration = (time.time() - start_time) * 1000
            logger.info(f"Vision Engine: Model initialized successfully in {duration:.2f}ms on {self.device}")
        except Exception as e:
            logger.error(f"Vision Engine INIT FAILURE: Failed to load models: {str(e)}")
            self._initialized = False
            raise RuntimeError(f"Vision Engine could not be initialized: {str(e)}")

    def estimate_volume(self, image_bytes: bytes) -> float:
        """
        Calculates physical volume in cubic centimeters (cm^3).
        Assumptions:
        - Mobile camera distance: 30 cm
        - Mobile camera Field of View (FOV): 70 degrees
        """
        start_time = time.time()
        logger.debug(f"Vision Engine: Starting volume estimation for {len(image_bytes)} bytes of image data")

        try:
            self._initialize()
        except Exception as e:
            logger.error(f"Vision Engine FAILURE: Initialization failed: {str(e)}")
            raise

        # Convert bytes to PIL Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
            logger.debug(f"Vision Engine: Image decoded. Size: {width}x{height}")
        except Exception as e:
            logger.error(f"Vision Engine FAILURE: Failed to decode image: {str(e)}")
            raise ValueError(f"Failed to decode image: {e}") from e

        try:
            # Run inference to generate a relative depth map
            logger.debug("Vision Engine: Running depth estimation inference...")
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
            logger.debug(f"Vision Engine: Depth map generated. Raw range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")

        except Exception as e:
            logger.error(f"Vision Engine FAILURE: Inference or interpolation failed: {str(e)}")
            raise RuntimeError(f"Depth estimation failed: {str(e)}")

        try:
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
                logger.warning("Vision Engine: Flat depth map detected (max == min)")
                norm_depth = np.zeros_like(depth_map)

            # Invert so higher values = further away
            norm_depth = 1.0 - norm_depth

            # Map to physical range
            physical_depth_map = min_depth_cm + norm_depth * (max_depth_cm - min_depth_cm)

            # Geometric assumptions:
            camera_distance_cm = 30.0
            fov_degrees = 70.0

            # Calculate pixel dimensions in physical space
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
            object_pixel_count = np.sum(object_mask)

            if object_pixel_count == 0:
                logger.warning("Vision Engine: No object detected (protrusion < 1cm across all pixels)")
                volume_cm3 = 250.0 # fallback
            else:
                volume_cm3 = np.sum(protrusion_height[object_mask] * pixel_area_cm2)
                logger.debug(f"Vision Engine: Object detected. Pixels: {object_pixel_count}, Volume: {volume_cm3:.2f}cm3")

            # Sensible fallback for tiny values
            if volume_cm3 < 10.0:
                logger.warning(f"Vision Engine: Calculated volume {volume_cm3:.2f}cm3 is too small. Using fallback 250.0cm3")
                volume_cm3 = 250.0

            duration = (time.time() - start_time) * 1000
            logger.info(f"Vision Engine: Volume estimation completed: {volume_cm3:.2f}cm3 in {duration:.2f}ms")
            return float(volume_cm3)

        except Exception as e:
            logger.error(f"Vision Engine FAILURE: Geometry calculation failed: {str(e)}")
            raise RuntimeError(f"Volume geometric calculation failed: {str(e)}")

    def run_mobile_sam(self, image_bytes: bytes):
        """
        Stub function for Phase 4 integration. Returns a dummy bounding box.
        """
        logger.debug("Vision Engine: MobileSAM stub called")
        return {"masks": [{"bbox": [0, 0, 100, 100]}]}

# Export singleton instance
vision_engine = VisionPipeline()

