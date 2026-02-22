import io
import math

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class VisionPipeline:
    """Singleton depth-to-volume estimator running on Apple Silicon (MPS).

    Loads Depth-Anything-V2-Small and uses pinhole camera geometry to
    convert a food photograph into an estimated physical volume (cm³).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionPipeline, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Prefer MPS on Apple Silicon, fallback to CUDA then CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Initializing Depth-Anything-V2-Small on device: {self.device}")

        # Load the HuggingFace depth-estimation pipeline with the
        # Depth-Anything-V2-Small checkpoint.
        self.depth_estimator = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self.device,
        )
        print("VisionPipeline initialized successfully.")

    # ── Geometric Constants ───────────────────────────────────────
    # These encode the "standard mobile camera" assumptions mandated
    # by the master plan (Phase 3, line 113).
    #
    # FOV_DEGREES (70°):
    #   Typical horizontal field-of-view for a smartphone rear camera.
    #   This determines how wide the real-world scene is at a given
    #   distance.  The formula comes directly from the pinhole camera
    #   model:  Width_real = 2 · Z · tan(FOV / 2).
    #
    # Z_DISTANCE_CM (30 cm):
    #   Assumed perpendicular distance from the phone camera to the
    #   food plate.  A user typically holds the phone roughly 25-35 cm
    #   above the table; 30 cm is a reasonable mean.
    #
    # DISPARITY_THRESHOLD (50):
    #   Depth-Anything-V2 outputs a relative disparity map (0-255)
    #   where 255 = closest to camera and 0 = farthest.  Pixels
    #   below this threshold are treated as background/plate and
    #   excluded from the volume integral.
    #
    # MAX_HEIGHT_CM (10 cm):
    #   The maximum physical height a food item is assumed to have.
    #   Disparity = 255 is linearly mapped to this value.  Most plated
    #   food items (rice, steak, salad) sit well within 10 cm.
    FOV_DEGREES = 70.0
    Z_DISTANCE_CM = 30.0
    DISPARITY_THRESHOLD = 50
    MAX_HEIGHT_CM = 10.0

    def estimate_volume(self, image_bytes: bytes) -> float:
        """Convert a food image to an estimated volume in cm³.

        **Pinhole Camera Geometry**

        1. Real-world width at the focal plane:
           ``W_real = 2 · Z · tan(FOV / 2)``
        2. Each pixel covers:
           ``pixel_area = (W_real / img_w) × (H_real / img_h)``  [cm²]
        3. Depth-Anything-V2 yields relative disparity (0-255).
           Foreground pixels (> DISPARITY_THRESHOLD) are linearly
           rescaled to [0, MAX_HEIGHT_CM] cm.
        4. Volume = Σ pixel_area × estimated_height  [cm³]

        Returns a deterministic float ≥ 0.
        """
        # ── 1. Decode image ──────────────────────────────────────
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ── 2. Run Depth-Anything-V2 inference ───────────────────
        result = self.depth_estimator(image)
        # result["depth"] is a PIL Image with pixel values in [0, 255]
        # representing *relative* disparity (inverse depth).
        depth_array = np.array(result["depth"])

        img_width, img_height = image.size

        # ── 3. Compute real-world pixel area (pinhole model) ─────
        #   W_real = 2 · Z · tan(FOV/2)
        #   H_real = W_real × (img_h / img_w)   (preserve aspect ratio)
        #   pixel_area = (W_real / img_w) × (H_real / img_h)
        fov_rad = math.radians(self.FOV_DEGREES)
        width_real_cm = 2.0 * self.Z_DISTANCE_CM * math.tan(fov_rad / 2.0)
        aspect_ratio = img_height / img_width
        height_real_cm = width_real_cm * aspect_ratio
        pixel_area_cm2 = (width_real_cm / img_width) * (height_real_cm / img_height)

        # ── 4. Foreground segmentation via disparity threshold ───
        foreground_mask = depth_array > self.DISPARITY_THRESHOLD
        if not np.any(foreground_mask):
            return 0.0  # No object detected above the plate plane

        # ── 5. Map disparity to physical height ──────────────────
        #   Normalise foreground disparity to [0, 1], then scale to cm.
        disparity_normalized = (
            (depth_array - self.DISPARITY_THRESHOLD)
            / (255.0 - self.DISPARITY_THRESHOLD)
        )
        disparity_normalized = np.clip(disparity_normalized, 0.0, 1.0)
        estimated_heights_cm = disparity_normalized * self.MAX_HEIGHT_CM

        # ── 6. Integrate volume: Σ (area × height) ──────────────
        volume_cm3 = float(np.sum(pixel_area_cm2 * estimated_heights_cm))

        return volume_cm3

    def run_mobile_sam(self, image_bytes: bytes) -> list:
        """
        Stub function returning a dummy bounding box for Phase 4 integration.
        MobileSAM logic will be implemented here later.
        """
        print("Executing MobileSAM stub...")
        return [0, 0, 100, 100]

# --- Manual Test Block ---
if __name__ == "__main__":
    import os
    
    # Create sample dummy image bytes
    test_image = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    pipeline = VisionPipeline()
    vol = pipeline.estimate_volume(img_bytes)
    print(f"Test Execution Successful. Estimated Volume: {vol:.2f} cm^3")
    
    print("MobileSAM stub:", pipeline.run_mobile_sam(img_bytes))
