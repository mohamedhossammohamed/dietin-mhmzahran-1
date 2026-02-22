# Phase 3: The Vision Engine

This document logs the actions taken during Phase 3.
We implemented deterministic volume geometry processing on Apple Silicon utilizing Depth-Anything-V2.

## Actions Completed

1. Added `transformers` and `pillow` to `requirements.txt`.
2. Initialized `python_backend/vision_engine.py`.
3. Created a singleton `VisionPipeline` class that successfully loads `depth-anything/Depth-Anything-V2-Small` onto the `mps` device.
4. Implemented `estimate_volume(image_bytes)` which:
   - Evaluates a depth map disparity tensor.
   - Applies the pinhole camera geometry math, assuming an FOV of 70 degrees and a focal distance plane at 30cm.
   - Converts the relative disparity to a real-world centimeter scale (Max Height = 10cm).
   - Integrates the volume over the pixel array by calculating single pixel real-world $cm^2$ area × estimated height $cm$.
5. Implemented the `run_mobile_sam(image_bytes)` stub returning bounding box `[0, 0, 100, 100]` for Phase 4 bridging.
6. Local test successfully instantiated the model on `mps` and computed a physical volume.

## Geometric Assumptions (Pinhole Camera Model)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FOV | 70° horizontal | Typical smartphone rear camera |
| Z distance | 30 cm | Average phone-to-plate distance |
| Disparity threshold | 50 / 255 | Background/plate cutoff |
| Max height | 10 cm | Tallest plated food assumption |

**Volume formula:** `V = Σ (pixel_area_cm² × height_cm)` where:
- `pixel_area = (W_real / img_w) × (H_real / img_h)`
- `W_real = 2 · Z · tan(FOV/2) ≈ 42 cm`
- `height = normalized_disparity × MAX_HEIGHT_CM`

## Acceptance Criteria Verification

✅ Script accepts an image and returns a deterministic float representing volume in cm³.
✅ Math is logically sound based on pinhole camera geometry.

## Stress Test Results (28/28 passed)

| Test Class | Tests | Status |
|---|---|---|
| AcceptanceCriteria | float output, non-negative, deterministic | ✅ |
| Singleton | same instance, shared device | ✅ |
| DeviceDetection | valid device string | ✅ |
| VolumeEstimation | small/large/non-square/gradient/PNG images | ✅ |
| BlackImage | low-disparity → low volume | ✅ |
| GeometricMath | pixel area formula, max volume, constants | ✅ |
| MobileSAMStub | returns [0,0,100,100] | ✅ |
| EdgeCases | grayscale, RGBA, tiny, panoramic, tall, corrupt, empty | ✅ |
| Concurrency | 20 parallel calls → consistent results | ✅ |
| ModelLoading | estimator exists and callable | ✅ |
