from typing import Optional, Tuple


def apply_liquid_fallback(volume_cm3: Optional[float], confidence_score: float) -> Tuple[float, float]:
    """
    Down-weight liquid confidence and apply a conservative fallback volume when depth is unreliable.
    """
    inferred_volume = float(volume_cm3) if volume_cm3 is not None and volume_cm3 > 0 else 240.0
    adjusted_confidence = min(confidence_score, 0.55)
    return inferred_volume, adjusted_confidence
