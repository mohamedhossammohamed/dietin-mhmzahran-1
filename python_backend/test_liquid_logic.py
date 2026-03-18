from liquid_logic import apply_liquid_fallback


def test_apply_liquid_fallback_uses_existing_volume_when_positive():
    volume, confidence = apply_liquid_fallback(315.0, 0.95)
    assert volume == 315.0
    assert confidence == 0.55

    volume_low_conf, confidence_low_conf = apply_liquid_fallback(315.0, 0.3)
    assert volume_low_conf == 315.0
    assert confidence_low_conf == 0.3


def test_apply_liquid_fallback_uses_default_volume_when_zero_or_missing():
    volume_zero, confidence_zero = apply_liquid_fallback(0.0, 0.4)
    assert volume_zero == 240.0
    assert confidence_zero == 0.4

    volume_negative, confidence_negative = apply_liquid_fallback(-5.0, 0.9)
    assert volume_negative == 240.0
    assert confidence_negative == 0.55

    volume_none_like, confidence_none_like = apply_liquid_fallback(None, 0.9)
    assert volume_none_like == 240.0
    assert confidence_none_like == 0.55
