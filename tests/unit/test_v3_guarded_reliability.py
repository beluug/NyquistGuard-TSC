from nyquistguard.experiments.v3_guarded_reliability import select_reliability_mode


def test_guard_enables_only_strict_oof_improvement() -> None:
    assert select_reliability_mode(0.2, 0.19, 0.0) == "calibrated"
    assert select_reliability_mode(0.2, 0.2, 0.0) == "confidence"
    assert select_reliability_mode(0.2, 0.21, 0.0) == "confidence"


def test_guard_respects_minimum_absolute_gain() -> None:
    assert select_reliability_mode(0.2, 0.195, 0.01) == "confidence"
    assert select_reliability_mode(0.2, 0.18, 0.01) == "calibrated"
