import numpy as np
import pytest

from nyquistguard.experiments.v3_anchored_reliability import confidence_anchored_score


def test_anchor_weight_depends_only_on_independent_group_count() -> None:
    confidence = np.array([0.2, 0.8])
    calibrated = np.array([0.9, 0.1])
    score, weight = confidence_anchored_score(confidence, calibrated, 8, 32)
    assert weight == pytest.approx(0.2)
    assert np.all((score > 0.0) & (score < 1.0))


def test_zero_pseudo_groups_recovers_calibrator() -> None:
    confidence = np.array([0.2, 0.8])
    calibrated = np.array([0.7, 0.3])
    score, weight = confidence_anchored_score(confidence, calibrated, 8, 0)
    assert weight == 1.0
    assert score == pytest.approx(calibrated)
