import numpy as np

from nyquistguard.experiments.v3_calibrated_reliability import (
    cross_fitted_scores,
    reliability_features,
)


def test_reliability_feature_map_is_fixed_and_finite() -> None:
    logits = np.array([[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]])
    features = reliability_features(logits, np.array([0.8, 0.4]), 0.6)
    assert features.shape == (2, 6)
    assert np.isfinite(features).all()


def test_cross_fitting_keeps_groups_together_and_returns_oof_scores() -> None:
    groups = np.tile(np.arange(8), 5)
    x = np.column_stack([groups, np.linspace(-1.0, 1.0, len(groups))])
    correct = ((groups % 3) != 0).astype(np.int64)
    scores, finite = cross_fitted_scores(x, correct, groups, 4, 1.0, 17)
    assert scores.shape == correct.shape
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert finite
