from pathlib import Path

import numpy as np
import pytest
import yaml

from nyquistguard.experiments.v3_reliability import (
    nyquist_reliability_score,
    threshold_for_target_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v3_protocol_is_frozen_and_bounded() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "v3_reliability.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["datasets"] == [
        "basicmotions_uea",
        "epilepsy_uea",
        "pamap2_uci",
        "mhealth_uci",
    ]
    assert config["seeds"] == [17, 42, 2026]
    assert config["score"]["confidence_exponent"] == 1.0
    assert config["score"]["gate_quality_exponent"] == 1.0
    assert config["wall_time_budget_seconds"] == 540


def test_nrs_preserves_within_rate_confidence_ranking() -> None:
    confidence = np.array([0.9, 0.4, 0.7])
    score = nyquist_reliability_score(confidence, 0.35)
    assert np.array_equal(np.argsort(score), np.argsort(confidence))
    assert score == pytest.approx(confidence * 0.35)


def test_validation_threshold_hits_target_coverage_without_training() -> None:
    scores = np.linspace(0.01, 1.0, 100)
    threshold = threshold_for_target_coverage(scores, 0.8)
    assert np.mean(scores >= threshold) == pytest.approx(0.8)


def test_nrs_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        nyquist_reliability_score(np.array([1.2]), 0.5)
    with pytest.raises(ValueError):
        threshold_for_target_coverage(np.array([]), 0.8)
