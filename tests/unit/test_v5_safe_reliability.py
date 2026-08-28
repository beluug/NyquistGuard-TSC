from __future__ import annotations

from copy import deepcopy

import pytest

from nyquistguard.research.v5_safe_reliability import (
    derive_safe_reliability,
    select_consensus_mode,
)


def test_consensus_requires_every_seed_to_improve() -> None:
    selected = select_consensus_mode(
        [0.01, 0.02, 0.03],
        minimum_seed_gain=0.0,
        minimum_mean_gain=0.0,
        required_positive_fraction=1.0,
    )
    assert selected["mode"] == "observability"

    fallback = select_consensus_mode(
        [0.01, -0.001, 0.03],
        minimum_seed_gain=0.0,
        minimum_mean_gain=0.0,
        required_positive_fraction=1.0,
    )
    assert fallback["mode"] == "confidence_fallback"


def test_tie_selects_confidence_fallback() -> None:
    row = select_consensus_mode(
        [0.01, 0.0, 0.02],
        minimum_seed_gain=0.0,
        minimum_mean_gain=0.0,
        required_positive_fraction=1.0,
    )
    assert row["mode"] == "confidence_fallback"


def _source() -> dict:
    datasets = ("a", "b", "c", "d")
    seeds = (1, 2, 3)
    candidates = {}
    for dataset in datasets:
        for seed in seeds:
            confidence = 0.2
            validation_gain = 0.02 if dataset == "a" else -0.01
            test_gain = 0.01 if dataset == "a" else -0.02
            candidates[f"{dataset}__seed{seed}__v5_dual_path"] = {
                "validation": {
                    "pooled_confidence_aurc": confidence,
                    "pooled_observability_aurc": confidence - validation_gain,
                },
                "test": {
                    "pooled_confidence_aurc": confidence,
                    "pooled_observability_aurc": confidence - test_gain,
                    "selected_pooled_aurc": confidence,
                },
            }
    return {
        "decision": {
            "checks": {
                "average_dataset_unseen_gain": True,
                "positive_dataset_count": True,
                "single_dataset_unseen_floor": True,
                "average_dataset_full_rate_floor": True,
                "no_constant_prediction": True,
                "finite_metrics": True,
            }
        },
        "candidate_results": candidates,
    }


def _config() -> dict:
    return {
        "source": {
            "datasets": ["a", "b", "c", "d"],
            "seeds": [1, 2, 3],
            "required_candidate_role": "v5_dual_path",
        },
        "controller": {
            "minimum_seed_validation_aurc_gain": 0.0,
            "minimum_dataset_mean_validation_aurc_gain": 0.0,
            "required_positive_seed_fraction": 1.0,
        },
        "development_gates": {
            "require_source_classification_checks_passed": True,
            "require_each_dataset_selected_test_aurc_nonworse_than_confidence": True,
            "maximum_average_dataset_selected_test_aurc_delta_vs_confidence": 0.0,
            "require_finite_metrics": True,
        },
    }


def test_derivation_is_validation_selected_and_does_not_mutate_source() -> None:
    source = _source()
    frozen = deepcopy(source)
    results, decision = derive_safe_reliability(source, _config())
    assert source == frozen
    assert results["a"]["mode"] == "observability"
    assert results["b"]["mode"] == "confidence_fallback"
    assert decision["checks"]["source_classification"] is True
    assert decision["checks"]["each_dataset_reliability_safety"] is True
    assert decision["average_dataset_selected_test_aurc_delta_vs_confidence"] == pytest.approx(-0.0025)
