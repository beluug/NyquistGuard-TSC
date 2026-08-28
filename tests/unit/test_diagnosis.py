from __future__ import annotations

import math

import numpy as np

from experiment_dashboard import STAGE_TASKS
from nyquistguard.experiments.diagnosis import (
    PILOT_DATASETS,
    PILOT_SEEDS,
    RATE_IDS,
    _analyze_flip,
    _analyze_performance,
    _aurc,
    _run_summaries,
)
from nyquistguard.experiments.pilot import PILOT_METHODS


def _synthetic_runs():
    runs = {}
    method_f1 = {
        "nyquistguard": 0.70,
        "fixed_rate_tcn": 0.58,
        "multirate_tcn": 0.60,
        "minirocket": 0.90,
        "no_nyquist_gate": 0.65,
        "no_cbe": 0.66,
        "no_selective_head": 0.67,
    }
    method_flip = {
        "nyquistguard": 0.04,
        "fixed_rate_tcn": 0.12,
        "multirate_tcn": 0.10,
        "minirocket": 0.01,
        "no_nyquist_gate": 0.08,
        "no_cbe": 0.07,
        "no_selective_head": 0.06,
    }
    for dataset in PILOT_DATASETS:
        for method in PILOT_METHODS:
            for seed in PILOT_SEEDS:
                per_rate = {}
                for index, rate in enumerate(RATE_IDS):
                    per_rate[rate] = {
                        "macro_f1": method_f1[method] - 0.02 * index,
                        "disagreement_vs_original": 0.0 if index == 0 else method_flip[method],
                    }
                unseen = [per_rate[rate] for rate in RATE_IDS[1:]]
                runs[(dataset, method, seed)] = {
                    "evaluation": {
                        "mean_unseen_macro_f1": float(np.mean([item["macro_f1"] for item in unseen])),
                        "mean_unseen_aurc": 0.2,
                        "per_rate": per_rate,
                    }
                }
    return runs


def test_diagnosis_stage_is_available_in_dashboard() -> None:
    assert len(STAGE_TASKS["diagnosis"]) == 9
    assert "full" in STAGE_TASKS


def test_aurc_rewards_putting_errors_last() -> None:
    errors = np.asarray([0.0, 0.0, 1.0])
    good = _aurc(errors, np.asarray([0.9, 0.8, 0.1]))
    bad = _aurc(errors, np.asarray([0.1, 0.2, 0.9]))
    assert good < bad


def test_static_go_no_go_helpers_use_worst_rate_and_flip_reduction() -> None:
    runs = _synthetic_runs()
    summaries = _run_summaries(runs)
    performance = _analyze_performance(runs, summaries)
    flip = _analyze_flip(summaries)
    assert performance["criterion_1"]["passed"] is True
    assert performance["criterion_1"]["direction_count"] == 4
    assert math.isclose(performance["criterion_1"]["average_delta"], 0.10)
    assert flip["criterion_3"]["passed"] is True
    assert math.isclose(flip["criterion_3"]["relative_reduction"], 0.60)
