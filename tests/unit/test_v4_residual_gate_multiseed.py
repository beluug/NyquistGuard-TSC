from __future__ import annotations

import copy
from pathlib import Path

import yaml

from experiment_dashboard import STAGE_TASKS
from nyquistguard.research.v4_residual_gate_multiseed import (
    V4_STABILITY_TASKS,
    stability_decision,
    validate_stability_matrix,
)


def _config() -> dict:
    path = Path(__file__).parents[2] / "configs" / "experiments" / "v4_residual_gate_multiseed_stability.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_frozen_multiseed_matrix_is_complete_paired_and_excludes_seed17() -> None:
    matrix = validate_stability_matrix(_config())
    assert len(matrix) == 8
    assert {seed for _, seed, _ in matrix} == {42, 2026}
    assert all(seed != 17 for _, seed, _ in matrix)


def test_duplicate_or_reordered_pair_is_rejected() -> None:
    config = copy.deepcopy(_config())
    config["design"]["run_order"][1] = config["design"]["run_order"][0]
    try:
        validate_stability_matrix(config)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid matrix was accepted")


def test_stability_decision_uses_dataset_seed_pairs_not_rates() -> None:
    rows = {}
    for dataset in ("basicmotions_uea", "pamap2_uci"):
        for seed in (42, 2026):
            rows[f"{dataset}__seed{seed}"] = {
                "dataset_id": dataset, "seed": seed,
                "unseen_macro_f1_delta_vs_hard_gate": 0.03,
                "full_rate_macro_f1_delta_vs_hard_gate": 0.0,
                "selected_aurc_delta_vs_confidence": 0.0,
                "minimum_gate_floor": 0.4, "maximum_gate_floor": 0.6,
            }
    assert stability_decision(rows, _config()["stability_gates"])["passed"] is True


def test_dashboard_exposes_all_multiseed_tasks() -> None:
    assert STAGE_TASKS["v4_residual_gate_multiseed_stability"] == V4_STABILITY_TASKS
