from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiment_dashboard import STAGE_TASKS
from nyquistguard.data import SplitData
from nyquistguard.research.v4_observe_only_micro import DevelopmentDataset
from nyquistguard.research.v5_dual_path_micro import (
    _assert_no_active_other_stage,
    _paired_initial_states,
    run_v5_dual_path_micro,
    v5_micro_decision,
    validate_v5_micro_matrix,
)
from nyquistguard.research.v5_four_dataset_benchmark import (
    benchmark_decision,
    benchmark_tasks,
    run_v5_four_dataset_benchmark,
)


ROOT = Path(__file__).resolve().parents[2]


def _split(count: int, channels: int = 3, length: int = 32) -> SplitData:
    generator = np.random.default_rng(51 + count)
    return SplitData(
        generator.normal(size=(count, channels, length)).astype(np.float32),
        np.arange(count, dtype=np.int64) % 2,
        np.asarray([f"case{index}" for index in range(count)]),
    )


def test_v5_micro_protocol_and_dashboard_matrix_are_frozen() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v5_dual_path_micro.yaml").read_text(
            encoding="utf-8"
        )
    )
    matrix = validate_v5_micro_matrix(config)
    assert len(matrix) == 4
    assert len(STAGE_TASKS["v5_dual_path_micro"]) == 6
    assert len(STAGE_TASKS["v5_four_dataset_benchmark"]) == 18
    assert len(benchmark_tasks()) == 18
    assert len(set(benchmark_tasks())) == 18


def test_v5_pair_preserves_every_shared_initial_parameter() -> None:
    dataset = DevelopmentDataset(
        "synthetic", 64.0, ("a", "b"), _split(8), _split(4)
    )
    base = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "pilot.yaml").read_text(encoding="utf-8")
    )
    control, candidate, exact = _paired_initial_states(dataset, base, 17)
    assert exact is True
    assert set(control).issubset(candidate)
    assert any(key.startswith("spatial_encoder") for key in set(candidate) - set(control))


def _validation(unseen: float, full: float, reliability_delta: float = -0.001) -> dict:
    confidence = 0.10
    return {
        "mean_unseen_macro_f1": unseen,
        "full_rate_macro_f1": full,
        "pooled_confidence_aurc": confidence,
        "pooled_observability_aurc": confidence + reliability_delta,
        "selected_pooled_aurc": confidence + min(0.0, reliability_delta),
        "learned_gate_floor": [0.45, 0.55],
        "per_rate": {
            rate: {"prediction_class_count": 2}
            for rate in ("r1000", "r0900", "r0600", "r0400", "r0300")
        },
    }


def test_v5_micro_decision_uses_two_dataset_pairs() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v5_dual_path_micro.yaml").read_text(
            encoding="utf-8"
        )
    )
    results = {
        dataset: {
            "v4_1_residual_gate": {"validation": _validation(0.60, 0.70)},
            "v5_dual_path": {"validation": _validation(0.63, 0.70)},
        }
        for dataset in ("basicmotions_uea", "pamap2_uci")
    }
    decision = v5_micro_decision(results, config["development_gates"])
    assert decision["passed"] is True
    assert decision["average_unseen_macro_f1_delta_vs_v4_1"] == pytest.approx(0.03)


def test_v5_large_benchmark_decision_keeps_dataset_as_primary_unit() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "v5_four_dataset_benchmark.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = {
        f"d{index}": {
            "mean_unseen_macro_f1_delta_vs_v4_1": value,
            "mean_full_rate_macro_f1_delta_vs_v4_1": 0.0,
            "mean_selected_aurc_delta_vs_confidence": -0.001,
            "seed_rows": [{
                "minimum_gate_floor": 0.45,
                "maximum_gate_floor": 0.55,
                "minimum_prediction_class_count": 2,
            }],
        }
        for index, value in enumerate((0.03, 0.02, 0.01, -0.01))
    }
    decision = benchmark_decision(rows, config["benchmark_gates"])
    assert decision["passed"] is True
    assert decision["positive_dataset_count"] == 3


def test_v5_runners_require_manual_start_and_refuse_resource_collision(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="manual confirmation"):
        run_v5_dual_path_micro(tmp_path)
    with pytest.raises(PermissionError, match="manual confirmation"):
        run_v5_four_dataset_benchmark(tmp_path)
    status = tmp_path / "runs" / "dashboard_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"stage": "v4_new_dataset_confirmation", "status": "running"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refusing to start V5"):
        _assert_no_active_other_stage(tmp_path)
