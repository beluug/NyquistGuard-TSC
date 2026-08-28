from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import nyquistguard.data.v5_independent_datasets as data_module
from experiment_dashboard import STAGE_TASKS
from nyquistguard.data.v5_independent_datasets import (
    V5_INDEPENDENT_DATASETS,
    prepare_v5_independent_dataset,
    prepare_v5_independent_development_dataset,
)
from nyquistguard.research.v5_independent_confirmation import (
    INDEPENDENT_ROLES,
    INDEPENDENT_SEEDS,
    independent_confirmation_tasks,
    independent_decision,
    run_v5_1_independent_confirmation,
)


ROOT = Path(__file__).resolve().parents[2]


def _synthetic_selection() -> dict:
    return {
        "datasets": {
            dataset_id: {
                "archive_name": dataset_id,
                "domain": "synthetic",
                "sampling_rate_hz": 20.0,
                "expected_channels": 2,
                "expected_length": 12,
                "expected_train_cases": 20,
                "expected_test_cases": 8,
                "expected_classes": 2,
            }
            for dataset_id in V5_INDEPENDENT_DATASETS
        }
    }


def test_development_parser_never_opens_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    selection = _synthetic_selection()
    monkeypatch.setattr(data_module, "_selection", lambda _root: selection)
    calls = []

    def fake_load(path: Path):
        calls.append(path.name)
        cases = 20 if "TRAIN" in path.name else 8
        x = np.arange(cases * 2 * 12, dtype=np.float32).reshape(cases, 2, 12)
        y = np.asarray([str(index % 2) for index in range(cases)])
        return x, y

    monkeypatch.setattr(data_module, "_load_ts", fake_load)
    dataset_id = V5_INDEPENDENT_DATASETS[0]
    for split in ("TRAIN", "TEST"):
        path = data_module.raw_split_path(tmp_path, dataset_id, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    dataset = prepare_v5_independent_development_dataset(tmp_path, dataset_id)
    assert not hasattr(dataset, "test")
    assert calls == [f"{dataset_id}_TRAIN.ts"]
    with np.load(
        data_module.independent_cache_path(tmp_path, dataset_id, development=True),
        allow_pickle=False,
    ) as payload:
        assert not any(name.startswith("test_") for name in payload.files)


def test_formal_parser_requires_manual_confirmation(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="manual confirmation"):
        prepare_v5_independent_dataset(tmp_path, V5_INDEPENDENT_DATASETS[0])
    with pytest.raises(PermissionError, match="manual confirmation"):
        run_v5_1_independent_confirmation(tmp_path)


def test_frozen_matrix_and_dashboard_have_55_unique_tasks() -> None:
    tasks = independent_confirmation_tasks()
    assert len(tasks) == 55
    assert len(set(tasks)) == 55
    assert STAGE_TASKS["v5_1_independent_confirmation"] == tasks
    assert len(INDEPENDENT_SEEDS) == 3
    assert INDEPENDENT_ROLES == ("v4_1_residual_gate", "v5_dual_path")


def test_yaml_matches_frozen_registry() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v5_1_independent_confirmation.yaml").read_text(
            encoding="utf-8"
        )
    )
    selection = yaml.safe_load(
        (ROOT / "configs/experiments/v5_1_independent_confirmation_selection.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(config["design"]["datasets"]) == V5_INDEPENDENT_DATASETS
    assert tuple(selection["scientific_boundary"]["selected_dataset_ids"]) == V5_INDEPENDENT_DATASETS
    assert config["reliability_controller"]["freeze_all_dataset_modes_before_any_test_read"] is True


def test_decision_uses_four_dataset_means() -> None:
    rows = {}
    for index, dataset_id in enumerate(V5_INDEPENDENT_DATASETS):
        delta = 0.03 if index < 3 else -0.01
        rows[dataset_id] = {
            "mean_unseen_macro_f1_delta_vs_v4_1": delta,
            "mean_full_rate_macro_f1_delta_vs_v4_1": 0.0,
            "mean_selected_aurc_delta_vs_confidence": 0.0,
            "seed_rows": [
                {
                    "minimum_gate_floor": 0.2,
                    "maximum_gate_floor": 0.8,
                    "minimum_prediction_class_count": 2,
                }
                for _ in INDEPENDENT_SEEDS
            ],
        }
    gates = {
        "minimum_average_dataset_unseen_macro_f1_delta_vs_v4_1": 0.0,
        "minimum_positive_dataset_count": 3,
        "maximum_single_dataset_unseen_macro_f1_drop": 0.02,
        "maximum_average_dataset_full_rate_macro_f1_drop": 0.01,
        "maximum_average_dataset_selected_aurc_delta_vs_confidence": 0.0,
        "require_no_constant_prediction_at_any_test_rate": True,
        "require_finite_all_metrics_and_gate_floors": True,
    }
    decision = independent_decision(rows, gates)
    assert decision["passed"] is True
    assert decision["positive_dataset_count"] == 3
    assert decision["average_dataset_unseen_macro_f1_delta_vs_v4_1"] == pytest.approx(0.02)

