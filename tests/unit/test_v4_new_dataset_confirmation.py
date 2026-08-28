from pathlib import Path

import numpy as np
import pytest
import yaml

from nyquistguard.data.new_confirmation_datasets import (
    ConfirmationDevelopmentDataset,
    _encode_from_training,
    _ragged_stats,
    _ragged_to_fixed,
    load_confirmation_development_cache,
    prepare_confirmation_dataset,
    _save_development,
)
from nyquistguard.data.pilot_datasets import SplitData
from nyquistguard.research.v4_new_dataset_confirmation import (
    _validate_frozen_protocol,
    confirmation_decision,
    confirmation_tasks,
)


def _split(count: int, offset: int = 0) -> SplitData:
    return SplitData(
        np.arange(count * 2 * 5, dtype=np.float32).reshape(count, 2, 5),
        np.asarray([(index + offset) % 2 for index in range(count)], dtype=np.int64),
        np.asarray([f"id{offset + index}" for index in range(count)]),
    )


def test_formal_test_access_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="manual confirmation"):
        prepare_confirmation_dataset(tmp_path, "character_trajectories_uea")


def test_development_cache_has_no_test_arrays_or_attribute(tmp_path: Path) -> None:
    dataset = ConfirmationDevelopmentDataset(
        "synthetic", 100.0, ("a", "b"), _split(4), _split(2, 10),
        {"test_accessed": False},
    )
    path = tmp_path / "development.npz"
    _save_development(path, dataset)
    with np.load(path, allow_pickle=False) as payload:
        assert not any(name.startswith("test_") for name in payload.files)
    loaded = load_confirmation_development_cache(path)
    assert not hasattr(loaded, "test")
    assert loaded.metadata["test_accessed"] is False


def test_training_labels_alone_define_class_mapping() -> None:
    train, others, names = _encode_from_training(["b", "a", "b"], [["a", "b"]])
    assert names == ("a", "b")
    assert train.tolist() == [1, 0, 1]
    assert others[0].tolist() == [0, 1]
    with pytest.raises(ValueError, match="absent"):
        _encode_from_training(["a", "b"], [["c"]])


def test_ragged_padding_uses_training_statistics_and_zero_mean_padding() -> None:
    cases = [
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0], [6.0]], dtype=np.float32),
    ]
    mean, scale = _ragged_stats(cases)
    fixed = _ragged_to_fixed(cases, 3, mean, scale)
    assert fixed.shape == (2, 2, 3)
    assert np.all(fixed[:, :, 2] == 0.0)
    assert np.isfinite(fixed).all()


def test_confirmation_matrix_has_29_dashboard_tasks() -> None:
    tasks = confirmation_tasks()
    assert len(tasks) == 29
    assert len(set(tasks)) == 29
    assert tasks[-1].startswith("Aggregate")


def test_confirmation_yaml_and_registry_parse() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "configs/experiments/v4_new_dataset_confirmation.yaml",
        "configs/experiments/v4_new_dataset_confirmation_selection.yaml",
        "data/registry.yaml",
    ):
        assert isinstance(yaml.safe_load((root / relative).read_text(encoding="utf-8")), dict)


def test_current_confirmation_protocol_matches_passed_v4_source() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs/experiments/v4_new_dataset_confirmation.yaml").read_text(encoding="utf-8")
    )
    _validate_frozen_protocol(root, config)


def test_confirmation_decision_uses_four_dataset_means() -> None:
    rows = {
        f"d{index}": {
            "mean_unseen_macro_f1_delta_vs_hard_gate": value,
            "mean_full_rate_macro_f1_delta_vs_hard_gate": 0.0,
            "mean_selected_aurc_delta_vs_confidence": -0.001,
            "seed_rows": [{"minimum_gate_floor": 0.5, "maximum_gate_floor": 0.8}],
        }
        for index, value in enumerate((0.03, 0.02, 0.01, -0.01))
    }
    gates = {
        "minimum_average_dataset_unseen_macro_f1_delta_vs_hard_gate": 0.0,
        "minimum_positive_dataset_count": 3,
        "maximum_single_dataset_unseen_macro_f1_drop": 0.02,
        "maximum_average_dataset_full_rate_macro_f1_drop": 0.01,
        "maximum_average_dataset_selected_aurc_delta_vs_confidence": 0.0,
        "require_finite_all_metrics_and_gate_floors": True,
    }
    decision = confirmation_decision(rows, gates)
    assert decision["passed"] is True
    assert decision["positive_dataset_count"] == 3
