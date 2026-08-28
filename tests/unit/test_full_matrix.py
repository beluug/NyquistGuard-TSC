from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from nyquistguard.data.full_datasets import _fixed_segment, _group_partition, _normalize_eegmmi_trial
from nyquistguard.experiments.full import FULL_METHODS, _generate_figures, build_full_matrix, paired_statistics
from nyquistguard.experiments.metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return yaml.safe_load((ROOT / "configs/experiments/full.yaml").read_text(encoding="utf-8"))


def test_full_matrix_is_frozen_and_unique() -> None:
    matrix = build_full_matrix(_config())
    assert len(matrix) == 210
    assert len({spec.run_key for spec in matrix}) == 210
    assert {spec.method for spec in matrix} == set(FULL_METHODS)


def test_full_matrix_rejects_seed_changes() -> None:
    config = _config()
    config["seeds"] = [17, 42, 99]
    with pytest.raises(ValueError, match="seeds"):
        build_full_matrix(config)


def test_group_partition_is_deterministic_and_disjoint() -> None:
    first = _group_partition([f"s{index:02d}" for index in range(20)])
    second = _group_partition([f"s{index:02d}" for index in reversed(range(20))])
    assert first == second
    assert not (first["train"] & first["validation"])
    assert not (first["train"] & first["test"])
    assert not (first["validation"] & first["test"])
    assert set.union(*first.values()) == {f"s{index:02d}" for index in range(20)}


def test_short_segment_is_interpolated_to_fixed_length() -> None:
    values = np.asarray([[0.0, 10.0], [1.0, 20.0]], dtype=np.float32)
    result = _fixed_segment(values, 5)
    assert result.shape == (2, 5)
    assert np.allclose(result[:, 0], [0.0, 10.0])
    assert np.allclose(result[:, -1], [1.0, 20.0])


def test_polyphase_ratio_used_by_eegmmi_produces_640_samples() -> None:
    source = np.zeros((64, 512), dtype=np.float32)
    normalized = _normalize_eegmmi_trial(source, 128.0)
    assert normalized.shape == (64, 640)


def test_eegmmi_rejects_an_unfrozen_native_rate() -> None:
    with pytest.raises(ValueError, match="unsupported EEGMMI source rate"):
        _normalize_eegmmi_trial(np.zeros((64, 400), dtype=np.float32), 100.0)


def test_balanced_accuracy_is_exposed() -> None:
    logits = np.asarray([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    targets = np.asarray([0, 0, 0, 1])
    metrics = classification_metrics(targets, logits, np.ones(4))
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.5


def test_paired_statistics_uses_dataset_level_units_and_holm() -> None:
    config = deepcopy(_config())
    config["statistics"]["bootstrap_resamples"] = 100
    rows = []
    for dataset_index, dataset_id in enumerate(config["datasets"]):
        for method in config["methods"]:
            for seed in config["seeds"]:
                score = 0.70 + dataset_index * 0.001
                if method != "v3_10":
                    score -= 0.05
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "seed": seed,
                        "mean_unseen_macro_f1": score,
                    }
                )
    result = paired_statistics(rows, config)
    assert set(result) == set(FULL_METHODS) - {"v3_10"}
    assert all(row["dataset_count"] == 10 for row in result.values())
    assert all(row["mean_delta"] == pytest.approx(0.05) for row in result.values())
    assert all(0.0 <= row["holm_adjusted_p"] <= 1.0 for row in result.values())


def test_full_figures_are_generated_without_display(tmp_path: Path) -> None:
    report = {
        "aggregate": {
            "method_summary": {
                method: {"mean_unseen_macro_f1": 0.6 + index * 0.01}
                for index, method in enumerate(FULL_METHODS)
            },
            "rate_summary": {
                method: {
                    rate: 0.7 - rate_index * 0.03
                    for rate_index, rate in enumerate(("r1000", "r0900", "r0600", "r0400", "r0300"))
                }
                for method in FULL_METHODS
            },
        }
    }
    paths = _generate_figures(report, [tmp_path])
    assert len(paths) == 2
    assert (tmp_path / "full_method_comparison.png").stat().st_size > 0
    assert (tmp_path / "full_rate_curves.png").stat().st_size > 0
