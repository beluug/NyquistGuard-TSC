from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from nyquistguard.data.pilot_datasets import (
    PreparedDataset,
    SplitData,
    _require_closed_set,
    load_prepared_dataset,
    save_prepared_dataset,
)
from nyquistguard.experiments.metrics import align_probability_columns, classification_metrics
from nyquistguard.experiments.pilot import PILOT_METHODS, build_pilot_matrix
from nyquistguard.models import TCNClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_pilot_matrix_has_84_unique_runs() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "pilot.yaml").read_text(encoding="utf-8"))
    matrix = build_pilot_matrix(config)
    assert len(matrix) == 84
    assert len({spec.run_key for spec in matrix}) == 84
    assert tuple(config["methods"]) == PILOT_METHODS


def test_prepared_dataset_cache_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)

    def split(prefix: str, count: int) -> SplitData:
        return SplitData(
            rng.normal(size=(count, 3, 24)).astype(np.float32),
            np.arange(count, dtype=np.int64) % 2,
            np.asarray([f"{prefix}_{index}" for index in range(count)]),
        )

    dataset = PreparedDataset("synthetic", 20.0, ("a", "b"), split("train", 8), split("val", 4), split("test", 4), {"kind": "test"})
    path = tmp_path / "prepared.npz"
    save_prepared_dataset(path, dataset)
    restored = load_prepared_dataset(path)
    assert restored.dataset_id == "synthetic"
    assert restored.train.x.shape == (8, 3, 24)
    assert np.array_equal(restored.test.ids, dataset.test.ids)
    assert not (set(restored.train.ids) & set(restored.test.ids))


def test_metrics_and_tcn_interface() -> None:
    model = TCNClassifier(3, 2, hidden_dim=8, depth=2, dropout=0.0)
    output = model(torch.randn(5, 3, 32), 20.0)
    assert output["logits"].shape == (5, 2)
    targets = np.asarray([0, 1, 0, 1])
    logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]])
    metrics = classification_metrics(targets, logits, np.asarray([0.9, 0.8, 0.7, 0.6]))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["aurc"] == 0.0


def test_probability_columns_follow_estimator_class_labels() -> None:
    local = np.asarray([[0.25, 0.75], [0.60, 0.40]])
    aligned = align_probability_columns(local, np.asarray([0, 2]), num_classes=3)
    assert np.allclose(aligned[:, 0], local[:, 0])
    assert np.allclose(aligned[:, 1], 0.0)
    assert np.allclose(aligned[:, 2], local[:, 1])
    assert np.allclose(aligned.sum(axis=1), 1.0)


def test_closed_set_guard_rejects_test_only_class() -> None:
    x = np.zeros((2, 1, 8), dtype=np.float32)
    train = SplitData(x, np.asarray([0, 0]), np.asarray(["a", "b"]))
    validation = SplitData(x, np.asarray([0, 0]), np.asarray(["c", "d"]))
    test = SplitData(x, np.asarray([0, 1]), np.asarray(["e", "f"]))
    with pytest.raises(ValueError, match="training split does not cover"):
        _require_closed_set(train, validation, test, ("zero", "one"))


def test_pilot_cli_refuses_to_start_without_manual_confirmation() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_experiments.py"), "--stage", "pilot"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 3
    assert "manual confirmation" in result.stderr
