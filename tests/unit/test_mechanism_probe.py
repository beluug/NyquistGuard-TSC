from __future__ import annotations

import numpy as np
import torch
from torch import nn

from experiment_dashboard import STAGE_TASKS
from nyquistguard.data import PreparedDataset, SplitData
from nyquistguard.experiments.mechanism_probe import (
    _balanced_indices,
    _fingerprint_probe,
    _gradient_norm,
    _render_markdown,
)


def test_mechanism_probe_stage_has_bounded_visible_tasks() -> None:
    assert len(STAGE_TASKS["mechanism_probe"]) == 7
    assert "checkpoint" in STAGE_TASKS["mechanism_probe"][0]


def test_balanced_indices_are_deterministic_and_cover_classes() -> None:
    labels = np.repeat(np.arange(4), 20)
    first = _balanced_indices(labels, 12, seed=17)
    second = _balanced_indices(labels, 12, seed=17)
    assert np.array_equal(first, second)
    assert len(first) == 12
    assert set(labels[first].tolist()) == {0, 1, 2, 3}


def test_autograd_probe_does_not_populate_or_update_parameters() -> None:
    model = nn.Linear(3, 2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = model(torch.ones(4, 3)).square().mean()
    norm = _gradient_norm(loss, list(model.parameters()))
    assert norm > 0
    for parameter, original in zip(model.parameters(), before):
        assert parameter.grad is None
        assert torch.equal(parameter.detach(), original)


def test_fingerprint_probe_detects_exact_cross_split_window() -> None:
    shared = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    other = np.ones((1, 3, 4), dtype=np.float32)
    train = SplitData(np.concatenate([shared, other]), np.asarray([0, 1]), np.asarray(["tr0", "tr1"]))
    validation = SplitData(other.copy(), np.asarray([1]), np.asarray(["v0"]))
    test = SplitData(shared.copy(), np.asarray([0]), np.asarray(["te0"]))
    dataset = PreparedDataset("synthetic", 10.0, ("a", "b"), train, validation, test, {})
    result = _fingerprint_probe(dataset)
    assert result["exact_train_test_duplicate_count"] == 1
    assert result["leakage_proven"] is True


def test_mechanism_report_table_renders_complete_rows() -> None:
    dataset_payload = {
        "filter_and_gate": {
            "center_frequencies_hz": [1.0, 2.0],
            "median_absolute_center_shift_hz": 0.1,
            "rates": {
                "r0300": {
                    "effective_band_sum": 1.5,
                    "near_zero_fraction_le_0_05": 0.25,
                }
            },
        },
        "loss_and_gradient": {
            "r0500": {
                "weighted_cbe_to_classification_gradient_ratio": None,
                "weighted_cbe_to_classification_filterbank_gradient_ratio": None,
                "contributions": {"cbe_unweighted": 0.2, "cbe_weighted": 0.02},
            }
        },
        "selectivity": {
            "mean_unseen_learned_aurc": 0.3,
            "mean_unseen_confidence_aurc": 0.2,
            "full_to_low_acceptance_drop": 0.1,
        },
        "fingerprint": {
            "exact_train_test_duplicate_count": 0,
            "nearest_signature_label_accuracy": 0.5,
        },
    }
    report = {
        "status": "completed",
        "pilot_root": "pilot",
        "device": {"name": "CPU"},
        "datasets": {
            dataset: dataset_payload
            for dataset in ("basicmotions_uea", "epilepsy_uea", "pamap2_uci", "mhealth_uci")
        },
        "aggregate": {"findings": []},
    }
    markdown = _render_markdown(report)
    rows = [line for line in markdown.splitlines() if line.startswith("| basicmotions_uea")]
    assert len(rows) == 1
    assert rows[0].count("|") == 8
    assert "| — | 0.3000/0.2000 |" in rows[0]
