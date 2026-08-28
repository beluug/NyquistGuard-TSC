from __future__ import annotations

import numpy as np

from nyquistguard.research.v4_observe_only_micro import (
    _decision,
    load_development_dataset,
)


def test_development_loader_does_not_require_or_expose_test_arrays(tmp_path) -> None:
    path = tmp_path / "development_only.npz"
    np.savez(
        path,
        dataset_id=np.asarray("tiny"),
        sampling_rate_hz=np.asarray(20.0),
        class_names=np.asarray(["a", "b"]),
        train_x=np.zeros((4, 2, 8), dtype=np.float32),
        train_y=np.asarray([0, 1, 0, 1]),
        train_ids=np.asarray(["t0", "t1", "t2", "t3"]),
        validation_x=np.zeros((2, 2, 8), dtype=np.float32),
        validation_y=np.asarray([0, 1]),
        validation_ids=np.asarray(["v0", "v1"]),
    )
    dataset = load_development_dataset(path)
    assert dataset.train.x.shape == (4, 2, 8)
    assert dataset.validation.x.shape == (2, 2, 8)
    assert not hasattr(dataset, "test")


def test_v4_micro_decision_uses_only_predeclared_validation_fields() -> None:
    def row(unseen: float, full: float, conf: float, obs: float, drop: float) -> dict:
        return {"validation": {
            "mean_unseen_macro_f1": unseen,
            "full_rate_macro_f1": full,
            "pooled_confidence_aurc": conf,
            "pooled_observability_aurc": obs,
            "full_to_low_observability_score_drop": drop,
        }}
    results = {
        "a": {"v3_10_hard_gate": row(0.50, 0.70, 0.2, 0.2, 0.1), "v4_observe_only": row(0.54, 0.70, 0.2, 0.19, 0.1)},
        "b": {"v3_10_hard_gate": row(0.60, 0.72, 0.2, 0.2, 0.1), "v4_observe_only": row(0.63, 0.71, 0.2, 0.20, 0.1)},
    }
    gates = {
        "minimum_average_unseen_macro_f1_delta_vs_hard_gate": 0.02,
        "maximum_single_dataset_unseen_macro_f1_drop": 0.01,
        "maximum_average_full_rate_macro_f1_drop": 0.01,
        "maximum_average_pooled_observability_aurc_delta_vs_confidence": 0.02,
        "require_rate_sufficiency_drop_both_datasets": True,
    }
    assert _decision(results, gates)["passed"] is True
