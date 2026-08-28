import json
from pathlib import Path

import pytest

from nyquistguard.analysis.v4_confirmation_artifacts import (
    ConfirmationArtifactError,
    audit_confirmation,
    build_confirmation_artifacts,
    exact_sign_flip_test,
    summarize_training_health,
)
from nyquistguard.data.new_confirmation_datasets import CONFIRMATION_DATASETS
from nyquistguard.research.v4_new_dataset_confirmation import (
    CONFIRMATION_ROLES,
    CONFIRMATION_SEEDS,
)


RATE_IDS = ("r1000", "r0900", "r0600", "r0400", "r0300")


def _evaluation(value: float, *, candidate: bool) -> dict:
    result = {
        "full_rate_macro_f1": value,
        "mean_unseen_macro_f1": value,
        "worst_unseen_macro_f1": value,
        "pooled_confidence_aurc": 0.20,
        "pooled_observability_aurc": 0.19,
        "per_rate": {
            rate_id: {
                "ratio": ratio,
                "macro_f1": value,
                "confidence_aurc": 0.20,
                "observability_aurc": 0.19,
                "relative_gate_mass": ratio,
            }
            for rate_id, ratio in zip(RATE_IDS, (1.0, 0.9, 0.6, 0.4, 0.3))
        },
    }
    if candidate:
        result.update(
            reliability_mode="observability",
            reliability_mode_selected_on_validation="observability",
            selected_pooled_aurc=0.19,
            learned_gate_floor=[0.5, 0.6],
        )
    return result


def _synthetic_completed_root(tmp_path: Path) -> Path:
    run_root = (
        tmp_path / "runs" / "v4_new_dataset_confirmation" /
        "v4_1_confirmation__4datasets__3seeds__20990101T000000Z"
    )
    run_root.mkdir(parents=True)
    protocol_hash = "synthetic-protocol"
    (run_root / "manifest.json").write_text(
        json.dumps({"status": "completed", "protocol_hash": protocol_hash}), encoding="utf-8"
    )
    role_results = {}
    dataset_results = {}
    for dataset_index, dataset_id in enumerate(CONFIRMATION_DATASETS):
        seed_rows = []
        for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
            hard_value = 0.50 + 0.01 * seed_index
            candidate_value = hard_value + 0.01 * (dataset_index + 1)
            for role, value in zip(CONFIRMATION_ROLES, (hard_value, candidate_value)):
                candidate = role == "v4_1_residual_gate"
                validation = _evaluation(value, candidate=candidate)
                test = _evaluation(value, candidate=candidate)
                if candidate:
                    test["reliability_mode_selected_on_validation"] = "observability"
                metric = {
                    "status": "completed", "protocol_hash": protocol_hash,
                    "dataset_id": dataset_id, "seed": seed, "role": role,
                    "validation": validation, "test": test, "test_accessed": True,
                    "test_used_for_checkpoint_or_threshold_selection": False,
                }
                key = f"{dataset_id}__seed{seed}__{role}"
                role_results[key] = metric
                role_dir = run_root / key
                role_dir.mkdir()
                (role_dir / "metrics.json").write_text(json.dumps(metric), encoding="utf-8")
                history = {"history": [{
                    "epoch": 1, "train_loss": 0.7,
                    "validation_selection_score": value,
                    "validation_full_rate_macro_f1": value,
                    "validation_mean_unseen_macro_f1": value,
                }]}
                (role_dir / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
            seed_rows.append({"seed": seed, "unseen_macro_f1_delta_vs_hard_gate": candidate_value - hard_value})
        dataset_results[dataset_id] = {
            "dataset_id": dataset_id, "seed_rows": seed_rows,
            "mean_unseen_macro_f1_delta_vs_hard_gate": 0.01 * (dataset_index + 1),
        }
        manifest_dir = tmp_path / "data" / "processed" / "v4_confirmation_v1"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / f"{dataset_id}__development.manifest.json").write_text(
            json.dumps({"class_counts": {"validation": [10, 10]}}), encoding="utf-8"
        )
    report = {
        "status": "completed", "protocol_hash": protocol_hash,
        "role_results": role_results, "dataset_results": dataset_results,
        "decision": {"passed": True, "checks": {"synthetic": True}},
    }
    (run_root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_root


def test_exact_sign_flip_respects_four_dataset_resolution() -> None:
    result = exact_sign_flip_test([0.01, 0.02, 0.03, 0.04])
    assert result["permutation_count"] == 16
    assert result["minimum_attainable_one_sided_p"] == pytest.approx(0.0625)
    assert result["one_sided_p_greater"] == pytest.approx(0.0625)


def test_partial_audit_reports_missing_without_failing(tmp_path: Path) -> None:
    run_root = (
        tmp_path / "runs" / "v4_new_dataset_confirmation" /
        "v4_1_confirmation__4datasets__3seeds__20990101T000000Z"
    )
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps({"status": "running", "protocol_hash": "x"}), encoding="utf-8"
    )
    result = audit_confirmation(tmp_path, require_complete=False)
    assert result["status"] == "pass"
    assert result["completed_metrics"] == 0
    assert len(result["missing_keys"]) == 24


def test_complete_audit_rejects_incomplete_matrix(tmp_path: Path) -> None:
    run_root = (
        tmp_path / "runs" / "v4_new_dataset_confirmation" /
        "v4_1_confirmation__4datasets__3seeds__20990101T000000Z"
    )
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps({"status": "running", "protocol_hash": "x"}), encoding="utf-8"
    )
    with pytest.raises(ConfirmationArtifactError, match="missing 24"):
        audit_confirmation(tmp_path, require_complete=True)


def test_synthetic_complete_build_writes_tables_figures_and_provenance(tmp_path: Path) -> None:
    _synthetic_completed_root(tmp_path)
    result = build_confirmation_artifacts(tmp_path)
    assert result["status"] == "completed"
    assert result["raw_or_test_arrays_loaded"] is False
    for output in result["outputs"]:
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0
    statistics = json.loads(
        (tmp_path / "reports" / "v4_confirmation_statistical_summary.json").read_text(encoding="utf-8")
    )
    assert statistics["n_primary_units"] == 4
    assert statistics["exact_sign_flip_sensitivity"]["minimum_attainable_one_sided_p"] == 0.0625
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_confirmation_artifacts(tmp_path)


def test_training_health_uses_only_histories_and_validation_manifests(tmp_path: Path) -> None:
    _synthetic_completed_root(tmp_path)
    rows = summarize_training_health(tmp_path)
    assert len(rows) == 24
    assert all(not any(key.startswith("test_") for key in row) for row in rows)
    assert all(row["epochs_completed"] == 1 for row in rows)
