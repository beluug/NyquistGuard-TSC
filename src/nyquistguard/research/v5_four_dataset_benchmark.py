"""Manual V5 benchmark reusing the completed V4.1 four-dataset controls.

This is intentionally labelled retrospective: the datasets' tests were already
accessed by V4 and therefore cannot become independent V5 confirmation data.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.data.new_confirmation_datasets import (
    CONFIRMATION_DATASETS,
    prepare_confirmation_dataset,
)
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.research.v4_new_dataset_confirmation import (
    CONFIRMATION_SEEDS,
    _evaluate_selected_checkpoint,
    _split_view,
)
from nyquistguard.research.v4_observe_only_micro import _train_variant
from nyquistguard.research.v5_dual_path_micro import (
    _assert_no_active_other_stage,
    _paired_initial_states,
)


def benchmark_tasks() -> list[str]:
    tasks = ["验证V4.1来源、互斥锁与retrospective边界"]
    for dataset_id in CONFIRMATION_DATASETS:
        tasks.append(f"Load frozen {dataset_id}")
        tasks.extend(
            f"{dataset_id} seed{seed} v5_dual_path" for seed in CONFIRMATION_SEEDS
        )
    tasks.append("Aggregate V5 vs V4.1 four-dataset benchmark")
    return tasks


def _protocol_hash(root: Path, config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        config_path,
        base_path,
        root / "src" / "nyquistguard" / "data" / "new_confirmation_datasets.py",
        root / "src" / "nyquistguard" / "research" / "v5_dual_path.py",
        root / "src" / "nyquistguard" / "research" / "v5_dual_path_micro.py",
        Path(__file__),
    ):
        digest.update(path.read_bytes())
    digest.update(b"v5-four-dataset-retrospective-benchmark-v1")
    return digest.hexdigest()


def _validate_protocol(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    design = config["design"]
    if tuple(design["datasets"]) != CONFIRMATION_DATASETS:
        raise ValueError("V5 benchmark dataset panel changed")
    if tuple(int(value) for value in design["seeds"]) != CONFIRMATION_SEEDS:
        raise ValueError("V5 benchmark seeds changed")
    boundary = config["scientific_boundary"]
    if boundary["may_be_called_new_or_independent_v5_confirmation"] is not False:
        raise ValueError("retrospective V5 benchmark cannot be independent confirmation")
    source_path = root / config["source"]["v4_confirmation_report"]
    if not source_path.exists():
        raise RuntimeError("completed V4 confirmation report is required before V5 benchmark")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != config["source"]["required_status"]:
        raise RuntimeError("V4 confirmation source is not completed")
    if source.get("protocol_hash") != config["source"]["required_protocol_hash"]:
        raise RuntimeError("V4 confirmation protocol hash changed")
    if source.get("test_accessed") is not True:
        raise RuntimeError("V4 source does not document formal test access")
    expected = {
        f"{dataset_id}__seed{seed}__v4_1_residual_gate"
        for dataset_id in CONFIRMATION_DATASETS for seed in CONFIRMATION_SEEDS
    }
    if not expected.issubset(source.get("role_results", {})):
        raise RuntimeError("V4 source lacks the complete 12-control matrix")
    return source


def _find_compatible_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for path in sorted(parent.glob("v5_benchmark__4datasets__3seeds__*"), reverse=True):
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("protocol_hash") == protocol_hash:
            return path
    return None


def benchmark_decision(
    dataset_rows: dict[str, dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    rows = list(dataset_rows.values())
    unseen = [float(row["mean_unseen_macro_f1_delta_vs_v4_1"]) for row in rows]
    full = [float(row["mean_full_rate_macro_f1_delta_vs_v4_1"]) for row in rows]
    reliability = [float(row["mean_selected_aurc_delta_vs_confidence"]) for row in rows]
    floors = [
        float(value) for row in rows for seed_row in row["seed_rows"]
        for value in (seed_row["minimum_gate_floor"], seed_row["maximum_gate_floor"])
    ]
    predicted = [
        int(seed_row["minimum_prediction_class_count"])
        for row in rows for seed_row in row["seed_rows"]
    ]
    finite = all(math.isfinite(value) for value in unseen + full + reliability + floors)
    finite = finite and all(0.0 <= value <= 1.0 for value in floors)
    checks = {
        "average_dataset_unseen_gain": float(np.mean(unseen)) > float(
            gates["minimum_average_dataset_unseen_macro_f1_delta_vs_v4_1"]
        ),
        "positive_dataset_count": sum(value > 0.0 for value in unseen) >= int(
            gates["minimum_positive_dataset_count"]
        ),
        "single_dataset_unseen_floor": float(np.min(unseen)) >= -float(
            gates["maximum_single_dataset_unseen_macro_f1_drop"]
        ),
        "average_dataset_full_rate_floor": float(np.mean(full)) >= -float(
            gates["maximum_average_dataset_full_rate_macro_f1_drop"]
        ),
        "average_dataset_reliability_safety": float(np.mean(reliability)) <= float(
            gates["maximum_average_dataset_selected_aurc_delta_vs_confidence"]
        ),
        "no_constant_prediction": all(value > 1 for value in predicted)
        if gates["require_no_constant_prediction_at_any_test_rate"] else True,
        "finite_metrics": finite if gates["require_finite_all_metrics_and_gate_floors"] else True,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "average_dataset_unseen_macro_f1_delta_vs_v4_1": float(np.mean(unseen)),
        "positive_dataset_count": int(sum(value > 0.0 for value in unseen)),
        "minimum_dataset_unseen_macro_f1_delta_vs_v4_1": float(np.min(unseen)),
        "average_dataset_full_rate_macro_f1_delta_vs_v4_1": float(np.mean(full)),
        "average_dataset_selected_aurc_delta_vs_confidence": float(np.mean(reliability)),
    }


def _dataset_rows(
    source_controls: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output = {}
    for dataset_id in CONFIRMATION_DATASETS:
        seed_rows = []
        for seed in CONFIRMATION_SEEDS:
            control = source_controls[f"{dataset_id}__seed{seed}__v4_1_residual_gate"]["test"]
            candidate_result = candidates[f"{dataset_id}__seed{seed}__v5_dual_path"]
            candidate = candidate_result["test"]
            counts = [int(row["prediction_class_count"]) for row in candidate["per_rate"].values()]
            floors = candidate_result["validation"]["learned_gate_floor"]
            seed_rows.append({
                "seed": seed,
                "unseen_macro_f1_delta_vs_v4_1": float(
                    candidate["mean_unseen_macro_f1"] - control["mean_unseen_macro_f1"]
                ),
                "full_rate_macro_f1_delta_vs_v4_1": float(
                    candidate["full_rate_macro_f1"] - control["full_rate_macro_f1"]
                ),
                "selected_aurc_delta_vs_confidence": float(
                    candidate["selected_pooled_aurc"] - candidate["pooled_confidence_aurc"]
                ),
                "minimum_prediction_class_count": min(counts),
                "minimum_gate_floor": float(min(floors)),
                "maximum_gate_floor": float(max(floors)),
            })
        output[dataset_id] = {
            "dataset_id": dataset_id, "seed_rows": seed_rows,
            "mean_unseen_macro_f1_delta_vs_v4_1": float(np.mean([
                row["unseen_macro_f1_delta_vs_v4_1"] for row in seed_rows
            ])),
            "mean_full_rate_macro_f1_delta_vs_v4_1": float(np.mean([
                row["full_rate_macro_f1_delta_vs_v4_1"] for row in seed_rows
            ])),
            "mean_selected_aurc_delta_vs_confidence": float(np.mean([
                row["selected_aurc_delta_vs_confidence"] for row in seed_rows
            ])),
        }
    return output


def run_v5_four_dataset_benchmark(
    project_root: str | Path, *, resume: bool = True, confirmed: bool = False
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("V5 four-dataset benchmark requires manual confirmation")
    root = Path(project_root).resolve()
    _assert_no_active_other_stage(root)
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v5_four_dataset_benchmark.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = _validate_protocol(root, config)
    design = config["design"]
    base_path = root / design["base_config"]
    base_template = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(root, config_path, base_path)
    parent = root / "runs" / "v5_four_dataset_benchmark"
    run_root = _find_compatible_root(parent, protocol_hash) if resume else None
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v5_benchmark__4datasets__3seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
    if resume and (run_root / "report.json").exists():
        cached = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return cached
    previous_elapsed = 0.0
    previous_manifest_path = run_root / "manifest.json"
    if resume and previous_manifest_path.exists():
        try:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
            previous_elapsed = float(previous_manifest.get("cumulative_elapsed_seconds", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            previous_elapsed = 0.0
    manifest = {
        "stage": config["stage"], "status": "running", "manual_confirmation": True,
        "protocol_hash": protocol_hash, "source_v4_protocol_hash": source["protocol_hash"],
        "retrospective_benchmark_only": True, "independent_confirmation": False,
        "test_accessed": True, "cumulative_elapsed_seconds": previous_elapsed,
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "manifest.json", manifest)
    tasks = benchmark_tasks()
    progress = DashboardProgress(
        root / "runs" / "dashboard_status.json", config["stage"], tasks, run_root.name
    )
    deadline = started + float(design["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates: dict[str, dict[str, Any]] = {}
    current_task = 0
    try:
        progress.start_task(current_task)
        progress.complete_task(current_task)
        current_task += 1
        for dataset_id in CONFIRMATION_DATASETS:
            progress.start_task(current_task)
            dataset = prepare_confirmation_dataset(root, dataset_id, confirmed_test_access=True)
            progress.complete_task(current_task)
            current_task += 1
            base_config = dict(base_template)
            base_config["batch_size"] = int(design["dataset_batch_sizes"][dataset_id])
            for seed in CONFIRMATION_SEEDS:
                progress.start_task(current_task)
                key = f"{dataset_id}__seed{seed}__v5_dual_path"
                role_dir = run_root / key
                metrics_path = role_dir / "metrics.json"
                result = None
                if resume and metrics_path.exists():
                    cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if cached.get("protocol_hash") == protocol_hash and cached.get("test_accessed") is True:
                        result = cached
                if result is None:
                    _, initial_state, exact = _paired_initial_states(dataset, base_config, seed)
                    if not exact:
                        raise RuntimeError(f"shared initialization mismatch for {dataset_id} seed{seed}")

                    def update_epoch(epoch: int, total: int, row: dict[str, Any]) -> None:
                        progress.current_task = (
                            f"{dataset_id} seed{seed} V5: epoch {epoch}/{total}, "
                            f"val={row['validation_selection_score']:.4f}"
                        )
                        progress.write()

                    role_started = time.monotonic()
                    model, history = _train_variant(
                        _split_view(dataset, "validation"), base_config, design,
                        "v5_dual_path", initial_state, protocol_hash, role_dir,
                        device, deadline, resume, seed=seed, epoch_callback=update_epoch,
                    )
                    validation, test = _evaluate_selected_checkpoint(
                        model, dataset, base_config, design, "v5_dual_path", device
                    )
                    result = {
                        "status": "completed", "protocol_hash": protocol_hash,
                        "dataset_id": dataset_id, "seed": seed, "role": "v5_dual_path",
                        "epochs_completed": len(history),
                        "duration_seconds": time.monotonic() - role_started,
                        "validation": validation, "test": test, "test_accessed": True,
                        "test_used_for_checkpoint_or_threshold_selection": False,
                    }
                    atomic_write_json(metrics_path, result)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                candidates[key] = result
                progress.complete_task(current_task)
                current_task += 1
            del dataset
        progress.start_task(current_task)
        dataset_rows = _dataset_rows(source["role_results"], candidates)
        decision = benchmark_decision(dataset_rows, config["benchmark_gates"])
        session_elapsed = time.monotonic() - started
        elapsed = previous_elapsed + session_elapsed
        report = {
            "status": "completed", "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash, "source_v4_protocol_hash": source["protocol_hash"],
            "manual_confirmation": True, "retrospective_benchmark_only": True,
            "independent_confirmation_claim_allowed": False,
            "new_untouched_datasets_still_required": 4,
            "test_accessed": True, "test_used_for_model_or_threshold_selection": False,
            "datasets": list(CONFIRMATION_DATASETS), "seeds": list(CONFIRMATION_SEEDS),
            "primary_unit": "dataset", "elapsed_seconds": elapsed,
            "final_resume_session_elapsed_seconds": session_elapsed,
            "elapsed_seconds_are_cumulative_across_recorded_resume_sessions": True,
            "device": str(device),
            "dataset_results": dataset_rows, "candidate_results": candidates,
            "decision": decision, "run_root": str(run_root),
            "later_stage_started": False, "finished_at_utc": utc_now(),
        }
        lines = [
            "# V5 vs V4.1 four-dataset retrospective benchmark", "",
            f"- Frozen benchmark gate: **{'PASS' if decision['passed'] else 'FAIL'}**.",
            "- This is not new or independent V5 confirmation: V4 already accessed these tests.",
            "- The completed V4.1 controls are reused; only 12 V5 candidates are trained.",
            "- Validation alone selects checkpoints and reliability mode; test is not used for tuning.",
            f"- Device / elapsed: `{device}` / {elapsed:.1f} s.", "",
            "| Dataset | Mean unseen F1 delta vs V4.1 | Full-rate F1 delta | Selected AURC delta |",
            "|---|---:|---:|---:|",
        ]
        for dataset_id, row in dataset_rows.items():
            lines.append(
                f"| {dataset_id} | {row['mean_unseen_macro_f1_delta_vs_v4_1']:+.4f} | "
                f"{row['mean_full_rate_macro_f1_delta_vs_v4_1']:+.4f} | "
                f"{row['mean_selected_aurc_delta_vs_confidence']:+.4f} |"
            )
        lines.extend(["", "## Frozen checks", ""])
        lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["checks"].items())
        lines.extend(["", "A pass still requires at least four new untouched datasets for V5 confirmation.", ""])
        markdown = "\n".join(lines)
        atomic_write_json(run_root / "report.json", report)
        _atomic_write_text(run_root / "report.md", markdown)
        atomic_write_json(root / "reports" / "v5_four_dataset_benchmark_report.json", report)
        _atomic_write_text(root / "reports" / "v5_four_dataset_benchmark_report.md", markdown)
        progress.complete_task(current_task)
        progress.finish(
            f"V5 four-dataset benchmark {'PASS' if decision['passed'] else 'FAIL'}; no later stage auto-started"
        )
        manifest.update(
            status="completed", decision=decision["passed"],
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(run_root / "manifest.json", manifest)
        return report
    except BaseException as error:
        elapsed = previous_elapsed + (time.monotonic() - started)
        progress.fail_task(min(current_task, len(tasks) - 1), error)
        manifest.update(
            status="failed", error=f"{type(error).__name__}: {error}",
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(run_root / "manifest.json", manifest)
        raise
