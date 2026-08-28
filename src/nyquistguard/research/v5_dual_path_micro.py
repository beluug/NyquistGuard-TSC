"""Manual, resumable and validation-only V5 paired micro experiment."""

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

from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.pilot import _seed_everything
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.research.v4_observe_only_micro import (
    _evaluate_validation,
    _new_model,
    _train_variant,
    load_development_dataset,
)


V5_MICRO_DATASETS = ("basicmotions_uea", "pamap2_uci")
V5_MICRO_SEED = 17
V5_MICRO_ROLES = ("v4_1_residual_gate", "v5_dual_path")
V5_MICRO_TASKS = [
    "协议、资源互斥与train/validation边界预检",
    "BasicMotions seed17：冻结V4.1控制",
    "BasicMotions seed17：V5双路径候选",
    "PAMAP2 seed17：冻结V4.1控制",
    "PAMAP2 seed17：V5双路径候选",
    "汇总V5 validation开发门并生成报告",
]


def _protocol_hash(config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        config_path,
        base_path,
        Path(__file__),
        Path(__file__).with_name("v5_dual_path.py"),
        Path(__file__).with_name("v4_residual_gate.py"),
        Path(__file__).with_name("v4_observe_only_micro.py"),
    ):
        digest.update(path.read_bytes())
    digest.update(b"v5-dual-path-validation-micro-v1")
    return digest.hexdigest()


def validate_v5_micro_matrix(config: dict[str, Any]) -> list[tuple[str, int, str]]:
    design = config["design"]
    datasets = tuple(str(value) for value in design["datasets"])
    roles = tuple(str(value) for value in design["roles"])
    seed = int(design["seed"])
    matrix = [(str(d), int(s), str(r)) for d, s, r in design["run_order"]]
    if datasets != V5_MICRO_DATASETS or roles != V5_MICRO_ROLES or seed != V5_MICRO_SEED:
        raise ValueError("V5 micro datasets, roles, or seed changed after freeze")
    expected = [(d, seed, r) for d in datasets for r in roles]
    if matrix != expected:
        raise ValueError("V5 micro run_order must contain ordered matched pairs")
    if config["data_boundary"]["v4_confirmation_datasets_eligible_for_v5_confirmation"] is not False:
        raise ValueError("V4 confirmation datasets must remain ineligible for V5 confirmation")
    return matrix


def _assert_no_active_other_stage(root: Path) -> None:
    path = root / "runs" / "dashboard_status.json"
    if not path.exists():
        return
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if state.get("status") == "running" and state.get("stage") != "v5_dual_path_micro":
        raise RuntimeError(
            f"refusing to start V5 while {state.get('stage', 'another stage')} is running"
        )


def _paired_initial_states(
    dataset: Any, base_config: dict[str, Any], seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], bool]:
    _seed_everything(seed)
    control = _new_model(dataset, base_config, "v4_1_residual_gate", torch.device("cpu"))
    _seed_everything(seed)
    candidate = _new_model(dataset, base_config, "v5_dual_path", torch.device("cpu"))
    control_state = {key: value.detach().clone() for key, value in control.state_dict().items()}
    candidate_state = {key: value.detach().clone() for key, value in candidate.state_dict().items()}
    common = set(control_state) & set(candidate_state)
    exact = bool(common) and all(
        torch.equal(control_state[key], candidate_state[key]) for key in common
    )
    return control_state, candidate_state, exact


def _select_reliability(validation: dict[str, Any]) -> None:
    use_observability = (
        validation["pooled_observability_aurc"] <= validation["pooled_confidence_aurc"]
    )
    validation["reliability_mode"] = (
        "observability" if use_observability else "confidence_fallback"
    )
    validation["selected_pooled_aurc"] = validation[
        "pooled_observability_aurc" if use_observability else "pooled_confidence_aurc"
    ]


def v5_micro_decision(
    results: dict[str, dict[str, dict[str, Any]]], gates: dict[str, Any]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for dataset_id, roles in results.items():
        control = roles["v4_1_residual_gate"]["validation"]
        candidate = roles["v5_dual_path"]["validation"]
        counts = [int(row["prediction_class_count"]) for row in candidate["per_rate"].values()]
        floors = [float(value) for value in candidate["learned_gate_floor"]]
        rows[dataset_id] = {
            "unseen_delta": float(candidate["mean_unseen_macro_f1"] - control["mean_unseen_macro_f1"]),
            "full_delta": float(candidate["full_rate_macro_f1"] - control["full_rate_macro_f1"]),
            "selected_aurc_delta_vs_confidence": float(
                candidate["selected_pooled_aurc"] - candidate["pooled_confidence_aurc"]
            ),
            "minimum_prediction_class_count": min(counts),
            "minimum_gate_floor": min(floors),
            "maximum_gate_floor": max(floors),
        }
    unseen = [row["unseen_delta"] for row in rows.values()]
    full = [row["full_delta"] for row in rows.values()]
    finite = all(
        math.isfinite(float(value))
        for row in rows.values()
        for value in (
            row["unseen_delta"], row["full_delta"],
            row["selected_aurc_delta_vs_confidence"],
            row["minimum_gate_floor"], row["maximum_gate_floor"],
        )
    ) and all(
        0.0 <= row["minimum_gate_floor"] <= row["maximum_gate_floor"] <= 1.0
        for row in rows.values()
    )
    checks = {
        "average_unseen_gain": float(np.mean(unseen)) >= float(
            gates["minimum_average_unseen_macro_f1_delta_vs_v4_1"]
        ),
        "single_dataset_unseen_floor": float(np.min(unseen)) >= -float(
            gates["maximum_single_dataset_unseen_macro_f1_drop"]
        ),
        "average_full_rate_floor": float(np.mean(full)) >= -float(
            gates["maximum_average_full_rate_macro_f1_drop"]
        ),
        "no_constant_prediction": all(
            row["minimum_prediction_class_count"] > 1 for row in rows.values()
        ) if gates["require_no_constant_prediction_at_any_validation_rate"] else True,
        "selected_reliability_safety": all(
            row["selected_aurc_delta_vs_confidence"] <= 1e-12 for row in rows.values()
        ) if gates["require_selected_reliability_nonworse_than_confidence"] else True,
        "finite_metrics_and_floors": finite
        if gates["require_finite_metrics_and_gate_floors"] else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dataset_rows": rows,
        "average_unseen_macro_f1_delta_vs_v4_1": float(np.mean(unseen)),
        "minimum_dataset_unseen_macro_f1_delta_vs_v4_1": float(np.min(unseen)),
        "average_full_rate_macro_f1_delta_vs_v4_1": float(np.mean(full)),
    }


def _find_compatible_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for path in sorted(parent.glob("v5_micro__2datasets__seed17__*"), reverse=True):
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


def run_v5_dual_path_micro(
    project_root: str | Path, *, resume: bool = True, confirmed: bool = False
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("V5 validation micro requires manual confirmation")
    root = Path(project_root).resolve()
    _assert_no_active_other_stage(root)
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v5_dual_path_micro.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = validate_v5_micro_matrix(config)
    design = config["design"]
    base_path = root / design["base_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path, base_path)
    parent = root / "runs" / "v5_dual_path_micro"
    run_root = _find_compatible_root(parent, protocol_hash) if resume else None
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v5_micro__2datasets__seed17__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
    completed_report = run_root / "report.json"
    if resume and completed_report.exists():
        cached = json.loads(completed_report.read_text(encoding="utf-8"))
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
        "protocol_hash": protocol_hash, "run_root": str(run_root),
        "test_accessed": False, "cumulative_elapsed_seconds": previous_elapsed,
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "manifest.json", manifest)
    progress = DashboardProgress(
        root / "runs" / "dashboard_status.json", config["stage"], V5_MICRO_TASKS, run_root.name
    )
    deadline = started + float(design["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    role_results: dict[str, dict[str, Any]] = {}
    current_task = 0
    try:
        progress.start_task(current_task)
        for dataset_id in V5_MICRO_DATASETS:
            dataset = load_development_dataset(
                root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
            )
            if hasattr(dataset, "test"):
                raise RuntimeError("V5 development dataset unexpectedly exposes test")
        progress.complete_task(current_task)
        current_task += 1
        state_cache: dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}
        for index, (dataset_id, seed, role) in enumerate(matrix):
            progress.start_task(current_task)
            print(f"[{index + 1:02d}/04] Starting {dataset_id} seed{seed} {role}", flush=True)
            dataset = load_development_dataset(
                root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
            )
            if dataset_id not in state_cache:
                control_state, candidate_state, exact = _paired_initial_states(
                    dataset, base_config, seed
                )
                if not exact:
                    raise RuntimeError(f"shared initialization mismatch for {dataset_id}")
                state_cache[dataset_id] = (control_state, candidate_state)
            initial_state = state_cache[dataset_id][0 if role == "v4_1_residual_gate" else 1]
            role_dir = run_root / f"{dataset_id}__seed{seed}__{role}"
            metrics_path = role_dir / "metrics.json"
            result = None
            if resume and metrics_path.exists():
                cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                if cached.get("protocol_hash") == protocol_hash and cached.get("test_accessed") is False:
                    result = cached
            if result is None:
                role_started = time.monotonic()

                def update_epoch(epoch: int, total: int, row: dict[str, Any]) -> None:
                    progress.current_task = (
                        f"{dataset_id} seed{seed} {role}: epoch {epoch}/{total}, "
                        f"val={row['validation_selection_score']:.4f}"
                    )
                    progress.write()

                model, history = _train_variant(
                    dataset, base_config, design, role, initial_state, protocol_hash,
                    role_dir, device, deadline, resume, seed=seed, epoch_callback=update_epoch,
                )
                validation = _evaluate_validation(
                    model, dataset, base_config,
                    tuple(float(value) for value in design["validation_rate_ratios"]), device,
                )
                _select_reliability(validation)
                validation["learned_gate_floor"] = model.gate_floor.detach().cpu().tolist()
                result = {
                    "status": "completed", "protocol_hash": protocol_hash,
                    "dataset_id": dataset_id, "seed": seed, "role": role,
                    "epochs_completed": len(history),
                    "duration_seconds": time.monotonic() - role_started,
                    "validation": validation, "test_accessed": False,
                }
                atomic_write_json(metrics_path, result)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            role_results[f"{dataset_id}__{role}"] = result
            progress.complete_task(current_task)
            current_task += 1
        progress.start_task(current_task)
        grouped = {
            dataset_id: {
                role: role_results[f"{dataset_id}__{role}"] for role in V5_MICRO_ROLES
            }
            for dataset_id in V5_MICRO_DATASETS
        }
        decision = v5_micro_decision(grouped, config["development_gates"])
        session_elapsed = time.monotonic() - started
        elapsed = previous_elapsed + session_elapsed
        report = {
            "status": "completed", "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash, "primary_split": "validation_only",
            "test_accessed": False, "manual_confirmation": True,
            "independent_confirmation_claim_allowed": False,
            "v4_confirmation_datasets_eligible_for_v5_confirmation": False,
            "elapsed_seconds": elapsed,
            "final_resume_session_elapsed_seconds": session_elapsed,
            "elapsed_seconds_are_cumulative_across_recorded_resume_sessions": True,
            "device": str(device), "results": grouped,
            "decision": decision, "run_root": str(run_root),
            "pilot_started": False, "full_started": False,
            "later_stage_started": False, "finished_at_utc": utc_now(),
        }
        lines = [
            "# V5 dual-path validation-only micro", "",
            f"- Frozen development decision: **{'PASS' if decision['passed'] else 'FAIL'}**.",
            "- Data: BasicMotions/PAMAP2 train and validation only; no test loaded.",
            "- Paired control: frozen V4.1; same seed, schedule, budget, and selector.",
            f"- Device / elapsed: `{device}` / {elapsed:.1f} s.", "",
            "| Dataset | Unseen F1 delta | Full-rate F1 delta | Minimum predicted classes | Reliability delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for dataset_id, row in decision["dataset_rows"].items():
            lines.append(
                f"| {dataset_id} | {row['unseen_delta']:+.4f} | {row['full_delta']:+.4f} | "
                f"{row['minimum_prediction_class_count']} | "
                f"{row['selected_aurc_delta_vs_confidence']:+.4f} |"
            )
        lines.extend(["", "## Frozen checks", ""])
        for name, passed in decision["checks"].items():
            lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
        lines.extend([
            "", "This development screen cannot establish superiority or independent confirmation.",
            "No later experiment was started automatically.", "",
        ])
        markdown = "\n".join(lines)
        atomic_write_json(run_root / "report.json", report)
        _atomic_write_text(run_root / "report.md", markdown)
        atomic_write_json(root / "reports" / "v5_dual_path_micro_report.json", report)
        _atomic_write_text(root / "reports" / "v5_dual_path_micro_report.md", markdown)
        progress.complete_task(current_task)
        progress.finish(
            f"V5 validation micro {'PASS' if decision['passed'] else 'FAIL'}; no later stage auto-started"
        )
        manifest.update(
            status="completed", decision=decision["passed"],
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(run_root / "manifest.json", manifest)
        return report
    except BaseException as error:
        elapsed = previous_elapsed + (time.monotonic() - started)
        progress.fail_task(min(current_task, len(V5_MICRO_TASKS) - 1), error)
        manifest.update(
            status="failed", error=f"{type(error).__name__}: {error}",
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(run_root / "manifest.json", manifest)
        raise
