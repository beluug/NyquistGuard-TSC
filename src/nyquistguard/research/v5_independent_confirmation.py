"""Manual two-phase independent confirmation for frozen V5.1."""

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

from nyquistguard.data.new_confirmation_datasets import ConfirmationDevelopmentDataset
from nyquistguard.data.v5_independent_datasets import (
    V5_INDEPENDENT_DATASETS,
    prepare_v5_independent_dataset,
    prepare_v5_independent_development_dataset,
    raw_split_path,
)
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.research.v4_observe_only_micro import (
    _evaluate_validation,
    _new_model,
    _train_variant,
)
from nyquistguard.research.v5_dual_path_micro import (
    _paired_initial_states,
)
from nyquistguard.research.v5_safe_reliability import select_consensus_mode


INDEPENDENT_SEEDS = (314159, 271828, 161803)
INDEPENDENT_ROLES = ("v4_1_residual_gate", "v5_dual_path")


def _assert_no_active_other_stage(root: Path) -> None:
    path = root / "runs" / "dashboard_status.json"
    if not path.exists():
        return
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if (
        state.get("status") == "running"
        and state.get("stage") != "v5_1_independent_confirmation"
    ):
        raise RuntimeError(
            f"refusing to start while {state.get('stage', 'another stage')} is running"
        )


def independent_confirmation_tasks() -> list[str]:
    tasks = ["Validate frozen panel, sources, files, and zero-TEST development caches"]
    tasks.extend(
        f"Validation train {dataset_id} seed{seed} {role}"
        for dataset_id in V5_INDEPENDENT_DATASETS
        for seed in INDEPENDENT_SEEDS
        for role in INDEPENDENT_ROLES
    )
    tasks.append("Freeze four dataset-level reliability modes from validation only")
    tasks.extend(f"Unlock official TEST {dataset_id}" for dataset_id in V5_INDEPENDENT_DATASETS)
    tasks.extend(
        f"One-shot TEST {dataset_id} seed{seed} {role}"
        for dataset_id in V5_INDEPENDENT_DATASETS
        for seed in INDEPENDENT_SEEDS
        for role in INDEPENDENT_ROLES
    )
    tasks.append("Aggregate four-dataset independent confirmation and write report")
    return tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_hash(root: Path, config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        config_path,
        root / "configs" / "experiments" / "v5_1_independent_confirmation_selection.yaml",
        base_path,
        root / "src" / "nyquistguard" / "data" / "v5_independent_datasets.py",
        root / "src" / "nyquistguard" / "research" / "v4_residual_gate.py",
        root / "src" / "nyquistguard" / "research" / "v5_dual_path.py",
        root / "src" / "nyquistguard" / "research" / "v5_safe_reliability.py",
        Path(__file__),
    ):
        digest.update(path.read_bytes())
    digest.update(b"v5.1-four-untouched-dataset-confirmation-v1")
    return digest.hexdigest()


def _validate_protocol(root: Path, config: dict[str, Any]) -> None:
    design = config["design"]
    if tuple(design["datasets"]) != V5_INDEPENDENT_DATASETS:
        raise ValueError("frozen V5.1 independent panel changed")
    if tuple(int(value) for value in design["seeds"]) != INDEPENDENT_SEEDS:
        raise ValueError("frozen V5.1 independent seeds changed")
    if tuple(design["roles"]) != INDEPENDENT_ROLES:
        raise ValueError("frozen matched roles changed")
    selection = yaml.safe_load(
        (root / config["source"]["selection_config"]).read_text(encoding="utf-8")
    )
    boundary = selection["scientific_boundary"]
    if tuple(boundary["selected_dataset_ids"]) != V5_INDEPENDENT_DATASETS:
        raise ValueError("selection and executable panel disagree")
    if boundary["all_selected_tests_untouched_at_freeze"] is not True:
        raise ValueError("panel was not frozen with untouched TEST")
    if boundary["prior_full_or_v4_v5_datasets_allowed"] is not False:
        raise ValueError("prior datasets cannot be independent evidence")
    source = json.loads(
        (root / config["source"]["v5_benchmark_report"]).read_text(encoding="utf-8")
    )
    if source.get("protocol_hash") != config["source"]["required_v5_benchmark_protocol_hash"]:
        raise ValueError("V5 classification source hash changed")
    safe = json.loads(
        (root / config["source"]["v5_safe_reliability_report"]).read_text(encoding="utf-8")
    )
    if bool(safe.get("decision", {}).get("passed")) is not bool(
        config["source"]["required_safe_reliability_pass"]
    ):
        raise ValueError("V5.1 safe reliability source did not pass")
    for dataset_id in V5_INDEPENDENT_DATASETS:
        for split in ("TRAIN", "TEST"):
            path = raw_split_path(root, dataset_id, split)
            if not path.exists() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"download required before formal start: {path}")


def _find_compatible_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for path in sorted(parent.glob("v5_1_confirmation__4datasets__3seeds__*"), reverse=True):
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


def _view(dataset: Any, split_name: str) -> ConfirmationDevelopmentDataset:
    return ConfirmationDevelopmentDataset(
        dataset.dataset_id,
        dataset.sampling_rate_hz,
        dataset.class_names,
        dataset.train,
        getattr(dataset, split_name),
        dataset.metadata,
    )


def _validation_hashes(run_root: Path) -> dict[str, str]:
    paths = sorted(run_root.glob("*__seed*/validation_metrics.json"))
    paths += sorted(run_root.glob("*__seed*/checkpoint_best.pt"))
    if len(paths) != 48:
        raise RuntimeError(f"expected 48 frozen validation artifacts, found {len(paths)}")
    return {str(path.relative_to(run_root)): _sha256(path) for path in paths}


def _freeze_reliability_modes(
    run_root: Path, protocol_hash: str, validations: dict[str, dict[str, Any]], controller: dict[str, Any]
) -> dict[str, Any]:
    path = run_root / "reliability_modes_frozen.json"
    hashes = _validation_hashes(run_root)
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("protocol_hash") != protocol_hash or frozen.get("source_hashes") != hashes:
            raise RuntimeError("frozen reliability modes or their validation sources changed")
        return frozen
    modes = {}
    for dataset_id in V5_INDEPENDENT_DATASETS:
        gains = []
        for seed in INDEPENDENT_SEEDS:
            validation = validations[f"{dataset_id}__seed{seed}__v5_dual_path"]["validation"]
            gains.append(
                float(validation["pooled_confidence_aurc"])
                - float(validation["pooled_observability_aurc"])
            )
        modes[dataset_id] = select_consensus_mode(
            gains,
            minimum_seed_gain=float(controller["minimum_seed_validation_aurc_gain"]),
            minimum_mean_gain=float(controller["minimum_dataset_mean_validation_aurc_gain"]),
            required_positive_fraction=float(controller["required_positive_seed_fraction"]),
        )
    frozen = {
        "status": "frozen_before_any_test_read",
        "protocol_hash": protocol_hash,
        "selection_split": "validation_only_across_three_matched_seeds",
        "test_accessed": False,
        "modes": modes,
        "source_hashes": hashes,
        "frozen_at_utc": utc_now(),
    }
    atomic_write_json(path, frozen)
    return frozen


def independent_decision(
    dataset_rows: dict[str, dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    rows = list(dataset_rows.values())
    unseen = [float(row["mean_unseen_macro_f1_delta_vs_v4_1"]) for row in rows]
    full = [float(row["mean_full_rate_macro_f1_delta_vs_v4_1"]) for row in rows]
    reliability = [float(row["mean_selected_aurc_delta_vs_confidence"]) for row in rows]
    floors = [
        float(value)
        for row in rows
        for seed_row in row["seed_rows"]
        for value in (seed_row["minimum_gate_floor"], seed_row["maximum_gate_floor"])
    ]
    counts = [
        int(seed_row["minimum_prediction_class_count"])
        for row in rows
        for seed_row in row["seed_rows"]
    ]
    finite = all(math.isfinite(value) for value in unseen + full + reliability + floors)
    finite = finite and all(0.0 <= value <= 1.0 for value in floors)
    checks = {
        "average_dataset_unseen_gain": float(np.mean(unseen))
        > float(gates["minimum_average_dataset_unseen_macro_f1_delta_vs_v4_1"]),
        "positive_dataset_count": sum(value > 0.0 for value in unseen)
        >= int(gates["minimum_positive_dataset_count"]),
        "single_dataset_unseen_floor": float(np.min(unseen))
        >= -float(gates["maximum_single_dataset_unseen_macro_f1_drop"]),
        "average_dataset_full_rate_floor": float(np.mean(full))
        >= -float(gates["maximum_average_dataset_full_rate_macro_f1_drop"]),
        "average_dataset_reliability_safety": float(np.mean(reliability))
        <= float(gates["maximum_average_dataset_selected_aurc_delta_vs_confidence"]),
        "no_constant_prediction": all(value > 1 for value in counts)
        if gates["require_no_constant_prediction_at_any_test_rate"]
        else True,
        "finite_metrics": finite
        if gates["require_finite_all_metrics_and_gate_floors"]
        else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "average_dataset_unseen_macro_f1_delta_vs_v4_1": float(np.mean(unseen)),
        "positive_dataset_count": int(sum(value > 0.0 for value in unseen)),
        "minimum_dataset_unseen_macro_f1_delta_vs_v4_1": float(np.min(unseen)),
        "average_dataset_full_rate_macro_f1_delta_vs_v4_1": float(np.mean(full)),
        "average_dataset_selected_aurc_delta_vs_confidence": float(np.mean(reliability)),
    }


def run_v5_1_independent_preflight(project_root: str | Path) -> dict[str, Any]:
    """Materialize TRAIN-derived caches only; structurally cannot access TEST."""

    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v5_1_independent_confirmation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection = yaml.safe_load(
        (root / config["source"]["selection_config"]).read_text(encoding="utf-8")
    )
    if tuple(selection["scientific_boundary"]["selected_dataset_ids"]) != V5_INDEPENDENT_DATASETS:
        raise ValueError("preflight panel differs from frozen selection")
    rows = {}
    for dataset_id in V5_INDEPENDENT_DATASETS:
        dataset = prepare_v5_independent_development_dataset(root, dataset_id)
        if hasattr(dataset, "test") or dataset.metadata.get("test_accessed") is not False:
            raise RuntimeError(f"TRAIN-only preflight exposed TEST for {dataset_id}")
        train_ids = set(dataset.train.ids.astype(str).tolist())
        validation_ids = set(dataset.validation.ids.astype(str).tolist())
        rows[dataset_id] = {
            "train_shape": list(dataset.train.x.shape),
            "validation_shape": list(dataset.validation.x.shape),
            "class_count": len(dataset.class_names),
            "split_id_overlap": bool(train_ids & validation_ids),
            "test_accessed": False,
        }
    checks = {
        "four_datasets": len(rows) == 4,
        "no_train_validation_overlap": all(not row["split_id_overlap"] for row in rows.values()),
        "no_test_access": all(row["test_accessed"] is False for row in rows.values()),
        "finite_arrays": all(
            np.isfinite(dataset.train.x).all() and np.isfinite(dataset.validation.x).all()
            for dataset in (
                prepare_v5_independent_development_dataset(root, dataset_id)
                for dataset_id in V5_INDEPENDENT_DATASETS
            )
        ),
    }
    report = {
        "status": "completed", "stage": "v5_1_independent_preflight",
        "passed": all(checks.values()), "test_accessed": False,
        "datasets": rows, "checks": checks,
        "elapsed_seconds": time.monotonic() - started,
        "later_stage_started": False, "finished_at_utc": utc_now(),
    }
    lines = [
        "# V5.1 independent panel TRAIN-only preflight", "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**.",
        "- Official TEST files were not opened or cached.", "",
        "| Dataset | Train | Validation | Classes | TEST accessed |", "|---|---:|---:|---:|---|",
    ]
    for dataset_id, row in rows.items():
        lines.append(
            f"| {dataset_id} | {row['train_shape']} | {row['validation_shape']} | "
            f"{row['class_count']} | no |"
        )
    lines.extend(["", "No training or later stage was started.", ""])
    atomic_write_json(root / "reports" / "v5_1_independent_preflight_report.json", report)
    _atomic_write_text(root / "reports" / "v5_1_independent_preflight_report.md", "\n".join(lines))
    return report


def _aggregate(
    results: dict[str, dict[str, Any]], modes: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    output = {}
    for dataset_id in V5_INDEPENDENT_DATASETS:
        seed_rows = []
        for seed in INDEPENDENT_SEEDS:
            control = results[f"{dataset_id}__seed{seed}__v4_1_residual_gate"]["test"]
            candidate_result = results[f"{dataset_id}__seed{seed}__v5_dual_path"]
            candidate = candidate_result["test"]
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
                "minimum_prediction_class_count": min(
                    int(row["prediction_class_count"])
                    for row in candidate["per_rate"].values()
                ),
                "minimum_gate_floor": min(candidate_result["validation"]["learned_gate_floor"]),
                "maximum_gate_floor": max(candidate_result["validation"]["learned_gate_floor"]),
            })
        output[dataset_id] = {
            "dataset_id": dataset_id,
            "reliability_mode": modes[dataset_id]["mode"],
            "seed_rows": seed_rows,
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


def run_v5_1_independent_confirmation(
    project_root: str | Path, *, resume: bool = True, confirmed: bool = False
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("V5.1 independent confirmation requires manual confirmation")
    root = Path(project_root).resolve()
    _assert_no_active_other_stage(root)
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v5_1_independent_confirmation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_protocol(root, config)
    design = config["design"]
    base_path = root / design["base_config"]
    base_template = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(root, config_path, base_path)
    parent = root / "runs" / "v5_1_independent_confirmation"
    run_root = _find_compatible_root(parent, protocol_hash) if resume else None
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v5_1_confirmation__4datasets__3seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
        shutil.copy2(root / config["source"]["selection_config"], run_root / "selection_frozen.yaml")
    report_path = run_root / "report.json"
    if resume and report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return cached
    previous_elapsed = 0.0
    manifest_path = run_root / "manifest.json"
    if resume and manifest_path.exists():
        try:
            previous_elapsed = float(json.loads(manifest_path.read_text(encoding="utf-8")).get("cumulative_elapsed_seconds", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            previous_elapsed = 0.0
    manifest = {
        "stage": config["stage"], "status": "running", "phase": "validation_training",
        "manual_confirmation": True, "protocol_hash": protocol_hash,
        "test_accessed": False, "cumulative_elapsed_seconds": previous_elapsed,
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    tasks = independent_confirmation_tasks()
    progress = DashboardProgress(root / "runs" / "dashboard_status.json", config["stage"], tasks, run_root.name)
    deadline = started + float(design["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_task = 0
    validations: dict[str, dict[str, Any]] = {}
    try:
        progress.start_task(current_task)
        development = {
            dataset_id: prepare_v5_independent_development_dataset(root, dataset_id)
            for dataset_id in V5_INDEPENDENT_DATASETS
        }
        if any(hasattr(dataset, "test") for dataset in development.values()):
            raise RuntimeError("development preflight exposed TEST")
        progress.complete_task(current_task)
        current_task += 1
        for dataset_id in V5_INDEPENDENT_DATASETS:
            dataset = development[dataset_id]
            base_config = dict(base_template)
            base_config["batch_size"] = int(design["dataset_batch_sizes"][dataset_id])
            for seed in INDEPENDENT_SEEDS:
                control_state, candidate_state, exact = _paired_initial_states(dataset, base_config, seed)
                if not exact:
                    raise RuntimeError(f"shared initialization mismatch for {dataset_id} seed{seed}")
                for role, initial_state in zip(INDEPENDENT_ROLES, (control_state, candidate_state)):
                    progress.start_task(current_task)
                    key = f"{dataset_id}__seed{seed}__{role}"
                    run_dir = run_root / key
                    metrics_path = run_dir / "validation_metrics.json"
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
                            run_dir, device, deadline, resume, seed=seed,
                            epoch_callback=update_epoch,
                        )
                        validation = _evaluate_validation(
                            model, dataset, base_config,
                            tuple(float(value) for value in design["validation_rate_ratios"]), device,
                        )
                        validation["learned_gate_floor"] = model.gate_floor.detach().cpu().tolist()
                        result = {
                            "status": "validation_completed", "protocol_hash": protocol_hash,
                            "dataset_id": dataset_id, "seed": seed, "role": role,
                            "epochs_completed": len(history),
                            "duration_seconds": time.monotonic() - role_started,
                            "validation": validation, "test_accessed": False,
                        }
                        atomic_write_json(metrics_path, result)
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    validations[key] = result
                    print(f"Validation completed {key}", flush=True)
                    progress.complete_task(current_task)
                    current_task += 1
        progress.start_task(current_task)
        frozen = _freeze_reliability_modes(
            run_root, protocol_hash, validations, config["reliability_controller"]
        )
        manifest.update(phase="test_evaluation", reliability_modes_frozen=True, updated_at_utc=utc_now())
        atomic_write_json(manifest_path, manifest)
        progress.complete_task(current_task)
        current_task += 1
        formal = {}
        for dataset_id in V5_INDEPENDENT_DATASETS:
            progress.start_task(current_task)
            formal[dataset_id] = prepare_v5_independent_dataset(
                root, dataset_id, confirmed_test_access=True
            )
            manifest.update(test_accessed=True, updated_at_utc=utc_now())
            atomic_write_json(manifest_path, manifest)
            progress.complete_task(current_task)
            current_task += 1
        results: dict[str, dict[str, Any]] = {}
        for dataset_id in V5_INDEPENDENT_DATASETS:
            dataset = formal[dataset_id]
            base_config = dict(base_template)
            base_config["batch_size"] = int(design["dataset_batch_sizes"][dataset_id])
            for seed in INDEPENDENT_SEEDS:
                for role in INDEPENDENT_ROLES:
                    progress.start_task(current_task)
                    key = f"{dataset_id}__seed{seed}__{role}"
                    run_dir = run_root / key
                    metrics_path = run_dir / "metrics.json"
                    result = None
                    if resume and metrics_path.exists():
                        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if cached.get("protocol_hash") == protocol_hash and cached.get("test_accessed") is True:
                            result = cached
                    if result is None:
                        model = _new_model(_view(dataset, "validation"), base_config, role, device)
                        model.load_state_dict(
                            torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=True),
                            strict=True,
                        )
                        test = _evaluate_validation(
                            model, _view(dataset, "test"), base_config,
                            tuple(float(value) for value in design["test_rate_ratios"]), device,
                        )
                        if role == "v5_dual_path":
                            mode = frozen["modes"][dataset_id]["mode"]
                            selected_key = (
                                "pooled_observability_aurc"
                                if mode == "observability"
                                else "pooled_confidence_aurc"
                            )
                            test["reliability_mode_selected_on_validation_consensus"] = mode
                            test["selected_pooled_aurc"] = float(test[selected_key])
                        result = {
                            **validations[key], "status": "completed", "test": test,
                            "test_accessed": True,
                            "test_used_for_any_selection": False,
                        }
                        atomic_write_json(metrics_path, result)
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    results[key] = result
                    progress.complete_task(current_task)
                    current_task += 1
        progress.start_task(current_task)
        dataset_rows = _aggregate(results, frozen["modes"])
        decision = independent_decision(dataset_rows, config["confirmation_gates"])
        elapsed = previous_elapsed + (time.monotonic() - started)
        report = {
            "status": "completed", "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash, "manual_confirmation": True,
            "independent_confirmation": True, "datasets": list(V5_INDEPENDENT_DATASETS),
            "seeds": list(INDEPENDENT_SEEDS), "primary_unit": "dataset",
            "test_accessed": True, "test_used_for_any_selection": False,
            "reliability_modes_frozen_before_test": frozen,
            "dataset_results": dataset_rows, "role_results": results,
            "decision": decision, "elapsed_seconds": elapsed, "device": str(device),
            "run_root": str(run_root), "later_stage_started": False,
            "finished_at_utc": utc_now(),
        }
        lines = [
            "# V5.1 independent confirmation on four untouched datasets", "",
            f"- Frozen decision: **{'PASS' if decision['passed'] else 'FAIL'}**.",
            "- All reliability modes were frozen from validation across three seeds before any TEST parse.",
            "- TEST was evaluated once per validation-selected checkpoint and never selected a model or mode.",
            f"- Device / cumulative elapsed: `{device}` / `{elapsed:.1f} s`.", "",
            "| Dataset | Reliability mode | Mean unseen F1 delta | Full-rate F1 delta | Selected AURC delta |",
            "|---|---|---:|---:|---:|",
        ]
        for dataset_id, row in dataset_rows.items():
            lines.append(
                f"| {dataset_id} | {row['reliability_mode']} | "
                f"{row['mean_unseen_macro_f1_delta_vs_v4_1']:+.4f} | "
                f"{row['mean_full_rate_macro_f1_delta_vs_v4_1']:+.4f} | "
                f"{row['mean_selected_aurc_delta_vs_confidence']:+.4f} |"
            )
        lines.extend(["", "## Frozen checks", ""])
        lines.extend(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in decision["checks"].items()
        )
        lines.extend(["", "No later stage was started automatically.", ""])
        markdown = "\n".join(lines)
        atomic_write_json(report_path, report)
        _atomic_write_text(run_root / "report.md", markdown)
        atomic_write_json(root / "reports" / "v5_1_independent_confirmation_report.json", report)
        _atomic_write_text(root / "reports" / "v5_1_independent_confirmation_report.md", markdown)
        progress.complete_task(current_task)
        progress.finish(f"V5.1 independent confirmation {'PASS' if decision['passed'] else 'FAIL'}")
        manifest.update(
            status="completed", phase="completed", decision=decision["passed"],
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(manifest_path, manifest)
        return report
    except BaseException as error:
        elapsed = previous_elapsed + (time.monotonic() - started)
        progress.fail_task(min(current_task, len(tasks) - 1), error)
        manifest.update(
            status="failed", error=f"{type(error).__name__}: {error}",
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(manifest_path, manifest)
        raise
