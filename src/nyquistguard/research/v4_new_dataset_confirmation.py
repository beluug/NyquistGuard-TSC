"""Manual, leakage-locked V4.1 confirmation on four previously unused datasets."""

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
    ConfirmationDevelopmentDataset,
    prepare_confirmation_dataset,
)
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.research.v4_observe_only_micro import (
    _evaluate_validation,
    _train_variant,
)
from nyquistguard.research.v4_residual_gate_multiseed import _paired_initial_states


CONFIRMATION_SEEDS = (16180, 57721, 94613)
CONFIRMATION_ROLES = ("v3_10_hard_gate", "v4_1_residual_gate")


def confirmation_tasks() -> list[str]:
    tasks: list[str] = []
    for dataset_id in CONFIRMATION_DATASETS:
        tasks.append(f"Prepare frozen {dataset_id}")
        tasks.extend(
            f"{dataset_id} seed{seed} {role}"
            for seed in CONFIRMATION_SEEDS
            for role in CONFIRMATION_ROLES
        )
    tasks.append("Aggregate four-dataset confirmation and write report")
    return tasks


def _protocol_hash(root: Path, config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        config_path,
        root / "configs" / "experiments" / "v4_new_dataset_confirmation_selection.yaml",
        base_path,
        root / "src" / "nyquistguard" / "data" / "new_confirmation_datasets.py",
        root / "src" / "nyquistguard" / "research" / "v4_observe_only_micro.py",
        root / "src" / "nyquistguard" / "research" / "v4_residual_gate.py",
        Path(__file__),
    ):
        digest.update(path.read_bytes())
    digest.update(b"v4.1-four-new-dataset-confirmation-v1")
    return digest.hexdigest()


def _validate_frozen_protocol(root: Path, config: dict[str, Any]) -> None:
    design = config["design"]
    if tuple(design["datasets"]) != CONFIRMATION_DATASETS:
        raise ValueError("the four frozen confirmation datasets changed")
    if tuple(int(value) for value in design["seeds"]) != CONFIRMATION_SEEDS:
        raise ValueError("the three frozen confirmation seeds changed")
    if tuple(design["roles"]) != CONFIRMATION_ROLES:
        raise ValueError("the matched confirmation roles changed")
    if float(design["initial_gate_floor"]) != 0.5:
        raise ValueError("the frozen V4.1 initial gate floor changed")
    if tuple(float(value) for value in design["test_rate_ratios"]) != (1.0, 0.9, 0.6, 0.4, 0.3):
        raise ValueError("the frozen test-rate grid changed")
    selection = yaml.safe_load(
        (root / config["source"]["selection_config"]).read_text(encoding="utf-8")
    )
    boundary = selection["scientific_boundary"]
    if tuple(boundary["selected_dataset_ids"]) != CONFIRMATION_DATASETS:
        raise ValueError("selection document and executable protocol disagree")
    if boundary["existing_full_datasets_allowed"] or boundary["existing_full_test_allowed"]:
        raise ValueError("existing Full data must remain forbidden as confirmation evidence")
    source = json.loads(
        (root / config["source"]["stability_report"]).read_text(encoding="utf-8")
    )
    if source.get("status") != config["source"]["required_stability_status"]:
        raise ValueError("V4.1 stability source is not completed")
    if bool(source.get("decision", {}).get("passed")) is not bool(config["source"]["required_stability_pass"]):
        raise ValueError("V4.1 stability source did not pass")
    if source.get("protocol_hash") != config["source"]["required_stability_protocol_hash"]:
        raise ValueError("V4.1 stability protocol hash changed")
    if source.get("test_accessed") is not False:
        raise ValueError("V4.1 stability source must be validation-only")


def _find_compatible_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for path in sorted(parent.glob("v4_1_confirmation__4datasets__3seeds__*"), reverse=True):
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


def _split_view(dataset: Any, split_name: str) -> ConfirmationDevelopmentDataset:
    return ConfirmationDevelopmentDataset(
        dataset.dataset_id,
        dataset.sampling_rate_hz,
        dataset.class_names,
        dataset.train,
        getattr(dataset, split_name),
        dataset.metadata,
    )


def _evaluate_selected_checkpoint(
    model: Any,
    dataset: Any,
    base_config: dict[str, Any],
    design: dict[str, Any],
    role: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rates = tuple(float(value) for value in design["validation_rate_ratios"])
    validation = _evaluate_validation(model, _split_view(dataset, "validation"), base_config, rates, device)
    if role in {"v4_1_residual_gate", "v5_dual_path"}:
        use_observability = validation["pooled_observability_aurc"] <= validation["pooled_confidence_aurc"]
        validation["reliability_mode"] = "observability" if use_observability else "confidence_fallback"
        validation["selected_pooled_aurc"] = validation[
            "pooled_observability_aurc" if use_observability else "pooled_confidence_aurc"
        ]
        validation["learned_gate_floor"] = model.gate_floor.detach().cpu().tolist()
    test_rates = tuple(float(value) for value in design["test_rate_ratios"])
    test = _evaluate_validation(model, _split_view(dataset, "test"), base_config, test_rates, device)
    if role in {"v4_1_residual_gate", "v5_dual_path"}:
        mode = validation["reliability_mode"]
        test["reliability_mode_selected_on_validation"] = mode
        test["selected_pooled_aurc"] = test[
            "pooled_observability_aurc" if mode == "observability" else "pooled_confidence_aurc"
        ]
        test["learned_gate_floor"] = validation["learned_gate_floor"]
    return validation, test


def _dataset_rows(role_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for dataset_id in CONFIRMATION_DATASETS:
        seed_rows = []
        for seed in CONFIRMATION_SEEDS:
            hard = role_results[f"{dataset_id}__seed{seed}__v3_10_hard_gate"]["test"]
            candidate = role_results[f"{dataset_id}__seed{seed}__v4_1_residual_gate"]["test"]
            seed_rows.append(
                {
                    "seed": seed,
                    "unseen_macro_f1_delta_vs_hard_gate": float(
                        candidate["mean_unseen_macro_f1"] - hard["mean_unseen_macro_f1"]
                    ),
                    "full_rate_macro_f1_delta_vs_hard_gate": float(
                        candidate["full_rate_macro_f1"] - hard["full_rate_macro_f1"]
                    ),
                    "selected_aurc_delta_vs_confidence": float(
                        candidate["selected_pooled_aurc"] - candidate["pooled_confidence_aurc"]
                    ),
                    "reliability_mode": candidate["reliability_mode_selected_on_validation"],
                    "minimum_gate_floor": float(min(
                        role_results[f"{dataset_id}__seed{seed}__v4_1_residual_gate"]["validation"]["learned_gate_floor"]
                    )),
                    "maximum_gate_floor": float(max(
                        role_results[f"{dataset_id}__seed{seed}__v4_1_residual_gate"]["validation"]["learned_gate_floor"]
                    )),
                }
            )
        output[dataset_id] = {
            "dataset_id": dataset_id,
            "seed_rows": seed_rows,
            "mean_unseen_macro_f1_delta_vs_hard_gate": float(np.mean([
                row["unseen_macro_f1_delta_vs_hard_gate"] for row in seed_rows
            ])),
            "mean_full_rate_macro_f1_delta_vs_hard_gate": float(np.mean([
                row["full_rate_macro_f1_delta_vs_hard_gate"] for row in seed_rows
            ])),
            "mean_selected_aurc_delta_vs_confidence": float(np.mean([
                row["selected_aurc_delta_vs_confidence"] for row in seed_rows
            ])),
        }
    return output


def confirmation_decision(dataset_rows: dict[str, dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    rows = list(dataset_rows.values())
    unseen = [float(row["mean_unseen_macro_f1_delta_vs_hard_gate"]) for row in rows]
    full = [float(row["mean_full_rate_macro_f1_delta_vs_hard_gate"]) for row in rows]
    reliability = [float(row["mean_selected_aurc_delta_vs_confidence"]) for row in rows]
    floors = [
        float(value)
        for row in rows
        for seed_row in row["seed_rows"]
        for value in (seed_row["minimum_gate_floor"], seed_row["maximum_gate_floor"])
    ]
    finite = all(math.isfinite(value) for value in unseen + full + reliability + floors)
    finite = finite and all(0.0 <= value <= 1.0 for value in floors)
    checks = {
        "average_dataset_unseen_gain": float(np.mean(unseen))
        > float(gates["minimum_average_dataset_unseen_macro_f1_delta_vs_hard_gate"]),
        "positive_dataset_count": sum(value > 0.0 for value in unseen)
        >= int(gates["minimum_positive_dataset_count"]),
        "single_dataset_unseen_floor": float(np.min(unseen))
        >= -float(gates["maximum_single_dataset_unseen_macro_f1_drop"]),
        "average_dataset_full_rate_floor": float(np.mean(full))
        >= -float(gates["maximum_average_dataset_full_rate_macro_f1_drop"]),
        "average_dataset_reliability_safety": float(np.mean(reliability))
        <= float(gates["maximum_average_dataset_selected_aurc_delta_vs_confidence"]),
        "finite_metrics": finite if gates["require_finite_all_metrics_and_gate_floors"] else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "average_dataset_unseen_macro_f1_delta_vs_hard_gate": float(np.mean(unseen)),
        "positive_dataset_count": int(sum(value > 0.0 for value in unseen)),
        "minimum_dataset_unseen_macro_f1_delta_vs_hard_gate": float(np.min(unseen)),
        "average_dataset_full_rate_macro_f1_delta_vs_hard_gate": float(np.mean(full)),
        "average_dataset_selected_aurc_delta_vs_confidence": float(np.mean(reliability)),
    }


def run_v4_new_dataset_confirmation(
    project_root: str | Path, *, resume: bool = True, confirmed: bool = False
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("four-dataset formal confirmation requires explicit manual confirmation")
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v4_new_dataset_confirmation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_frozen_protocol(root, config)
    design = config["design"]
    base_path = root / design["base_config"]
    base_template = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(root, config_path, base_path)
    parent = root / "runs" / "v4_new_dataset_confirmation"
    run_root = _find_compatible_root(parent, protocol_hash) if resume else None
    if run_root is not None and (run_root / "report.json").exists():
        cached = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return cached
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v4_1_confirmation__4datasets__3seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
        shutil.copy2(
            root / config["source"]["selection_config"], run_root / "selection_frozen.yaml"
        )
    manifest = {
        "stage": config["stage"], "status": "running", "manual_confirmation": True,
        "protocol_hash": protocol_hash, "run_root": str(run_root), "test_accessed": False,
        "test_access_authorized": True,
        "test_access_reason": "explicit formal manual start", "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "manifest.json", manifest)
    tasks = confirmation_tasks()
    progress = DashboardProgress(root / "runs" / "dashboard_status.json", config["stage"], tasks, run_root.name)
    deadline = started + float(design["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    role_results: dict[str, dict[str, Any]] = {}
    current_task = 0
    try:
        for dataset_id in CONFIRMATION_DATASETS:
            progress.start_task(current_task)
            dataset = prepare_confirmation_dataset(root, dataset_id, confirmed_test_access=True)
            if dataset.metadata.get("test_accessed") is not True:
                raise RuntimeError(f"formal cache is not marked test-accessed: {dataset_id}")
            manifest.update(test_accessed=True, updated_at_utc=utc_now())
            atomic_write_json(run_root / "manifest.json", manifest)
            progress.complete_task(current_task)
            current_task += 1
            base_config = dict(base_template)
            base_config["batch_size"] = int(design["dataset_batch_sizes"][dataset_id])
            for seed in CONFIRMATION_SEEDS:
                hard_state, candidate_state, exact = _paired_initial_states(dataset, base_config, seed)
                if not exact:
                    raise RuntimeError(f"paired shared initialization mismatch for {dataset_id} seed{seed}")
                for role_index, role in enumerate(CONFIRMATION_ROLES):
                    progress.start_task(current_task)
                    key = f"{dataset_id}__seed{seed}__{role}"
                    role_dir = run_root / key
                    metrics_path = role_dir / "metrics.json"
                    result: dict[str, Any] | None = None
                    if resume and metrics_path.exists():
                        candidate_cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if candidate_cached.get("protocol_hash") == protocol_hash and candidate_cached.get("test_accessed") is True:
                            result = candidate_cached
                    if result is None:
                        role_started = time.monotonic()

                        def update_epoch(epoch: int, total: int, row: dict[str, Any]) -> None:
                            progress.current_task = (
                                f"{dataset_id} seed{seed} {role}: epoch {epoch}/{total}, "
                                f"val={row['validation_selection_score']:.4f}"
                            )
                            progress.write()

                        initial_state = hard_state if role_index == 0 else candidate_state
                        model, history = _train_variant(
                            _split_view(dataset, "validation"), base_config, design, role,
                            initial_state, protocol_hash, role_dir, device, deadline, resume,
                            seed=seed, epoch_callback=update_epoch,
                        )
                        validation, test = _evaluate_selected_checkpoint(
                            model, dataset, base_config, design, role, device
                        )
                        result = {
                            "status": "completed", "protocol_hash": protocol_hash,
                            "dataset_id": dataset_id, "seed": seed, "role": role,
                            "batch_size": base_config["batch_size"],
                            "epochs_completed": len(history),
                            "duration_seconds": time.monotonic() - role_started,
                            "validation": validation, "test": test,
                            "test_accessed": True,
                            "test_used_for_checkpoint_or_threshold_selection": False,
                        }
                        atomic_write_json(metrics_path, result)
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    role_results[key] = result
                    print(f"Completed {key}", flush=True)
                    progress.complete_task(current_task)
                    current_task += 1
            del dataset
        progress.start_task(current_task)
        dataset_rows = _dataset_rows(role_results)
        decision = confirmation_decision(dataset_rows, config["confirmation_gates"])
        elapsed = time.monotonic() - started
        report = {
            "status": "completed", "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash, "manual_confirmation": True,
            "datasets": list(CONFIRMATION_DATASETS), "seeds": list(CONFIRMATION_SEEDS),
            "primary_unit": "dataset", "test_accessed": True,
            "test_used_for_model_checkpoint_or_threshold_selection": False,
            "elapsed_seconds": elapsed, "device": str(device),
            "dataset_results": dataset_rows, "role_results": role_results,
            "confirmation_gates": config["confirmation_gates"], "decision": decision,
            "run_root": str(run_root), "later_stage_started": False,
            "finished_at_utc": utc_now(),
        }
        lines = [
            "# V4.1 confirmation on four new datasets", "",
            f"- Frozen decision: **{'PASS' if decision['passed'] else 'FAIL'}**.",
            "- Primary unit: dataset; each dataset value is averaged over three matched seeds.",
            "- Test was evaluated once per validation-selected checkpoint after manual start.",
            "- Test labels did not select models, checkpoints, thresholds, or reliability mode.",
            f"- Device / elapsed: `{device}` / {elapsed:.1f} s.", "",
            "| Dataset | Mean unseen F1 delta | Full-rate F1 delta | Selected AURC delta |",
            "|---|---:|---:|---:|",
        ]
        for dataset_id, row in dataset_rows.items():
            lines.append(
                f"| {dataset_id} | {row['mean_unseen_macro_f1_delta_vs_hard_gate']:+.4f} | "
                f"{row['mean_full_rate_macro_f1_delta_vs_hard_gate']:+.4f} | "
                f"{row['mean_selected_aurc_delta_vs_confidence']:+.4f} |"
            )
        lines.extend(["", "## Frozen checks", ""])
        lines.extend(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in decision["checks"].items()
        )
        lines.append("")
        markdown = "\n".join(lines)
        atomic_write_json(run_root / "report.json", report)
        _atomic_write_text(run_root / "report.md", markdown)
        atomic_write_json(root / "reports" / "v4_new_dataset_confirmation_report.json", report)
        _atomic_write_text(root / "reports" / "v4_new_dataset_confirmation_report.md", markdown)
        progress.complete_task(current_task)
        progress.finish(f"V4.1 four-new-dataset confirmation {'PASS' if decision['passed'] else 'FAIL'}")
        manifest.update(status="completed", decision=decision["passed"], updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        return report
    except BaseException as error:
        progress.fail_task(min(current_task, len(tasks) - 1), error)
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        raise
