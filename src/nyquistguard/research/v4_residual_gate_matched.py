"""Matched-budget audit of V4.1 against the hard-gate control."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.pilot import _seed_everything
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.research.v4_observe_only_micro import (
    V4_MICRO_DATASETS,
    V4_MICRO_SEED,
    _evaluate_validation,
    _new_model,
    _train_variant,
    load_development_dataset,
)
from nyquistguard.research.v4_residual_gate_micro import _decision


def _protocol_hash(config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(b"v4.1-matched-budget-control-runner-v1")
    return digest.hexdigest()


def _hard_initial_state(dataset: Any, base_config: dict[str, Any]) -> dict[str, torch.Tensor]:
    _seed_everything(V4_MICRO_SEED)
    model = _new_model(dataset, base_config, "v3_10_hard_gate", torch.device("cpu"))
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def run_v4_residual_gate_matched(project_root: str | Path, *, resume: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v4_residual_gate_matched_control.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    screen = config["matched_screen"]
    if tuple(screen["datasets"]) != V4_MICRO_DATASETS or int(screen["seed"]) != V4_MICRO_SEED:
        raise ValueError("matched V4.1 dataset/seed design changed")
    base_path = root / screen["base_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path, base_path)
    candidate_path = root / config["candidate_source"]
    candidate_source = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate_source.get("test_accessed") is not False:
        raise RuntimeError("candidate source is not leakage-locked")
    for dataset_id in V4_MICRO_DATASETS:
        epochs = candidate_source["results"][dataset_id]["v4_1_residual_gate"]["epochs_completed"]
        if int(epochs) != int(screen["epochs"]):
            raise RuntimeError(f"candidate did not receive matched epoch cap for {dataset_id}")
    run_root = root / "runs" / "v4_residual_gate_matched" / f"matched__seed17__{protocol_hash[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)
    deadline = started + float(screen["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    for dataset_id in V4_MICRO_DATASETS:
        dataset = load_development_dataset(root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz")
        metrics_path = run_root / dataset_id / "v3_10_hard_gate" / "metrics.json"
        if resume and metrics_path.exists():
            hard_result = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            model, history = _train_variant(
                dataset, base_config, screen, "v3_10_hard_gate",
                _hard_initial_state(dataset, base_config), protocol_hash,
                metrics_path.parent, device, deadline, resume,
            )
            validation = _evaluate_validation(
                model, dataset, base_config,
                tuple(float(value) for value in screen["validation_rate_ratios"]), device,
            )
            hard_result = {
                "protocol_hash": protocol_hash, "epochs_completed": len(history),
                "validation": validation, "test_accessed": False,
            }
            atomic_write_json(metrics_path, hard_result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        candidate = candidate_source["results"][dataset_id]["v4_1_residual_gate"]
        if int(hard_result["epochs_completed"]) > int(screen["epochs"]):
            raise RuntimeError(f"control exceeded the matched epoch cap for {dataset_id}")
        results[dataset_id] = {
            "v3_10_hard_gate_control": hard_result,
            "v4_1_residual_gate": candidate,
        }
    decision = _decision(results, screen["development_gates"])
    report = {
        "status": "completed_matched_candidate_pass" if decision["passed"] else "completed_matched_candidate_fail",
        "protocol_version": config["protocol_version"], "protocol_hash": protocol_hash,
        "primary_split": "validation_only", "test_accessed": False,
        "matched_epoch_cap": int(screen["epochs"]),
        "matched_early_stopping_patience": int(screen["early_stopping_patience"]),
        "realized_epochs_may_differ_under_the_same_early_stopping_rule": True,
        "candidate_source_report": str(candidate_path),
        "candidate_source_protocol_hash": candidate_source["protocol_hash"],
        "independent_confirmation_claim_allowed": False,
        "minimum_new_untouched_confirmation_datasets": int(config["data_boundary"]["minimum_new_confirmation_datasets"]),
        "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "results": results, "decision": decision, "run_root": str(run_root),
        "pilot_started": False, "full_started": False, "finished_at_utc": utc_now(),
    }
    lines = [
        "# V4.1 matched-budget validation audit", "",
        f"- Status: **{'PASS' if decision['passed'] else 'FAIL'}** (development only)",
        f"- Both arms: {screen['epochs']} epoch cap, patience {screen['early_stopping_patience']}, seed {V4_MICRO_SEED}.",
        "- Existing test arrays were not loaded or scored.", "",
        "| Dataset | Hard unseen F1 | V4.1 unseen F1 | Delta | Hard full F1 | V4.1 full F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, row in results.items():
        hard = row["v3_10_hard_gate_control"]["validation"]
        candidate = row["v4_1_residual_gate"]["validation"]
        lines.append(
            f"| {dataset_id} | {hard['mean_unseen_macro_f1']:.4f} | {candidate['mean_unseen_macro_f1']:.4f} | "
            f"{candidate['mean_unseen_macro_f1'] - hard['mean_unseen_macro_f1']:+.4f} | "
            f"{hard['full_rate_macro_f1']:.4f} | {candidate['full_rate_macro_f1']:.4f} |"
        )
    lines.extend(["", "## Frozen matched decision", "", f"```json\n{json.dumps(decision, ensure_ascii=False, indent=2)}\n```", ""])
    markdown = "\n".join(lines)
    atomic_write_json(run_root / "report.json", report)
    _atomic_write_text(run_root / "report.md", markdown)
    atomic_write_json(root / "reports" / "v4_residual_gate_matched_report.json", report)
    _atomic_write_text(root / "reports" / "v4_residual_gate_matched_report.md", markdown)
    return report
