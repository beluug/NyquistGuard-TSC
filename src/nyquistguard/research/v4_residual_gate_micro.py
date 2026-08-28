"""Adaptive V4.1 validation-only micro screen following the frozen V4.0 result."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.pilot import _seed_everything
from nyquistguard.research.v4_observe_only_micro import (
    V4_MICRO_DATASETS,
    V4_MICRO_SEED,
    _evaluate_validation,
    _new_model,
    _train_variant,
    load_development_dataset,
)


def _hash(config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(b"v4.1-residual-gate-adaptive-runner-v1")
    return digest.hexdigest()


def _shared_initialization_exact(
    dataset: Any, base_config: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], bool]:
    _seed_everything(V4_MICRO_SEED)
    hard = _new_model(dataset, base_config, "v3_10_hard_gate", torch.device("cpu"))
    _seed_everything(V4_MICRO_SEED)
    candidate = _new_model(dataset, base_config, "v4_1_residual_gate", torch.device("cpu"))
    hard_state = hard.state_dict()
    candidate_state = candidate.state_dict()
    common = set(hard_state) & set(candidate_state)
    exact = bool(common) and all(torch.equal(hard_state[key], candidate_state[key]) for key in common)
    initial = {key: value.detach().clone() for key, value in candidate_state.items()}
    return initial, exact


def _decision(results: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    unseen, full, reliability, sufficiency = [], [], [], []
    for row in results.values():
        hard = row["v3_10_hard_gate_control"]["validation"]
        candidate = row["v4_1_residual_gate"]["validation"]
        unseen.append(candidate["mean_unseen_macro_f1"] - hard["mean_unseen_macro_f1"])
        full.append(candidate["full_rate_macro_f1"] - hard["full_rate_macro_f1"])
        reliability.append(candidate["selected_pooled_aurc"] - candidate["pooled_confidence_aurc"])
        sufficiency.append(candidate["full_to_low_observability_score_drop"] > 0.0)
    checks = {
        "average_unseen_gain": float(np.mean(unseen)) >= float(gates["minimum_average_unseen_macro_f1_delta_vs_hard_gate"]),
        "single_dataset_unseen_floor": float(np.min(unseen)) >= -float(gates["maximum_single_dataset_unseen_macro_f1_drop"]),
        "average_full_rate_floor": float(np.mean(full)) >= -float(gates["maximum_average_full_rate_macro_f1_drop"]),
        "selected_reliability_safety": all(value <= 1e-12 for value in reliability)
        if gates["require_selected_reliability_nonworse_than_confidence"] else True,
        "rate_sufficiency": all(sufficiency)
        if gates["require_rate_sufficiency_drop_both_datasets"] else True,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "average_unseen_macro_f1_delta_vs_hard_gate": float(np.mean(unseen)),
        "minimum_dataset_unseen_macro_f1_delta_vs_hard_gate": float(np.min(unseen)),
        "average_full_rate_macro_f1_delta_vs_hard_gate": float(np.mean(full)),
        "average_selected_pooled_aurc_delta_vs_confidence": float(np.mean(reliability)),
    }


def run_v4_residual_gate_micro(project_root: str | Path, *, resume: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v4_residual_gate_development.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    screen = config["micro_screen"]
    if tuple(screen["datasets"]) != V4_MICRO_DATASETS or int(screen["seed"]) != V4_MICRO_SEED:
        raise ValueError("V4.1 frozen dataset/seed design changed")
    base_path = root / screen["base_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _hash(config_path, base_path)
    source_path = root / "reports" / "v4_observe_only_micro_report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("test_accessed") is not False or source.get("primary_split") != "validation_only":
        raise RuntimeError("V4.0 control source is not leakage-locked")
    run_root = root / "runs" / "v4_residual_gate_micro" / f"v4_residual_gate__seed17__{protocol_hash[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)
    deadline = started + float(screen["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    initialization_checks: dict[str, bool] = {}
    for dataset_id in V4_MICRO_DATASETS:
        dataset = load_development_dataset(root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz")
        initial_state, exact = _shared_initialization_exact(dataset, base_config)
        initialization_checks[dataset_id] = exact
        if not exact:
            raise RuntimeError(f"shared initialization mismatch for {dataset_id}")
        metrics_path = run_root / dataset_id / "v4_1_residual_gate" / "metrics.json"
        if resume and metrics_path.exists():
            candidate_result = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            model, history = _train_variant(
                dataset, base_config, screen, "v4_1_residual_gate", initial_state,
                protocol_hash, metrics_path.parent, device, deadline, resume,
            )
            validation = _evaluate_validation(
                model, dataset, base_config,
                tuple(float(value) for value in screen["validation_rate_ratios"]), device,
            )
            use_observability = validation["pooled_observability_aurc"] <= validation["pooled_confidence_aurc"]
            validation["reliability_mode"] = "observability" if use_observability else "confidence_fallback"
            validation["selected_pooled_aurc"] = (
                validation["pooled_observability_aurc"] if use_observability
                else validation["pooled_confidence_aurc"]
            )
            validation["learned_gate_floor"] = model.gate_floor.detach().cpu().tolist()
            candidate_result = {
                "protocol_hash": protocol_hash, "epochs_completed": len(history),
                "validation": validation, "test_accessed": False,
            }
            atomic_write_json(metrics_path, candidate_result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[dataset_id] = {
            "v3_10_hard_gate_control": source["results"][dataset_id]["v3_10_hard_gate"],
            "v4_1_residual_gate": candidate_result,
        }
    decision = _decision(results, screen["development_gates"])
    report = {
        "status": "completed_candidate_pass" if decision["passed"] else "completed_candidate_fail",
        "protocol_version": config["protocol_version"], "protocol_hash": protocol_hash,
        "adaptive_development": True,
        "adapted_after": "v4_observe_only_micro_completed_candidate_fail",
        "primary_split": "validation_only", "test_accessed": False,
        "independent_confirmation_claim_allowed": False,
        "minimum_new_untouched_confirmation_datasets": int(config["data_boundary"]["minimum_new_confirmation_datasets"]),
        "shared_parameter_initialization_exact": initialization_checks,
        "control_source_report": str(source_path), "control_source_protocol_hash": source["protocol_hash"],
        "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "results": results, "development_gates": screen["development_gates"],
        "decision": decision, "run_root": str(run_root),
        "pilot_started": False, "full_started": False, "finished_at_utc": utc_now(),
    }
    lines = [
        "# V4.1 residual-gate validation micro screen", "",
        f"- Status: **{'PASS' if decision['passed'] else 'FAIL'}** (adaptive development only)",
        f"- Device / elapsed: `{device}` / {report['elapsed_seconds']:.1f} s",
        "- Existing test arrays were not loaded or scored; the hard-gate control is reused from leakage-locked V4.0.",
        "- At least four new untouched datasets remain mandatory before any independent claim.", "",
        "| Dataset | Hard unseen F1 | V4.1 unseen F1 | Delta | Hard full F1 | V4.1 full F1 | Reliability mode | Gate-floor range |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset_id, row in results.items():
        hard = row["v3_10_hard_gate_control"]["validation"]
        candidate = row["v4_1_residual_gate"]["validation"]
        floors = candidate["learned_gate_floor"]
        lines.append(
            f"| {dataset_id} | {hard['mean_unseen_macro_f1']:.4f} | {candidate['mean_unseen_macro_f1']:.4f} | "
            f"{candidate['mean_unseen_macro_f1'] - hard['mean_unseen_macro_f1']:+.4f} | "
            f"{hard['full_rate_macro_f1']:.4f} | {candidate['full_rate_macro_f1']:.4f} | "
            f"{candidate['reliability_mode']} | {min(floors):.3f}-{max(floors):.3f} |"
        )
    lines.extend(["", "## Frozen gate decision", "", f"```json\n{json.dumps(decision, ensure_ascii=False, indent=2)}\n```", ""])
    markdown = "\n".join(lines)
    atomic_write_json(run_root / "report.json", report)
    _atomic_write_text(run_root / "report.md", markdown)
    atomic_write_json(root / "reports" / "v4_residual_gate_micro_report.json", report)
    _atomic_write_text(root / "reports" / "v4_residual_gate_micro_report.md", markdown)
    return report
