"""Bounded seeds 42/2026 confirmation for the v3.5 end-to-end candidate."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.pilot import _deep_model, _seed_everything
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v2_micro_pilot import _evaluate_model
from nyquistguard.experiments.v3_core_micro import (
    _anchored_reliability,
    _train_core,
)
from nyquistguard.experiments.v3_guarded_reliability import _guarded_evaluation


def _protocol_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(b"v3-multiseed-confirmation-v1")
    return digest.hexdigest()


def _validate_matrix(config: dict[str, Any]) -> list[tuple[str, int]]:
    datasets = tuple(config["datasets"])
    seeds = tuple(int(value) for value in config["confirmation_seeds"])
    matrix = [(str(dataset), int(seed)) for dataset, seed in config["run_order"]]
    expected = {(dataset, seed) for dataset in datasets for seed in seeds}
    if set(matrix) != expected or len(matrix) != len(expected):
        raise ValueError("confirmation run_order must contain each dataset×seed once")
    if int(config["development_seed_excluded_from_primary_gate"]) in seeds:
        raise ValueError("development seed17 must remain excluded from confirmation")
    return matrix


def _run_row(
    candidate_classification: dict[str, Any],
    control_classification: dict[str, Any],
    candidate_reliability: dict[str, Any],
    control_reliability: dict[str, Any],
) -> dict[str, float | str]:
    candidate_validation = candidate_classification["validation"]
    control_validation = control_classification["validation"]
    guarded_validation = candidate_reliability["validation"]
    control_reliability_validation = control_reliability["validation"]
    return {
        "unseen_macro_f1_delta_vs_v1": float(
            candidate_validation["mean_unseen_macro_f1"]
            - control_validation["mean_unseen_macro_f1"]
        ),
        "full_rate_macro_f1_delta_vs_v1": float(
            candidate_validation["full_rate_macro_f1"]
            - control_validation["full_rate_macro_f1"]
        ),
        "selected_mode": candidate_reliability["selected_mode"],
        "selected_aurc_delta_vs_confidence": float(
            guarded_validation["selected_aurc"] - guarded_validation["confidence_aurc"]
        ),
        "selected_aurc_delta_vs_v1": float(
            guarded_validation["selected_aurc"]
            - control_reliability_validation["pooled_calibrated_aurc"]
        ),
        "target_risk_delta_vs_confidence": float(
            guarded_validation["selected_target"]["risk"]
            - guarded_validation["confidence_target"]["risk"]
        ),
    }


def _decision(rows: dict[str, dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    values = list(rows.values())
    unseen = [float(row["unseen_macro_f1_delta_vs_v1"]) for row in values]
    full = [float(row["full_rate_macro_f1_delta_vs_v1"]) for row in values]
    selected_confidence = [
        float(row["selected_aurc_delta_vs_confidence"]) for row in values
    ]
    selected_v1 = [float(row["selected_aurc_delta_vs_v1"]) for row in values]
    risk = [float(row["target_risk_delta_vs_confidence"]) for row in values]
    calibrated_count = sum(row["selected_mode"] == "calibrated" for row in values)
    checks = {
        "average_unseen_f1": float(np.mean(unseen))
        >= float(gates["minimum_average_unseen_macro_f1_delta_vs_v1"]),
        "positive_unseen_f1_runs": sum(value > 0.0 for value in unseen)
        >= int(gates["minimum_positive_unseen_macro_f1_run_count"]),
        "single_run_unseen_f1": float(np.min(unseen))
        >= -float(gates["maximum_single_run_unseen_macro_f1_drop"]),
        "average_full_f1": float(np.mean(full))
        >= -float(gates["maximum_average_full_rate_macro_f1_drop"]),
        "guarded_nonworse_confidence": all(value <= 1e-12 for value in selected_confidence)
        if gates["require_guarded_aurc_nonworse_vs_confidence_all_runs"]
        else True,
        "calibrator_used": calibrated_count >= int(gates["minimum_calibrated_run_count"]),
        "average_selected_vs_v1": float(np.mean(selected_v1))
        <= float(gates["maximum_average_selected_aurc_delta_vs_v1"]),
        "average_target_risk": float(np.mean(risk))
        <= float(gates["maximum_average_target_risk_delta_vs_confidence"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "average_unseen_macro_f1_delta_vs_v1": float(np.mean(unseen)),
        "positive_unseen_macro_f1_run_count": int(sum(value > 0.0 for value in unseen)),
        "minimum_run_unseen_macro_f1_delta_vs_v1": float(np.min(unseen)),
        "average_full_rate_macro_f1_delta_vs_v1": float(np.mean(full)),
        "calibrated_run_count": calibrated_count,
        "average_selected_aurc_delta_vs_v1": float(np.mean(selected_v1)),
        "average_target_risk_delta_vs_confidence": float(np.mean(risk)),
    }


def run_v3_multiseed_confirmation(
    project_root: str | Path, resume: bool = True
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v3_multiseed_confirmation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = _validate_matrix(config)
    base_path = root / config["base_config"]
    reliability_path = root / config["reliability_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    reliability_config = yaml.safe_load(reliability_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path, base_path, reliability_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        root
        / "runs"
        / "v3_multiseed_confirmation"
        / f"v3_multiseed_confirmation__2datasets__2seeds__{stamp}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    deadline = started + float(config["wall_time_budget_seconds"])
    pilot_root = _latest_completed_pilot(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    for dataset_id, seed in matrix:
        if time.monotonic() >= deadline:
            raise TimeoutError("v3.5 multiseed confirmation wall-time budget exceeded")
        run_key = f"{dataset_id}__seed{seed}"
        dataset = load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        _seed_everything(seed)
        candidate, history = _train_core(
            dataset,
            base_config,
            config,
            protocol_hash,
            run_root / run_key,
            seed=seed,
            resume=resume,
            deadline=deadline,
        )
        candidate.eval()
        control = _deep_model(dataset, base_config, "nyquistguard", device)
        control_path = (
            pilot_root / f"{dataset_id}__nyquistguard__seed{seed}" / "checkpoint_best.pt"
        )
        control.load_state_dict(
            torch.load(control_path, map_location=device, weights_only=True), strict=True
        )
        control.eval()
        candidate_classification = _evaluate_model(candidate, dataset, base_config, device)
        control_classification = _evaluate_model(control, dataset, base_config, device)
        candidate_reliability = _guarded_evaluation(
            candidate,
            dataset,
            base_config,
            reliability_config,
            seed,
            float(
                yaml.safe_load(
                    (root / "configs" / "experiments" / "v3_guarded_reliability.yaml").read_text(
                        encoding="utf-8"
                    )
                )["controller"]["minimum_absolute_aurc_gain_to_enable_calibrator"]
            ),
        )
        control_reliability = _anchored_reliability(
            control, dataset, base_config, reliability_config, seed
        )
        row = _run_row(
            candidate_classification,
            control_classification,
            candidate_reliability,
            control_reliability,
        )
        rows[run_key] = row
        results[run_key] = {
            "dataset_id": dataset_id,
            "seed": seed,
            "epochs_completed": len(history),
            "candidate_classification": candidate_classification,
            "v1_control_classification": control_classification,
            "candidate_reliability": candidate_reliability,
            "v1_control_reliability": control_reliability,
            "summary": row,
            "v1_control_checkpoint": str(control_path),
        }
        atomic_write_json(run_root / run_key / "metrics.json", results[run_key])
        del candidate, control, dataset
    decision = _decision(rows, config["confirmation_gates"])
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "protocol_hash": protocol_hash,
        "primary_units": "datasets×confirmation_seeds_42_2026",
        "development_seed17_in_primary_gate": False,
        "primary_split": "validation",
        "test_role": "exploratory_appendix",
        "elapsed_seconds": elapsed,
        "device": str(device),
        "pilot_started": False,
        "full_started": False,
        "results": results,
        "confirmation_gates": config["confirmation_gates"],
        "decision": decision,
        "run_root": str(run_root),
        "finished_at_utc": utc_now(),
    }
    lines = [
        "# NyquistGuard-TSC v3.5 Multi-seed Confirmation",
        "",
        "- 主确认单元：BasicMotions/PAMAP2 × seeds 42/2026；seed17 不进入主门。",
        "- 训练、validation checkpoint、anchored calibrator 与 safety controller 均保持冻结。",
        f"- 墙钟 {elapsed:.1f} 秒；test 仅研发附录。",
        f"- 确认门：{'PASS' if decision['passed'] else 'FAIL'}；不授权 Pilot/Full。",
        "",
        "| run | unseen F1 Δ vs v1 | full F1 Δ vs v1 | reliability mode | selected AURC Δ vs v1 | target risk Δ |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for run_key, row in rows.items():
        lines.append(
            f"| {run_key} | {row['unseen_macro_f1_delta_vs_v1']:+.4f} | "
            f"{row['full_rate_macro_f1_delta_vs_v1']:+.4f} | {row['selected_mode']} | "
            f"{row['selected_aurc_delta_vs_v1']:+.4f} | "
            f"{row['target_risk_delta_vs_confidence']:+.4f} |"
        )
    lines.extend(["", "## 决策检查", ""])
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "v3_multiseed_confirmation_report.json", report)
    _atomic_write_text(run_root / "v3_multiseed_confirmation_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_multiseed_confirmation_report.json", report)
    _atomic_write_text(root / "reports" / "v3_multiseed_confirmation_report.md", markdown)
    return report
