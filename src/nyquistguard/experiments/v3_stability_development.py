"""Targeted v3.6 rate-robust checkpoint development on PAMAP2 seed42."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.pilot import _deep_model, _seed_everything
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v2_micro_pilot import _evaluate_model
from nyquistguard.experiments.v3_core_micro import _anchored_reliability, _train_core
from nyquistguard.experiments.v3_guarded_reliability import _guarded_evaluation


def _hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(b"v3-stability-development-v1")
    return digest.hexdigest()


def _decision(
    summary: dict[str, float], gates: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "unseen_vs_v1": summary["unseen_macro_f1_delta_vs_v1"]
        >= float(gates["minimum_unseen_macro_f1_delta_vs_v1"]),
        "unseen_vs_failed_v3_5": summary["unseen_macro_f1_improvement_vs_failed_v3_5"]
        >= float(gates["minimum_unseen_macro_f1_improvement_vs_v3_5_failed_run"]),
        "full_vs_v1": summary["full_rate_macro_f1_delta_vs_v1"]
        >= float(gates["minimum_full_rate_macro_f1_delta_vs_v1"]),
        "guarded_nonworse_confidence": summary["selected_aurc_delta_vs_confidence"] <= 1e-12
        if gates["require_guarded_aurc_nonworse_vs_confidence"]
        else True,
        "selected_vs_v1": summary["selected_aurc_delta_vs_v1"]
        <= float(gates["maximum_selected_aurc_delta_vs_v1"]),
        "target_risk": summary["target_risk_delta_vs_confidence"]
        <= float(gates["maximum_target_risk_delta_vs_confidence"]),
    }
    return {"passed": all(checks.values()), "checks": checks, **summary}


def run_v3_stability_development(project_root: str | Path, resume: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v3_stability_development.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != "pamap2_uci" or int(config["seed"]) != 42:
        raise ValueError("v3.6 targeted development must remain PAMAP2-seed42")
    base_path = root / config["base_config"]
    reliability_path = root / config["reliability_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    reliability_config = yaml.safe_load(reliability_path.read_text(encoding="utf-8"))
    protocol_hash = _hash(config_path, base_path, reliability_path)
    seed = int(config["seed"])
    dataset_id = str(config["dataset"])
    parent = root / "runs" / "v3_stability_development"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = parent / f"v3_stability__pamap2__seed42__{stamp}"
    if resume and parent.exists():
        incomplete = [
            path
            for path in sorted(parent.glob("v3_stability__pamap2__seed42__*"), reverse=True)
            if (path / "pamap2_uci__seed42" / "checkpoint_last.pt").exists()
            and not (path / "v3_stability_development_report.json").exists()
        ]
        if incomplete:
            run_root = incomplete[0]
    run_root.mkdir(parents=True, exist_ok=True)
    deadline = started + float(config["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_prepared_dataset(
        root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
    )
    _seed_everything(seed)
    candidate, history = _train_core(
        dataset,
        base_config,
        config,
        protocol_hash,
        run_root / f"{dataset_id}__seed{seed}",
        seed=seed,
        resume=resume,
        deadline=deadline,
    )
    candidate.eval()
    pilot_root = _latest_completed_pilot(root)
    control_path = pilot_root / f"{dataset_id}__nyquistguard__seed{seed}" / "checkpoint_best.pt"
    control = _deep_model(dataset, base_config, "nyquistguard", device)
    control.load_state_dict(
        torch.load(control_path, map_location=device, weights_only=True), strict=True
    )
    control.eval()
    candidate_class = _evaluate_model(candidate, dataset, base_config, device)
    control_class = _evaluate_model(control, dataset, base_config, device)
    guarded_config = yaml.safe_load(
        (root / "configs" / "experiments" / "v3_guarded_reliability.yaml").read_text(
            encoding="utf-8"
        )
    )
    candidate_reliability = _guarded_evaluation(
        candidate,
        dataset,
        base_config,
        reliability_config,
        seed,
        float(guarded_config["controller"]["minimum_absolute_aurc_gain_to_enable_calibrator"]),
    )
    control_reliability = _anchored_reliability(
        control, dataset, base_config, reliability_config, seed
    )
    failed_report = json.loads(
        (root / config["source_failed_confirmation_report"]).read_text(encoding="utf-8")
    )
    failed_delta = float(
        failed_report["results"]["pamap2_uci__seed42"]["summary"][
            "unseen_macro_f1_delta_vs_v1"
        ]
    )
    candidate_validation = candidate_class["validation"]
    control_validation = control_class["validation"]
    reliability_validation = candidate_reliability["validation"]
    summary = {
        "unseen_macro_f1_delta_vs_v1": float(
            candidate_validation["mean_unseen_macro_f1"]
            - control_validation["mean_unseen_macro_f1"]
        ),
        "unseen_macro_f1_improvement_vs_failed_v3_5": float(
            candidate_validation["mean_unseen_macro_f1"]
            - control_validation["mean_unseen_macro_f1"]
            - failed_delta
        ),
        "full_rate_macro_f1_delta_vs_v1": float(
            candidate_validation["full_rate_macro_f1"]
            - control_validation["full_rate_macro_f1"]
        ),
        "selected_aurc_delta_vs_confidence": float(
            reliability_validation["selected_aurc"]
            - reliability_validation["confidence_aurc"]
        ),
        "selected_aurc_delta_vs_v1": float(
            reliability_validation["selected_aurc"]
            - control_reliability["validation"]["pooled_calibrated_aurc"]
        ),
        "target_risk_delta_vs_confidence": float(
            reliability_validation["selected_target"]["risk"]
            - reliability_validation["confidence_target"]["risk"]
        ),
    }
    decision = _decision(summary, config["development_gates"])
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "protocol_hash": protocol_hash,
        "dataset": dataset_id,
        "seed": seed,
        "epochs_completed": len(history),
        "checkpoint_selection": config["checkpoint_selection"],
        "primary_split": "validation",
        "test_role": "exploratory_appendix",
        "elapsed_seconds": elapsed,
        "device": str(device),
        "candidate_classification": candidate_class,
        "v1_control_classification": control_class,
        "candidate_reliability": candidate_reliability,
        "v1_control_reliability": control_reliability,
        "development_gates": config["development_gates"],
        "decision": decision,
        "run_root": str(run_root),
        "pilot_started": False,
        "full_started": False,
        "finished_at_utc": utc_now(),
    }
    lines = [
        "# NyquistGuard-TSC v3.6 Rate-Robust Checkpoint Development",
        "",
        "- 目标：只修复 PAMAP2-seed42 分类 seed 稳定性。",
        "- 唯一改动：checkpoint score = 0.5×full F1 + 0.5×mean unseen-rate F1。",
        f"- 墙钟 {elapsed:.1f} 秒；test 仅研发附录。",
        f"- 冻结开发门：{'PASS' if decision['passed'] else 'FAIL'}；不授权 Pilot/Full。",
        "",
        f"- unseen F1 Δ vs v1: {summary['unseen_macro_f1_delta_vs_v1']:+.4f}",
        f"- unseen F1 improvement vs failed v3.5: {summary['unseen_macro_f1_improvement_vs_failed_v3_5']:+.4f}",
        f"- full-rate F1 Δ vs v1: {summary['full_rate_macro_f1_delta_vs_v1']:+.4f}",
        f"- reliability mode: {candidate_reliability['selected_mode']}",
        "",
        "## 决策检查",
        "",
    ]
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "v3_stability_development_report.json", report)
    _atomic_write_text(run_root / "v3_stability_development_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_stability_development_report.json", report)
    _atomic_write_text(root / "reports" / "v3_stability_development_report.md", markdown)
    return report
