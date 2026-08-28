"""Targeted v3.9 low-rate exposure development on PAMAP2 seed42."""

from __future__ import annotations

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
from nyquistguard.experiments.v3_stability_development import _decision, _hash


def resolve_secondary_ratios(config: dict[str, Any]) -> list[float]:
    ratios = [float(value) for value in config["secondary_train_rate_ratios"]]
    if ratios != [0.75, 0.5, 0.3]:
        raise ValueError("v3.9 secondary-rate schedule must remain [0.75, 0.5, 0.3]")
    return ratios


def _run_rate_development(
    project_root: str | Path,
    resume: bool,
    *,
    config_filename: str,
    run_namespace: str,
    run_prefix: str,
    report_stem: str,
    title: str,
    rate_description: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / config_filename
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != "pamap2_uci" or int(config["seed"]) != 42:
        raise ValueError("targeted rate development must remain PAMAP2-seed42")
    base_path = root / config["base_config"]
    reliability_path = root / config["reliability_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if "secondary_train_rate_ratios" in config:
        base_config["train_rate_ratios"] = resolve_secondary_ratios(config)
    reliability_config = yaml.safe_load(reliability_path.read_text(encoding="utf-8"))
    protocol_hash = _hash(config_path, base_path, reliability_path)
    seed = int(config["seed"])
    dataset_id = str(config["dataset"])
    parent = root / "runs" / run_namespace
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = parent / f"{run_prefix}__pamap2__seed42__{stamp}"
    if resume and parent.exists():
        incomplete = [
            path
            for path in sorted(parent.glob(f"{run_prefix}__pamap2__seed42__*"), reverse=True)
            if (path / "pamap2_uci__seed42" / "checkpoint_last.pt").exists()
            and not (path / f"{report_stem}.json").exists()
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
        failed_report["results"]["pamap2_uci__seed42"]["summary"]["unseen_macro_f1_delta_vs_v1"]
    )
    candidate_validation = candidate_class["validation"]
    control_validation = control_class["validation"]
    reliability_validation = candidate_reliability["validation"]
    unseen_delta = float(
        candidate_validation["mean_unseen_macro_f1"]
        - control_validation["mean_unseen_macro_f1"]
    )
    summary = {
        "unseen_macro_f1_delta_vs_v1": unseen_delta,
        "unseen_macro_f1_improvement_vs_failed_v3_5": unseen_delta - failed_delta,
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
        "secondary_train_rate_ratios": base_config["train_rate_ratios"],
        "secondary_rate_sampling": config.get("secondary_rate_sampling"),
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
        title,
        "",
        "- Fixed primary view: 1.0 rate for every batch.",
        rate_description,
        "- Model, objective, optimizer, seed, epochs, and controller are unchanged.",
        f"- Elapsed: {elapsed:.1f} seconds; test is exploratory only.",
        f"- Frozen development decision: {'PASS' if decision['passed'] else 'FAIL'}.",
        "- Pilot and Full were not started.",
        "",
        f"- unseen F1 delta vs v1: {summary['unseen_macro_f1_delta_vs_v1']:+.4f}",
        f"- improvement vs failed v3.5: {summary['unseen_macro_f1_improvement_vs_failed_v3_5']:+.4f}",
        f"- full-rate F1 delta vs v1: {summary['full_rate_macro_f1_delta_vs_v1']:+.4f}",
        f"- reliability mode: {candidate_reliability['selected_mode']}",
        "",
        "## Decision checks",
        "",
    ]
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / f"{report_stem}.json", report)
    _atomic_write_text(run_root / f"{report_stem}.md", markdown)
    atomic_write_json(root / "reports" / f"{report_stem}.json", report)
    _atomic_write_text(root / "reports" / f"{report_stem}.md", markdown)
    return report


def run_v3_low_rate_development(
    project_root: str | Path, resume: bool = True
) -> dict[str, Any]:
    return _run_rate_development(
        project_root,
        resume,
        config_filename="v3_low_rate_development.yaml",
        run_namespace="v3_low_rate_development",
        run_prefix="v3_low_rate",
        report_stem="v3_low_rate_development_report",
        title="# NyquistGuard-TSC v3.9 Low-Rate Exposure Development",
        rate_description="- Frozen secondary schedule: [0.75, 0.5, 0.3]; no sweep was performed.",
    )


def run_v3_continuous_rate_development(
    project_root: str | Path, resume: bool = True
) -> dict[str, Any]:
    return _run_rate_development(
        project_root,
        resume,
        config_filename="v3_continuous_rate_development.yaml",
        run_namespace="v3_continuous_rate_development",
        run_prefix="v3_continuous_rate",
        report_stem="v3_continuous_rate_development_report",
        title="# NyquistGuard-TSC v3.10 Continuous-Rate Augmentation Development",
        rate_description=(
            "- Frozen secondary schedule: one identity slot plus two deterministic "
            "Uniform[0.3, 0.75] slots per three batches; no sweep was performed."
        ),
    )
