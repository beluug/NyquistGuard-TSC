"""Fixed two-checkpoint prediction ensemble for v3.8 stability development."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.pilot import _deep_model
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v2_micro_pilot import _evaluate_model
from nyquistguard.experiments.v3_core_micro import _anchored_reliability
from nyquistguard.experiments.v3_guarded_reliability import _guarded_evaluation
from nyquistguard.experiments.v3_spectral_reliability import sample_retained_band_energy
from nyquistguard.experiments.v3_stability_development import _decision


class FixedLogitEnsemble(nn.Module):
    """Average two frozen predictions and their physical retention summaries."""

    def __init__(
        self,
        first: nn.Module,
        second: nn.Module,
        coefficient_first: float = 0.5,
        coefficient_second: float = 0.5,
    ) -> None:
        super().__init__()
        if coefficient_first < 0 or coefficient_second < 0:
            raise ValueError("ensemble coefficients must be non-negative")
        total = float(coefficient_first + coefficient_second)
        if total <= 0:
            raise ValueError("ensemble coefficients cannot both be zero")
        self.first = first
        self.second = second
        self.coefficient_first = float(coefficient_first / total)
        self.coefficient_second = float(coefficient_second / total)

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        left = self.first(*args, **kwargs)
        right = self.second(*args, **kwargs)
        alpha = self.coefficient_first
        beta = self.coefficient_second
        left_retention = sample_retained_band_energy(
            left["band_features"], left["gated_band_features"]
        )
        right_retention = sample_retained_band_energy(
            right["band_features"], right["gated_band_features"]
        )
        return {
            "logits": left["logits"] * alpha + right["logits"] * beta,
            "accept_probability": (
                left["accept_probability"] * alpha
                + right["accept_probability"] * beta
            ),
            "retained_band_energy": left_retention * alpha + right_retention * beta,
            # Kept for compatibility with callers that inspect the physical tensors.
            "band_features": left["band_features"],
            "gated_band_features": left["gated_band_features"],
        }


def run_v3_logit_ensemble(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config = yaml.safe_load(
        (root / "configs" / "experiments" / "v3_logit_ensemble.yaml").read_text(
            encoding="utf-8"
        )
    )
    dataset_id = str(config["dataset"])
    seed = int(config["seed"])
    if dataset_id != "pamap2_uci" or seed != 42:
        raise ValueError("v3.8 must remain targeted at PAMAP2-seed42")

    source = json.loads(
        (root / config["source_stability_report"]).read_text(encoding="utf-8")
    )
    failed = json.loads(
        (root / config["source_failed_confirmation_report"]).read_text(encoding="utf-8")
    )
    base_config = yaml.safe_load((root / config["base_config"]).read_text(encoding="utf-8"))
    reliability_config = yaml.safe_load(
        (root / config["reliability_config"]).read_text(encoding="utf-8")
    )
    guarded_config = yaml.safe_load(
        (root / "configs" / "experiments" / "v3_guarded_reliability.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_dir = Path(source["run_root"]) / f"{dataset_id}__seed{seed}"
    ensemble_config = config["prediction_ensemble"]
    first_state = torch.load(
        source_dir / ensemble_config["checkpoint_a"], map_location="cpu", weights_only=True
    )
    last_payload = torch.load(
        source_dir / ensemble_config["checkpoint_b"], map_location="cpu", weights_only=False
    )
    second_state = last_payload["model_state_dict"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_prepared_dataset(
        root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
    )
    first = _deep_model(dataset, base_config, "no_selective_head", device)
    second = _deep_model(dataset, base_config, "no_selective_head", device)
    first.load_state_dict(first_state, strict=True)
    second.load_state_dict(second_state, strict=True)
    candidate = FixedLogitEnsemble(
        first,
        second,
        float(ensemble_config["coefficient_a"]),
        float(ensemble_config["coefficient_b"]),
    ).to(device)
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
    failed_delta = float(
        failed["results"]["pamap2_uci__seed42"]["summary"]["unseen_macro_f1_delta_vs_v1"]
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / "v3_logit_ensemble" / f"v3_logit_ensemble__pamap2__seed42__{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "dataset": dataset_id,
        "seed": seed,
        "prediction_ensemble": ensemble_config,
        "primary_split": "validation",
        "test_role": "exploratory_appendix",
        "elapsed_seconds": elapsed,
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
        "# NyquistGuard-TSC v3.8 Fixed Logit Ensemble",
        "",
        "- Candidate: fixed 0.5 x rate-robust best + 0.5 x epoch-30 final logits.",
        "- Coefficients were frozen; no search was performed; test is exploratory only.",
        f"- Elapsed: {elapsed:.1f} seconds.",
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
    atomic_write_json(run_root / "v3_logit_ensemble_report.json", report)
    _atomic_write_text(run_root / "v3_logit_ensemble_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_logit_ensemble_report.json", report)
    _atomic_write_text(root / "reports" / "v3_logit_ensemble_report.md", markdown)
    return report
