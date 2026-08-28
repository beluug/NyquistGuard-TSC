"""Validation-gated safe reliability controller for the v3.5 candidate."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.metrics import classification_metrics
from nyquistguard.experiments.pilot import _deep_model
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v3_anchored_reliability import confidence_anchored_score
from nyquistguard.experiments.v3_calibrated_reliability import (
    _collect_split,
    _fit_and_score,
)
from nyquistguard.experiments.v3_reliability import (
    _risk_and_coverage,
    threshold_for_target_coverage,
)


def select_reliability_mode(
    confidence_aurc: float,
    calibrated_aurc: float,
    minimum_absolute_gain: float,
) -> str:
    """Enable calibration only after an OOF validation improvement."""

    gain = float(confidence_aurc - calibrated_aurc)
    return "calibrated" if gain > float(minimum_absolute_gain) else "confidence"


def _aurc(split: dict[str, np.ndarray], score: np.ndarray) -> float:
    return float(
        classification_metrics(split["targets"], split["logits"], score)["aurc"]
    )


def _guarded_evaluation(
    model: torch.nn.Module,
    dataset: Any,
    base_config: dict[str, Any],
    reliability_config: dict[str, Any],
    seed: int,
    minimum_gain: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ratios = tuple(float(value) for value in reliability_config["rate_ratios"])
    calibration = reliability_config["calibrator"]
    target_coverage = float(
        reliability_config["threshold_calibration"]["pooled_validation_target_coverage"]
    )
    validation = _collect_split(
        model, dataset.validation, dataset.sampling_rate_hz, ratios, base_config, device
    )
    test = _collect_split(
        model, dataset.test, dataset.sampling_rate_hz, ratios, base_config, device
    )
    validation_raw, test_raw, finite = _fit_and_score(
        validation,
        test,
        float(calibration["regularization_c"]),
        int(calibration["group_folds"]),
        seed,
    )
    groups = len(np.unique(validation["groups"]))
    pseudo = float(calibration["shrinkage_pseudo_groups"])
    validation_calibrated, weight = confidence_anchored_score(
        validation["confidence"], validation_raw, groups, pseudo
    )
    test_calibrated, _ = confidence_anchored_score(
        test["confidence"], test_raw, groups, pseudo
    )
    confidence_aurc = _aurc(validation, validation["confidence"])
    calibrated_aurc = _aurc(validation, validation_calibrated)
    mode = select_reliability_mode(confidence_aurc, calibrated_aurc, minimum_gain)
    validation_selected = (
        validation_calibrated if mode == "calibrated" else validation["confidence"]
    )
    test_selected = test_calibrated if mode == "calibrated" else test["confidence"]
    threshold = threshold_for_target_coverage(validation_selected, target_coverage)
    confidence_threshold = threshold_for_target_coverage(
        validation["confidence"], target_coverage
    )
    return {
        "selected_mode": mode,
        "calibrator_finite": finite,
        "independent_validation_groups": groups,
        "shrinkage_weight": weight,
        "validation": {
            "confidence_aurc": confidence_aurc,
            "calibrated_aurc": calibrated_aurc,
            "selected_aurc": _aurc(validation, validation_selected),
            "selected_target": _risk_and_coverage(
                validation["correct"], validation_selected, threshold
            ),
            "confidence_target": _risk_and_coverage(
                validation["correct"], validation["confidence"], confidence_threshold
            ),
        },
        "test_exploratory": {
            "confidence_aurc": _aurc(test, test["confidence"]),
            "calibrated_aurc": _aurc(test, test_calibrated),
            "selected_aurc": _aurc(test, test_selected),
            "selected_target": _risk_and_coverage(
                test["correct"], test_selected, threshold
            ),
            "confidence_target": _risk_and_coverage(
                test["correct"], test["confidence"], confidence_threshold
            ),
        },
    }


def _decision(
    source: dict[str, Any], results: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    source_checks = source["decision"]["checks"]
    classification_pass = all(
        source_checks[name]
        for name in ("average_unseen_f1", "single_dataset_unseen_f1", "average_full_f1")
    )
    nonworse = all(
        row["validation"]["selected_aurc"]
        <= row["validation"]["confidence_aurc"] + 1e-12
        for row in results.values()
    )
    calibrated_count = sum(
        row["selected_mode"] == "calibrated" for row in results.values()
    )
    selected_vs_v1 = []
    risk_deltas = []
    for dataset_id, row in results.items():
        v1_aurc = source["results"][dataset_id]["v1_control_reliability"]["validation"][
            "pooled_calibrated_aurc"
        ]
        selected_vs_v1.append(row["validation"]["selected_aurc"] - v1_aurc)
        risk_deltas.append(
            row["validation"]["selected_target"]["risk"]
            - row["validation"]["confidence_target"]["risk"]
        )
    checks = {
        "source_classification": classification_pass
        if gates["require_source_classification_checks_passed"]
        else True,
        "nonworse_vs_confidence": nonworse
        if gates["require_nonworse_validation_aurc_vs_confidence_all_datasets"]
        else True,
        "calibrator_used": calibrated_count
        >= int(gates["minimum_calibrated_dataset_count"]),
        "average_vs_v1": float(np.mean(selected_vs_v1))
        <= float(gates["maximum_average_selected_aurc_delta_vs_v1"]),
        "target_risk": float(np.mean(risk_deltas))
        <= float(gates["maximum_average_target_risk_delta_vs_confidence"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "calibrated_dataset_count": calibrated_count,
        "average_selected_aurc_delta_vs_v1": float(np.mean(selected_vs_v1)),
        "average_target_risk_delta_vs_confidence": float(np.mean(risk_deltas)),
    }


def run_v3_guarded_reliability(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config = yaml.safe_load(
        (root / "configs" / "experiments" / "v3_guarded_reliability.yaml").read_text(
            encoding="utf-8"
        )
    )
    source = json.loads((root / config["source_core_report"]).read_text(encoding="utf-8"))
    base_config = yaml.safe_load(
        (root / config["base_config"]).read_text(encoding="utf-8")
    )
    reliability_config = yaml.safe_load(
        (root / config["reliability_config"]).read_text(encoding="utf-8")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    minimum_gain = float(
        config["controller"]["minimum_absolute_aurc_gain_to_enable_calibrator"]
    )
    results: dict[str, Any] = {}
    source_root = Path(source["run_root"])
    for dataset_id in config["datasets"]:
        if time.monotonic() - started > float(config["wall_time_budget_seconds"]):
            raise TimeoutError("v3.5 guarded reliability wall-time budget exceeded")
        dataset = load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        model = _deep_model(dataset, base_config, "no_selective_head", device)
        checkpoint = source_root / dataset_id / "checkpoint_final.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True), strict=True
        )
        model.eval()
        results[dataset_id] = _guarded_evaluation(
            model,
            dataset,
            base_config,
            reliability_config,
            int(config["seed"]),
            minimum_gain,
        )
        results[dataset_id]["checkpoint"] = str(checkpoint)
        del model, dataset
    decision = _decision(source, results, config["development_gates"])
    elapsed = time.monotonic() - started
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / "v3_guarded_reliability" / f"v3_guarded_reliability__seed17__{stamp}"
    report: dict[str, Any] = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "source_core_report": str(root / config["source_core_report"]),
        "primary_split": "grouped_out_of_fold_validation",
        "test_role": "exploratory_appendix",
        "elapsed_seconds": elapsed,
        "device": str(device),
        "pilot_started": False,
        "full_started": False,
        "results": results,
        "development_gates": config["development_gates"],
        "decision": decision,
        "run_root": str(run_root),
        "finished_at_utc": utc_now(),
    }
    lines = [
        "# NyquistGuard-TSC v3.5 Guarded Reliability Controller",
        "",
        "- grouped OOF validation 决定启用 anchored calibrator 或回退 max-softmax confidence。",
        "- 分类核心直接复用 v3.4 validation-selected checkpoint，不重新训练。",
        f"- 墙钟 {elapsed:.1f} 秒；主判据 validation；test 仅研发附录。",
        f"- 冻结开发门：{'PASS' if decision['passed'] else 'FAIL'}；不授权 Pilot/Full。",
        "",
        "| 数据集 | 选择模式 | confidence AURC | calibrated AURC | selected AURC | target risk Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset_id, row in results.items():
        validation = row["validation"]
        risk_delta = (
            validation["selected_target"]["risk"]
            - validation["confidence_target"]["risk"]
        )
        lines.append(
            f"| {dataset_id} | {row['selected_mode']} | "
            f"{validation['confidence_aurc']:.4f} | {validation['calibrated_aurc']:.4f} | "
            f"{validation['selected_aurc']:.4f} | {risk_delta:+.4f} |"
        )
    lines.extend(["", "## 决策检查", ""])
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "v3_guarded_reliability_report.json", report)
    _atomic_write_text(run_root / "v3_guarded_reliability_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_guarded_reliability_report.json", report)
    _atomic_write_text(root / "reports" / "v3_guarded_reliability_report.md", markdown)
    return report
