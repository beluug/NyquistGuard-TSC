"""Confidence-anchored Nyquist-aware reliability calibration for v3.3."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.pilot import _deep_model
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v3_calibrated_reliability import (
    _collect_split,
    _decision,
    _fit_and_score,
    _split_summary,
)
from nyquistguard.experiments.v3_reliability import threshold_for_target_coverage


def confidence_anchored_score(
    confidence: np.ndarray,
    calibrated_score: np.ndarray,
    independent_groups: int,
    pseudo_groups: float,
) -> tuple[np.ndarray, float]:
    """Shrink calibration log-odds toward confidence for small validation sets."""

    base = np.asarray(confidence, dtype=np.float64)
    calibrated = np.asarray(calibrated_score, dtype=np.float64)
    if base.shape != calibrated.shape:
        raise ValueError("confidence and calibrated_score must have matching shapes")
    if independent_groups <= 0 or pseudo_groups < 0:
        raise ValueError("group counts must be positive/non-negative")
    weight = float(independent_groups / (independent_groups + pseudo_groups))
    base_clipped = base.clip(1e-6, 1.0 - 1e-6)
    calibrated_clipped = calibrated.clip(1e-6, 1.0 - 1e-6)
    base_logit = np.log(base_clipped / (1.0 - base_clipped))
    calibrated_logit = np.log(calibrated_clipped / (1.0 - calibrated_clipped))
    blended_logit = base_logit + weight * (calibrated_logit - base_logit)
    return 1.0 / (1.0 + np.exp(-blended_logit)), weight


def run_v3_anchored_reliability(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config = yaml.safe_load(
        (root / "configs" / "experiments" / "v3_anchored_reliability.yaml").read_text(
            encoding="utf-8"
        )
    )
    base_config = yaml.safe_load(
        (root / config["base_config"]).read_text(encoding="utf-8")
    )
    pilot_root = _latest_completed_pilot(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratios = tuple(float(value) for value in config["rate_ratios"])
    calibrator_config = config["calibrator"]
    target_coverage = float(
        config["threshold_calibration"]["pooled_validation_target_coverage"]
    )
    pseudo_groups = float(calibrator_config["shrinkage_pseudo_groups"])
    budget = float(config["wall_time_budget_seconds"])
    results: dict[str, Any] = {}
    for dataset_id in config["datasets"]:
        dataset = load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        results[dataset_id] = {}
        for seed_value in config["seeds"]:
            if time.monotonic() - started > budget:
                raise TimeoutError("v3.3 anchored reliability wall-time budget exceeded")
            seed = int(seed_value)
            model = _deep_model(dataset, base_config, "nyquistguard", device)
            checkpoint = (
                pilot_root
                / f"{dataset_id}__nyquistguard__seed{seed}"
                / "checkpoint_best.pt"
            )
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True), strict=True
            )
            model.eval()
            validation = _collect_split(
                model,
                dataset.validation,
                dataset.sampling_rate_hz,
                ratios,
                base_config,
                device,
            )
            test = _collect_split(
                model,
                dataset.test,
                dataset.sampling_rate_hz,
                ratios,
                base_config,
                device,
            )
            validation_raw, test_raw, finite = _fit_and_score(
                validation,
                test,
                float(calibrator_config["regularization_c"]),
                int(calibrator_config["group_folds"]),
                seed,
            )
            independent_groups = len(np.unique(validation["groups"]))
            validation_score, weight = confidence_anchored_score(
                validation["confidence"],
                validation_raw,
                independent_groups,
                pseudo_groups,
            )
            test_score, test_weight = confidence_anchored_score(
                test["confidence"], test_raw, independent_groups, pseudo_groups
            )
            if abs(weight - test_weight) > 1e-12:
                raise RuntimeError("validation/test shrinkage weights diverged")
            score_threshold = threshold_for_target_coverage(
                validation_score, target_coverage
            )
            confidence_threshold = threshold_for_target_coverage(
                validation["confidence"], target_coverage
            )
            results[dataset_id][str(seed)] = {
                "checkpoint": str(checkpoint),
                "independent_validation_groups": independent_groups,
                "shrinkage_weight": weight,
                "calibrator_finite": finite,
                "validation_score_threshold": score_threshold,
                "validation_confidence_threshold": confidence_threshold,
                "validation": _split_summary(
                    validation,
                    validation_score,
                    score_threshold,
                    confidence_threshold,
                ),
                "test_exploratory": _split_summary(
                    test, test_score, score_threshold, confidence_threshold
                ),
            }
            del model
        del dataset
    decision = _decision(results, config["development_gates"])
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "calibrator": calibrator_config,
        "primary_split": "group_cross_fitted_validation",
        "test_role": "exploratory_appendix",
        "target_coverage": target_coverage,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "classification_model_trained": False,
        "calibrator_fitted": True,
        "pilot_started": False,
        "full_started": False,
        "results": results,
        "development_gates": config["development_gates"],
        "decision": decision,
        "finished_at_utc": utc_now(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        root
        / "runs"
        / "v3_anchored_reliability"
        / f"v3_anchored_reliability__3seeds__{stamp}"
    )
    report["run_root"] = str(run_root)
    lines = [
        "# NyquistGuard-TSC v3.3 Confidence-Anchored Reliability",
        "",
        "- 方法：v3.2 correctness calibrator 的 log-odds 向 max-softmax confidence 收缩。",
        f"- 固定收缩伪样本组：{pseudo_groups:g}；主判据仍为 grouped OOF validation。",
        "- test 仅为研发附录；分类模型不训练、不改 logits。",
        f"- 4 数据集 × 3 seeds；墙钟 {elapsed:.1f} 秒。",
        f"- 冻结开发门：{'PASS' if decision['passed'] else 'FAIL'}；不授权 Full。",
        "",
        "| 数据集 | OOF validation AURC 相对下降 | 改善 seeds | 80% coverage risk 差值 |",
        "|---|---:|---:|---:|",
    ]
    for dataset_id, row in decision["dataset_summary"].items():
        first = next(iter(results[dataset_id].values()))
        lines.append(
            f"| {dataset_id} | {row['mean_pooled_aurc_relative_reduction'] * 100:+.2f}% | "
            f"{row['improved_seed_count']}/3 | "
            f"{row['mean_selective_risk_delta_at_target_coverage']:+.4f} |"
        )
        lines.append(
            f"<!-- {dataset_id}: groups={first['independent_validation_groups']}, "
            f"weight={first['shrinkage_weight']:.4f} -->"
        )
    lines.extend(
        [
            "",
            f"总体 OOF validation AURC 相对下降：{decision['overall_mean_relative_reduction'] * 100:+.2f}% 。",
            "",
            "## 决策检查",
            "",
        ]
    )
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "v3_anchored_reliability_report.json", report)
    _atomic_write_text(run_root / "v3_anchored_reliability_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_anchored_reliability_report.json", report)
    _atomic_write_text(root / "reports" / "v3_anchored_reliability_report.md", markdown)
    return report
