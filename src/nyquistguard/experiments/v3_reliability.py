"""Three-seed, no-training development probe for Nyquist Reliability Score."""

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
from nyquistguard.experiments.deterministic_selector_probe import _gate_quality
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.metrics import classification_metrics
from nyquistguard.experiments.pilot import _deep_model, _predict_deep
from nyquistguard.experiments.progress import atomic_write_json, utc_now


def nyquist_reliability_score(
    confidence: np.ndarray,
    gate_quality: float | np.ndarray,
    confidence_exponent: float = 1.0,
    gate_quality_exponent: float = 1.0,
) -> np.ndarray:
    """Combine sample confidence and rate-level usable spectral capacity."""

    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("confidence must be a finite 1-D array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("confidence must lie in [0, 1]")
    quality = np.asarray(gate_quality, dtype=np.float64)
    if np.any(~np.isfinite(quality)) or np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("gate_quality must lie in [0, 1]")
    try:
        return np.power(values, confidence_exponent) * np.power(
            quality, gate_quality_exponent
        )
    except ValueError as error:
        raise ValueError("gate_quality must be scalar or match confidence") from error


def threshold_for_target_coverage(scores: np.ndarray, target_coverage: float) -> float:
    """Return a deterministic validation-only threshold near target coverage."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite 1-D array")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must lie in (0, 1]")
    rank = max(0, min(len(values) - 1, int(np.ceil(target_coverage * len(values))) - 1))
    return float(np.sort(values)[::-1][rank])


def _risk_and_coverage(
    correct: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    selected = np.asarray(scores) >= float(threshold)
    return {
        "coverage": float(selected.mean()),
        "risk": float(1.0 - np.asarray(correct)[selected].mean())
        if selected.any()
        else 1.0,
    }


def _split_scores(
    model: torch.nn.Module,
    split: Any,
    source_rate_hz: float,
    ratios: tuple[float, ...],
    base_config: dict[str, Any],
    device: torch.device,
    confidence_exponent: float,
    gate_quality_exponent: float,
) -> dict[str, Any]:
    pooled_logits: list[np.ndarray] = []
    pooled_targets: list[np.ndarray] = []
    pooled_confidence: list[np.ndarray] = []
    pooled_nrs: list[np.ndarray] = []
    per_rate: dict[str, Any] = {}
    for ratio in ratios:
        logits, _ = _predict_deep(
            model, split, source_rate_hz, ratio, base_config, device
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        confidence = probabilities.max(axis=1)
        quality = _gate_quality(model, source_rate_hz, ratio)
        nrs = nyquist_reliability_score(
            confidence,
            quality,
            confidence_exponent,
            gate_quality_exponent,
        )
        rate_id = f"r{int(round(ratio * 1000)):04d}"
        per_rate[rate_id] = {
            "ratio": ratio,
            "gate_quality_vs_full": quality,
            "mean_confidence": float(confidence.mean()),
            "mean_nrs": float(nrs.mean()),
        }
        pooled_logits.append(logits)
        pooled_targets.append(np.asarray(split.y, dtype=np.int64))
        pooled_confidence.append(confidence)
        pooled_nrs.append(nrs)
    logits = np.concatenate(pooled_logits)
    targets = np.concatenate(pooled_targets)
    confidence = np.concatenate(pooled_confidence)
    nrs = np.concatenate(pooled_nrs)
    predictions = logits.argmax(axis=1)
    return {
        "logits": logits,
        "targets": targets,
        "correct": predictions == targets,
        "confidence": confidence,
        "nrs": nrs,
        "confidence_aurc": classification_metrics(targets, logits, confidence)["aurc"],
        "nrs_aurc": classification_metrics(targets, logits, nrs)["aurc"],
        "per_rate": per_rate,
    }


def _relative_reduction(candidate: float, control: float) -> float:
    return float((control - candidate) / max(abs(control), 1e-12))


def _decision(results: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    dataset_rows: dict[str, Any] = {}
    improved_seed_count = 0
    monotonic_all = True
    for dataset_id, seeds in results.items():
        validation_reductions = [
            row["validation"]["pooled_aurc_relative_reduction"]
            for row in seeds.values()
        ]
        risk_deltas = [
            row["validation"]["nrs_target"]["risk"]
            - row["validation"]["confidence_target"]["risk"]
            for row in seeds.values()
        ]
        improved_seed_count += sum(value > 0.0 for value in validation_reductions)
        monotonic_all = monotonic_all and all(
            row["validation"]["full_to_low_mean_nrs_drop"] > 0.0
            for row in seeds.values()
        )
        dataset_rows[dataset_id] = {
            "mean_pooled_aurc_relative_reduction": float(np.mean(validation_reductions)),
            "improved_seed_count": int(sum(value > 0.0 for value in validation_reductions)),
            "mean_selective_risk_delta_at_target_coverage": float(np.mean(risk_deltas)),
        }
    threshold = float(gates["minimum_dataset_mean_pooled_aurc_relative_reduction"])
    improved_datasets = sum(
        row["mean_pooled_aurc_relative_reduction"] > threshold
        for row in dataset_rows.values()
    )
    lower_risk_datasets = sum(
        row["mean_selective_risk_delta_at_target_coverage"] < 0.0
        for row in dataset_rows.values()
    )
    overall = float(
        np.mean(
            [row["mean_pooled_aurc_relative_reduction"] for row in dataset_rows.values()]
        )
    )
    checks = {
        "dataset_mean_aurc": improved_datasets
        >= int(gates["minimum_improved_dataset_count"]),
        "dataset_seed_directions": improved_seed_count
        >= int(gates["minimum_improved_dataset_seed_count"]),
        "overall_effect": overall
        >= float(gates["minimum_overall_mean_relative_reduction"]),
        "full_to_low_score_drop": monotonic_all
        if gates["require_full_to_low_score_drop_all_dataset_seeds"]
        else True,
        "selective_risk": lower_risk_datasets
        >= int(gates["minimum_dataset_count_with_lower_selective_risk_at_target_coverage"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dataset_summary": dataset_rows,
        "improved_dataset_count": improved_datasets,
        "improved_dataset_seed_count": improved_seed_count,
        "lower_risk_dataset_count": lower_risk_datasets,
        "overall_mean_relative_reduction": overall,
    }


def _public_split_summary(
    split_result: dict[str, Any], threshold: float, confidence_threshold: float
) -> dict[str, Any]:
    return {
        "pooled_confidence_aurc": split_result["confidence_aurc"],
        "pooled_nrs_aurc": split_result["nrs_aurc"],
        "pooled_aurc_relative_reduction": _relative_reduction(
            split_result["nrs_aurc"], split_result["confidence_aurc"]
        ),
        "nrs_target": _risk_and_coverage(
            split_result["correct"], split_result["nrs"], threshold
        ),
        "confidence_target": _risk_and_coverage(
            split_result["correct"],
            split_result["confidence"],
            confidence_threshold,
        ),
        "full_to_low_mean_nrs_drop": float(
            split_result["per_rate"]["r1000"]["mean_nrs"]
            - split_result["per_rate"]["r0300"]["mean_nrs"]
        ),
        "per_rate": split_result["per_rate"],
    }


def run_v3_reliability(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v3_reliability.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config = yaml.safe_load(
        (root / config["base_config"]).read_text(encoding="utf-8")
    )
    pilot_root = _latest_completed_pilot(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratios = (1.0,) + tuple(float(value) for value in config["unseen_rate_ratios"])
    target_coverage = float(
        config["threshold_calibration"]["pooled_validation_target_coverage"]
    )
    score_config = config["score"]
    budget = float(config["wall_time_budget_seconds"])
    results: dict[str, Any] = {}
    for dataset_id in config["datasets"]:
        dataset = load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        results[dataset_id] = {}
        for seed_value in config["seeds"]:
            if time.monotonic() - started > budget:
                raise TimeoutError("v3 reliability wall-time budget exceeded")
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
            validation = _split_scores(
                model,
                dataset.validation,
                dataset.sampling_rate_hz,
                ratios,
                base_config,
                device,
                float(score_config["confidence_exponent"]),
                float(score_config["gate_quality_exponent"]),
            )
            nrs_threshold = threshold_for_target_coverage(
                validation["nrs"], target_coverage
            )
            confidence_threshold = threshold_for_target_coverage(
                validation["confidence"], target_coverage
            )
            test = _split_scores(
                model,
                dataset.test,
                dataset.sampling_rate_hz,
                ratios,
                base_config,
                device,
                float(score_config["confidence_exponent"]),
                float(score_config["gate_quality_exponent"]),
            )
            results[dataset_id][str(seed)] = {
                "checkpoint": str(checkpoint),
                "validation_nrs_threshold": nrs_threshold,
                "validation_confidence_threshold": confidence_threshold,
                "validation": _public_split_summary(
                    validation, nrs_threshold, confidence_threshold
                ),
                "test_exploratory": _public_split_summary(
                    test, nrs_threshold, confidence_threshold
                ),
            }
            del model
        del dataset
    decision = _decision(results, config["development_gates"])
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "score": score_config,
        "primary_split": "validation",
        "test_role": "exploratory_appendix",
        "target_coverage": target_coverage,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "trained_models": False,
        "parameters_updated": False,
        "pilot_started": False,
        "full_started": False,
        "results": results,
        "development_gates": config["development_gates"],
        "decision": decision,
        "finished_at_utc": utc_now(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / "v3_reliability" / f"v3_reliability__3seeds__{stamp}"
    report["run_root"] = str(run_root)
    lines = [
        "# NyquistGuard-TSC v3 Nyquist Reliability Score",
        "",
        "- 公式：`NRS = max-softmax confidence × relative effective gate mass`。",
        "- 主判据：validation；test 仅为研发附录。",
        f"- 4 数据集 × 3 seeds；墙钟 {elapsed:.1f} 秒；无训练、无参数更新。",
        f"- 冻结开发门：{'PASS' if decision['passed'] else 'FAIL'}；只决定是否进入下一轮小型 v3 研发，不授权 Full。",
        "",
        "| 数据集 | validation pooled AURC 相对下降 | 改善 seeds | 80% coverage risk 差值 |",
        "|---|---:|---:|---:|",
    ]
    for dataset_id, row in decision["dataset_summary"].items():
        lines.append(
            f"| {dataset_id} | {row['mean_pooled_aurc_relative_reduction'] * 100:+.2f}% | "
            f"{row['improved_seed_count']}/3 | "
            f"{row['mean_selective_risk_delta_at_target_coverage']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"总体 validation pooled AURC 相对下降：{decision['overall_mean_relative_reduction'] * 100:+.2f}% 。",
            "",
            "## 决策检查",
            "",
        ]
    )
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "分类 logits 与预测未被修改；NRS 仅改变跨 rate 的可靠性排序和 validation 阈值。",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "v3_reliability_report.json", report)
    _atomic_write_text(run_root / "v3_reliability_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_reliability_report.json", report)
    _atomic_write_text(root / "reports" / "v3_reliability_report.md", markdown)
    return report
