"""Cross-fitted Nyquist-aware correctness calibration for v3.2 development."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.metrics import classification_metrics
from nyquistguard.experiments.pilot import _deep_model
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v3_reliability import (
    _relative_reduction,
    _risk_and_coverage,
    threshold_for_target_coverage,
)
from nyquistguard.experiments.v3_spectral_reliability import _predict_with_retention


def reliability_features(
    logits: np.ndarray, retention: np.ndarray, rate_ratio: float
) -> np.ndarray:
    """Fixed v3.2 feature map; all features are available at inference time."""

    values = np.asarray(logits, dtype=np.float64)
    retained = np.asarray(retention, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(probabilities * np.log(probabilities.clip(1e-12))).sum(axis=1)
    normalized_entropy = entropy / np.log(probabilities.shape[1])
    confidence_logit = np.log(
        confidence.clip(1e-6, 1.0 - 1e-6)
        / (1.0 - confidence.clip(1e-6, 1.0 - 1e-6))
    )
    return np.column_stack(
        [
            confidence_logit,
            normalized_entropy,
            margin,
            retained,
            np.full(len(values), np.log(float(rate_ratio))),
            confidence * retained,
        ]
    )


def _collect_split(
    model: torch.nn.Module,
    split: Any,
    source_rate_hz: float,
    ratios: tuple[float, ...],
    base_config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for ratio in ratios:
        logits, retention = _predict_with_retention(
            model, split, source_rate_hz, ratio, base_config, device
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        features.append(reliability_features(logits, retention, ratio))
        logits_all.append(logits)
        targets.append(np.asarray(split.y, dtype=np.int64))
        confidence.append(probabilities.max(axis=1))
        groups.append(np.arange(len(split.y), dtype=np.int64))
    merged_logits = np.concatenate(logits_all)
    merged_targets = np.concatenate(targets)
    return {
        "features": np.concatenate(features),
        "logits": merged_logits,
        "targets": merged_targets,
        "correct": (merged_logits.argmax(axis=1) == merged_targets).astype(np.int64),
        "confidence": np.concatenate(confidence),
        "groups": np.concatenate(groups),
    }


def _new_calibrator(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=1000, random_state=seed),
    )


def cross_fitted_scores(
    features: np.ndarray,
    correct: np.ndarray,
    groups: np.ndarray,
    folds: int,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, bool]:
    """Out-of-fold correctness probabilities with sample-group isolation."""

    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(int(folds), len(unique_groups)))
    scores = np.full(len(correct), np.nan, dtype=np.float64)
    finite = True
    for train_index, heldout_index in splitter.split(features, correct, groups):
        if len(np.unique(correct[train_index])) < 2:
            scores[heldout_index] = float(correct[train_index].mean())
            continue
        model = _new_calibrator(c_value, seed)
        model.fit(features[train_index], correct[train_index])
        scores[heldout_index] = model.predict_proba(features[heldout_index])[:, 1]
        finite = finite and bool(np.isfinite(model[-1].coef_).all())
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("cross-fitted reliability scores are incomplete")
    return scores, finite


def _fit_and_score(
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    c_value: float,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    validation_scores, finite = cross_fitted_scores(
        validation["features"],
        validation["correct"],
        validation["groups"],
        folds,
        c_value,
        seed,
    )
    if len(np.unique(validation["correct"])) < 2:
        test_scores = np.full(len(test["correct"]), validation["correct"].mean())
    else:
        model = _new_calibrator(c_value, seed)
        model.fit(validation["features"], validation["correct"])
        test_scores = model.predict_proba(test["features"])[:, 1]
        finite = finite and bool(np.isfinite(model[-1].coef_).all())
    return validation_scores, test_scores, finite


def _split_summary(
    split: dict[str, np.ndarray],
    score: np.ndarray,
    score_threshold: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    confidence_aurc = classification_metrics(
        split["targets"], split["logits"], split["confidence"]
    )["aurc"]
    score_aurc = classification_metrics(
        split["targets"], split["logits"], score
    )["aurc"]
    return {
        "pooled_confidence_aurc": confidence_aurc,
        "pooled_calibrated_aurc": score_aurc,
        "pooled_aurc_relative_reduction": _relative_reduction(
            score_aurc, confidence_aurc
        ),
        "calibrated_target": _risk_and_coverage(
            split["correct"], score, score_threshold
        ),
        "confidence_target": _risk_and_coverage(
            split["correct"], split["confidence"], confidence_threshold
        ),
    }


def _decision(results: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    dataset_summary: dict[str, Any] = {}
    improved_seed_count = 0
    all_finite = True
    for dataset_id, seeds in results.items():
        reductions = [
            row["validation"]["pooled_aurc_relative_reduction"]
            for row in seeds.values()
        ]
        risk_deltas = [
            row["validation"]["calibrated_target"]["risk"]
            - row["validation"]["confidence_target"]["risk"]
            for row in seeds.values()
        ]
        improved_seed_count += sum(value > 0.0 for value in reductions)
        all_finite = all_finite and all(row["calibrator_finite"] for row in seeds.values())
        dataset_summary[dataset_id] = {
            "mean_pooled_aurc_relative_reduction": float(np.mean(reductions)),
            "improved_seed_count": int(sum(value > 0.0 for value in reductions)),
            "mean_selective_risk_delta_at_target_coverage": float(np.mean(risk_deltas)),
        }
    improved_dataset_count = sum(
        row["mean_pooled_aurc_relative_reduction"] > 0.0
        for row in dataset_summary.values()
    )
    lower_risk_dataset_count = sum(
        row["mean_selective_risk_delta_at_target_coverage"] < 0.0
        for row in dataset_summary.values()
    )
    overall = float(
        np.mean(
            [row["mean_pooled_aurc_relative_reduction"] for row in dataset_summary.values()]
        )
    )
    checks = {
        "dataset_mean_aurc": improved_dataset_count
        >= int(gates["minimum_improved_dataset_count"]),
        "dataset_seed_directions": improved_seed_count
        >= int(gates["minimum_improved_dataset_seed_count"]),
        "overall_effect": overall
        >= float(gates["minimum_overall_mean_pooled_aurc_relative_reduction"]),
        "selective_risk": lower_risk_dataset_count
        >= int(gates["minimum_dataset_count_with_lower_selective_risk_at_target_coverage"]),
        "finite_calibrators": all_finite
        if gates["require_all_calibrators_finite"]
        else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dataset_summary": dataset_summary,
        "improved_dataset_count": improved_dataset_count,
        "improved_dataset_seed_count": improved_seed_count,
        "lower_risk_dataset_count": lower_risk_dataset_count,
        "overall_mean_relative_reduction": overall,
    }


def run_v3_calibrated_reliability(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config = yaml.safe_load(
        (root / "configs" / "experiments" / "v3_calibrated_reliability.yaml").read_text(
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
    budget = float(config["wall_time_budget_seconds"])
    results: dict[str, Any] = {}
    for dataset_id in config["datasets"]:
        dataset = load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        results[dataset_id] = {}
        for seed_value in config["seeds"]:
            if time.monotonic() - started > budget:
                raise TimeoutError("v3.2 calibrated reliability wall-time budget exceeded")
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
            validation_score, test_score, finite = _fit_and_score(
                validation,
                test,
                float(calibrator_config["regularization_c"]),
                int(calibrator_config["group_folds"]),
                seed,
            )
            score_threshold = threshold_for_target_coverage(
                validation_score, target_coverage
            )
            confidence_threshold = threshold_for_target_coverage(
                validation["confidence"], target_coverage
            )
            results[dataset_id][str(seed)] = {
                "checkpoint": str(checkpoint),
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
        / "v3_calibrated_reliability"
        / f"v3_calibrated_reliability__3seeds__{stamp}"
    )
    report["run_root"] = str(run_root)
    lines = [
        "# NyquistGuard-TSC v3.2 Cross-fitted Reliability Calibrator",
        "",
        "- 可靠性头：固定 6 特征的标准化逻辑回归；预测 correctness。",
        "- 主判据：按原始样本分组的 4-fold OOF validation；同一样本的各 rate 不跨折。",
        "- test 仅为研发附录；分类模型不训练、不改 logits。",
        f"- 4 数据集 × 3 seeds；墙钟 {elapsed:.1f} 秒。",
        f"- 冻结开发门：{'PASS' if decision['passed'] else 'FAIL'}；不授权 Full。",
        "",
        "| 数据集 | OOF validation AURC 相对下降 | 改善 seeds | 80% coverage risk 差值 |",
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
            f"总体 OOF validation AURC 相对下降：{decision['overall_mean_relative_reduction'] * 100:+.2f}% 。",
            "",
            "## 决策检查",
            "",
        ]
    )
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "v3_calibrated_reliability_report.json", report)
    _atomic_write_text(run_root / "v3_calibrated_reliability_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_calibrated_reliability_report.json", report)
    _atomic_write_text(root / "reports" / "v3_calibrated_reliability_report.md", markdown)
    return report
