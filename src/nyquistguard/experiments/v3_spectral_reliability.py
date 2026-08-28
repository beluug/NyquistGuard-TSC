"""Sample-level spectral reliability development using frozen v1 checkpoints."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.metrics import classification_metrics
from nyquistguard.experiments.pilot import _deep_model, _view
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v3_reliability import (
    _decision,
    _public_split_summary,
    nyquist_reliability_score,
    threshold_for_target_coverage,
)


def sample_retained_band_energy(
    band_features: torch.Tensor, gated_band_features: torch.Tensor
) -> torch.Tensor:
    """Fraction of each sample's absolute band energy retained by the gate."""

    if band_features.shape != gated_band_features.shape or band_features.ndim < 2:
        raise ValueError("band feature tensors must have the same [B,...] shape")
    dimensions = tuple(range(1, band_features.ndim))
    total = band_features.abs().sum(dim=dimensions).clamp_min(1e-8)
    retained = gated_band_features.abs().sum(dim=dimensions)
    return (retained / total).clamp(0.0, 1.0)


def _predict_with_retention(
    model: torch.nn.Module,
    split: Any,
    source_rate_hz: float,
    ratio: float,
    base_config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(split.x), torch.from_numpy(split.y)),
        batch_size=int(base_config["batch_size"]),
        shuffle=False,
        num_workers=int(base_config["num_workers"]),
    )
    logits: list[np.ndarray] = []
    retention: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, _ in loader:
            x_view, rate = _view(x.to(device), source_rate_hz, ratio, base_config)
            output = model(x_view, rate)
            logits.append(output["logits"].float().cpu().numpy())
            if "retained_band_energy" in output:
                retained = output["retained_band_energy"]
            else:
                retained = sample_retained_band_energy(
                    output["band_features"], output["gated_band_features"]
                )
            retention.append(retained.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(retention)


def _split_scores(
    model: torch.nn.Module,
    split: Any,
    source_rate_hz: float,
    ratios: tuple[float, ...],
    base_config: dict[str, Any],
    device: torch.device,
    confidence_exponent: float,
    retained_energy_exponent: float,
) -> dict[str, Any]:
    pooled_logits: list[np.ndarray] = []
    pooled_targets: list[np.ndarray] = []
    pooled_confidence: list[np.ndarray] = []
    pooled_score: list[np.ndarray] = []
    per_rate: dict[str, Any] = {}
    for ratio in ratios:
        logits, retention = _predict_with_retention(
            model, split, source_rate_hz, ratio, base_config, device
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        confidence = probabilities.max(axis=1)
        score = nyquist_reliability_score(
            confidence,
            retention,
            confidence_exponent,
            retained_energy_exponent,
        )
        rate_id = f"r{int(round(ratio * 1000)):04d}"
        per_rate[rate_id] = {
            "ratio": ratio,
            "mean_retained_band_energy": float(retention.mean()),
            "mean_confidence": float(confidence.mean()),
            "mean_nrs": float(score.mean()),
        }
        pooled_logits.append(logits)
        pooled_targets.append(np.asarray(split.y, dtype=np.int64))
        pooled_confidence.append(confidence)
        pooled_score.append(score)
    logits = np.concatenate(pooled_logits)
    targets = np.concatenate(pooled_targets)
    confidence = np.concatenate(pooled_confidence)
    score = np.concatenate(pooled_score)
    return {
        "logits": logits,
        "targets": targets,
        "correct": logits.argmax(axis=1) == targets,
        "confidence": confidence,
        "nrs": score,
        "confidence_aurc": classification_metrics(targets, logits, confidence)["aurc"],
        "nrs_aurc": classification_metrics(targets, logits, score)["aurc"],
        "per_rate": per_rate,
    }


def run_v3_spectral_reliability(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config = yaml.safe_load(
        (root / "configs" / "experiments" / "v3_spectral_reliability.yaml").read_text(
            encoding="utf-8"
        )
    )
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
                raise TimeoutError("v3.1 spectral reliability wall-time budget exceeded")
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
                float(score_config["retained_energy_exponent"]),
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
                float(score_config["retained_energy_exponent"]),
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
    run_root = (
        root
        / "runs"
        / "v3_spectral_reliability"
        / f"v3_spectral_reliability__3seeds__{stamp}"
    )
    report["run_root"] = str(run_root)
    lines = [
        "# NyquistGuard-TSC v3.1 Sample Spectral Reliability",
        "",
        "- 公式：`SSR = max-softmax confidence × sample retained band energy`。",
        "- 主判据：validation；test 仅为研发附录。",
        f"- 4 数据集 × 3 seeds；墙钟 {elapsed:.1f} 秒；无训练、无参数更新。",
        f"- 冻结开发门：{'PASS' if decision['passed'] else 'FAIL'}；只决定下一轮研发，不授权 Full。",
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
    lines.append("")
    lines.append("SSR 只读取已有模型的物理频带张量，不修改分类 logits 或预测。")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "v3_spectral_reliability_report.json", report)
    _atomic_write_text(run_root / "v3_spectral_reliability_report.md", markdown)
    atomic_write_json(root / "reports" / "v3_spectral_reliability_report.json", report)
    _atomic_write_text(root / "reports" / "v3_spectral_reliability_report.md", markdown)
    return report
