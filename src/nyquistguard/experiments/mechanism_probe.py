"""Bounded checkpoint mechanism probes for a completed Pilot.

This module is deliberately diagnostic: it loads frozen seed-17 checkpoints,
runs forward passes and ``torch.autograd.grad`` probes, and inspects processed
window fingerprints.  It never constructs an optimizer, calls ``backward()``,
updates parameters, writes checkpoints, or starts another experiment stage.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from nyquistguard.data import PreparedDataset, load_prepared_dataset
from nyquistguard.experiments.diagnosis import (
    PILOT_DATASETS,
    _atomic_write_text,
    _aurc,
    _latest_completed_pilot,
    _load_active_runs,
)
from nyquistguard.experiments.pilot import (
    _deep_model,
    _resolved_objective_config,
    _view,
)
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.losses import NyquistGuardObjective


PROBE_SEED = 17
PROBE_RATIOS = (1.0, 0.9, 0.6, 0.4, 0.3)
GRADIENT_PAIR_RATIOS = (0.5, 0.3)
MAX_TEST_SAMPLES = 128
MAX_GRADIENT_SAMPLES = 24

MECHANISM_PROBE_TASKS = [
    "预检 Pilot、checkpoint、配置与设备",
    "BasicMotions：gate/CBE/选择性/指纹探针",
    "Epilepsy：gate/CBE/选择性/指纹探针",
    "PAMAP2：gate/CBE/选择性/指纹探针",
    "MHEALTH：gate/CBE/选择性/指纹探针",
    "汇总候选根因与证据强度",
    "生成机制探针报告",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return float(statistics.mean(materialized)) if materialized else math.nan


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _balanced_indices(labels: np.ndarray, maximum: int, seed: int = PROBE_SEED) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    base, remainder = divmod(maximum, len(classes))
    selected: list[np.ndarray] = []
    for position, label in enumerate(classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = min(len(indices), base + (1 if position < remainder else 0))
        selected.append(indices[:count])
    combined = np.concatenate(selected)
    if len(combined) < maximum:
        remaining = np.setdiff1d(np.arange(len(labels)), combined, assume_unique=False)
        rng.shuffle(remaining)
        combined = np.concatenate([combined, remaining[: maximum - len(combined)]])
    return np.sort(combined[:maximum])


def _gradient_norm(loss: Tensor, parameters: Sequence[nn.Parameter]) -> float:
    if not loss.requires_grad or not parameters:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(torch.sqrt(squared).detach().cpu())


def _load_model(
    dataset: PreparedDataset,
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, np.ndarray]]:
    model = _deep_model(dataset, config, "nyquistguard", device)
    initial = {
        "centers": model.filterbank.center_frequencies_hz.detach().cpu().numpy().copy(),  # type: ignore[attr-defined]
        "sigmas": model.filterbank.time_scales_seconds.detach().cpu().numpy().copy(),  # type: ignore[attr-defined]
    }
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, initial


def _filter_and_gate_probe(
    model: nn.Module,
    initial: dict[str, np.ndarray],
    source_rate_hz: float,
) -> dict[str, Any]:
    filterbank = model.filterbank  # type: ignore[attr-defined]
    centers_tensor = filterbank.center_frequencies_hz
    sigmas_tensor = filterbank.time_scales_seconds
    bandwidth_tensor = filterbank.bandwidth_std_hz
    centers = centers_tensor.detach().float().cpu().numpy()
    sigmas = sigmas_tensor.detach().float().cpu().numpy()
    bandwidths = bandwidth_tensor.detach().float().cpu().numpy()
    sorted_centers = np.sort(centers)
    span = float(filterbank.max_center_hz - filterbank.min_center_hz)
    boundary_distance = np.minimum(
        centers - float(filterbank.min_center_hz),
        float(filterbank.max_center_hz) - centers,
    )
    rate_results: dict[str, Any] = {}
    for ratio in PROBE_RATIOS:
        rate = float(source_rate_hz * ratio)
        with torch.inference_mode():
            gate = model.nyquist_gate(  # type: ignore[attr-defined]
                rate,
                centers_tensor,
                sigmas_tensor,
                batch_size=1,
            )[0].detach().float().cpu().numpy()
        rate_results[f"r{int(round(ratio * 1000)):04d}"] = {
            "rate_hz": rate,
            "nyquist_hz": rate / 2.0,
            "gate_mean": float(np.mean(gate)),
            "gate_min": float(np.min(gate)),
            "gate_max": float(np.max(gate)),
            "effective_band_sum": float(np.sum(gate)),
            "near_zero_fraction_le_0_05": float(np.mean(gate <= 0.05)),
            "near_one_fraction_ge_0_95": float(np.mean(gate >= 0.95)),
            "mid_gate_fraction": float(np.mean((gate > 0.05) & (gate < 0.95))),
            "center_above_nyquist_count": int(np.sum(centers > rate / 2.0)),
            "gate_values": gate.tolist(),
        }
    low = rate_results["r0300"]
    return {
        "num_bands": len(centers),
        "center_frequencies_hz": centers.tolist(),
        "time_scales_seconds": sigmas.tolist(),
        "bandwidth_std_hz": bandwidths.tolist(),
        "initial_center_frequencies_hz": initial["centers"].tolist(),
        "initial_time_scales_seconds": initial["sigmas"].tolist(),
        "median_absolute_center_shift_hz": float(np.median(np.abs(centers - initial["centers"]))),
        "median_absolute_sigma_shift_seconds": float(np.median(np.abs(sigmas - initial["sigmas"]))),
        "minimum_sorted_center_spacing_hz": float(np.min(np.diff(sorted_centers))),
        "center_order_inversions": int(np.sum(np.diff(centers) < 0)),
        "center_boundary_fraction_within_2_percent": float(np.mean(boundary_distance <= 0.02 * span)),
        "rates": rate_results,
        "candidate_low_rate_gate_collapse": bool(
            low["effective_band_sum"] < max(2.0, 0.25 * len(centers))
            or low["near_zero_fraction_le_0_05"] >= 0.75
        ),
    }


def _inference_probe(
    model: nn.Module,
    dataset: PreparedDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    indices = _balanced_indices(dataset.test.y, MAX_TEST_SAMPLES)
    values = dataset.test.x[indices]
    targets = dataset.test.y[indices]
    batch_size = min(int(config["batch_size"]), 32)
    acceptance_by_rate: dict[str, np.ndarray] = {}
    confidence_by_rate: dict[str, np.ndarray] = {}
    correctness_by_rate: dict[str, np.ndarray] = {}
    entropy_by_rate: dict[str, np.ndarray] = {}
    per_rate: dict[str, Any] = {}
    for ratio in PROBE_RATIOS:
        acceptances: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        correctness: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(values[start : start + batch_size]).to(device)
            target = targets[start : start + batch_size]
            viewed, rate = _view(x, dataset.sampling_rate_hz, ratio, config)
            with torch.inference_mode():
                output = model(viewed, rate)
                probabilities = torch.softmax(output["logits"].float(), dim=-1)  # type: ignore[index]
                acceptance = output["accept_probability"].float()  # type: ignore[index]
                entropy = output["aux"]["normalized_prediction_entropy"].float()  # type: ignore[index]
            probability_np = probabilities.cpu().numpy()
            acceptances.append(acceptance.cpu().numpy())
            confidences.append(probability_np.max(axis=1))
            correctness.append((probability_np.argmax(axis=1) == target).astype(np.float64))
            entropies.append(entropy.cpu().numpy())
        rate_id = f"r{int(round(ratio * 1000)):04d}"
        acceptance_np = np.concatenate(acceptances)
        confidence_np = np.concatenate(confidences)
        correctness_np = np.concatenate(correctness)
        entropy_np = np.concatenate(entropies)
        errors = 1.0 - correctness_np
        acceptance_by_rate[rate_id] = acceptance_np
        confidence_by_rate[rate_id] = confidence_np
        correctness_by_rate[rate_id] = correctness_np
        entropy_by_rate[rate_id] = entropy_np
        wrong = acceptance_np[correctness_np == 0]
        correct = acceptance_np[correctness_np == 1]
        per_rate[rate_id] = {
            "ratio": ratio,
            "sample_count": len(acceptance_np),
            "accuracy": float(np.mean(correctness_np)),
            "acceptance_mean": float(np.mean(acceptance_np)),
            "acceptance_std": float(np.std(acceptance_np)),
            "acceptance_correct_mean": float(np.mean(correct)) if len(correct) else None,
            "acceptance_wrong_mean": float(np.mean(wrong)) if len(wrong) else None,
            "correct_minus_wrong_acceptance": (
                float(np.mean(correct) - np.mean(wrong)) if len(correct) and len(wrong) else None
            ),
            "coverage_at_0_5": float(np.mean(acceptance_np >= 0.5)),
            "learned_acceptance_aurc": _aurc(errors, acceptance_np),
            "max_softmax_confidence_aurc": _aurc(errors, confidence_np),
            "acceptance_correctness_correlation": _pearson(acceptance_np, correctness_np),
            "acceptance_entropy_correlation": _pearson(acceptance_np, entropy_np),
        }
    full = acceptance_by_rate["r1000"]
    monotonic_violations = {
        rate_id: float(np.mean(acceptance_by_rate[rate_id] > full + 1e-6))
        for rate_id in ("r0900", "r0600", "r0400", "r0300")
    }
    unseen = [per_rate[rate] for rate in ("r0900", "r0600", "r0400", "r0300")]
    return {
        "sample_count": len(indices),
        "selection_indices": indices.tolist(),
        "per_rate": per_rate,
        "full_to_low_acceptance_drop": float(np.mean(full) - np.mean(acceptance_by_rate["r0300"])),
        "monotonic_violation_vs_full": monotonic_violations,
        "mean_unseen_learned_aurc": _mean(item["learned_acceptance_aurc"] for item in unseen),
        "mean_unseen_confidence_aurc": _mean(item["max_softmax_confidence_aurc"] for item in unseen),
        "learned_better_than_confidence": bool(
            _mean(item["learned_acceptance_aurc"] for item in unseen)
            < _mean(item["max_softmax_confidence_aurc"] for item in unseen)
        ),
    }


def _loss_gradient_probe(
    model: nn.Module,
    dataset: PreparedDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    objective_config = _resolved_objective_config(dataset, config, "nyquistguard")
    objective = NyquistGuardObjective(**objective_config).to(device)
    indices = _balanced_indices(dataset.train.y, MAX_GRADIENT_SAMPLES)
    x = torch.from_numpy(dataset.train.x[indices]).to(device)
    targets = torch.from_numpy(dataset.train.y[indices]).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    filter_parameters = [parameter for parameter in model.filterbank.parameters() if parameter.requires_grad]  # type: ignore[attr-defined]
    selective_module = model.selective_head  # type: ignore[attr-defined]
    selective_parameters = (
        [parameter for parameter in selective_module.parameters() if parameter.requires_grad]
        if selective_module is not None
        else []
    )
    results: dict[str, Any] = {}
    for ratio in GRADIENT_PAIR_RATIOS:
        high, high_rate = _view(x, dataset.sampling_rate_hz, 1.0, config)
        low, low_rate = _view(x, dataset.sampling_rate_hz, ratio, config)
        output_high = model(high, high_rate)
        output_low = model(low, low_rate)
        losses = objective(output_high, targets, output_low, targets)
        lambda_cbe = float(objective.lambda_cbe)
        lambda_selective = float(objective.lambda_selective)
        lambda_monotonicity = float(objective.lambda_monotonicity)
        lambda_regularization = float(objective.lambda_filter_regularization)
        contributions = {
            "classification": float(losses["classification"].detach().cpu()),
            "cbe_unweighted": float(losses["equivariance"].detach().cpu()),
            "cbe_weighted": float((lambda_cbe * losses["equivariance"]).detach().cpu()),
            "selective_unweighted": float(losses["selective"].detach().cpu()),
            "selective_weighted": float((lambda_selective * losses["selective"]).detach().cpu()),
            "monotonicity_unweighted": float(losses["monotonicity"].detach().cpu()),
            "monotonicity_weighted": float((lambda_monotonicity * losses["monotonicity"]).detach().cpu()),
            "filter_regularization_unweighted": float(losses["filter_regularization"].detach().cpu()),
            "filter_regularization_weighted": float(
                (lambda_regularization * losses["filter_regularization"]).detach().cpu()
            ),
            "total": float(losses["total"].detach().cpu()),
            "coverage": float(losses["coverage"].detach().cpu()),
            "coverage_penalty": float(losses["coverage_penalty"].detach().cpu()),
            "selective_risk": float(losses["selective_risk"].detach().cpu()),
        }
        common_mask = objective.equivariance.common_mask(
            output_high["nyquist_gate"], output_low["nyquist_gate"]  # type: ignore[arg-type]
        )
        gradient_norms = {
            "classification_all": _gradient_norm(losses["classification"], parameters),
            "classification_filterbank": _gradient_norm(losses["classification"], filter_parameters),
            "weighted_cbe_all": _gradient_norm(lambda_cbe * losses["equivariance"], parameters),
            "weighted_cbe_filterbank": _gradient_norm(
                lambda_cbe * losses["equivariance"], filter_parameters
            ),
            "weighted_selective_all": _gradient_norm(
                lambda_selective * losses["selective"], parameters
            ),
            "weighted_selective_head": _gradient_norm(
                lambda_selective * losses["selective"], selective_parameters
            ),
            "weighted_monotonicity_all": _gradient_norm(
                lambda_monotonicity * losses["monotonicity"], parameters
            ),
            "total_all": _gradient_norm(losses["total"], parameters),
        }
        results[f"r{int(round(ratio * 1000)):04d}"] = {
            "ratio": ratio,
            "sample_count": len(indices),
            "contributions": contributions,
            "common_mask_mean": float(common_mask.detach().float().mean().cpu()),
            "common_mask_effective_band_sum": float(
                common_mask.detach().float().sum(dim=1).mean().cpu()
            ),
            "gradient_norms": gradient_norms,
            "weighted_cbe_to_classification_gradient_ratio": _safe_ratio(
                gradient_norms["weighted_cbe_all"], gradient_norms["classification_all"]
            ),
            "weighted_cbe_to_classification_filterbank_gradient_ratio": _safe_ratio(
                gradient_norms["weighted_cbe_filterbank"],
                gradient_norms["classification_filterbank"],
            ),
            "weighted_selective_to_classification_gradient_ratio": _safe_ratio(
                gradient_norms["weighted_selective_all"], gradient_norms["classification_all"]
            ),
        }
        del output_high, output_low, losses, high, low, common_mask
    del x, targets, objective
    return results


def _window_signatures(values: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    signatures = np.empty((len(values), values.shape[1] * 3), dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        batch = np.asarray(values[start : start + chunk_size], dtype=np.float32)
        mean = batch.mean(axis=2)
        standard_deviation = batch.std(axis=2)
        differences = np.diff(batch, axis=2)
        difference_rms = np.sqrt(np.mean(np.square(differences), axis=2))
        signatures[start : start + len(batch)] = np.concatenate(
            [mean, standard_deviation, difference_rms], axis=1
        )
    return signatures


def _sample_hashes(values: np.ndarray) -> set[str]:
    result: set[str] = set()
    for sample in values:
        contiguous = np.ascontiguousarray(sample, dtype=np.float32)
        result.add(hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest())
    return result


def _fingerprint_probe(dataset: PreparedDataset) -> dict[str, Any]:
    train_hashes = _sample_hashes(dataset.train.x)
    test_digests = [
        hashlib.blake2b(
            np.ascontiguousarray(sample, dtype=np.float32).view(np.uint8), digest_size=16
        ).hexdigest()
        for sample in dataset.test.x
    ]
    exact_overlap = np.asarray([digest in train_hashes for digest in test_digests], dtype=bool)
    train_signature = _window_signatures(dataset.train.x)
    test_signature = _window_signatures(dataset.test.x)
    center = train_signature.mean(axis=0, keepdims=True)
    scale = train_signature.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    train_normalized = (train_signature - center) / scale
    test_normalized = (test_signature - center) / scale
    train_norm = np.sum(np.square(train_normalized), axis=1)
    nearest_distances: list[np.ndarray] = []
    nearest_indices: list[np.ndarray] = []
    for start in range(0, len(test_normalized), 128):
        batch = test_normalized[start : start + 128]
        distances = (
            np.sum(np.square(batch), axis=1, keepdims=True)
            + train_norm.reshape(1, -1)
            - 2.0 * batch @ train_normalized.T
        )
        distances = np.maximum(distances, 0.0)
        indices = np.argmin(distances, axis=1)
        nearest_indices.append(indices)
        nearest_distances.append(np.sqrt(distances[np.arange(len(batch)), indices]))
    nearest = np.concatenate(nearest_indices)
    distance = np.concatenate(nearest_distances)
    nearest_label_accuracy = float(np.mean(dataset.train.y[nearest] == dataset.test.y))
    return {
        "scope": "processed standardized windows; not raw-source byte fingerprints",
        "train_sample_count": len(dataset.train.x),
        "test_sample_count": len(dataset.test.x),
        "unique_train_hash_count": len(train_hashes),
        "exact_train_test_duplicate_count": int(exact_overlap.sum()),
        "exact_train_test_duplicate_fraction": float(exact_overlap.mean()),
        "nearest_signature_distance_min": float(np.min(distance)),
        "nearest_signature_distance_p01": float(np.quantile(distance, 0.01)),
        "nearest_signature_distance_median": float(np.median(distance)),
        "nearest_signature_distance_p99": float(np.quantile(distance, 0.99)),
        "nearest_signature_label_accuracy": nearest_label_accuracy,
        "leakage_proven": bool(exact_overlap.any()),
        "interpretation": (
            "发现处理后 train/test 完全相同窗口，需要阻断后续实验并定位来源。"
            if exact_overlap.any()
            else "未发现处理后窗口的精确 train/test 重复；近邻签名只能衡量任务相似度，不能排除所有泄漏。"
        ),
    }


def _probe_dataset(
    project_root: Path,
    pilot_root: Path,
    dataset_id: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    cache_path = project_root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
    dataset = load_prepared_dataset(cache_path)
    checkpoint = pilot_root / f"{dataset_id}__nyquistguard__seed{PROBE_SEED}" / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"缺少冻结 checkpoint：{checkpoint}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model, initial = _load_model(dataset, config, checkpoint, device)
    filter_gate = _filter_and_gate_probe(model, initial, dataset.sampling_rate_hz)
    inference = _inference_probe(model, dataset, config, device)
    gradients = _loss_gradient_probe(model, dataset, config, device)
    fingerprint = _fingerprint_probe(dataset)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    cuda_peak_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    result = {
        "dataset_id": dataset_id,
        "seed": PROBE_SEED,
        "checkpoint": str(checkpoint),
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "input_shape": list(dataset.test.x.shape[1:]),
        "num_classes": len(dataset.class_names),
        "parameter_count": parameter_count,
        "device": str(device),
        "filter_and_gate": filter_gate,
        "selectivity": inference,
        "loss_and_gradient": gradients,
        "fingerprint": fingerprint,
        "duration_seconds": time.monotonic() - started,
        "cuda_peak_memory_mb": cuda_peak_mb,
    }
    del model, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate_findings(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gate_collapse = [
        dataset
        for dataset, payload in datasets.items()
        if payload["filter_and_gate"]["candidate_low_rate_gate_collapse"]
    ]
    cbe_inactive: list[str] = []
    cbe_dominant: list[str] = []
    for dataset, payload in datasets.items():
        ratio = payload["loss_and_gradient"]["r0500"][
            "weighted_cbe_to_classification_filterbank_gradient_ratio"
        ]
        if ratio is not None and ratio < 1e-3:
            cbe_inactive.append(dataset)
        if ratio is not None and ratio > 10.0:
            cbe_dominant.append(dataset)
    selection_failures = [
        dataset
        for dataset, payload in datasets.items()
        if not payload["selectivity"]["learned_better_than_confidence"]
    ]
    exact_duplicates = [
        dataset
        for dataset, payload in datasets.items()
        if payload["fingerprint"]["exact_train_test_duplicate_count"] > 0
    ]
    findings: list[dict[str, Any]] = []
    findings.append(
        {
            "issue": "selective_head_risk_ranking",
            "severity": "critical" if len(selection_failures) >= 3 else "important",
            "datasets": selection_failures,
            "observation": "learned acceptance AURC is not better than same-checkpoint max-softmax confidence",
            "interpretation": "The selective head/objective is a supported failure candidate; the exact loss-design cause is not yet causal proof.",
        }
    )
    findings.append(
        {
            "issue": "low_rate_gate_collapse",
            "severity": "important" if gate_collapse else "not_detected",
            "datasets": gate_collapse,
            "observation": "r0300 effective gate mass is below the pre-frozen collapse threshold",
            "interpretation": "If present, too few observable bands may contribute to low-rate failure; expected Nyquist suppression and harmful collapse must be distinguished.",
        }
    )
    findings.append(
        {
            "issue": "cbe_gradient_inactive_or_dominant",
            "severity": "important" if cbe_inactive or cbe_dominant else "not_detected",
            "inactive_datasets": cbe_inactive,
            "dominant_datasets": cbe_dominant,
            "observation": "weighted CBE/classification gradient ratio on the shared filterbank parameter group at r0500",
            "interpretation": "Using a shared parameter group avoids an invalid global-denominator comparison; the ratio remains a scale diagnostic, not proof that a particular weight is optimal.",
        }
    )
    findings.append(
        {
            "issue": "processed_window_exact_overlap",
            "severity": "critical" if exact_duplicates else "not_detected",
            "datasets": exact_duplicates,
            "observation": "exact BLAKE2b fingerprints across processed train/test windows",
            "interpretation": "No exact overlap does not exclude subject, preprocessing, or semantic leakage.",
        }
    )
    return {
        "findings": findings,
        "selection_failure_datasets": selection_failures,
        "gate_collapse_datasets": gate_collapse,
        "cbe_inactive_datasets": cbe_inactive,
        "cbe_dominant_datasets": cbe_dominant,
        "exact_duplicate_datasets": exact_duplicates,
        "causal_conclusion_allowed": False,
        "next_action": (
            "Stop and repair split construction before any model experiment."
            if exact_duplicates
            else "Use these probes to define a small, separately versioned targeted experiment; do not rerun 84 runs blindly."
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NyquistGuard-TSC 机制探针报告",
        "",
        f"- 状态：{report['status']}",
        f"- Pilot：{report['pilot_root']}",
        f"- 设备：{report['device']['name']}",
        f"- 固定范围：seed{PROBE_SEED}；测试样本最多 {MAX_TEST_SAMPLES}；梯度样本最多 {MAX_GRADIENT_SAMPLES}",
        "- 安全边界：只执行 forward 与 autograd.grad；无优化器、无参数更新、无 checkpoint 写入、无 Pilot/Full 启动。",
        "- 证据边界：下列为机制候选诊断，不是因果证明。",
        "",
        "## 数据集汇总",
        "",
        "| 数据集 | r0.3有效band | r0.3 gate≤0.05 | CBE/filterbank分类梯度比(r0.5) | learned/conf AURC | q全率→低率变化 | 精确跨split重复 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in PILOT_DATASETS:
        payload = report["datasets"][dataset]
        gate = payload["filter_and_gate"]["rates"]["r0300"]
        cbe_ratio = payload["loss_and_gradient"]["r0500"][
            "weighted_cbe_to_classification_filterbank_gradient_ratio"
        ]
        selection = payload["selectivity"]
        cbe_ratio_cell = f"{cbe_ratio:.4g}" if cbe_ratio is not None else "—"
        lines.append(
            f"| {dataset} | {gate['effective_band_sum']:.2f}/16 | "
            f"{gate['near_zero_fraction_le_0_05'] * 100:.1f}% | "
            f"{cbe_ratio_cell} | "
            f"{selection['mean_unseen_learned_aurc']:.4f}/"
            f"{selection['mean_unseen_confidence_aurc']:.4f} | "
            f"{selection['full_to_low_acceptance_drop']:+.4f} | "
            f"{payload['fingerprint']['exact_train_test_duplicate_count']} |"
        )
    lines.extend(["", "## 候选问题", ""])
    for finding in report["aggregate"]["findings"]:
        datasets = finding.get("datasets") or finding.get("inactive_datasets") or []
        lines.append(
            f"- **{finding['severity']} · {finding['issue']}**："
            f"{', '.join(datasets) if datasets else '未触发阈值'}。{finding['interpretation']}"
        )
    lines.extend(
        [
            "",
            "## 每个数据集的关键数值",
            "",
        ]
    )
    for dataset in PILOT_DATASETS:
        payload = report["datasets"][dataset]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- 过滤器中心频率：{', '.join(f'{value:.3f}' for value in payload['filter_and_gate']['center_frequencies_hz'])} Hz",
                f"- 中心频率相对初始化的绝对位移中位数：{payload['filter_and_gate']['median_absolute_center_shift_hz']:.4f} Hz",
                f"- r0.3 gate有效band和：{payload['filter_and_gate']['rates']['r0300']['effective_band_sum']:.3f}",
                f"- r0.5 CBE未加权/加权loss：{payload['loss_and_gradient']['r0500']['contributions']['cbe_unweighted']:.6g} / {payload['loss_and_gradient']['r0500']['contributions']['cbe_weighted']:.6g}",
                f"- r0.5 weighted CBE/filterbank分类梯度比：{payload['loss_and_gradient']['r0500']['weighted_cbe_to_classification_filterbank_gradient_ratio']}",
                f"- 未见率 learned/confidence AURC：{payload['selectivity']['mean_unseen_learned_aurc']:.4f} / {payload['selectivity']['mean_unseen_confidence_aurc']:.4f}",
                f"- full→r0.3 接受概率均值下降：{payload['selectivity']['full_to_low_acceptance_drop']:+.4f}",
                f"- 处理后 train/test 精确重复：{payload['fingerprint']['exact_train_test_duplicate_count']}；最近签名标签一致率：{payload['fingerprint']['nearest_signature_label_accuracy']:.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一步",
            "",
            "1. 不启动 Full，不盲目重跑 84-run Pilot。",
            "2. 依据触发的候选问题设计单 seed、代表性数据集的小规模 v2 开发实验。",
            "3. 若属于代码 bug，保留本报告和旧结果后按原协议修复；若修改 loss/结构，明确命名为 v2。",
            "4. 处理后无精确重复不等于排除所有泄漏；MiniROCKET 满分仍需原始文件级审计。",
            "",
        ]
    )
    return "\n".join(lines)


def run_mechanism_probe(project_root: str | Path) -> dict[str, Any]:
    """Run bounded mechanism probes without modifying any learned state."""

    root = Path(project_root).resolve()
    probe_id = f"mechanism_probe__seed{PROBE_SEED}__{_utc_stamp()}"
    run_root = root / "runs" / "mechanism_probe" / probe_id
    progress = DashboardProgress(
        root / "runs" / "dashboard_status.json",
        "mechanism_probe",
        MECHANISM_PROBE_TASKS,
        probe_id,
    )
    task_index = 0
    try:
        progress.start_task(task_index)
        pilot_root = _latest_completed_pilot(root)
        runs = _load_active_runs(pilot_root)
        for dataset in PILOT_DATASETS:
            checkpoint = Path(str(runs[(dataset, "nyquistguard", PROBE_SEED)]["_run_dir"])) / "checkpoint_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"缺少 checkpoint：{checkpoint}")
        config = yaml.safe_load(
            (root / "configs" / "experiments" / "pilot.yaml").read_text(encoding="utf-8")
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        print(f"[1/7] Pilot/checkpoint 预检通过；设备：{device_name}", flush=True)
        progress.complete_task(task_index)

        dataset_results: dict[str, Any] = {}
        for offset, dataset in enumerate(PILOT_DATASETS, start=1):
            task_index = offset
            progress.start_task(task_index)
            print(f"[{offset + 1}/7] 开始 {dataset} 机制探针", flush=True)
            dataset_results[dataset] = _probe_dataset(root, pilot_root, dataset, config, device)
            print(
                f"[{offset + 1}/7] {dataset} 完成；"
                f"耗时 {dataset_results[dataset]['duration_seconds']:.1f}s；"
                f"峰值显存 {dataset_results[dataset]['cuda_peak_memory_mb']:.1f} MiB",
                flush=True,
            )
            progress.complete_task(task_index)

        task_index = 5
        progress.start_task(task_index)
        aggregate = _aggregate_findings(dataset_results)
        print(
            "[6/7] 候选问题：selection="
            f"{len(aggregate['selection_failure_datasets'])}/4，gate collapse="
            f"{len(aggregate['gate_collapse_datasets'])}/4，exact duplicate="
            f"{len(aggregate['exact_duplicate_datasets'])}/4",
            flush=True,
        )
        progress.complete_task(task_index)

        task_index = 6
        progress.start_task(task_index)
        report = {
            "status": "completed",
            "probe_id": probe_id,
            "pilot_root": str(pilot_root),
            "created_at_utc": utc_now(),
            "device": {
                "type": device.type,
                "name": device_name,
                "cuda_available": torch.cuda.is_available(),
            },
            "frozen_scope": {
                "seed": PROBE_SEED,
                "ratios": list(PROBE_RATIOS),
                "gradient_pair_ratios": list(GRADIENT_PAIR_RATIOS),
                "max_test_samples": MAX_TEST_SAMPLES,
                "max_gradient_samples": MAX_GRADIENT_SAMPLES,
                "trained_models": False,
                "optimizer_constructed": False,
                "parameters_updated": False,
                "checkpoints_modified": False,
                "pilot_started": False,
                "full_started": False,
            },
            "datasets": dataset_results,
            "aggregate": aggregate,
            "causal_boundary": (
                "Mechanism probes identify scale, saturation, ranking and overlap candidates. "
                "They do not by themselves establish causality or authorize Full experiments."
            ),
        }
        run_root.mkdir(parents=True, exist_ok=True)
        json_path = run_root / "mechanism_probe_report.json"
        markdown_path = run_root / "mechanism_probe_report.md"
        atomic_write_json(json_path, report)
        markdown = _render_markdown(report)
        _atomic_write_text(markdown_path, markdown)
        atomic_write_json(root / "reports" / "mechanism_probe_report.json", report)
        _atomic_write_text(root / "reports" / "mechanism_probe_report.md", markdown)
        progress.complete_task(task_index)
        progress.finish("机制探针完成；未训练模型；等待人工审阅")
        print(f"[7/7] 机制探针完成：{markdown_path}", flush=True)
        print("未构造优化器、未更新参数、未修改 checkpoint、未启动 Pilot/Full。", flush=True)
        return report
    except BaseException as error:
        progress.fail_task(task_index, error)
        raise


__all__ = ["MECHANISM_PROBE_TASKS", "run_mechanism_probe"]
