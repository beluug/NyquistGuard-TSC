"""No-training probe for a deterministic confidence × gate selector."""

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
from nyquistguard.experiments.diagnosis import PILOT_DATASETS, _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.metrics import classification_metrics
from nyquistguard.experiments.pilot import _deep_model, _predict_deep
from nyquistguard.experiments.progress import atomic_write_json, utc_now


SEED = 17


def _gate_quality(model: torch.nn.Module, source_rate_hz: float, ratio: float) -> float:
    filterbank = model.filterbank  # type: ignore[attr-defined]
    with torch.inference_mode():
        full = model.nyquist_gate(  # type: ignore[attr-defined]
            source_rate_hz,
            filterbank.center_frequencies_hz,
            filterbank.time_scales_seconds,
            batch_size=1,
        )[0].sum()
        current = model.nyquist_gate(  # type: ignore[attr-defined]
            source_rate_hz * ratio,
            filterbank.center_frequencies_hz,
            filterbank.time_scales_seconds,
            batch_size=1,
        )[0].sum()
    return float((current / full.clamp_min(1e-8)).clamp(0.0, 1.0).cpu())


def run_deterministic_selector_probe(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config = yaml.safe_load(
        (root / "configs" / "experiments" / "pilot.yaml").read_text(encoding="utf-8")
    )
    pilot_root = _latest_completed_pilot(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    for dataset_id in PILOT_DATASETS:
        dataset = load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        model = _deep_model(dataset, config, "nyquistguard", device)
        checkpoint = pilot_root / f"{dataset_id}__nyquistguard__seed{SEED}" / "checkpoint_best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        model.eval()
        per_rate: dict[str, Any] = {}
        q_by_rate: dict[str, np.ndarray] = {}
        for ratio_value in config["test_rate_ratios"]:
            ratio = float(ratio_value)
            logits, learned_q = _predict_deep(
                model, dataset.test, dataset.sampling_rate_hz, ratio, config, device
            )
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            confidence = probabilities.max(axis=1)
            quality = _gate_quality(model, dataset.sampling_rate_hz, ratio)
            deterministic_q = confidence * quality
            confidence_metrics = classification_metrics(dataset.test.y, logits, confidence)
            deterministic_metrics = classification_metrics(
                dataset.test.y, logits, deterministic_q
            )
            learned_metrics = classification_metrics(dataset.test.y, logits, learned_q)
            rate_id = f"r{int(round(ratio * 1000)):04d}"
            q_by_rate[rate_id] = deterministic_q
            per_rate[rate_id] = {
                "ratio": ratio,
                "gate_quality_vs_full": quality,
                "accuracy": deterministic_metrics["accuracy"],
                "macro_f1": deterministic_metrics["macro_f1"],
                "deterministic_aurc": deterministic_metrics["aurc"],
                "confidence_aurc": confidence_metrics["aurc"],
                "learned_v1_aurc": learned_metrics["aurc"],
                "deterministic_acceptance_mean": float(np.mean(deterministic_q)),
                "confidence_mean": float(np.mean(confidence)),
                "coverage_at_0_5": float(np.mean(deterministic_q >= 0.5)),
            }
        full = q_by_rate["r1000"]
        low = q_by_rate["r0300"]
        unseen = [per_rate[key] for key in ("r0900", "r0600", "r0400", "r0300")]
        results[dataset_id] = {
            "checkpoint": str(checkpoint),
            "per_rate": per_rate,
            "mean_unseen_deterministic_aurc": float(
                np.mean([row["deterministic_aurc"] for row in unseen])
            ),
            "mean_unseen_confidence_aurc": float(
                np.mean([row["confidence_aurc"] for row in unseen])
            ),
            "mean_unseen_learned_v1_aurc": float(
                np.mean([row["learned_v1_aurc"] for row in unseen])
            ),
            "full_to_low_acceptance_drop": float(np.mean(full) - np.mean(low)),
            "low_above_full_fraction": float(np.mean(low > full + 1e-8)),
        }
        del model, dataset
    ranking_preserved = all(
        abs(
            row["mean_unseen_deterministic_aurc"]
            - row["mean_unseen_confidence_aurc"]
        )
        <= 1e-12
        for row in results.values()
    )
    rate_sufficiency_added = all(
        row["full_to_low_acceptance_drop"] > 0 for row in results.values()
    )
    report = {
        "status": "completed",
        "rule": "q = max_softmax_confidence * (effective_gate_mass(rate) / effective_gate_mass(full))",
        "seed": SEED,
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
        "trained_models": False,
        "parameters_updated": False,
        "pilot_started": False,
        "full_started": False,
        "results": results,
        "decision": {
            "ranking_preserved_exactly": ranking_preserved,
            "rate_sufficiency_added_all_datasets": rate_sufficiency_added,
            "candidate_passed": ranking_preserved and rate_sufficiency_added,
            "confirmatory_claim_allowed": False,
        },
        "finished_at_utc": utc_now(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / "deterministic_selector_probe" / f"deterministic_selector__seed17__{stamp}"
    report["run_root"] = str(run_root)
    lines = [
        "# NyquistGuard-TSC 确定性选择器探针",
        "",
        f"- 规则：`{report['rule']}`",
        f"- 状态：completed；候选门：{'PASS' if report['decision']['candidate_passed'] else 'FAIL'}",
        f"- 墙钟：{report['elapsed_seconds']:.1f} 秒；无训练、无参数更新。",
        "- 结论边界：这是确定性候选验证，不是 confirmatory 结果，不授权 Full。",
        "",
        "| 数据集 | v1 learned AURC | confidence/deterministic AURC | full→r0.3 q下降 | r0.3 q>full 比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset_id, row in results.items():
        lines.append(
            f"| {dataset_id} | {row['mean_unseen_learned_v1_aurc']:.4f} | "
            f"{row['mean_unseen_confidence_aurc']:.4f} | "
            f"{row['full_to_low_acceptance_drop']:+.4f} | "
            f"{row['low_above_full_fraction'] * 100:.1f}% |"
        )
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / "deterministic_selector_report.json", report)
    _atomic_write_text(run_root / "deterministic_selector_report.md", markdown)
    atomic_write_json(root / "reports" / "deterministic_selector_report.json", report)
    _atomic_write_text(root / "reports" / "deterministic_selector_report.md", markdown)
    return report

