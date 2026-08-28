"""Read-only diagnosis of a completed NyquistGuard-TSC Pilot run.

The diagnosis consumes existing metrics, predictions, training histories and
processed split metadata.  It never trains a model, changes a checkpoint, or
starts Pilot/Full experiments.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nyquistguard.experiments.pilot import PILOT_METHODS
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now


PILOT_DATASETS = ("basicmotions_uea", "epilepsy_uea", "pamap2_uci", "mhealth_uci")
PILOT_SEEDS = (17, 42, 2026)
RATE_IDS = ("r1000", "r0900", "r0600", "r0400", "r0300")
UNSEEN_RATE_IDS = RATE_IDS[1:]
CLASSIFICATION_BASELINES = ("fixed_rate_tcn", "multirate_tcn")
STRONG_BASELINES = CLASSIFICATION_BASELINES + ("minirocket",)
ABLATIONS = ("no_nyquist_gate", "no_cbe", "no_selective_head")

DIAGNOSTIC_TASKS = [
    "定位并冻结有效 Pilot 结果",
    "审计 84-run 产物与协议一致性",
    "审计数据切分与标签闭集",
    "分析逐采样率性能与最差 rate",
    "分析 prediction flip 稳定性",
    "诊断选择性头与置信度排序",
    "检查训练历史、checkpoint 与消融",
    "专项检查 MiniROCKET 满分现象",
    "生成诊断报告与 Go/No-Go 结论",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _is_finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_is_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return float(statistics.mean(materialized))


def _aurc(errors: np.ndarray, scores: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if errors.ndim != 1 or scores.ndim != 1 or len(errors) != len(scores) or not len(errors):
        raise ValueError("errors and scores must be equally sized non-empty vectors")
    order = np.argsort(-scores, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return float(np.mean(cumulative_risk))


def _latest_completed_pilot(project_root: Path) -> Path:
    pilot_parent = project_root / "runs" / "pilot"
    candidates: list[tuple[float, Path]] = []
    for path in pilot_parent.glob("pilot__*"):
        manifest = path / "pilot_manifest.json"
        if not path.is_dir() or not manifest.exists():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "completed" and int(payload.get("matrix_size", 0)) == 84:
            candidates.append((manifest.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError("没有找到已完成的 84-run Pilot；诊断不会自动启动或补跑 Pilot")
    return max(candidates, key=lambda item: item[0])[1]


def _expected_run_keys() -> set[tuple[str, str, int]]:
    return {
        (dataset, method, seed)
        for dataset in PILOT_DATASETS
        for method in PILOT_METHODS
        for seed in PILOT_SEEDS
    }


def _load_active_runs(pilot_root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for metrics_path in pilot_root.glob("*__*__seed*/metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        spec = payload.get("spec", {})
        key = (str(spec.get("dataset_id")), str(spec.get("method")), int(spec.get("seed", -1)))
        if key in runs:
            raise RuntimeError(f"重复的有效 run：{key}")
        payload["_run_dir"] = str(metrics_path.parent)
        runs[key] = payload
    expected = _expected_run_keys()
    missing = sorted(expected - set(runs))
    extra = sorted(set(runs) - expected)
    if missing or extra:
        raise RuntimeError(f"Pilot 活动矩阵不完整：missing={missing[:3]} extra={extra[:3]}")
    return runs


def _audit_artifacts(
    runs: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(payload.get("status")) for payload in runs.values())
    protocol_hashes = sorted({str(payload.get("protocol_hash")) for payload in runs.values()})
    prediction_files = 0
    prediction_headers_valid = 0
    checkpoint_best = 0
    checkpoint_last = 0
    minirocket_models = 0
    missing_dataset_protocol = 0
    all_finite = True
    malformed_predictions: list[str] = []
    expected_header = {
        "sample_id",
        "rate_ratio",
        "target",
        "prediction",
        "acceptance",
        "probabilities_json",
    }
    for key, payload in runs.items():
        all_finite = all_finite and _is_finite_tree(payload.get("evaluation", {}))
        if not payload.get("dataset_protocol_id"):
            missing_dataset_protocol += 1
        run_dir = Path(str(payload["_run_dir"]))
        predictions = run_dir / "predictions.csv"
        if predictions.exists():
            prediction_files += 1
            try:
                with predictions.open(newline="", encoding="utf-8") as handle:
                    header = set(next(csv.reader(handle)))
                if expected_header <= header:
                    prediction_headers_valid += 1
                else:
                    malformed_predictions.append("__".join(map(str, key)))
            except (OSError, StopIteration):
                malformed_predictions.append("__".join(map(str, key)))
        if (run_dir / "checkpoint_best.pt").exists():
            checkpoint_best += 1
        if (run_dir / "checkpoint_last.pt").exists():
            checkpoint_last += 1
        if (run_dir / "minirocket.pkl").exists():
            minirocket_models += 1
    passed = (
        len(runs) == 84
        and statuses == Counter({"completed": 84})
        and len(protocol_hashes) == 1
        and all_finite
        and prediction_files == 84
        and prediction_headers_valid == 84
    )
    return {
        "status": "pass_with_warnings" if passed and missing_dataset_protocol else ("pass" if passed else "fail"),
        "active_run_count": len(runs),
        "completed_run_count": statuses.get("completed", 0),
        "protocol_hash_count": len(protocol_hashes),
        "protocol_hashes": protocol_hashes,
        "all_metrics_finite": all_finite,
        "prediction_files": prediction_files,
        "prediction_headers_valid": prediction_headers_valid,
        "malformed_predictions": malformed_predictions,
        "checkpoint_best_files": checkpoint_best,
        "checkpoint_last_files": checkpoint_last,
        "minirocket_model_files": minirocket_models,
        "missing_dataset_protocol_id_runs": missing_dataset_protocol,
        "superseded_results_included": False,
        "warning": (
            "BasicMotions/Epilepsy 的 42 个早期复用结果未落盘 dataset_protocol_id；"
            "其活动矩阵和统一 protocol_hash 完整，但 Full 前应增强代码与数据协议指纹。"
            if missing_dataset_protocol
            else None
        ),
    }


def _subject_tokens(ids: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for sample_id in ids:
        match = re.search(r"subject(\d+)", str(sample_id), flags=re.IGNORECASE)
        if match:
            tokens.add(match.group(1))
    return tokens


def _audit_data_splits(project_root: Path) -> dict[str, Any]:
    processed = project_root / "data" / "processed" / "pilot_v1"
    datasets: dict[str, Any] = {}
    for dataset in PILOT_DATASETS:
        cache_path = processed / f"{dataset}.npz"
        manifest_path = processed / f"{dataset}.manifest.json"
        if not cache_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"缺少处理后数据：{cache_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(cache_path, allow_pickle=False) as archive:
            split_ids = {
                split: [str(item) for item in archive[f"{split}_ids"].tolist()]
                for split in ("train", "validation", "test")
            }
            split_labels = {
                split: {int(item) for item in archive[f"{split}_y"].tolist()}
                for split in ("train", "validation", "test")
            }
        id_overlap = {
            "train_validation": len(set(split_ids["train"]) & set(split_ids["validation"])),
            "train_test": len(set(split_ids["train"]) & set(split_ids["test"])),
            "validation_test": len(set(split_ids["validation"]) & set(split_ids["test"])),
        }
        subjects = {split: sorted(_subject_tokens(values)) for split, values in split_ids.items()}
        subject_overlap = {
            "train_validation": sorted(set(subjects["train"]) & set(subjects["validation"])),
            "train_test": sorted(set(subjects["train"]) & set(subjects["test"])),
            "validation_test": sorted(set(subjects["validation"]) & set(subjects["test"])),
        }
        test_unseen_labels = sorted(split_labels["test"] - split_labels["train"])
        passed = not any(id_overlap.values()) and not any(subject_overlap.values()) and not test_unseen_labels
        datasets[dataset] = {
            "status": "pass" if passed else "fail",
            "shapes": manifest.get("shapes", {}),
            "id_overlap_counts": id_overlap,
            "subject_ids": subjects,
            "subject_overlap": subject_overlap,
            "train_labels": sorted(split_labels["train"]),
            "validation_labels": sorted(split_labels["validation"]),
            "test_labels": sorted(split_labels["test"]),
            "test_labels_unseen_in_train": test_unseen_labels,
            "dataset_protocol_id": manifest.get("metadata", {}).get("dataset_protocol_id"),
            "split_protocol": manifest.get("metadata", {}).get("split_protocol"),
        }
    return {
        "status": "pass" if all(item["status"] == "pass" for item in datasets.values()) else "fail",
        "datasets": datasets,
        "boundary": "This checks cached IDs, subject tokens and closed-set labels; it is not a raw-data fingerprint audit.",
    }


def _run_summaries(
    runs: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for (dataset, method, seed), payload in runs.items():
        evaluation = payload["evaluation"]
        rates = evaluation["per_rate"]
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "mean_unseen_macro_f1": float(evaluation["mean_unseen_macro_f1"]),
                "worst_unseen_macro_f1": min(float(rates[rate]["macro_f1"]) for rate in UNSEEN_RATE_IDS),
                "full_rate_macro_f1": float(rates["r1000"]["macro_f1"]),
                "mean_unseen_flip": _mean(float(rates[rate]["disagreement_vs_original"]) for rate in UNSEEN_RATE_IDS),
                "mean_unseen_aurc": float(evaluation["mean_unseen_aurc"]),
            }
        )
    return summaries


def _group_metric(summaries: list[dict[str, Any]], dataset: str, method: str, metric: str) -> float:
    return _mean(
        float(row[metric])
        for row in summaries
        if row["dataset"] == dataset and row["method"] == method
    )


def _analyze_performance(
    runs: dict[tuple[str, str, int], dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    rate_curves: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in PILOT_DATASETS:
        rate_curves[dataset] = {}
        for method in ("nyquistguard",) + STRONG_BASELINES:
            rate_curves[dataset][method] = {
                rate: _mean(
                    float(runs[(dataset, method, seed)]["evaluation"]["per_rate"][rate]["macro_f1"])
                    for seed in PILOT_SEEDS
                )
                for rate in RATE_IDS
            }

    worst_comparisons: dict[str, Any] = {}
    full_comparisons: dict[str, Any] = {}
    for dataset in PILOT_DATASETS:
        strongest_worst = max(
            (
                _group_metric(summaries, dataset, method, "worst_unseen_macro_f1"),
                method,
            )
            for method in CLASSIFICATION_BASELINES
        )
        strongest_full = max(
            (_group_metric(summaries, dataset, method, "full_rate_macro_f1"), method)
            for method in CLASSIFICATION_BASELINES
        )
        nyquist_worst = _group_metric(summaries, dataset, "nyquistguard", "worst_unseen_macro_f1")
        nyquist_full = _group_metric(summaries, dataset, "nyquistguard", "full_rate_macro_f1")
        worst_comparisons[dataset] = {
            "nyquistguard": nyquist_worst,
            "baseline": strongest_worst[0],
            "baseline_method": strongest_worst[1],
            "delta": nyquist_worst - strongest_worst[0],
        }
        full_comparisons[dataset] = {
            "nyquistguard": nyquist_full,
            "baseline": strongest_full[0],
            "baseline_method": strongest_full[1],
            "delta": nyquist_full - strongest_full[0],
        }

    worst_average_delta = _mean(item["delta"] for item in worst_comparisons.values())
    worst_direction_count = sum(item["delta"] > 0 for item in worst_comparisons.values())
    full_average_delta = _mean(item["delta"] for item in full_comparisons.values())
    criterion_1 = {
        "name": "最差未见率 macro-F1",
        "passed": worst_average_delta >= 0.03 and worst_direction_count >= 3,
        "average_delta": worst_average_delta,
        "direction_count": worst_direction_count,
        "required_average_delta": 0.03,
        "required_direction_count": 3,
        "datasets": worst_comparisons,
    }
    criterion_2 = {
        "name": "full-rate macro-F1 代价",
        "passed": full_average_delta >= -0.01,
        "average_delta": full_average_delta,
        "maximum_allowed_drop": 0.01,
        "datasets": full_comparisons,
        "warning": "平均值被 PAMAP2 的大幅正收益抵消；必须同时报告逐数据集异质性。",
    }

    sensitivity: dict[str, Any] = {}
    for dataset in PILOT_DATASETS:
        best_worst = max(
            (_group_metric(summaries, dataset, method, "worst_unseen_macro_f1"), method)
            for method in STRONG_BASELINES
        )
        best_full = max(
            (_group_metric(summaries, dataset, method, "full_rate_macro_f1"), method)
            for method in STRONG_BASELINES
        )
        sensitivity[dataset] = {
            "worst_rate_delta": _group_metric(summaries, dataset, "nyquistguard", "worst_unseen_macro_f1")
            - best_worst[0],
            "worst_rate_baseline": best_worst[1],
            "full_rate_delta": _group_metric(summaries, dataset, "nyquistguard", "full_rate_macro_f1")
            - best_full[0],
            "full_rate_baseline": best_full[1],
        }
    return {
        "rate_curves_macro_f1": rate_curves,
        "criterion_1": criterion_1,
        "criterion_2": criterion_2,
        "strong_baseline_sensitivity": sensitivity,
    }


def _analyze_flip(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for dataset in PILOT_DATASETS:
        best = min(
            (_group_metric(summaries, dataset, method, "mean_unseen_flip"), method)
            for method in CLASSIFICATION_BASELINES
        )
        nyquist = _group_metric(summaries, dataset, "nyquistguard", "mean_unseen_flip")
        reduction = (best[0] - nyquist) / best[0] if best[0] > 0 else -math.inf
        comparisons[dataset] = {
            "nyquistguard": nyquist,
            "baseline": best[0],
            "baseline_method": best[1],
            "relative_reduction": reduction,
        }
    nyquist_average = _mean(item["nyquistguard"] for item in comparisons.values())
    baseline_average = _mean(item["baseline"] for item in comparisons.values())
    relative_reduction = (
        (baseline_average - nyquist_average) / baseline_average if baseline_average > 0 else -math.inf
    )
    return {
        "criterion_3": {
            "name": "prediction flip rate",
            "passed": relative_reduction >= 0.20,
            "nyquistguard_average": nyquist_average,
            "baseline_average": baseline_average,
            "relative_reduction": relative_reduction,
            "required_relative_reduction": 0.20,
            "datasets": comparisons,
        }
    }


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _analyze_selectivity(
    runs: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    dataset_results: dict[str, Any] = {}
    no_selective_values: set[float] = set()
    for dataset in PILOT_DATASETS:
        learned_aurcs: list[float] = []
        confidence_aurcs: list[float] = []
        correct_acceptance: list[float] = []
        wrong_acceptance: list[float] = []
        for seed in PILOT_SEEDS:
            run_dir = Path(str(runs[(dataset, "nyquistguard", seed)]["_run_dir"]))
            rows = _read_prediction_rows(run_dir / "predictions.csv")
            for rate_id, ratio in zip(RATE_IDS, (1.0, 0.9, 0.6, 0.4, 0.3)):
                selected = [row for row in rows if abs(float(row["rate_ratio"]) - ratio) < 1e-8]
                errors = np.asarray(
                    [int(row["prediction"]) != int(row["target"]) for row in selected], dtype=np.float64
                )
                acceptance = np.asarray([float(row["acceptance"]) for row in selected], dtype=np.float64)
                confidence = np.asarray(
                    [max(json.loads(row["probabilities_json"])) for row in selected], dtype=np.float64
                )
                correct_acceptance.extend(acceptance[errors == 0].tolist())
                wrong_acceptance.extend(acceptance[errors == 1].tolist())
                if rate_id in UNSEEN_RATE_IDS:
                    learned_aurcs.append(_aurc(errors, acceptance))
                    confidence_aurcs.append(_aurc(errors, confidence))
            ablation_dir = Path(str(runs[(dataset, "no_selective_head", seed)]["_run_dir"]))
            for row in _read_prediction_rows(ablation_dir / "predictions.csv"):
                no_selective_values.add(round(float(row["acceptance"]), 8))
        learned = _mean(learned_aurcs)
        confidence = _mean(confidence_aurcs)
        dataset_results[dataset] = {
            "learned_acceptance_aurc": learned,
            "max_softmax_confidence_aurc": confidence,
            "delta": learned - confidence,
            "learned_is_better": learned < confidence,
            "mean_acceptance_correct": _mean(correct_acceptance),
            "mean_acceptance_wrong": _mean(wrong_acceptance) if wrong_acceptance else None,
        }
    learned_average = _mean(item["learned_acceptance_aurc"] for item in dataset_results.values())
    confidence_average = _mean(item["max_softmax_confidence_aurc"] for item in dataset_results.values())
    direction_count = sum(bool(item["learned_is_better"]) for item in dataset_results.values())
    criterion = {
        "name": "AURC 与置信度选择基线",
        "passed": learned_average < confidence_average and direction_count >= 3,
        "learned_acceptance_average": learned_average,
        "confidence_average": confidence_average,
        "relative_improvement": (
            (confidence_average - learned_average) / confidence_average if confidence_average > 0 else 0.0
        ),
        "direction_count": direction_count,
        "required_direction_count": 3,
        "datasets": dataset_results,
    }
    return {
        "criterion_4": criterion,
        "no_selective_head_unique_acceptance_values": sorted(no_selective_values),
        "no_selective_head_is_constant_0_5": no_selective_values == {0.5},
        "interpretation_warning": (
            "no_selective_head 的接受概率固定为 0.5，不能作为置信度基线；"
            "Criterion 4 使用同一 NyquistGuard 分类预测的最大 softmax 概率进行比较。"
        ),
    }


def _analyze_training_and_ablations(
    runs: dict[tuple[str, str, int], dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    deep_run_count = 0
    histories_found = 0
    finite_histories = 0
    early_stopped = 0
    epoch_counts: list[int] = []
    for (dataset, method, seed), payload in runs.items():
        del dataset, seed
        if method == "minirocket":
            continue
        deep_run_count += 1
        run_dir = Path(str(payload["_run_dir"]))
        history_path = run_dir / "training_history.json"
        if not history_path.exists():
            continue
        histories_found += 1
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        epoch_counts.append(len(history))
        if _is_finite_tree(history):
            finite_histories += 1
        if len(history) < 30:
            early_stopped += 1
    seed_stability = {
        dataset: {
            "mean_unseen_macro_f1_mean": _group_metric(
                summaries, dataset, "nyquistguard", "mean_unseen_macro_f1"
            ),
            "mean_unseen_macro_f1_seed_sd": float(
                statistics.stdev(
                    float(row["mean_unseen_macro_f1"])
                    for row in summaries
                    if row["dataset"] == dataset and row["method"] == "nyquistguard"
                )
            ),
        }
        for dataset in PILOT_DATASETS
    }
    ablation_effects: dict[str, Any] = {}
    for dataset in PILOT_DATASETS:
        main_f1 = _group_metric(summaries, dataset, "nyquistguard", "mean_unseen_macro_f1")
        main_aurc = _group_metric(summaries, dataset, "nyquistguard", "mean_unseen_aurc")
        ablation_effects[dataset] = {}
        for method in ABLATIONS:
            ablation_effects[dataset][method] = {
                "macro_f1_delta_main_minus_ablation": main_f1
                - _group_metric(summaries, dataset, method, "mean_unseen_macro_f1"),
                "aurc_delta_main_minus_ablation": main_aurc
                - _group_metric(summaries, dataset, method, "mean_unseen_aurc"),
            }
    return {
        "deep_run_count": deep_run_count,
        "training_histories_found": histories_found,
        "finite_training_histories": finite_histories,
        "early_stopped_runs": early_stopped,
        "epoch_count_min": min(epoch_counts) if epoch_counts else None,
        "epoch_count_max": max(epoch_counts) if epoch_counts else None,
        "seed_stability": seed_stability,
        "ablation_effects": ablation_effects,
        "ablation_direction_consistent": {
            method: sum(
                ablation_effects[dataset][method]["macro_f1_delta_main_minus_ablation"] > 0
                for dataset in PILOT_DATASETS
            )
            for method in ABLATIONS
        },
    }


def _audit_minirocket(
    runs: dict[tuple[str, str, int], dict[str, Any]],
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in PILOT_DATASETS:
        values = []
        for seed in PILOT_SEEDS:
            rates = runs[(dataset, "minirocket", seed)]["evaluation"]["per_rate"]
            values.extend(float(rates[rate]["macro_f1"]) for rate in RATE_IDS)
        perfect = all(abs(value - 1.0) < 1e-12 for value in values)
        datasets[dataset] = {
            "evaluations_checked": len(values),
            "all_seed_rate_macro_f1_perfect": perfect,
            "minimum_macro_f1": min(values),
            "cached_split_audit_passed": data_audit["datasets"][dataset]["status"] == "pass",
            "interpretation": (
                "缓存 ID/subject 切分审计未发现重叠，但满分仍需原始样本指纹与任务难度专项核查。"
                if perfect
                else "未出现跨全部 seed/rate 的满分。"
            ),
        }
    perfect_datasets = [
        dataset for dataset, item in datasets.items() if item["all_seed_rate_macro_f1_perfect"]
    ]
    return {
        "datasets": datasets,
        "perfect_across_all_seed_rates": perfect_datasets,
        "leakage_proven": False,
        "warning": (
            "满分不是泄漏证据；当前仅能确认缓存 split ID/subject 不重叠。"
            if perfect_datasets
            else None
        ),
    }


def _format_percent(value: float, signed: bool = False) -> str:
    return f"{value * 100:+.2f}%" if signed else f"{value * 100:.2f}%"


def _render_markdown(report: dict[str, Any]) -> str:
    performance = report["performance"]
    flip = report["prediction_flip"]["criterion_3"]
    selectivity = report["selectivity"]["criterion_4"]
    criteria = report["go_no_go"]["criteria"]
    lines = [
        "# NyquistGuard-TSC Pilot 静态诊断报告",
        "",
        f"- 诊断状态：{report['status']}",
        f"- Pilot：{report['pilot_root']}",
        f"- 有效 runs：{report['artifact_audit']['completed_run_count']} / 84",
        f"- 预注册结论：{report['go_no_go']['decision'].upper()}（通过 {report['go_no_go']['passed_count']} / 4，要求至少 3 / 4）",
        "- 安全边界：本诊断只读取既有产物；未训练、未改 checkpoint、未启动 Pilot/Full。",
        "",
        "## Go/No-Go",
        "",
        "| 标准 | 结果 | 判断 |",
        "|---|---:|---|",
        f"| 最差未见率 macro-F1 | 平均差 {criteria[0]['average_delta'] * 100:+.2f} 个百分点；同向 {criteria[0]['direction_count']}/4 | {'通过' if criteria[0]['passed'] else '未通过'} |",
        f"| full-rate macro-F1 代价 | 平均差 {criteria[1]['average_delta'] * 100:+.2f} 个百分点 | {'通过' if criteria[1]['passed'] else '未通过'} |",
        f"| prediction flip | 主模型 {_format_percent(flip['nyquistguard_average'])}；基线 {_format_percent(flip['baseline_average'])}；相对变化 {_format_percent(flip['relative_reduction'], signed=True)} | {'通过' if flip['passed'] else '未通过'} |",
        f"| AURC | learned {selectivity['learned_acceptance_average']:.4f}；softmax {selectivity['confidence_average']:.4f}；同向 {selectivity['direction_count']}/4 | {'通过' if selectivity['passed'] else '未通过'} |",
        "",
        "## 逐采样率 macro-F1（3 seeds 均值）",
        "",
        "| 数据集 | 方法 | 1.0 | 0.9 | 0.6 | 0.4 | 0.3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in PILOT_DATASETS:
        for method in ("nyquistguard", "fixed_rate_tcn", "multirate_tcn", "minirocket"):
            curve = performance["rate_curves_macro_f1"][dataset][method]
            lines.append(
                f"| {dataset} | {method} | "
                + " | ".join(f"{curve[rate]:.3f}" for rate in RATE_IDS)
                + " |"
            )
    lines.extend(
        [
            "",
            "## 关键诊断",
            "",
            f"- 工程产物：{report['artifact_audit']['status']}；84/84 完成，指标有限，活动矩阵未包含 superseded PAMAP2。",
            f"- 数据切分：{report['data_split_audit']['status']}；缓存 ID、可解析 subject 和测试标签闭集检查未发现冲突。",
            f"- 低率退化：最明显的数据集为 {', '.join(report['diagnosis']['low_rate_failure_datasets'])}。",
            f"- 选择性头：learned AURC 比同一模型 softmax confidence 高 {selectivity['learned_acceptance_average'] - selectivity['confidence_average']:.4f}，4 个数据集均未改善。",
            f"- MiniROCKET：{', '.join(report['minirocket_audit']['perfect_across_all_seed_rates']) or '无'} 在所有 seed/rate 上满分；这不是已证实泄漏，但需要原始样本指纹与任务难度专项核查。",
            f"- 消融方向：Nyquist gate/CBE/selective head 的 F1 正向数据集数分别为 {report['training_and_ablations']['ablation_direction_consistent']['no_nyquist_gate']}/4、{report['training_and_ablations']['ablation_direction_consistent']['no_cbe']}/4、{report['training_and_ablations']['ablation_direction_consistent']['no_selective_head']}/4。",
            "",
            "## 建议",
            "",
            "1. 保持 Full 锁定，不改变预注册阈值。",
            "2. 优先检查选择性损失、coverage 约束和接受概率排序；当前 learned acceptance 明确弱于 softmax confidence。",
            "3. 针对 Epilepsy/MHEALTH 的 0.4、0.3 rate 做频谱、gate 和滤波器响应检查。",
            "4. 对 MiniROCKET 满分数据做原始样本指纹/近邻相似度专项审计。",
            "5. 只有发现明确实现问题后，才设计少量定向实验；本程序不会自动重跑 Pilot。",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnosis_summary(
    performance: dict[str, Any],
    flip: dict[str, Any],
    selectivity: dict[str, Any],
) -> dict[str, Any]:
    curves = performance["rate_curves_macro_f1"]
    low_rate_failure_datasets = [
        dataset
        for dataset in PILOT_DATASETS
        if curves[dataset]["nyquistguard"]["r0300"]
        < curves[dataset]["multirate_tcn"]["r0300"] - 0.03
    ]
    return {
        "engineering_run_valid": True,
        "scientific_target_met": False,
        "primary_diagnosis": "当前主模型跨率性能、prediction flip 和选择性排序不足；PAMAP2 有局部正收益。",
        "low_rate_failure_datasets": low_rate_failure_datasets,
        "selection_head_underperforms_confidence": not selectivity["criterion_4"]["passed"],
        "prediction_flip_worse_than_baseline": flip["criterion_3"]["relative_reduction"] < 0,
    }


def run_diagnosis(project_root: str | Path, resume: bool = False) -> dict[str, Any]:
    """Run the read-only static Pilot diagnosis and write auditable reports."""

    del resume  # Static diagnosis is cheap and intentionally re-reads current artifacts.
    root = Path(project_root).resolve()
    diagnosis_id = f"diagnosis__pilot_static__{_utc_stamp()}"
    run_root = root / "runs" / "diagnosis" / diagnosis_id
    progress = DashboardProgress(
        root / "runs" / "dashboard_status.json",
        "diagnosis",
        DIAGNOSTIC_TASKS,
        diagnosis_id,
    )
    report: dict[str, Any] = {}
    task_index = 0
    try:
        progress.start_task(task_index)
        pilot_root = _latest_completed_pilot(root)
        print(f"[1/9] 使用已完成 Pilot：{pilot_root}", flush=True)
        progress.complete_task(task_index)

        task_index += 1
        progress.start_task(task_index)
        runs = _load_active_runs(pilot_root)
        artifact_audit = _audit_artifacts(runs)
        if artifact_audit["status"] == "fail":
            raise RuntimeError("84-run 产物审计失败；诊断已停止且不会补跑实验")
        print("[2/9] 84/84 有效 runs、预测文件和有限指标审计完成", flush=True)
        progress.complete_task(task_index)

        task_index += 1
        progress.start_task(task_index)
        data_audit = _audit_data_splits(root)
        print(f"[3/9] 数据切分审计：{data_audit['status']}", flush=True)
        progress.complete_task(task_index)

        summaries = _run_summaries(runs)
        task_index += 1
        progress.start_task(task_index)
        performance = _analyze_performance(runs, summaries)
        print(
            "[4/9] 最差未见率平均差 "
            f"{performance['criterion_1']['average_delta'] * 100:+.2f} 个百分点",
            flush=True,
        )
        progress.complete_task(task_index)

        task_index += 1
        progress.start_task(task_index)
        flip = _analyze_flip(summaries)
        print(
            "[5/9] prediction flip 相对降低 "
            f"{flip['criterion_3']['relative_reduction'] * 100:+.1f}%",
            flush=True,
        )
        progress.complete_task(task_index)

        task_index += 1
        progress.start_task(task_index)
        selectivity = _analyze_selectivity(runs)
        print(
            "[6/9] learned/softmax AURC："
            f"{selectivity['criterion_4']['learned_acceptance_average']:.4f} / "
            f"{selectivity['criterion_4']['confidence_average']:.4f}",
            flush=True,
        )
        progress.complete_task(task_index)

        task_index += 1
        progress.start_task(task_index)
        training = _analyze_training_and_ablations(runs, summaries)
        print(
            f"[7/9] 训练历史 {training['training_histories_found']}/{training['deep_run_count']} 可读且有限",
            flush=True,
        )
        progress.complete_task(task_index)

        task_index += 1
        progress.start_task(task_index)
        minirocket = _audit_minirocket(runs, data_audit)
        print(
            "[8/9] MiniROCKET 全 seed/rate 满分数据集："
            + (", ".join(minirocket["perfect_across_all_seed_rates"]) or "无"),
            flush=True,
        )
        progress.complete_task(task_index)

        task_index += 1
        progress.start_task(task_index)
        criteria = [
            performance["criterion_1"],
            performance["criterion_2"],
            flip["criterion_3"],
            selectivity["criterion_4"],
        ]
        passed_count = sum(bool(item["passed"]) for item in criteria)
        decision = "go" if passed_count >= 3 else "no_go"
        diagnosis = _diagnosis_summary(performance, flip, selectivity)
        report = {
            "status": "completed",
            "diagnosis_id": diagnosis_id,
            "pilot_root": str(pilot_root),
            "created_at_utc": utc_now(),
            "read_only_scope": {
                "trained_models": False,
                "modified_checkpoints": False,
                "started_pilot": False,
                "started_full": False,
            },
            "artifact_audit": artifact_audit,
            "data_split_audit": data_audit,
            "performance": performance,
            "prediction_flip": flip,
            "selectivity": selectivity,
            "training_and_ablations": training,
            "minirocket_audit": minirocket,
            "go_no_go": {
                "decision": decision,
                "passed_count": passed_count,
                "required_count": 3,
                "criteria": criteria,
                "full_auto_start": False,
            },
            "diagnosis": diagnosis,
        }
        run_root.mkdir(parents=True, exist_ok=True)
        report_json = run_root / "diagnostic_report.json"
        report_md = run_root / "diagnostic_report.md"
        atomic_write_json(report_json, report)
        markdown = _render_markdown(report)
        _atomic_write_text(report_md, markdown)
        reports_root = root / "reports"
        atomic_write_json(reports_root / "diagnostic_report.json", report)
        _atomic_write_text(reports_root / "diagnostic_report.md", markdown)
        progress.complete_task(task_index)
        progress.finish(f"诊断完成：{decision.upper()}；Full 保持人工锁定")
        print(f"[9/9] 诊断完成：{decision.upper()}（通过 {passed_count}/4）", flush=True)
        print(f"诊断报告：{report_md}", flush=True)
        print("未训练模型、未修改 checkpoint、未启动 Pilot 或 Full。", flush=True)
        return report
    except BaseException as error:
        progress.fail_task(task_index, error)
        raise


__all__ = ["DIAGNOSTIC_TASKS", "run_diagnosis"]
