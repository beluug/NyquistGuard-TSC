"""Manual, resumable Full benchmark for the frozen v3.10 candidate."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.data import FULL_DATASETS, PreparedDataset, prepare_full_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.metrics import align_probability_columns, classification_metrics
from nyquistguard.experiments.pilot import (
    PilotRunSpec,
    _deep_model,
    _fixed_length_view,
    _predict_deep,
    _seed_everything,
    _train_deep,
)
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.experiments.v3_core_micro import _train_core
from nyquistguard.experiments.v3_guarded_reliability import _guarded_evaluation


FULL_METHODS = (
    "v3_10",
    "v1_nyquistguard",
    "fixed_rate_tcn",
    "multirate_tcn",
    "minirocket",
    "multirocket",
    "v3_10_no_nyquist_gate",
)
FULL_SEEDS = (17, 42, 2026)


@dataclass(frozen=True)
class FullRunSpec:
    dataset_id: str
    method: str
    seed: int

    @property
    def run_key(self) -> str:
        return f"{self.dataset_id}__{self.method}__seed{self.seed}"


def build_full_matrix(config: dict[str, Any]) -> list[FullRunSpec]:
    datasets = tuple(str(value) for value in config["datasets"])
    methods = tuple(str(value) for value in config["methods"])
    seeds = tuple(int(value) for value in config["seeds"])
    if datasets != FULL_DATASETS:
        raise ValueError(f"Full datasets must remain frozen as {FULL_DATASETS}")
    if methods != FULL_METHODS:
        raise ValueError(f"Full methods must remain frozen as {FULL_METHODS}")
    if seeds != FULL_SEEDS:
        raise ValueError(f"Full seeds must remain frozen as {FULL_SEEDS}")
    matrix = [FullRunSpec(dataset, method, seed) for dataset in datasets for method in methods for seed in seeds]
    if len(matrix) != 210 or len({spec.run_key for spec in matrix}) != 210:
        raise ValueError("Full matrix must contain exactly 210 unique runs")
    rate_protocol = config["rate_protocol"]
    if tuple(float(value) for value in rate_protocol["test_ratios"]) != (1.0, 0.9, 0.6, 0.4, 0.3):
        raise ValueError("Full test-rate protocol changed after freeze")
    return matrix


def _protocol_hash(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update((root / "src/nyquistguard/experiments/full.py").read_bytes())
    digest.update((root / "src/nyquistguard/data/full_datasets.py").read_bytes())
    digest.update(b"nyquistguard-full-v1-v3.10")
    return digest.hexdigest()


def _validate_confirmation(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = json.loads((root / config["source_confirmation_report"]).read_text(encoding="utf-8"))
    if report.get("protocol_version") != "v3_10_independent_confirmation_v1":
        raise ValueError("Full requires the frozen v3.10 independent confirmation report")
    if not report.get("decision", {}).get("passed", False):
        raise ValueError("Full remains locked because independent confirmation did not pass")
    return report


def _find_resume_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for manifest_path in sorted(parent.glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("protocol_hash") == protocol_hash:
            return manifest_path.parent
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _jsd(reference: np.ndarray, current: np.ndarray) -> float:
    p = np.asarray(reference, dtype=np.float64).clip(1e-12, 1.0)
    q = np.asarray(current, dtype=np.float64).clip(1e-12, 1.0)
    middle = 0.5 * (p + q)
    values = 0.5 * np.sum(p * np.log(p / middle), axis=1) + 0.5 * np.sum(q * np.log(q / middle), axis=1)
    return float(np.mean(values))


def _classical_estimator(
    dataset: PreparedDataset,
    spec: FullRunSpec,
    config: dict[str, Any],
    run_dir: Path,
    resume: bool,
) -> object:
    estimator_path = run_dir / "estimator.pkl"
    if resume and estimator_path.exists():
        with estimator_path.open("rb") as handle:
            return pickle.load(handle)
    settings = config["classical_baselines"]
    if spec.method == "minirocket":
        # Pilot helper is deliberately bypassed here: Full's frozen baseline is
        # trained at the reference rate, matching published default practice.
        from aeon.classification.convolution_based import MiniRocketClassifier

        estimator = MiniRocketClassifier(
            n_kernels=int(settings["n_kernels"]),
            n_jobs=int(settings["n_jobs"]),
            random_state=spec.seed,
        )
    elif spec.method == "multirocket":
        from aeon.classification.convolution_based import MultiRocketClassifier

        estimator = MultiRocketClassifier(
            n_kernels=int(settings["n_kernels"]),
            n_jobs=int(settings["n_jobs"]),
            random_state=spec.seed,
        )
    else:
        raise ValueError(f"not a classical Full method: {spec.method}")
    estimator.fit(dataset.train.x, dataset.train.y)
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = estimator_path.with_name(estimator_path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(estimator, handle)
    os.replace(temporary, estimator_path)
    return estimator


def _fit_method(
    dataset: PreparedDataset,
    spec: FullRunSpec,
    base_config: dict[str, Any],
    full_config: dict[str, Any],
    protocol_hash: str,
    run_dir: Path,
    resume: bool,
) -> object:
    if spec.method in {"minirocket", "multirocket"}:
        return _classical_estimator(dataset, spec, full_config, run_dir, resume)
    if spec.method in {"v3_10", "v3_10_no_nyquist_gate"}:
        model, _history = _train_core(
            dataset,
            base_config,
            full_config,
            protocol_hash,
            run_dir,
            seed=spec.seed,
            resume=resume,
            deadline=math.inf,
            model_variant=("v3_no_nyquist_gate" if spec.method == "v3_10_no_nyquist_gate" else "no_selective_head"),
        )
        return model
    pilot_method = {
        "v1_nyquistguard": "nyquistguard",
        "fixed_rate_tcn": "fixed_rate_tcn",
        "multirate_tcn": "multirate_tcn",
    }[spec.method]
    return _train_deep(
        dataset,
        PilotRunSpec(dataset.dataset_id, pilot_method, spec.seed),
        base_config,
        protocol_hash,
        run_dir,
        resume,
    )


def _evaluate_method(
    model: object,
    dataset: PreparedDataset,
    spec: FullRunSpec,
    base_config: dict[str, Any],
    ratios: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_rate: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {"targets": dataset.test.y, "sample_ids": dataset.test.ids}
    reference_probabilities: np.ndarray | None = None
    reference_predictions: np.ndarray | None = None
    for ratio in ratios:
        if spec.method in {"minirocket", "multirocket"}:
            values = _fixed_length_view(dataset.test.x, dataset.sampling_rate_hz, ratio, base_config)
            probabilities = align_probability_columns(
                model.predict_proba(values),  # type: ignore[attr-defined]
                np.asarray(model.classes_),  # type: ignore[attr-defined]
                len(dataset.class_names),
            )
            logits = np.log(probabilities.clip(1e-12))
            model_acceptance = probabilities.max(axis=1)
        else:
            logits, model_acceptance = _predict_deep(
                model, dataset.test, dataset.sampling_rate_hz, ratio, base_config, device  # type: ignore[arg-type]
            )
            probabilities = _softmax(logits)
        confidence = probabilities.max(axis=1)
        metrics = classification_metrics(dataset.test.y, logits, confidence)
        model_metrics = classification_metrics(dataset.test.y, logits, model_acceptance)
        predictions = probabilities.argmax(axis=1)
        if reference_probabilities is None:
            reference_probabilities = probabilities
            reference_predictions = predictions
        rate_key = f"r{int(round(ratio * 1000)):04d}"
        per_rate[rate_key] = {
            **metrics,
            "model_acceptance_aurc": float(model_metrics["aurc"]),
            "disagreement_vs_full_rate": float(np.mean(predictions != reference_predictions)),
            "jsd_vs_full_rate": _jsd(reference_probabilities, probabilities),
        }
        arrays[f"probabilities_{rate_key}"] = probabilities.astype(np.float32)
        arrays[f"acceptance_{rate_key}"] = np.asarray(model_acceptance, dtype=np.float32)
    unseen = [per_rate[f"r{int(round(ratio * 1000)):04d}"] for ratio in ratios if ratio != 1.0]
    summary = {
        "mean_unseen_macro_f1": float(np.mean([row["macro_f1"] for row in unseen])),
        "worst_unseen_macro_f1": float(np.min([row["macro_f1"] for row in unseen])),
        "mean_unseen_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in unseen])),
        "mean_unseen_aurc": float(np.mean([row["aurc"] for row in unseen])),
        "mean_unseen_disagreement": float(np.mean([row["disagreement_vs_full_rate"] for row in unseen])),
        "mean_unseen_jsd": float(np.mean([row["jsd_vs_full_rate"] for row in unseen])),
        "full_rate_macro_f1": float(per_rate["r1000"]["macro_f1"]),
        "full_rate_balanced_accuracy": float(per_rate["r1000"]["balanced_accuracy"]),
        "per_rate": per_rate,
    }
    return summary, arrays


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _profile_deep_model(model: object, dataset: PreparedDataset) -> dict[str, Any]:
    if not isinstance(model, torch.nn.Module):
        return {"parameters": None, "flops_per_sample": None}
    device = next(model.parameters()).device
    parameters = sum(parameter.numel() for parameter in model.parameters())
    sample = torch.from_numpy(dataset.test.x[:1]).to(device)
    rate = float(dataset.sampling_rate_hz)
    flops: int | None = None
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profile:
            model(sample, rate)
        flops = int(sum(event.flops for event in profile.key_averages() if event.flops))
    except (RuntimeError, AttributeError):
        flops = None
    return {"parameters": int(parameters), "flops_per_sample": flops}


def _run_one(
    root: Path,
    run_root: Path,
    dataset: PreparedDataset,
    spec: FullRunSpec,
    base_config: dict[str, Any],
    full_config: dict[str, Any],
    reliability_config: dict[str, Any],
    protocol_hash: str,
    resume: bool,
) -> dict[str, Any]:
    run_dir = run_root / spec.run_key
    metrics_path = run_dir / "metrics.json"
    if resume and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol_hash") == protocol_hash:
            print(f"[{spec.run_key}] completed result reused", flush=True)
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "status.json", {"status": "running", "spec": asdict(spec), "protocol_hash": protocol_hash, "started_at_utc": utc_now()})
    started = time.monotonic()
    try:
        _seed_everything(spec.seed)
        train_started = time.monotonic()
        model = _fit_method(dataset, spec, base_config, full_config, protocol_hash, run_dir, resume)
        training_seconds = time.monotonic() - train_started
        evaluation_started = time.monotonic()
        ratios = tuple(float(value) for value in full_config["rate_protocol"]["test_ratios"])
        evaluation, arrays = _evaluate_method(model, dataset, spec, base_config, ratios)
        reliability = None
        if spec.method == "v3_10":
            reliability = _guarded_evaluation(
                model, dataset, base_config, reliability_config, spec.seed,
                float(full_config["reliability_controller"]["minimum_absolute_validation_aurc_gain"]),
            )
        evaluation_seconds = time.monotonic() - evaluation_started
        _atomic_npz(run_dir / "predictions.npz", arrays)
        efficiency = _profile_deep_model(model, dataset)
        efficiency.update(
            training_seconds=float(training_seconds),
            evaluation_seconds=float(evaluation_seconds),
            artifact_bytes=int(sum(path.stat().st_size for path in run_dir.iterdir() if path.is_file())),
        )
        result = {
            "status": "completed",
            "protocol_hash": protocol_hash,
            "spec": asdict(spec),
            "dataset_protocol_id": dataset.metadata.get("dataset_protocol_id"),
            "sampling_rate_hz": dataset.sampling_rate_hz,
            "class_names": list(dataset.class_names),
            "evaluation": evaluation,
            "reliability": reliability,
            "efficiency": efficiency,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": utc_now(),
        }
        atomic_write_json(metrics_path, result)
        atomic_write_json(run_dir / "status.json", result)
        return result
    except BaseException as error:
        atomic_write_json(run_dir / "status.json", {"status": "failed", "spec": asdict(spec), "protocol_hash": protocol_hash, "error": f"{type(error).__name__}: {error}", "failed_at_utc": utc_now()})
        raise


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, (count - index) * float(p_values[name]))
        adjusted[name] = min(1.0, running)
    return adjusted


def paired_statistics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    candidate = "v3_10"
    baselines = [method for method in FULL_METHODS if method != candidate]
    by_key = {(row["dataset_id"], row["method"], row["seed"]): row for row in rows}
    seed_level: dict[str, list[float]] = {}
    dataset_level: dict[str, list[float]] = {}
    raw_p: dict[str, float] = {}
    intervals: dict[str, list[float]] = {}
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    resamples = int(config["statistics"]["bootstrap_resamples"])
    for baseline in baselines:
        deltas = []
        dataset_means = []
        for dataset_id in FULL_DATASETS:
            dataset_deltas = []
            for seed in FULL_SEEDS:
                delta = float(by_key[(dataset_id, candidate, seed)]["mean_unseen_macro_f1"] - by_key[(dataset_id, baseline, seed)]["mean_unseen_macro_f1"])
                deltas.append(delta)
                dataset_deltas.append(delta)
            dataset_means.append(float(np.mean(dataset_deltas)))
        seed_level[baseline] = deltas
        dataset_level[baseline] = dataset_means
        if np.allclose(dataset_means, 0.0):
            raw_p[baseline] = 1.0
        else:
            raw_p[baseline] = float(wilcoxon(dataset_means, alternative="two-sided", method="auto").pvalue)
        boot = np.mean(rng.choice(dataset_means, size=(resamples, len(dataset_means)), replace=True), axis=1)
        intervals[baseline] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    adjusted = _holm(raw_p)
    return {
        baseline: {
            "mean_delta": float(np.mean(dataset_level[baseline])),
            "median_delta": float(np.median(dataset_level[baseline])),
            "dataset_clustered_bootstrap_95_ci": intervals[baseline],
            "wilcoxon_p": raw_p[baseline],
            "holm_adjusted_p": adjusted[baseline],
            "positive_dataset_count": int(sum(value > 0 for value in dataset_level[baseline])),
            "dataset_count": len(FULL_DATASETS),
        }
        for baseline in baselines
    }


def _aggregate(run_root: Path, matrix: list[FullRunSpec], config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rate_values: dict[str, dict[str, list[float]]] = {
        method: {
            f"r{int(round(ratio * 1000)):04d}": []
            for ratio in config["rate_protocol"]["test_ratios"]
        }
        for method in FULL_METHODS
    }
    efficiency_values: dict[str, dict[str, list[float]]] = {
        method: {key: [] for key in ("parameters", "flops_per_sample", "training_seconds", "evaluation_seconds", "artifact_bytes")}
        for method in FULL_METHODS
    }
    reliability_rows: list[dict[str, Any]] = []
    for spec in matrix:
        payload = json.loads((run_root / spec.run_key / "metrics.json").read_text(encoding="utf-8"))
        summary = payload["evaluation"]
        rows.append({"dataset_id": spec.dataset_id, "method": spec.method, "seed": spec.seed, **{key: summary[key] for key in summary if key != "per_rate"}, "duration_seconds": payload["duration_seconds"]})
        for rate_key, rate_row in summary["per_rate"].items():
            rate_values[spec.method][rate_key].append(float(rate_row["macro_f1"]))
        for key, value in payload["efficiency"].items():
            if key in efficiency_values[spec.method] and value is not None:
                efficiency_values[spec.method][key].append(float(value))
        if payload.get("reliability") is not None:
            reliability = payload["reliability"]
            test = reliability["test_exploratory"]
            reliability_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "seed": spec.seed,
                    "selected_mode": reliability["selected_mode"],
                    "test_selected_aurc": test["selected_aurc"],
                    "test_confidence_aurc": test["confidence_aurc"],
                    "test_selected_risk_at_target": test["selected_target"]["risk"],
                    "test_selected_coverage": test["selected_target"]["coverage"],
                }
            )
    method_summary = {
        method: {
            key: float(np.mean([row[key] for row in rows if row["method"] == method]))
            for key in (
                "mean_unseen_macro_f1", "worst_unseen_macro_f1", "mean_unseen_balanced_accuracy",
                "mean_unseen_aurc", "mean_unseen_disagreement", "mean_unseen_jsd", "full_rate_macro_f1",
            )
        }
        for method in FULL_METHODS
    }
    rate_summary = {
        method: {rate_key: float(np.mean(values)) for rate_key, values in rates.items()}
        for method, rates in rate_values.items()
    }
    efficiency_summary = {
        method: {
            key: (float(np.mean(values)) if values else None)
            for key, values in metrics.items()
        }
        for method, metrics in efficiency_values.items()
    }
    return {
        "rows": rows,
        "method_summary": method_summary,
        "rate_summary": rate_summary,
        "efficiency_summary": efficiency_summary,
        "v3_10_reliability_rows": reliability_rows,
        "paired_statistics": paired_statistics(rows, config),
    }


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _generate_figures(report: dict[str, Any], destinations: list[Path]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = report["aggregate"]
    labels = list(FULL_METHODS)
    display = [label.replace("_", "\n") for label in labels]
    values = [aggregate["method_summary"][label]["mean_unseen_macro_f1"] for label in labels]
    colors = ["#31d0aa" if label == "v3_10" else "#4f8cff" for label in labels]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(display, values, color=colors)
    axis.set_ylabel("Mean unseen-rate macro-F1")
    axis.set_title("Frozen Full benchmark: mean across 10 datasets and 3 seeds")
    axis.set_ylim(0.0, min(1.0, max(values) + 0.12))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    written: list[str] = []
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / "full_method_comparison.png", dpi=220)
        written.append(str(destination / "full_method_comparison.png"))
    plt.close(figure)

    ratios = [1.0, 0.9, 0.6, 0.4, 0.3]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for method in labels:
        curve = [aggregate["rate_summary"][method][f"r{int(round(ratio * 1000)):04d}"] for ratio in ratios]
        axis.plot(ratios, curve, marker="o", linewidth=2.5 if method == "v3_10" else 1.3, label=method)
    axis.invert_xaxis()
    axis.set_xlabel("Sampling-rate ratio (1.0 to 0.3)")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Rate-degradation curves")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for destination in destinations:
        figure.savefig(destination / "full_rate_curves.png", dpi=220)
        written.append(str(destination / "full_rate_curves.png"))
    plt.close(figure)
    return written


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NyquistGuard-TSC Full benchmark",
        "",
        f"- Protocol: `{report['protocol_version']}`",
        f"- Runs: {report['completed_runs']}/210; elapsed {report['elapsed_seconds']:.1f} seconds.",
        "- Test was used only after the Full protocol and matrix were frozen.",
        "- No follow-up experiment was started automatically.",
        "",
        "## Aggregate test metrics",
        "",
        "| method | mean unseen F1 | worst unseen F1 | full-rate F1 | unseen balanced acc. | unseen AURC | unseen JSD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["aggregate"]["method_summary"].items():
        lines.append(
            f"| {method} | {row['mean_unseen_macro_f1']:.4f} | {row['worst_unseen_macro_f1']:.4f} | "
            f"{row['full_rate_macro_f1']:.4f} | {row['mean_unseen_balanced_accuracy']:.4f} | "
            f"{row['mean_unseen_aurc']:.4f} | {row['mean_unseen_jsd']:.4f} |"
        )
    lines.extend(["", "## Paired v3.10 comparisons", "", "| baseline | mean delta | 95% dataset-bootstrap CI | Wilcoxon p | Holm p | positive datasets |", "|---|---:|---:|---:|---:|---:|"])
    for baseline, row in report["aggregate"]["paired_statistics"].items():
        ci = row["dataset_clustered_bootstrap_95_ci"]
        lines.append(f"| {baseline} | {row['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['wilcoxon_p']:.4g} | {row['holm_adjusted_p']:.4g} | {row['positive_dataset_count']}/10 |")
    lines.extend(["", "## Efficiency", "", "| method | parameters | FLOPs/sample | train seconds/run | eval seconds/run |", "|---|---:|---:|---:|---:|"])
    for method, row in report["aggregate"]["efficiency_summary"].items():
        parameters = "—" if row["parameters"] is None else f"{row['parameters']:.0f}"
        flops = "—" if row["flops_per_sample"] is None else f"{row['flops_per_sample']:.0f}"
        lines.append(f"| {method} | {parameters} | {flops} | {row['training_seconds']:.1f} | {row['evaluation_seconds']:.1f} |")
    reliability = report["aggregate"]["v3_10_reliability_rows"]
    if reliability:
        calibrated = sum(row["selected_mode"] == "calibrated" for row in reliability)
        lines.extend(
            [
                "",
                "## v3.10 guarded reliability",
                "",
                f"- Calibrated mode selected by frozen validation rule in {calibrated}/{len(reliability)} dataset-seed runs; otherwise confidence fallback was used.",
                f"- Mean Full-test selected AURC: {np.mean([row['test_selected_aurc'] for row in reliability]):.4f}.",
                f"- Mean Full-test risk at the validation-calibrated target coverage: {np.mean([row['test_selected_risk_at_target'] for row in reliability]):.4f}.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_full(
    project_root: str | Path,
    *,
    resume: bool = True,
    confirmed: bool = False,
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("Full requires a separate explicit manual confirmation")
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "full.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = build_full_matrix(config)
    _validate_confirmation(root, config)
    base_path = root / config["base_config"]
    reliability_path = root / config["reliability_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    reliability_config = yaml.safe_load(reliability_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(root, [config_path, base_path, reliability_path])
    parent = root / "runs" / "full"
    run_root = _find_resume_root(parent, protocol_hash) if resume else None
    if run_root is not None:
        completed_report = run_root / "full_report.json"
        if completed_report.exists():
            existing = json.loads(completed_report.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                return existing
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"full__10datasets__7methods__3seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
    manifest = {"stage": "full", "status": "running", "manual_confirmation": True, "protocol_hash": protocol_hash, "run_root": str(run_root), "updated_at_utc": utc_now()}
    atomic_write_json(run_root / "manifest.json", manifest)
    tasks = [f"Prepare/cache {dataset_id}" for dataset_id in FULL_DATASETS]
    tasks.extend(f"Run {spec.run_key}" for spec in matrix)
    tasks.append("Aggregate paired statistics and Full report")
    progress = DashboardProgress(root / "runs" / "dashboard_status.json", "full", tasks, run_root.name)
    current = 0
    try:
        for dataset_id in FULL_DATASETS:
            progress.start_task(current)
            print(f"[data {current + 1:02d}/10] Preparing or loading {dataset_id}", flush=True)
            dataset = prepare_full_dataset(root, dataset_id)
            del dataset
            progress.complete_task(current)
            current += 1
        spec_index = 0
        for dataset_id in FULL_DATASETS:
            dataset = prepare_full_dataset(root, dataset_id)
            for spec in [item for item in matrix if item.dataset_id == dataset_id]:
                progress.start_task(current)
                spec_index += 1
                print(f"[{spec_index:03d}/210] Starting {spec.run_key}", flush=True)
                _run_one(root, run_root, dataset, spec, base_config, config, reliability_config, protocol_hash, resume)
                progress.complete_task(current)
                current += 1
            del dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        progress.start_task(current)
        aggregate = _aggregate(run_root, matrix, config)
        report = {
            "status": "completed",
            "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash,
            "manual_confirmation": True,
            "completed_runs": 210,
            "datasets": list(FULL_DATASETS),
            "methods": list(FULL_METHODS),
            "seeds": list(FULL_SEEDS),
            "primary_split": "frozen Full test splits",
            "aggregate": aggregate,
            "elapsed_seconds": time.monotonic() - started,
            "run_root": str(run_root),
            "automatic_followup_started": False,
            "finished_at_utc": utc_now(),
        }
        _write_aggregate_csv(run_root / "full_results.csv", aggregate["rows"])
        _write_aggregate_csv(root / "reports" / "full_results.csv", aggregate["rows"])
        report["figure_paths"] = _generate_figures(report, [run_root, root / "reports"])
        atomic_write_json(run_root / "full_report.json", report)
        _atomic_write_text(run_root / "full_report.md", _report_markdown(report))
        atomic_write_json(root / "reports" / "full_report.json", report)
        _atomic_write_text(root / "reports" / "full_report.md", _report_markdown(report))
        manifest.update(status="completed", updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        progress.complete_task(current)
        progress.finish("Full completed; no follow-up experiment was started")
        return report
    except BaseException as error:
        progress.fail_task(current, error)
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        raise
