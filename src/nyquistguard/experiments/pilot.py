"""Recoverable, manually gated 84-run pilot experiment matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from nyquistguard.data import PILOT_DATASETS, PreparedDataset, SplitData, prepare_pilot_dataset, resample_antialiased
from nyquistguard.experiments.metrics import align_probability_columns, classification_metrics
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.losses import NyquistGuardObjective
from nyquistguard.models import NyquistGuardTSC, TCNClassifier
from nyquistguard.training.checkpointing import load_training_checkpoint, save_training_checkpoint


PILOT_METHODS = (
    "nyquistguard",
    "fixed_rate_tcn",
    "multirate_tcn",
    "minirocket",
    "no_nyquist_gate",
    "no_cbe",
    "no_selective_head",
)


@dataclass(frozen=True)
class PilotRunSpec:
    dataset_id: str
    method: str
    seed: int

    @property
    def run_key(self) -> str:
        return f"{self.dataset_id}__{self.method}__seed{self.seed}"


def build_pilot_matrix(config: dict[str, Any]) -> list[PilotRunSpec]:
    datasets = tuple(config["datasets"])
    methods = tuple(config["methods"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    if datasets != PILOT_DATASETS:
        raise ValueError(f"pilot datasets must remain frozen as {PILOT_DATASETS}")
    if methods != PILOT_METHODS:
        raise ValueError(f"pilot methods must remain frozen as {PILOT_METHODS}")
    matrix = [PilotRunSpec(dataset, method, seed) for dataset in datasets for method in methods for seed in seeds]
    if len(matrix) != 84 or len({spec.run_key for spec in matrix}) != 84:
        raise ValueError("pilot matrix must contain exactly 84 unique runs")
    return matrix


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _protocol_hash(config_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    digest.update(b"pilot-runner-v1")
    return digest.hexdigest()


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _view(x: torch.Tensor, source_rate: float, ratio: float, config: dict[str, Any]) -> tuple[torch.Tensor, float]:
    viewed, rate, _ = resample_antialiased(
        x,
        source_rate,
        ratio,
        taps=int(config["antialias_fir_taps"]),
        rolloff=float(config["antialias_rolloff"]),
    )
    return viewed, rate


def _resolved_model_config(dataset: PreparedDataset, config: dict[str, Any], method: str) -> dict[str, Any]:
    maximum_hz = max(0.25, 0.45 * dataset.sampling_rate_hz)
    result = dict(config["model"])
    result.update(
        input_channels=int(dataset.train.x.shape[1]),
        num_classes=len(dataset.class_names),
        max_center_hz=maximum_hz,
        reference_sampling_rate_hz=dataset.sampling_rate_hz,
        use_nyquist_gate=method not in {"no_nyquist_gate", "v3_no_nyquist_gate"},
        use_selective_head=method not in {"no_selective_head", "v3_no_nyquist_gate"},
    )
    return result


def _resolved_objective_config(dataset: PreparedDataset, config: dict[str, Any], method: str) -> dict[str, Any]:
    result = dict(config["objective"])
    result["max_center_hz"] = max(0.25, 0.45 * dataset.sampling_rate_hz)
    if method == "no_cbe":
        result["lambda_cbe"] = 0.0
    if method in {"no_selective_head", "v3_no_nyquist_gate"}:
        result["lambda_selective"] = 0.0
        result["lambda_monotonicity"] = 0.0
    return result


def _deep_model(dataset: PreparedDataset, config: dict[str, Any], method: str, device: torch.device) -> nn.Module:
    if method in {"fixed_rate_tcn", "multirate_tcn"}:
        return TCNClassifier(
            int(dataset.train.x.shape[1]), len(dataset.class_names), **dict(config["baseline"])
        ).to(device)
    return NyquistGuardTSC(**_resolved_model_config(dataset, config, method)).to(device)


def _predict_deep(
    model: nn.Module,
    split: SplitData,
    source_rate_hz: float,
    ratio: float,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(split.x), torch.from_numpy(split.y)),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    logits: list[np.ndarray] = []
    acceptance: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, _ in loader:
            x_view, rate = _view(x.to(device), source_rate_hz, ratio, config)
            output = model(x_view, rate)
            logits.append(output["logits"].float().cpu().numpy())
            acceptance.append(output["accept_probability"].float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(acceptance)


def _validation_score(
    model: nn.Module, dataset: PreparedDataset, config: dict[str, Any], device: torch.device
) -> float:
    scores: list[float] = []
    for ratio in (1.0, 0.6):
        logits, acceptance = _predict_deep(model, dataset.validation, dataset.sampling_rate_hz, ratio, config, device)
        scores.append(classification_metrics(dataset.validation.y, logits, acceptance)["macro_f1"])
    return float(np.mean(scores))


def _train_deep(
    dataset: PreparedDataset,
    spec: PilotRunSpec,
    config: dict[str, Any],
    protocol_hash: str,
    run_dir: Path,
    resume: bool,
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _deep_model(dataset, config, spec.method, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=float(config["amp_initial_scale"]),
        growth_interval=int(config["amp_growth_interval"]),
    )
    objective = None
    if isinstance(model, NyquistGuardTSC):
        objective = NyquistGuardObjective(**_resolved_objective_config(dataset, config, spec.method)).to(device)
    checkpoint = run_dir / "checkpoint_last.pt"
    best_path = run_dir / "checkpoint_best.pt"
    start_epoch = 0
    best_score = -math.inf
    patience_used = 0
    if resume and checkpoint.exists():
        payload = load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_protocol_hash=protocol_hash,
            map_location=device,
        )
        start_epoch = int(payload["step"])
        best_score = float(payload.get("extra", {}).get("best_score", -math.inf))
        patience_used = int(payload.get("extra", {}).get("patience_used", 0))

    train_tensor_dataset = TensorDataset(torch.from_numpy(dataset.train.x), torch.from_numpy(dataset.train.y))
    train_ratios = tuple(float(value) for value in config["train_rate_ratios"])
    history: list[dict[str, float | int]] = []
    history_path = run_dir / "training_history.json"
    if resume and history_path.exists():
        try:
            history = list(json.loads(history_path.read_text(encoding="utf-8")).get("history", []))
        except (OSError, json.JSONDecodeError):
            history = []
    for epoch in range(start_epoch, int(config["epochs"])):
        # Epoch-derived shuffling makes a resumed run reproduce the same order
        # without depending on an uncheckpointed DataLoader generator state.
        loader = DataLoader(
            train_tensor_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(spec.seed + epoch),
            num_workers=int(config["num_workers"]),
        )
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_index, (x_cpu, targets_cpu) in enumerate(loader):
            x = x_cpu.to(device)
            targets = targets_cpu.to(device)
            ratio = train_ratios[(epoch + batch_index) % len(train_ratios)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                if objective is not None:
                    high, high_rate = _view(x, dataset.sampling_rate_hz, 1.0, config)
                    low, low_rate = _view(x, dataset.sampling_rate_hz, ratio, config)
                    losses = objective(model(high, high_rate), targets, model(low, low_rate), targets)
                    loss = losses["total"]
                else:
                    used_ratio = ratio if spec.method == "multirate_tcn" else 1.0
                    viewed, rate = _view(x, dataset.sampling_rate_hz, used_ratio, config)
                    loss = nn.functional.cross_entropy(model(viewed, rate)["logits"], targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        score = _validation_score(model, dataset, config, device)
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            patience_used = 0
            _atomic_torch_save(model.state_dict(), best_path)
        else:
            patience_used += 1
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / max(1, batches), "validation_macro_f1": score})
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=epoch + 1,
            protocol_hash=protocol_hash,
            model_config={"method": spec.method},
            extra={"best_score": best_score, "patience_used": patience_used},
        )
        atomic_write_json(history_path, {"history": history, "updated_at_utc": utc_now()})
        print(
            f"[{spec.run_key}] epoch {epoch + 1}/{config['epochs']} loss={epoch_loss / max(1, batches):.4f} "
            f"val_macro_f1={score:.4f}",
            flush=True,
        )
        if patience_used >= int(config["early_stopping_patience"]):
            break
    if not best_path.exists():
        _atomic_torch_save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model


def _fixed_length_view(values: np.ndarray, source_rate: float, ratio: float, config: dict[str, Any]) -> np.ndarray:
    tensor = torch.from_numpy(values)
    viewed, _ = _view(tensor, source_rate, ratio, config)
    if viewed.shape[-1] != tensor.shape[-1]:
        viewed = nn.functional.interpolate(viewed, size=tensor.shape[-1], mode="linear", align_corners=False)
    return viewed.numpy()


def _fit_minirocket(dataset: PreparedDataset, spec: PilotRunSpec, config: dict[str, Any], run_dir: Path, resume: bool):
    estimator_path = run_dir / "minirocket.pkl"
    if resume and estimator_path.exists():
        with estimator_path.open("rb") as handle:
            return pickle.load(handle)
    from aeon.classification.convolution_based import MiniRocketClassifier

    views = [
        _fixed_length_view(dataset.train.x, dataset.sampling_rate_hz, ratio, config)
        for ratio in config["train_rate_ratios"]
    ]
    train_x = np.concatenate(views, axis=0)
    train_y = np.tile(dataset.train.y, len(views))
    estimator = MiniRocketClassifier(
        n_kernels=int(config["minirocket_kernels"]),
        n_jobs=int(config["minirocket_jobs"]),
        random_state=spec.seed,
    )
    estimator.fit(train_x, train_y)
    temporary = estimator_path.with_name(estimator_path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(estimator, handle)
    os.replace(temporary, estimator_path)
    return estimator


def _evaluate_run(
    dataset: PreparedDataset,
    spec: PilotRunSpec,
    config: dict[str, Any],
    model: object,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_rate: dict[str, dict[str, float]] = {}
    prediction_rows: list[dict[str, object]] = []
    reference_predictions: np.ndarray | None = None
    for ratio_value in config["test_rate_ratios"]:
        ratio = float(ratio_value)
        if spec.method == "minirocket":
            values = _fixed_length_view(dataset.test.x, dataset.sampling_rate_hz, ratio, config)
            probabilities = align_probability_columns(
                model.predict_proba(values),  # type: ignore[attr-defined]
                np.asarray(model.classes_),  # type: ignore[attr-defined]
                len(dataset.class_names),
            )
            logits = np.log(np.asarray(probabilities).clip(1e-12))
            acceptance = np.asarray(probabilities).max(axis=1)
        else:
            logits, acceptance = _predict_deep(
                model, dataset.test, dataset.sampling_rate_hz, ratio, config, device  # type: ignore[arg-type]
            )
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        metrics = classification_metrics(dataset.test.y, logits, acceptance)
        predictions = np.asarray(probabilities).argmax(axis=1)
        if reference_predictions is None:
            reference_predictions = predictions
            metrics["disagreement_vs_original"] = 0.0
        else:
            metrics["disagreement_vs_original"] = float(np.mean(predictions != reference_predictions))
        rate_id = f"r{int(round(ratio * 1000)):04d}"
        per_rate[rate_id] = metrics
        for index, sample_id in enumerate(dataset.test.ids):
            prediction_rows.append(
                {
                    "sample_id": str(sample_id),
                    "rate_ratio": ratio,
                    "target": int(dataset.test.y[index]),
                    "prediction": int(predictions[index]),
                    "acceptance": float(acceptance[index]),
                    "probabilities_json": json.dumps(np.asarray(probabilities[index]).tolist()),
                }
            )
    unseen = [
        per_rate[f"r{int(round(float(ratio) * 1000)):04d}"]
        for ratio in config["test_rate_ratios"]
        if float(ratio) != 1.0
    ]
    summary = {
        "mean_unseen_accuracy": float(np.mean([item["accuracy"] for item in unseen])),
        "mean_unseen_macro_f1": float(np.mean([item["macro_f1"] for item in unseen])),
        "mean_unseen_aurc": float(np.mean([item["aurc"] for item in unseen])),
        "per_rate": per_rate,
    }
    return summary, prediction_rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def _run_one(
    dataset: PreparedDataset,
    spec: PilotRunSpec,
    config: dict[str, Any],
    protocol_hash: str,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if resume and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol_hash") == protocol_hash:
            print(f"[{spec.run_key}] completed result reused", flush=True)
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "status.json",
        {
            "status": "running",
            "spec": asdict(spec),
            "protocol_hash": protocol_hash,
            "dataset_protocol_id": dataset.metadata.get("dataset_protocol_id", "pilot_v1"),
            "class_names": dataset.class_names,
            "started_at_utc": utc_now(),
        },
    )
    started = time.monotonic()
    try:
        _seed_everything(spec.seed)
        if spec.method == "minirocket":
            model = _fit_minirocket(dataset, spec, config, run_dir, resume)
        else:
            model = _train_deep(dataset, spec, config, protocol_hash, run_dir, resume)
        evaluation, rows = _evaluate_run(dataset, spec, config, model)
        _write_csv(run_dir / "predictions.csv", rows)
        result = {
            "status": "completed",
            "spec": asdict(spec),
            "protocol_hash": protocol_hash,
            "duration_seconds": time.monotonic() - started,
            "sampling_rate_hz": dataset.sampling_rate_hz,
            "dataset_protocol_id": dataset.metadata.get("dataset_protocol_id", "pilot_v1"),
            "class_names": dataset.class_names,
            "evaluation": evaluation,
            "finished_at_utc": utc_now(),
        }
        atomic_write_json(metrics_path, result)
        atomic_write_json(run_dir / "status.json", result)
        return result
    except BaseException as error:
        atomic_write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "spec": asdict(spec),
                "protocol_hash": protocol_hash,
                "error": f"{type(error).__name__}: {error}",
                "failed_at_utc": utc_now(),
            },
        )
        raise


def _find_resume_root(pilot_root: Path, protocol_hash: str) -> Path | None:
    for manifest in sorted(pilot_root.glob("*/pilot_manifest.json"), reverse=True):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("protocol_hash") == protocol_hash:
            return manifest.parent
    return None


def _aggregate(run_root: Path, matrix: list[PilotRunSpec]) -> dict[str, Any]:
    rows: list[dict[str, object]] = []
    for spec in matrix:
        payload = json.loads((run_root / spec.run_key / "metrics.json").read_text(encoding="utf-8"))
        evaluation = payload["evaluation"]
        rows.append(
            {
                **asdict(spec),
                "mean_unseen_accuracy": evaluation["mean_unseen_accuracy"],
                "mean_unseen_macro_f1": evaluation["mean_unseen_macro_f1"],
                "mean_unseen_aurc": evaluation["mean_unseen_aurc"],
                "duration_seconds": payload["duration_seconds"],
            }
        )
    _write_csv(run_root / "pilot_summary.csv", rows)
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in PILOT_DATASETS:
        grouped[dataset] = {}
        for method in PILOT_METHODS:
            selected = [row for row in rows if row["dataset_id"] == dataset and row["method"] == method]
            grouped[dataset][method] = {
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in ("mean_unseen_accuracy", "mean_unseen_macro_f1", "mean_unseen_aurc")
            }
    report = {
        "status": "completed",
        "run_count": len(rows),
        "grouped_seed_means": grouped,
        "decision": "manual_review_required",
        "full_auto_start": False,
        "scientific_boundary": "Pilot aggregation does not authorize or launch full experiments.",
        "finished_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "pilot_summary.json", report)
    return report


def _write_report(path: Path, run_root: Path, report: dict[str, Any]) -> None:
    lines = [
        "# NyquistGuard-TSC Pilot 报告",
        "",
        f"- 状态：{report['status']}",
        f"- 已完成 runs：{report['run_count']} / 84",
        f"- 结果目录：`{run_root}`",
        "- Full：不会自动启动，必须人工审阅并在窗口中手动开始。",
        "",
        "详细种子均值见 `pilot_summary.json`，逐 run 指标和预测保留在结果目录。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(project_root: str | Path, *, resume: bool = False, confirmed: bool = False) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("pilot requires an explicit manual confirmation")
    root = Path(project_root)
    smoke_report = root / "reports" / "smoke_report.json"
    if not smoke_report.exists() or json.loads(smoke_report.read_text(encoding="utf-8")).get("status") != "completed":
        raise RuntimeError("a completed smoke report is required before pilot")
    config_path = root / "configs" / "experiments" / "pilot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = build_pilot_matrix(config)
    protocol_hash = _protocol_hash(config_path)
    pilot_root = root / "runs" / "pilot"
    run_root = _find_resume_root(pilot_root, protocol_hash) if resume else None
    if run_root is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = pilot_root / f"pilot__4datasets__7methods__3seeds__{timestamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_resolved.yaml")
    manifest = {
        "stage": "pilot",
        "status": "running",
        "protocol_hash": protocol_hash,
        "manual_confirmation": True,
        "matrix_size": len(matrix),
        "matrix": [asdict(spec) for spec in matrix],
        "run_root": str(run_root),
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "pilot_manifest.json", manifest)
    task_names = [f"准备 {dataset}" for dataset in config["datasets"]]
    task_names.extend(f"{index + 1:02d}/84 {spec.run_key}" for index, spec in enumerate(matrix))
    task_names.append("聚合指标与人工 Go/No-Go 报告")
    progress = DashboardProgress(root / "runs" / "dashboard_status.json", "pilot", task_names, run_root.name)
    datasets: dict[str, PreparedDataset] = {}
    current_index = 0
    try:
        for dataset_id in config["datasets"]:
            progress.start_task(current_index)
            print(f"Preparing {dataset_id} (cached processed data will be reused)", flush=True)
            datasets[dataset_id] = prepare_pilot_dataset(root, dataset_id)
            progress.complete_task(current_index)
            current_index += 1
        for spec in matrix:
            progress.start_task(current_index)
            print(f"Starting pilot run {spec.run_key}", flush=True)
            _run_one(datasets[spec.dataset_id], spec, config, protocol_hash, run_root / spec.run_key, resume)
            progress.complete_task(current_index)
            current_index += 1
        progress.start_task(current_index)
        report = _aggregate(run_root, matrix)
        _write_report(root / "reports" / "pilot_go_no_go.md", run_root, report)
        atomic_write_json(root / "reports" / "pilot_summary.json", report)
        progress.complete_task(current_index)
        progress.finish("Pilot 已完成；Full 等待人工启动")
        manifest.update(status="completed", updated_at_utc=utc_now())
        atomic_write_json(run_root / "pilot_manifest.json", manifest)
        return report
    except BaseException as error:
        progress.fail_task(current_index, error)
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_utc=utc_now())
        atomic_write_json(run_root / "pilot_manifest.json", manifest)
        raise
