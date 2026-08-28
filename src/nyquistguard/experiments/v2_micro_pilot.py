"""Bounded, development-only v2 micro-pilot.

The experiment isolates two mechanism changes on BasicMotions and PAMAP2 at
seed 17.  It never launches Pilot or Full and enforces a total wall-clock cap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from nyquistguard.data import PreparedDataset, SplitData, load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.metrics import classification_metrics
from nyquistguard.experiments.pilot import (
    _atomic_torch_save,
    _deep_model,
    _predict_deep,
    _resolved_objective_config,
    _seed_everything,
)
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.losses import DetachedCorrectnessSelectiveLoss, NyquistGuardObjective
from nyquistguard.losses.monotonicity import AcceptanceMonotonicityLoss
from nyquistguard.models import NyquistGuardTSC
from nyquistguard.training.checkpointing import load_training_checkpoint, save_training_checkpoint


V2_DATASETS = ("basicmotions_uea", "pamap2_uci")
V2_VARIANTS = ("selector_detached_v2", "cbe_balanced_v2")
V2_SEED = 17


class MicroPilotTimeBudgetExceeded(RuntimeError):
    """Raised after a completed checkpoint when the bounded runtime is consumed."""


@dataclass(frozen=True)
class V2RunSpec:
    dataset_id: str
    variant: str
    seed: int = V2_SEED

    @property
    def run_key(self) -> str:
        return f"{self.dataset_id}__{self.variant}__seed{self.seed}"


def build_v2_matrix(config: dict[str, Any]) -> list[V2RunSpec]:
    if tuple(config["datasets"]) != V2_DATASETS:
        raise ValueError(f"v2 datasets must remain frozen as {V2_DATASETS}")
    if tuple(config["variants"]) != V2_VARIANTS:
        raise ValueError(f"v2 variants must remain frozen as {V2_VARIANTS}")
    if int(config["seed"]) != V2_SEED:
        raise ValueError(f"v2 seed must remain frozen as {V2_SEED}")
    # Alternate order within dataset blocks to reduce a simple run-order confound.
    matrix = [
        V2RunSpec("basicmotions_uea", "selector_detached_v2"),
        V2RunSpec("basicmotions_uea", "cbe_balanced_v2"),
        V2RunSpec("pamap2_uci", "cbe_balanced_v2"),
        V2RunSpec("pamap2_uci", "selector_detached_v2"),
    ]
    if len({spec.run_key for spec in matrix}) != 4:
        raise RuntimeError("v2 matrix must contain four unique runs")
    return matrix


def _protocol_hash(micro_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(micro_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(b"v2-micro-runner-v1")
    return digest.hexdigest()


def _detached_selector_output(model: NyquistGuardTSC, output: dict[str, Any]) -> dict[str, Any]:
    if model.selective_head is None:
        raise ValueError("selector_detached_v2 requires a selective head")
    aux = output["aux"]
    accept_logit = model.selective_head(
        output["embedding"].detach(),
        output["nyquist_gate"].detach(),
        aux["sampling_rate_hz"].detach(),
        aux["normalized_prediction_entropy"].detach().to(output["embedding"].dtype),
    )
    detached = dict(output)
    detached["accept_logit"] = accept_logit
    detached["accept_probability"] = torch.sigmoid(accept_logit)
    return detached


def _gradient_norm_tensor(loss: Tensor, parameters: Sequence[nn.Parameter]) -> Tensor:
    gradients = torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=True,
        allow_unused=True,
    )
    squared = loss.new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.float().square().sum()
    return torch.sqrt(squared)


def _balanced_cbe_scale(
    classification: Tensor,
    equivariance: Tensor,
    filter_parameters: Sequence[nn.Parameter],
    *,
    target_ratio: float,
    minimum_scale: float,
    maximum_scale: float,
    epsilon: float = 1e-12,
) -> dict[str, Tensor]:
    classification_norm = _gradient_norm_tensor(classification, filter_parameters)
    cbe_norm = _gradient_norm_tensor(equivariance, filter_parameters)
    raw_scale = target_ratio * classification_norm / (cbe_norm + epsilon)
    scale = raw_scale.clamp(minimum_scale, maximum_scale).detach()
    achieved = scale * cbe_norm.detach() / (classification_norm.detach() + epsilon)
    return {
        "scale": scale,
        "classification_filterbank_gradient_norm": classification_norm.detach(),
        "cbe_filterbank_gradient_norm": cbe_norm.detach(),
        "achieved_filterbank_gradient_ratio": achieved,
    }


def _selector_objective(
    model: NyquistGuardTSC,
    output_high: dict[str, Any],
    output_low: dict[str, Any],
    targets: Tensor,
    base_objective: NyquistGuardObjective,
    selector_loss: DetachedCorrectnessSelectiveLoss,
    selector_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    high = _detached_selector_output(model, output_high)
    low = _detached_selector_output(model, output_low)
    base = base_objective(high, targets, low, targets)
    selector_high = selector_loss(high["logits"], targets, high["accept_logit"])
    selector_low = selector_loss(low["logits"], targets, low["accept_logit"])
    selector = 0.5 * (selector_high["total"] + selector_low["total"])
    total = base["total"] + selector_weight * selector
    return total, {
        "selector_correctness_bce": float(selector.detach().cpu()),
        "cbe_scale": float(base_objective.lambda_cbe),
        "cbe_filterbank_gradient_ratio": math.nan,
    }


def _cbe_balanced_objective(
    model: NyquistGuardTSC,
    output_high: dict[str, Any],
    output_low: dict[str, Any],
    targets: Tensor,
    objective: NyquistGuardObjective,
    balancing: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    losses = objective(output_high, targets, output_low, targets)
    filter_parameters = [
        parameter for parameter in model.filterbank.parameters() if parameter.requires_grad
    ]
    balance = _balanced_cbe_scale(
        losses["classification"],
        losses["equivariance"],
        filter_parameters,
        target_ratio=float(balancing["target_gradient_ratio"]),
        minimum_scale=float(balancing["minimum_scale"]),
        maximum_scale=float(balancing["maximum_scale"]),
    )
    scale = balance["scale"]
    total = losses["total"] + (scale - float(objective.lambda_cbe)) * losses["equivariance"]
    return total, {
        "selector_correctness_bce": math.nan,
        "cbe_scale": float(scale.cpu()),
        "cbe_filterbank_gradient_ratio": float(
            balance["achieved_filterbank_gradient_ratio"].cpu()
        ),
    }


def _evaluate_split(
    model: nn.Module,
    split: SplitData,
    dataset: PreparedDataset,
    base_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    per_rate: dict[str, dict[str, float]] = {}
    for ratio_value in base_config["test_rate_ratios"]:
        ratio = float(ratio_value)
        logits, acceptance = _predict_deep(
            model, split, dataset.sampling_rate_hz, ratio, base_config, device
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        confidence = probabilities.max(axis=1)
        learned = classification_metrics(split.y, logits, acceptance)
        confidence_metrics = classification_metrics(split.y, logits, confidence)
        rate_id = f"r{int(round(ratio * 1000)):04d}"
        per_rate[rate_id] = {
            "accuracy": learned["accuracy"],
            "macro_f1": learned["macro_f1"],
            "learned_aurc": learned["aurc"],
            "confidence_aurc": confidence_metrics["aurc"],
            "acceptance_mean": float(np.mean(acceptance)),
        }
    unseen = [per_rate[key] for key in ("r0900", "r0600", "r0400", "r0300")]
    return {
        "mean_unseen_macro_f1": float(np.mean([item["macro_f1"] for item in unseen])),
        "mean_unseen_learned_aurc": float(np.mean([item["learned_aurc"] for item in unseen])),
        "mean_unseen_confidence_aurc": float(
            np.mean([item["confidence_aurc"] for item in unseen])
        ),
        "full_rate_macro_f1": per_rate["r1000"]["macro_f1"],
        "per_rate": per_rate,
    }


def _evaluate_model(
    model: nn.Module,
    dataset: PreparedDataset,
    base_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    return {
        "validation": _evaluate_split(model, dataset.validation, dataset, base_config, device),
        "test_exploratory": _evaluate_split(model, dataset.test, dataset, base_config, device),
    }


def _train_variant(
    dataset: PreparedDataset,
    spec: V2RunSpec,
    base_config: dict[str, Any],
    micro_config: dict[str, Any],
    protocol_hash: str,
    run_dir: Path,
    *,
    resume: bool,
    deadline: float,
) -> tuple[NyquistGuardTSC, list[dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _deep_model(dataset, base_config, "nyquistguard", device)
    if not isinstance(model, NyquistGuardTSC):
        raise TypeError("v2 micro-pilot requires NyquistGuardTSC")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(base_config["learning_rate"]),
        weight_decay=float(base_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=float(base_config["amp_initial_scale"]),
        growth_interval=int(base_config["amp_growth_interval"]),
    )
    objective_config = _resolved_objective_config(dataset, base_config, "nyquistguard")
    if spec.variant == "selector_detached_v2":
        objective_config["lambda_selective"] = 0.0
    objective = NyquistGuardObjective(**objective_config).to(device)
    selector_loss = DetachedCorrectnessSelectiveLoss().to(device)
    checkpoint = run_dir / "checkpoint_last.pt"
    start_epoch = 0
    history: list[dict[str, Any]] = []
    history_path = run_dir / "training_history.json"
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
        if history_path.exists():
            history = list(json.loads(history_path.read_text(encoding="utf-8")).get("history", []))

    train_data = TensorDataset(torch.from_numpy(dataset.train.x), torch.from_numpy(dataset.train.y))
    train_ratios = tuple(float(value) for value in base_config["train_rate_ratios"])
    for epoch in range(start_epoch, int(micro_config["epochs"])):
        if time.monotonic() >= deadline:
            raise MicroPilotTimeBudgetExceeded(
                f"time budget reached before epoch {epoch + 1} of {spec.run_key}"
            )
        loader = DataLoader(
            train_data,
            batch_size=int(base_config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(spec.seed + epoch),
            num_workers=int(base_config["num_workers"]),
        )
        model.train()
        epoch_loss = 0.0
        cbe_scales: list[float] = []
        cbe_ratios: list[float] = []
        selector_losses: list[float] = []
        batches = 0
        for batch_index, (x_cpu, targets_cpu) in enumerate(loader):
            if time.monotonic() >= deadline:
                raise MicroPilotTimeBudgetExceeded(
                    f"time budget reached during epoch {epoch + 1} of {spec.run_key}; prior epoch checkpoint retained"
                )
            x = x_cpu.to(device)
            targets = targets_cpu.to(device)
            ratio = train_ratios[(epoch + batch_index) % len(train_ratios)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                from nyquistguard.experiments.pilot import _view

                high, high_rate = _view(x, dataset.sampling_rate_hz, 1.0, base_config)
                low, low_rate = _view(x, dataset.sampling_rate_hz, ratio, base_config)
                output_high = model(high, high_rate)
                output_low = model(low, low_rate)
                if spec.variant == "selector_detached_v2":
                    loss, diagnostics = _selector_objective(
                        model,
                        output_high,
                        output_low,
                        targets,
                        objective,
                        selector_loss,
                        float(micro_config["selector_v2"]["lambda_correctness_bce"]),
                    )
                else:
                    loss, diagnostics = _cbe_balanced_objective(
                        model,
                        output_high,
                        output_low,
                        targets,
                        objective,
                        dict(micro_config["cbe_balancing"]),
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(base_config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            if math.isfinite(diagnostics["cbe_scale"]):
                cbe_scales.append(diagnostics["cbe_scale"])
            if math.isfinite(diagnostics["cbe_filterbank_gradient_ratio"]):
                cbe_ratios.append(diagnostics["cbe_filterbank_gradient_ratio"])
            if math.isfinite(diagnostics["selector_correctness_bce"]):
                selector_losses.append(diagnostics["selector_correctness_bce"])
            batches += 1
        row = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(1, batches),
            "selector_correctness_bce": float(np.mean(selector_losses)) if selector_losses else None,
            "cbe_scale_mean": float(np.mean(cbe_scales)) if cbe_scales else None,
            "cbe_filterbank_gradient_ratio_mean": (
                float(np.mean(cbe_ratios)) if cbe_ratios else None
            ),
        }
        history.append(row)
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=epoch + 1,
            protocol_hash=protocol_hash,
            model_config={"variant": spec.variant},
            extra={"fixed_epoch_endpoint": int(micro_config["epochs"])},
        )
        atomic_write_json(history_path, {"history": history, "updated_at_utc": utc_now()})
        diagnostic_text = (
            f"selector_bce={row['selector_correctness_bce']:.4f}"
            if row["selector_correctness_bce"] is not None
            else f"cbe_scale={row['cbe_scale_mean']:.2f} ratio={row['cbe_filterbank_gradient_ratio_mean']:.4f}"
        )
        print(
            f"[{spec.run_key}] epoch {epoch + 1}/{micro_config['epochs']} "
            f"loss={row['train_loss']:.4f} {diagnostic_text}",
            flush=True,
        )
    final_path = run_dir / "checkpoint_final.pt"
    _atomic_torch_save(model.state_dict(), final_path)
    return model, history


def _run_one(
    dataset: PreparedDataset,
    spec: V2RunSpec,
    base_config: dict[str, Any],
    micro_config: dict[str, Any],
    protocol_hash: str,
    run_dir: Path,
    *,
    resume: bool,
    deadline: float,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if resume and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol_hash") == protocol_hash:
            print(f"[{spec.run_key}] completed result reused", flush=True)
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    atomic_write_json(
        run_dir / "status.json",
        {
            "status": "running",
            "spec": asdict(spec),
            "protocol_hash": protocol_hash,
            "development_only": True,
            "started_at_utc": utc_now(),
        },
    )
    try:
        _seed_everything(spec.seed)
        model, history = _train_variant(
            dataset,
            spec,
            base_config,
            micro_config,
            protocol_hash,
            run_dir,
            resume=resume,
            deadline=deadline,
        )
        device = next(model.parameters()).device
        evaluation = _evaluate_model(model, dataset, base_config, device)
        result = {
            "status": "completed",
            "spec": asdict(spec),
            "protocol_hash": protocol_hash,
            "duration_seconds": time.monotonic() - started,
            "epochs_completed": len(history),
            "evaluation": evaluation,
            "training_diagnostics": {
                "median_cbe_scale": float(
                    np.median(
                        [row["cbe_scale_mean"] for row in history if row["cbe_scale_mean"] is not None]
                    )
                ) if spec.variant == "cbe_balanced_v2" else None,
                "median_cbe_filterbank_gradient_ratio": float(
                    np.median(
                        [
                            row["cbe_filterbank_gradient_ratio_mean"]
                            for row in history
                            if row["cbe_filterbank_gradient_ratio_mean"] is not None
                        ]
                    )
                ) if spec.variant == "cbe_balanced_v2" else None,
            },
            "finished_at_utc": utc_now(),
        }
        atomic_write_json(metrics_path, result)
        atomic_write_json(run_dir / "status.json", result)
        return result
    except BaseException as error:
        atomic_write_json(
            run_dir / "status.json",
            {
                "status": "budget_exhausted" if isinstance(error, MicroPilotTimeBudgetExceeded) else "failed",
                "spec": asdict(spec),
                "protocol_hash": protocol_hash,
                "error": f"{type(error).__name__}: {error}",
                "updated_at_utc": utc_now(),
            },
        )
        raise


def _load_v1_controls(
    project_root: Path,
    pilot_root: Path,
    datasets: dict[str, PreparedDataset],
    base_config: dict[str, Any],
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controls: dict[str, Any] = {}
    for dataset_id, dataset in datasets.items():
        model = _deep_model(dataset, base_config, "nyquistguard", device)
        checkpoint = pilot_root / f"{dataset_id}__nyquistguard__seed{V2_SEED}" / "checkpoint_best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        controls[dataset_id] = {
            "checkpoint": str(checkpoint),
            "evaluation": _evaluate_model(model, dataset, base_config, device),
        }
        del model
    return controls


def _development_decision(
    controls: dict[str, Any],
    results: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    selector_rows: list[dict[str, float | str]] = []
    cbe_rows: list[dict[str, float | str]] = []
    for dataset in V2_DATASETS:
        control = controls[dataset]["evaluation"]["validation"]
        selector = results[f"{dataset}__selector_detached_v2__seed{V2_SEED}"]["evaluation"][
            "validation"
        ]
        cbe = results[f"{dataset}__cbe_balanced_v2__seed{V2_SEED}"]
        cbe_validation = cbe["evaluation"]["validation"]
        selector_rows.append(
            {
                "dataset": dataset,
                "learned_aurc_improvement": control["mean_unseen_learned_aurc"]
                - selector["mean_unseen_learned_aurc"],
                "macro_f1_delta": selector["mean_unseen_macro_f1"]
                - control["mean_unseen_macro_f1"],
            }
        )
        cbe_rows.append(
            {
                "dataset": dataset,
                "macro_f1_delta": cbe_validation["mean_unseen_macro_f1"]
                - control["mean_unseen_macro_f1"],
                "full_rate_macro_f1_delta": cbe_validation["full_rate_macro_f1"]
                - control["full_rate_macro_f1"],
                "median_filterbank_gradient_ratio": cbe["training_diagnostics"][
                    "median_cbe_filterbank_gradient_ratio"
                ],
            }
        )
    selector_gate = gates["selector"]
    selector_pass = bool(
        all(float(row["learned_aurc_improvement"]) > 0 for row in selector_rows)
        and np.mean([float(row["macro_f1_delta"]) for row in selector_rows])
        >= -float(selector_gate["maximum_average_unseen_macro_f1_drop"])
    )
    cbe_gate = gates["cbe"]
    cbe_pass = bool(
        all(
            float(cbe_gate["minimum_median_filterbank_gradient_ratio"])
            <= float(row["median_filterbank_gradient_ratio"])
            <= float(cbe_gate["maximum_median_filterbank_gradient_ratio"])
            for row in cbe_rows
        )
        and np.mean([float(row["macro_f1_delta"]) for row in cbe_rows])
        >= float(cbe_gate["minimum_average_unseen_macro_f1_delta"])
        and min(float(row["macro_f1_delta"]) for row in cbe_rows)
        >= -float(cbe_gate["maximum_single_dataset_unseen_macro_f1_drop"])
        and np.mean([float(row["full_rate_macro_f1_delta"]) for row in cbe_rows])
        >= -float(cbe_gate["maximum_average_full_rate_macro_f1_drop"])
    )
    return {
        "selector": {"passed": selector_pass, "rows": selector_rows},
        "cbe": {"passed": cbe_pass, "rows": cbe_rows},
        "combined_v2_authorized": False,
        "interpretation": (
            "These are single-seed development gates, not confirmatory evidence. "
            "A combined v2 requires a separate decision after both isolated tracks are reviewed."
        ),
    }


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# NyquistGuard-TSC v2 微型开发实验",
        "",
        f"- 状态：{report['status']}",
        f"- run：`{report['run_root']}`",
        f"- 墙钟：{report['elapsed_seconds']:.1f} 秒（硬上限 {report['wall_time_budget_seconds']} 秒）",
        "- 设计：BasicMotions + PAMAP2，seed17；两条机制独立修改，各 12 epochs。",
        "- 主判据：validation；test 仅为探索性附录，因为 Pilot test 已经看过。",
        "- 边界：本实验不能授权 Full，也不允许直接把两条修改合并。",
        "",
        "## 决策",
        "",
        f"- selector_detached_v2：{'PASS' if report['decision']['selector']['passed'] else 'FAIL'}",
        f"- cbe_balanced_v2：{'PASS' if report['decision']['cbe']['passed'] else 'FAIL'}",
        "- combined_v2：未授权；必须单独审阅。",
        "",
        "## Selector validation",
        "",
        "| 数据集 | learned AURC 改善（正值好） | 未见率 macro-F1 Δ |",
        "|---|---:|---:|",
    ]
    for row in report["decision"]["selector"]["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['learned_aurc_improvement']:+.4f} | {row['macro_f1_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## CBE validation",
            "",
            "| 数据集 | filterbank 梯度比中位数 | 未见率 macro-F1 Δ | full-rate macro-F1 Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["decision"]["cbe"]["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['median_filterbank_gradient_ratio']:.4f} | "
            f"{row['macro_f1_delta']:+.4f} | {row['full_rate_macro_f1_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 单 seed、两个开发数据集不构成统计复制。",
            "- 通过开发门不等于论文主张成立；失败也只否定当前 v2 实例。",
            "- 任何后续较大实验仍须另行批准，不会自动启动。",
            "",
        ]
    )
    return "\n".join(lines)


def _find_resume_root(parent: Path, protocol_hash: str) -> Path | None:
    for manifest_path in sorted(parent.glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("protocol_hash") == protocol_hash and manifest.get("status") != "completed":
            return manifest_path.parent
    return None


def run_v2_micro_pilot(project_root: str | Path, *, resume: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = root / "configs" / "experiments" / "v2_micro.yaml"
    micro_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = root / str(micro_config["base_config"])
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    matrix = build_v2_matrix(micro_config)
    protocol_hash = _protocol_hash(config_path, base_path)
    parent = root / "runs" / "v2_micro"
    run_root = _find_resume_root(parent, protocol_hash) if resume else None
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v2_micro__2datasets__2isolated_variants__seed17__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_resolved.yaml")
    started = time.monotonic()
    budget = int(micro_config["wall_time_budget_seconds"])
    deadline = started + budget
    manifest = {
        "stage": "v2_micro",
        "status": "running",
        "development_only": True,
        "protocol_hash": protocol_hash,
        "matrix": [asdict(spec) for spec in matrix],
        "run_root": str(run_root),
        "wall_time_budget_seconds": budget,
        "pilot_auto_start": False,
        "full_auto_start": False,
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "manifest.json", manifest)
    datasets = {
        dataset: load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset}.npz"
        )
        for dataset in V2_DATASETS
    }
    pilot_root = _latest_completed_pilot(root)
    print("[1/6] Loading frozen v1 seed17 controls", flush=True)
    controls = _load_v1_controls(root, pilot_root, datasets, base_config)
    atomic_write_json(run_root / "v1_controls.json", controls)
    results: dict[str, dict[str, Any]] = {}
    try:
        for index, spec in enumerate(matrix, start=2):
            print(f"[{index}/6] Starting {spec.run_key}", flush=True)
            results[spec.run_key] = _run_one(
                datasets[spec.dataset_id],
                spec,
                base_config,
                micro_config,
                protocol_hash,
                run_root / spec.run_key,
                resume=resume,
                deadline=deadline,
            )
        print("[6/6] Applying pre-frozen development gates and writing report", flush=True)
        decision = _development_decision(
            controls, results, dict(micro_config["development_gates"])
        )
        report = {
            "status": "completed",
            "run_root": str(run_root),
            "protocol_hash": protocol_hash,
            "elapsed_seconds": time.monotonic() - started,
            "wall_time_budget_seconds": budget,
            "development_only": True,
            "controls": controls,
            "results": results,
            "decision": decision,
            "pilot_started": False,
            "full_started": False,
            "finished_at_utc": utc_now(),
        }
        atomic_write_json(run_root / "v2_micro_report.json", report)
        markdown = _render_report(report)
        _atomic_write_text(run_root / "v2_micro_report.md", markdown)
        atomic_write_json(root / "reports" / "v2_micro_report.json", report)
        _atomic_write_text(root / "reports" / "v2_micro_report.md", markdown)
        manifest.update(status="completed", updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        return report
    except BaseException as error:
        manifest.update(
            status="budget_exhausted" if isinstance(error, MicroPilotTimeBudgetExceeded) else "failed",
            error=f"{type(error).__name__}: {error}",
            updated_at_utc=utc_now(),
        )
        atomic_write_json(run_root / "manifest.json", manifest)
        raise


def _fit_selector_head_only(
    model: NyquistGuardTSC,
    dataset: PreparedDataset,
    base_config: dict[str, Any],
    selector_config: dict[str, Any],
    *,
    deadline: float,
) -> list[dict[str, float | int]]:
    if model.selective_head is None:
        raise ValueError("selector_v2b requires a selective head")
    device = next(model.parameters()).device
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.selective_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.selective_head.parameters(),
        lr=float(selector_config["learning_rate"]),
        weight_decay=float(selector_config["weight_decay"]),
    )
    selector_loss = DetachedCorrectnessSelectiveLoss().to(device)
    monotonicity = AcceptanceMonotonicityLoss().to(device)
    train_data = TensorDataset(torch.from_numpy(dataset.train.x), torch.from_numpy(dataset.train.y))
    ratios = tuple(float(value) for value in base_config["train_rate_ratios"])
    history: list[dict[str, float | int]] = []
    for epoch in range(int(selector_config["epochs"])):
        if time.monotonic() >= deadline:
            raise MicroPilotTimeBudgetExceeded("selector_v2b wall-clock budget reached")
        loader = DataLoader(
            train_data,
            batch_size=int(base_config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(V2_SEED + epoch),
            num_workers=int(base_config["num_workers"]),
        )
        losses: list[float] = []
        for batch_index, (x_cpu, targets_cpu) in enumerate(loader):
            if time.monotonic() >= deadline:
                raise MicroPilotTimeBudgetExceeded("selector_v2b wall-clock budget reached")
            x = x_cpu.to(device)
            targets = targets_cpu.to(device)
            ratio = ratios[(epoch + batch_index) % len(ratios)]
            from nyquistguard.experiments.pilot import _view

            model.eval()
            # no_grad tensors can be reused as fixed inputs to a trainable head;
            # inference_mode tensors cannot be saved for the head's backward pass.
            with torch.no_grad():
                high_x, high_rate = _view(x, dataset.sampling_rate_hz, 1.0, base_config)
                low_x, low_rate = _view(x, dataset.sampling_rate_hz, ratio, base_config)
                high = model(high_x, high_rate)
                low = model(low_x, low_rate)
            model.selective_head.train()
            high_logit = model.selective_head(
                high["embedding"],
                high["nyquist_gate"],
                high["aux"]["sampling_rate_hz"],
                high["aux"]["normalized_prediction_entropy"].to(high["embedding"].dtype),
            )
            low_logit = model.selective_head(
                low["embedding"],
                low["nyquist_gate"],
                low["aux"]["sampling_rate_hz"],
                low["aux"]["normalized_prediction_entropy"].to(low["embedding"].dtype),
            )
            high_loss = selector_loss(high["logits"], targets, high_logit)["total"]
            low_loss = selector_loss(low["logits"], targets, low_logit)["total"]
            ordered = monotonicity(
                torch.sigmoid(low_logit),
                torch.sigmoid(high_logit),
                torch.full_like(targets, ratio != 1.0, dtype=torch.bool),
            )
            loss = 0.5 * (high_loss + low_loss) + float(
                selector_config["lambda_monotonicity"]
            ) * ordered
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[selector_v2b:{dataset.dataset_id}] epoch {epoch + 1}/{selector_config['epochs']} "
                f"loss={history[-1]['train_loss']:.4f}",
                flush=True,
            )
    model.eval()
    return history


def run_selector_v2b(project_root: str | Path) -> dict[str, Any]:
    """Post-hoc selector recalibration with the v1 classifier fully frozen."""

    root = Path(project_root).resolve()
    config_path = root / "configs" / "experiments" / "selector_v2b.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if tuple(config["datasets"]) != V2_DATASETS or int(config["seed"]) != V2_SEED:
        raise ValueError("selector_v2b datasets/seed do not match the frozen design")
    base_path = root / str(config["base_config"])
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path, base_path)
    started = time.monotonic()
    budget = int(config["wall_time_budget_seconds"])
    deadline = started + budget
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / "selector_v2b" / f"selector_v2b__2datasets__seed17__{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_root / "config_resolved.yaml")
    pilot_root = _latest_completed_pilot(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        dataset_id: load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        for dataset_id in V2_DATASETS
    }
    results: dict[str, Any] = {}
    try:
        for index, dataset_id in enumerate(V2_DATASETS, start=1):
            dataset = datasets[dataset_id]
            model = _deep_model(dataset, base_config, "nyquistguard", device)
            if not isinstance(model, NyquistGuardTSC):
                raise TypeError("selector_v2b requires NyquistGuardTSC")
            checkpoint = pilot_root / f"{dataset_id}__nyquistguard__seed{V2_SEED}" / "checkpoint_best.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
            before = _evaluate_model(model, dataset, base_config, device)
            classifier_before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if not name.startswith("selective_head.")
            }
            print(f"[{index}/3] Recalibrating frozen selector for {dataset_id}", flush=True)
            history = _fit_selector_head_only(
                model, dataset, base_config, config, deadline=deadline
            )
            after = _evaluate_model(model, dataset, base_config, device)
            classification_unchanged = all(
                torch.equal(parameter.detach().cpu(), classifier_before[name])
                for name, parameter in model.named_parameters()
                if not name.startswith("selective_head.")
            )
            logits_metrics_unchanged = all(
                before[split][metric] == after[split][metric]
                for split in ("validation", "test_exploratory")
                for metric in ("mean_unseen_macro_f1", "full_rate_macro_f1")
            )
            head_path = run_root / dataset_id / "selective_head_state.pt"
            _atomic_torch_save(model.selective_head.state_dict(), head_path)
            atomic_write_json(
                run_root / dataset_id / "training_history.json",
                {"history": history, "updated_at_utc": utc_now()},
            )
            results[dataset_id] = {
                "checkpoint": str(checkpoint),
                "selective_head_state": str(head_path),
                "before": before,
                "after": after,
                "classification_parameters_unchanged": classification_unchanged,
                "classification_metrics_unchanged": logits_metrics_unchanged,
                "validation_learned_aurc_improvement": before["validation"][
                    "mean_unseen_learned_aurc"
                ] - after["validation"]["mean_unseen_learned_aurc"],
                "validation_after_minus_confidence_aurc": after["validation"][
                    "mean_unseen_learned_aurc"
                ] - after["validation"]["mean_unseen_confidence_aurc"],
            }
        gate = config["development_gates"]
        passed = bool(
            all(result["classification_parameters_unchanged"] for result in results.values())
            and all(result["classification_metrics_unchanged"] for result in results.values())
            and all(
                result["validation_learned_aurc_improvement"] > 0
                for result in results.values()
            )
            and all(
                result["validation_after_minus_confidence_aurc"] <= 0
                for result in results.values()
            )
        )
        report = {
            "status": "completed",
            "run_root": str(run_root),
            "protocol_hash": protocol_hash,
            "elapsed_seconds": time.monotonic() - started,
            "wall_time_budget_seconds": budget,
            "development_only": True,
            "results": results,
            "decision": {
                "passed": passed,
                "combined_v2_authorized": False,
                "gates": gate,
            },
            "pilot_started": False,
            "full_started": False,
            "finished_at_utc": utc_now(),
        }
        lines = [
            "# NyquistGuard-TSC selector_v2b 后校准实验",
            "",
            f"- 状态：completed；开发门：{'PASS' if passed else 'FAIL'}",
            f"- 墙钟：{report['elapsed_seconds']:.1f} 秒 / 上限 {budget} 秒",
            "- v1 分类器、filterbank、gate 与 encoder 全部冻结；只更新 selective head。",
            "- validation 为主，test 仅作探索性附录；不会授权 Full。",
            "",
            "| 数据集 | validation learned AURC 改善 | 校准后 learned-confidence AURC | 分类参数/指标不变 |",
            "|---|---:|---:|---:|",
        ]
        for dataset_id, result in results.items():
            lines.append(
                f"| {dataset_id} | {result['validation_learned_aurc_improvement']:+.4f} | "
                f"{result['validation_after_minus_confidence_aurc']:+.4f} | "
                f"{result['classification_parameters_unchanged'] and result['classification_metrics_unchanged']} |"
            )
        markdown = "\n".join(lines) + "\n"
        atomic_write_json(run_root / "selector_v2b_report.json", report)
        _atomic_write_text(run_root / "selector_v2b_report.md", markdown)
        atomic_write_json(root / "reports" / "selector_v2b_report.json", report)
        _atomic_write_text(root / "reports" / "selector_v2b_report.md", markdown)
        return report
    except BaseException as error:
        atomic_write_json(
            run_root / "status.json",
            {
                "status": "budget_exhausted" if isinstance(error, MicroPilotTimeBudgetExceeded) else "failed",
                "error": f"{type(error).__name__}: {error}",
                "updated_at_utc": utc_now(),
            },
        )
        raise
