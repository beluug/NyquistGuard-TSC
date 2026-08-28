"""Leakage-locked, bounded V4 observe-only development screen."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nyquistguard.data import SplitData
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.metrics import classification_metrics
from nyquistguard.experiments.pilot import (
    _atomic_torch_save,
    _resolved_model_config,
    _seed_everything,
    _view,
)
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v3_core_micro import (
    rate_robust_selection_score,
    secondary_training_ratio,
    v3_core_objective_config,
)
from nyquistguard.losses import NyquistGuardObjective
from nyquistguard.models import NyquistGuardTSC
from nyquistguard.research.v4_observe_only import ObserveOnlyNyquistGuardTSC
from nyquistguard.research.v4_residual_gate import ResidualGateNyquistGuardTSC
from nyquistguard.research.v5_dual_path import DualPathNyquistGuardTSC
from nyquistguard.training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)


V4_MICRO_DATASETS = ("basicmotions_uea", "pamap2_uci")
V4_MICRO_SEED = 17
V4_VARIANTS = ("v3_10_hard_gate", "v4_observe_only")


@dataclass(frozen=True)
class DevelopmentDataset:
    """Dataset view that structurally has no test split."""

    dataset_id: str
    sampling_rate_hz: float
    class_names: tuple[str, ...]
    train: SplitData
    validation: SplitData


class V4MicroTimeBudgetExceeded(RuntimeError):
    """Raised only after the latest completed epoch has been checkpointed."""


def load_development_dataset(path: str | Path) -> DevelopmentDataset:
    """Read only train/validation keys; test arrays are never materialized."""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "dataset_id",
            "sampling_rate_hz",
            "class_names",
            "train_x",
            "train_y",
            "train_ids",
            "validation_x",
            "validation_y",
            "validation_ids",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"development cache is missing keys: {missing}")
        return DevelopmentDataset(
            dataset_id=str(payload["dataset_id"].item()),
            sampling_rate_hz=float(payload["sampling_rate_hz"].item()),
            class_names=tuple(payload["class_names"].astype(str).tolist()),
            train=SplitData(payload["train_x"], payload["train_y"], payload["train_ids"]),
            validation=SplitData(
                payload["validation_x"], payload["validation_y"], payload["validation_ids"]
            ),
        )


def _protocol_hash(config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(b"v4-observe-only-micro-runner-v1")
    return digest.hexdigest()


def _new_model(
    dataset: DevelopmentDataset,
    base_config: dict[str, Any],
    variant: str,
    device: torch.device,
) -> NyquistGuardTSC:
    kwargs = _resolved_model_config(dataset, base_config, "no_selective_head")
    if variant == "v3_10_hard_gate":
        model: NyquistGuardTSC = NyquistGuardTSC(**kwargs)
    elif variant == "v4_observe_only":
        model = ObserveOnlyNyquistGuardTSC(**kwargs)
    elif variant == "v4_1_residual_gate":
        model = ResidualGateNyquistGuardTSC(**kwargs, initial_gate_floor=0.5)
    elif variant == "v5_dual_path":
        model = DualPathNyquistGuardTSC(
            **kwargs,
            initial_gate_floor=0.5,
            spatial_channels=24,
        )
    else:
        raise ValueError(f"unknown V4 micro variant: {variant}")
    if model.selective_head is not None:
        raise RuntimeError("V4 micro classification screen must not train a selective head")
    return model.to(device)


def _predict(
    model: NyquistGuardTSC,
    split: SplitData,
    source_rate_hz: float,
    ratio: float,
    base_config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(split.x), torch.from_numpy(split.y)),
        batch_size=int(base_config["batch_size"]),
        shuffle=False,
        num_workers=int(base_config["num_workers"]),
    )
    logits: list[np.ndarray] = []
    gate_mass: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x_cpu, _ in loader:
            viewed, rate = _view(x_cpu.to(device), source_rate_hz, ratio, base_config)
            output = model(viewed, rate)
            logits.append(output["logits"].float().cpu().numpy())
            gate_mass.append(output["nyquist_gate"].float().mean(dim=1).cpu().numpy())
    return np.concatenate(logits), np.concatenate(gate_mass), float(np.mean(np.concatenate(gate_mass)))


def _evaluate_validation(
    model: NyquistGuardTSC,
    dataset: DevelopmentDataset,
    base_config: dict[str, Any],
    ratios: tuple[float, ...],
    device: torch.device,
) -> dict[str, Any]:
    per_rate: dict[str, dict[str, float]] = {}
    pooled_targets: list[np.ndarray] = []
    pooled_logits: list[np.ndarray] = []
    pooled_confidence: list[np.ndarray] = []
    pooled_observability: list[np.ndarray] = []
    full_gate_mass: float | None = None
    cached: list[tuple[float, str, np.ndarray, np.ndarray, float]] = []
    for ratio in ratios:
        logits, _, gate_mass = _predict(
            model, dataset.validation, dataset.sampling_rate_hz, ratio, base_config, device
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        confidence = probabilities.max(axis=1)
        rate_id = f"r{int(round(ratio * 1000)):04d}"
        if ratio == 1.0:
            full_gate_mass = gate_mass
        cached.append((ratio, rate_id, logits, confidence, gate_mass))
    if full_gate_mass is None or not math.isfinite(full_gate_mass) or full_gate_mass <= 0:
        raise RuntimeError("invalid full-rate gate mass")
    for ratio, rate_id, logits, confidence, gate_mass in cached:
        relative_mass = float(np.clip(gate_mass / full_gate_mass, 0.0, 1.0))
        observability = confidence * relative_mass
        confidence_metrics = classification_metrics(dataset.validation.y, logits, confidence)
        observability_metrics = classification_metrics(dataset.validation.y, logits, observability)
        per_rate[rate_id] = {
            "ratio": ratio,
            "macro_f1": confidence_metrics["macro_f1"],
            "confidence_aurc": confidence_metrics["aurc"],
            "observability_aurc": observability_metrics["aurc"],
            "relative_gate_mass": relative_mass,
            "confidence_mean": float(np.mean(confidence)),
            "observability_score_mean": float(np.mean(observability)),
            "prediction_class_count": int(np.unique(logits.argmax(axis=1)).size),
        }
        pooled_targets.append(dataset.validation.y)
        pooled_logits.append(logits)
        pooled_confidence.append(confidence)
        pooled_observability.append(observability)
    unseen_ids = ("r0900", "r0600", "r0400", "r0300")
    unseen = [per_rate[key] for key in unseen_ids]
    pooled_y = np.concatenate(pooled_targets)
    pooled_z = np.concatenate(pooled_logits)
    confidence_metrics = classification_metrics(
        pooled_y, pooled_z, np.concatenate(pooled_confidence)
    )
    observability_metrics = classification_metrics(
        pooled_y, pooled_z, np.concatenate(pooled_observability)
    )
    return {
        "full_rate_macro_f1": per_rate["r1000"]["macro_f1"],
        "mean_unseen_macro_f1": float(np.mean([row["macro_f1"] for row in unseen])),
        "worst_unseen_macro_f1": float(np.min([row["macro_f1"] for row in unseen])),
        "pooled_confidence_aurc": confidence_metrics["aurc"],
        "pooled_observability_aurc": observability_metrics["aurc"],
        "full_to_low_observability_score_drop": (
            per_rate["r1000"]["observability_score_mean"]
            - per_rate["r0300"]["observability_score_mean"]
        ),
        "per_rate": per_rate,
    }


def _selection_score(evaluation: dict[str, Any], selection: dict[str, Any]) -> float:
    return rate_robust_selection_score(
        float(evaluation["full_rate_macro_f1"]),
        float(evaluation["mean_unseen_macro_f1"]),
        float(selection["full_rate_weight"]),
        float(selection["mean_unseen_rate_weight"]),
    )


def _train_variant(
    dataset: DevelopmentDataset,
    base_config: dict[str, Any],
    screen: dict[str, Any],
    variant: str,
    initial_state: dict[str, torch.Tensor],
    protocol_hash: str,
    run_dir: Path,
    device: torch.device,
    deadline: float,
    resume: bool,
    seed: int = V4_MICRO_SEED,
    epoch_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> tuple[NyquistGuardTSC, list[dict[str, Any]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    model = _new_model(dataset, base_config, variant, device)
    model.load_state_dict(initial_state, strict=True)
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
    objective_config = v3_core_objective_config(dataset, base_config)
    objective = NyquistGuardObjective(**objective_config).to(device)
    checkpoint = run_dir / "checkpoint_last.pt"
    best_path = run_dir / "checkpoint_best.pt"
    history_path = run_dir / "training_history.json"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_score = -math.inf
    stale = 0
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
        stale = int(payload.get("extra", {}).get("stale_epochs", 0))
        if history_path.exists():
            history = list(json.loads(history_path.read_text(encoding="utf-8"))["history"])
    train_data = TensorDataset(torch.from_numpy(dataset.train.x), torch.from_numpy(dataset.train.y))
    rates = tuple(float(value) for value in screen["validation_rate_ratios"])
    schedule = {"secondary_rate_sampling": screen["train_rate_schedule"]}
    for epoch in range(start_epoch, int(screen["epochs"])):
        if time.monotonic() >= deadline:
            raise V4MicroTimeBudgetExceeded(f"time budget reached before {variant} epoch {epoch + 1}")
        loader = DataLoader(
            train_data,
            batch_size=int(base_config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=int(base_config["num_workers"]),
        )
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        batches = 0
        for batch_index, (x_cpu, y_cpu) in enumerate(loader):
            if time.monotonic() >= deadline:
                raise V4MicroTimeBudgetExceeded(
                    f"time budget reached during {variant} epoch {epoch + 1}; prior epoch retained"
                )
            x = x_cpu.to(device)
            targets = y_cpu.to(device)
            ratio = secondary_training_ratio(
                schedule, tuple(float(v) for v in base_config["train_rate_ratios"]),
                seed, epoch, batch_index,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                high, high_rate = _view(x, dataset.sampling_rate_hz, 1.0, base_config)
                low, low_rate = _view(x, dataset.sampling_rate_hz, ratio, base_config)
                losses = objective(model(high, high_rate), targets, model(low, low_rate), targets)
                loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {variant} loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(base_config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            total_cls += float(losses["classification"].detach().cpu())
            batches += 1
        validation = _evaluate_validation(model, dataset, base_config, rates, device)
        score = _selection_score(validation, screen["checkpoint_selection"])
        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            stale = 0
            _atomic_torch_save(model.state_dict(), best_path)
        else:
            stale += 1
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batches),
            "classification_loss": total_cls / max(1, batches),
            "validation_selection_score": score,
            "validation_full_rate_macro_f1": validation["full_rate_macro_f1"],
            "validation_mean_unseen_macro_f1": validation["mean_unseen_macro_f1"],
        }
        history.append(row)
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=epoch + 1,
            protocol_hash=protocol_hash,
            model_config={"variant": variant, "selective_head": False},
            extra={"best_score": best_score, "stale_epochs": stale},
        )
        atomic_write_json(history_path, {"history": history, "updated_at_utc": utc_now()})
        print(
            f"[{dataset.dataset_id}][{variant}] epoch {epoch + 1}/{screen['epochs']} "
            f"loss={row['train_loss']:.4f} val={score:.4f}", flush=True,
        )
        if epoch_callback is not None:
            epoch_callback(epoch + 1, int(screen["epochs"]), row)
        if stale >= int(screen["early_stopping_patience"]):
            break
    if not best_path.exists():
        raise RuntimeError(f"no validation-selected checkpoint for {variant}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
    _atomic_torch_save(model.state_dict(), run_dir / "checkpoint_final.pt")
    return model, history


def _decision(results: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    unseen_deltas = []
    full_deltas = []
    reliability_deltas = []
    sufficiency = []
    for row in results.values():
        hard = row["v3_10_hard_gate"]["validation"]
        v4 = row["v4_observe_only"]["validation"]
        unseen_deltas.append(v4["mean_unseen_macro_f1"] - hard["mean_unseen_macro_f1"])
        full_deltas.append(v4["full_rate_macro_f1"] - hard["full_rate_macro_f1"])
        reliability_deltas.append(v4["pooled_observability_aurc"] - v4["pooled_confidence_aurc"])
        sufficiency.append(v4["full_to_low_observability_score_drop"] > 0.0)
    checks = {
        "average_unseen_gain": float(np.mean(unseen_deltas))
        >= float(gates["minimum_average_unseen_macro_f1_delta_vs_hard_gate"]),
        "single_dataset_unseen_floor": float(np.min(unseen_deltas))
        >= -float(gates["maximum_single_dataset_unseen_macro_f1_drop"]),
        "average_full_rate_floor": float(np.mean(full_deltas))
        >= -float(gates["maximum_average_full_rate_macro_f1_drop"]),
        "observability_aurc_safety": float(np.mean(reliability_deltas))
        <= float(gates["maximum_average_pooled_observability_aurc_delta_vs_confidence"]),
        "rate_sufficiency": all(sufficiency)
        if bool(gates["require_rate_sufficiency_drop_both_datasets"]) else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "average_unseen_macro_f1_delta_vs_hard_gate": float(np.mean(unseen_deltas)),
        "minimum_dataset_unseen_macro_f1_delta_vs_hard_gate": float(np.min(unseen_deltas)),
        "average_full_rate_macro_f1_delta_vs_hard_gate": float(np.mean(full_deltas)),
        "average_pooled_observability_aurc_delta_vs_confidence": float(np.mean(reliability_deltas)),
    }


def run_v4_observe_only_micro(project_root: str | Path, *, resume: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v4_observe_only_development.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    screen = config["micro_screen"]
    if tuple(screen["datasets"]) != V4_MICRO_DATASETS or int(screen["seed"]) != V4_MICRO_SEED:
        raise ValueError("V4 micro dataset/seed design changed after freeze")
    if tuple(screen["variants"]) != V4_VARIANTS:
        raise ValueError("V4 micro comparator set changed after freeze")
    if config["data_boundary"]["forbidden"] != ["existing_test_splits", "full_metrics", "full_predictions"]:
        raise ValueError("V4 data boundary changed")
    base_path = root / screen["base_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path, base_path)
    run_root = root / "runs" / "v4_observe_only_micro" / f"v4_observe_only_micro__seed17__{protocol_hash[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)
    deadline = started + float(screen["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    for dataset_id in V4_MICRO_DATASETS:
        dataset = load_development_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        _seed_everything(V4_MICRO_SEED)
        template = _new_model(dataset, base_config, "v3_10_hard_gate", torch.device("cpu"))
        initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
        del template
        dataset_results: dict[str, Any] = {}
        for variant in V4_VARIANTS:
            metrics_path = run_root / dataset_id / variant / "metrics.json"
            if resume and metrics_path.exists():
                cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                if cached.get("protocol_hash") == protocol_hash:
                    dataset_results[variant] = cached
                    continue
            model, history = _train_variant(
                dataset, base_config, screen, variant, initial_state, protocol_hash,
                run_root / dataset_id / variant, device, deadline, resume,
            )
            validation = _evaluate_validation(
                model, dataset, base_config,
                tuple(float(value) for value in screen["validation_rate_ratios"]), device,
            )
            result = {
                "protocol_hash": protocol_hash,
                "epochs_completed": len(history),
                "validation": validation,
                "test_accessed": False,
            }
            atomic_write_json(metrics_path, result)
            dataset_results[variant] = result
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[dataset_id] = dataset_results
    decision = _decision(results, screen["development_gates"])
    report = {
        "status": "completed_candidate_pass" if decision["passed"] else "completed_candidate_fail",
        "protocol_version": config["protocol_version"],
        "protocol_hash": protocol_hash,
        "generation": "post_full_test_review_development",
        "primary_split": "validation_only",
        "test_accessed": False,
        "independent_confirmation_claim_allowed": False,
        "minimum_new_untouched_confirmation_datasets": int(
            config["data_boundary"]["minimum_new_confirmation_datasets"]
        ),
        "paired_initialization": True,
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
        "development_gates": screen["development_gates"],
        "decision": decision,
        "run_root": str(run_root),
        "pilot_started": False,
        "full_started": False,
        "finished_at_utc": utc_now(),
    }
    lines = [
        "# V4 observe-only validation micro screen", "",
        f"- Status: **{'PASS' if decision['passed'] else 'FAIL'}** (development candidate screen only)",
        f"- Device / elapsed: `{device}` / {report['elapsed_seconds']:.1f} s",
        "- Data boundary: train + validation only; existing test arrays were not loaded or scored.",
        "- Confirmation boundary: no independent claim; at least four new untouched datasets remain required.",
        "", "| Dataset | Hard-gate unseen F1 | V4 unseen F1 | Delta | Hard full F1 | V4 full F1 | V4 obs-conf pooled AURC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, row in results.items():
        hard = row["v3_10_hard_gate"]["validation"]
        v4 = row["v4_observe_only"]["validation"]
        lines.append(
            f"| {dataset_id} | {hard['mean_unseen_macro_f1']:.4f} | {v4['mean_unseen_macro_f1']:.4f} | "
            f"{v4['mean_unseen_macro_f1'] - hard['mean_unseen_macro_f1']:+.4f} | "
            f"{hard['full_rate_macro_f1']:.4f} | {v4['full_rate_macro_f1']:.4f} | "
            f"{v4['pooled_observability_aurc'] - v4['pooled_confidence_aurc']:+.4f} |"
        )
    lines.extend(["", "## Frozen gate decision", "", f"```json\n{json.dumps(decision, ensure_ascii=False, indent=2)}\n```", ""])
    markdown = "\n".join(lines)
    atomic_write_json(run_root / "report.json", report)
    _atomic_write_text(run_root / "report.md", markdown)
    atomic_write_json(root / "reports" / "v4_observe_only_micro_report.json", report)
    _atomic_write_text(root / "reports" / "v4_observe_only_micro_report.md", markdown)
    return report
