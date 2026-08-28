"""Bounded end-to-end micro development for the v3.3 core and calibrator."""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from nyquistguard.data import PreparedDataset, load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text, _latest_completed_pilot
from nyquistguard.experiments.pilot import (
    _atomic_torch_save,
    _deep_model,
    _resolved_objective_config,
    _seed_everything,
    _validation_score,
    _view,
)
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.experiments.v2_micro_pilot import _evaluate_model, _evaluate_split
from nyquistguard.experiments.v3_anchored_reliability import confidence_anchored_score
from nyquistguard.experiments.v3_calibrated_reliability import (
    _collect_split,
    _fit_and_score,
    _split_summary,
)
from nyquistguard.experiments.v3_reliability import threshold_for_target_coverage
from nyquistguard.losses import NyquistGuardObjective
from nyquistguard.models import NyquistGuardTSC
from nyquistguard.training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)


V3_CORE_DATASETS = ("basicmotions_uea", "pamap2_uci")
V3_CORE_SEED = 17


class V3CoreTimeBudgetExceeded(RuntimeError):
    """Raised after retaining the last completed epoch checkpoint."""


def secondary_training_ratio(
    micro_config: dict[str, Any],
    base_ratios: tuple[float, ...],
    seed: int,
    epoch: int,
    batch_index: int,
) -> float:
    """Resolve a deterministic secondary-view rate without resume dependence."""

    sampling = micro_config.get("secondary_rate_sampling")
    if sampling is None:
        return base_ratios[(epoch + batch_index) % len(base_ratios)]
    if sampling.get("mode") != "identity_plus_continuous_uniform":
        raise ValueError("unsupported secondary-rate sampling mode")
    period = int(sampling["period_batches"])
    identity_slots = int(sampling["identity_slots"])
    minimum = float(sampling["uniform_min"])
    maximum = float(sampling["uniform_max"])
    if period != 3 or identity_slots != 1 or minimum != 0.3 or maximum != 0.75:
        raise ValueError("v3.10 continuous-rate schedule changed from its frozen design")
    if (epoch + batch_index) % period < identity_slots:
        return 1.0
    derived_seed = (seed * 1_000_003 + epoch * 10_007 + batch_index) & ((1 << 63) - 1)
    return random.Random(derived_seed).uniform(minimum, maximum)


def rate_robust_selection_score(
    full_rate_macro_f1: float,
    mean_unseen_macro_f1: float,
    full_weight: float = 0.5,
    unseen_weight: float = 0.5,
) -> float:
    if full_weight < 0 or unseen_weight < 0 or full_weight + unseen_weight <= 0:
        raise ValueError("checkpoint-selection weights must be non-negative and non-zero")
    total = full_weight + unseen_weight
    return float(
        (full_weight * full_rate_macro_f1 + unseen_weight * mean_unseen_macro_f1)
        / total
    )


def _checkpoint_selection_metrics(
    model: NyquistGuardTSC,
    dataset: PreparedDataset,
    base_config: dict[str, Any],
    device: torch.device,
    selection: dict[str, Any],
) -> dict[str, float]:
    metric = str(selection["metric"])
    if metric == "mean_macro_f1":
        score = _validation_score(model, dataset, base_config, device)
        return {"score": score, "full_rate_macro_f1": float("nan"), "mean_unseen_macro_f1": float("nan")}
    if metric == "balanced_full_unseen_macro_f1":
        evaluation = _evaluate_split(
            model, dataset.validation, dataset, base_config, device
        )
        full = float(evaluation["full_rate_macro_f1"])
        unseen = float(evaluation["mean_unseen_macro_f1"])
        return {
            "score": rate_robust_selection_score(
                full,
                unseen,
                float(selection["full_rate_weight"]),
                float(selection["mean_unseen_rate_weight"]),
            ),
            "full_rate_macro_f1": full,
            "mean_unseen_macro_f1": unseen,
        }
    raise ValueError(f"unsupported checkpoint selection metric: {metric}")


def v3_core_objective_config(
    dataset: PreparedDataset, base_config: dict[str, Any]
) -> dict[str, Any]:
    config = _resolved_objective_config(dataset, base_config, "no_selective_head")
    config["lambda_cbe"] = 0.0
    config["lambda_selective"] = 0.0
    config["lambda_monotonicity"] = 0.0
    return config


def _protocol_hash(micro_path: Path, base_path: Path, reliability_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (micro_path, base_path, reliability_path):
        digest.update(path.read_bytes())
    digest.update(b"v3-core-micro-runner-v1")
    return digest.hexdigest()


def _train_core(
    dataset: PreparedDataset,
    base_config: dict[str, Any],
    micro_config: dict[str, Any],
    protocol_hash: str,
    run_dir: Path,
    *,
    seed: int,
    resume: bool,
    deadline: float,
    model_variant: str = "no_selective_head",
) -> tuple[NyquistGuardTSC, list[dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_variant not in {"no_selective_head", "v3_no_nyquist_gate"}:
        raise ValueError(f"unsupported v3 core model variant: {model_variant}")
    model = _deep_model(dataset, base_config, model_variant, device)
    if not isinstance(model, NyquistGuardTSC) or model.selective_head is not None:
        raise TypeError("v3 core requires NyquistGuardTSC without a selective head")
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
    objective = NyquistGuardObjective(
        **v3_core_objective_config(dataset, base_config)
    ).to(device)
    checkpoint = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "training_history.json"
    start_epoch = 0
    history: list[dict[str, Any]] = []
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
            history = list(
                json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
            )
    selection = micro_config.get("checkpoint_selection")
    best_score = max(
        [float(row["validation_macro_f1"]) for row in history if row.get("validation_macro_f1") is not None],
        default=-float("inf"),
    )
    stale_epochs = 0
    if history and selection:
        best_index = max(
            range(len(history)),
            key=lambda index: float(history[index].get("validation_macro_f1", -float("inf"))),
        )
        stale_epochs = len(history) - best_index - 1

    train_data = TensorDataset(
        torch.from_numpy(dataset.train.x), torch.from_numpy(dataset.train.y)
    )
    ratios = tuple(float(value) for value in base_config["train_rate_ratios"])
    for epoch in range(start_epoch, int(micro_config["epochs"])):
        if time.monotonic() >= deadline:
            raise V3CoreTimeBudgetExceeded(
                f"time budget reached before epoch {epoch + 1}"
            )
        loader = DataLoader(
            train_data,
            batch_size=int(base_config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=int(base_config["num_workers"]),
        )
        model.train()
        total_loss = 0.0
        total_classification = 0.0
        batches = 0
        for batch_index, (x_cpu, targets_cpu) in enumerate(loader):
            if time.monotonic() >= deadline:
                raise V3CoreTimeBudgetExceeded(
                    f"time budget reached during epoch {epoch + 1}; prior checkpoint retained"
                )
            x = x_cpu.to(device)
            targets = targets_cpu.to(device)
            ratio = secondary_training_ratio(
                micro_config, ratios, seed, epoch, batch_index
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                high, high_rate = _view(x, dataset.sampling_rate_hz, 1.0, base_config)
                low, low_rate = _view(x, dataset.sampling_rate_hz, ratio, base_config)
                output_high = model(high, high_rate)
                output_low = model(low, low_rate)
                losses = objective(output_high, targets, output_low, targets)
                loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(base_config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            total_classification += float(losses["classification"].detach().cpu())
            batches += 1
        selection_metrics = (
            _checkpoint_selection_metrics(
                model, dataset, base_config, device, selection
            )
            if selection
            else None
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batches),
            "classification_loss": total_classification / max(1, batches),
            "validation_macro_f1": selection_metrics["score"] if selection_metrics else None,
            "validation_full_rate_macro_f1": selection_metrics["full_rate_macro_f1"] if selection_metrics else None,
            "validation_mean_unseen_macro_f1": selection_metrics["mean_unseen_macro_f1"] if selection_metrics else None,
        }
        history.append(row)
        if selection:
            score = float(row["validation_macro_f1"])
            if score > best_score + 1e-12:
                best_score = score
                stale_epochs = 0
                _atomic_torch_save(model.state_dict(), run_dir / "checkpoint_best.pt")
            else:
                stale_epochs += 1
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=epoch + 1,
            protocol_hash=protocol_hash,
            model_config={"variant": model_variant, "selector": False, "cbe": False},
            extra={"fixed_epoch_endpoint": int(micro_config["epochs"])},
        )
        atomic_write_json(history_path, {"history": history, "updated_at_utc": utc_now()})
        print(
            f"epoch {epoch + 1}/{micro_config['epochs']} "
            f"loss={row['train_loss']:.4f} cls={row['classification_loss']:.4f}"
            + (
                f" val={row['validation_macro_f1']:.4f}"
                if row["validation_macro_f1"] is not None
                else ""
            ),
            flush=True,
        )
        if selection and stale_epochs >= int(selection["early_stopping_patience"]):
            break
    if selection:
        best_path = run_dir / "checkpoint_best.pt"
        if not best_path.exists():
            raise RuntimeError("validation-selected v3 core checkpoint was not saved")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _atomic_torch_save(model.state_dict(), run_dir / "checkpoint_final.pt")
    return model, history


def _anchored_reliability(
    model: NyquistGuardTSC,
    dataset: PreparedDataset,
    base_config: dict[str, Any],
    reliability_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ratios = tuple(float(value) for value in reliability_config["rate_ratios"])
    calibration = reliability_config["calibrator"]
    target_coverage = float(
        reliability_config["threshold_calibration"]["pooled_validation_target_coverage"]
    )
    validation = _collect_split(
        model, dataset.validation, dataset.sampling_rate_hz, ratios, base_config, device
    )
    test = _collect_split(
        model, dataset.test, dataset.sampling_rate_hz, ratios, base_config, device
    )
    validation_raw, test_raw, finite = _fit_and_score(
        validation,
        test,
        float(calibration["regularization_c"]),
        int(calibration["group_folds"]),
        seed,
    )
    groups = len(np.unique(validation["groups"]))
    pseudo = float(calibration["shrinkage_pseudo_groups"])
    validation_score, weight = confidence_anchored_score(
        validation["confidence"], validation_raw, groups, pseudo
    )
    test_score, _ = confidence_anchored_score(
        test["confidence"], test_raw, groups, pseudo
    )
    score_threshold = threshold_for_target_coverage(validation_score, target_coverage)
    confidence_threshold = threshold_for_target_coverage(
        validation["confidence"], target_coverage
    )
    return {
        "calibrator_finite": finite,
        "independent_validation_groups": groups,
        "shrinkage_weight": weight,
        "validation": _split_summary(
            validation, validation_score, score_threshold, confidence_threshold
        ),
        "test_exploratory": _split_summary(
            test, test_score, score_threshold, confidence_threshold
        ),
    }


def _decision(results: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    unseen_deltas = [
        row["candidate_classification"]["validation"]["mean_unseen_macro_f1"]
        - row["v1_control_classification"]["validation"]["mean_unseen_macro_f1"]
        for row in results.values()
    ]
    full_deltas = [
        row["candidate_classification"]["validation"]["full_rate_macro_f1"]
        - row["v1_control_classification"]["validation"]["full_rate_macro_f1"]
        for row in results.values()
    ]
    reliability_reductions = [
        row["candidate_reliability"]["validation"]["pooled_aurc_relative_reduction"]
        for row in results.values()
    ]
    anchored_deltas = [
        row["candidate_reliability"]["validation"]["pooled_calibrated_aurc"]
        - row["v1_control_reliability"]["validation"]["pooled_calibrated_aurc"]
        for row in results.values()
    ]
    checks = {
        "average_unseen_f1": float(np.mean(unseen_deltas))
        >= float(gates["minimum_average_unseen_macro_f1_delta_vs_v1"]),
        "single_dataset_unseen_f1": float(np.min(unseen_deltas))
        >= -float(gates["maximum_single_dataset_unseen_macro_f1_drop"]),
        "average_full_f1": float(np.mean(full_deltas))
        >= -float(gates["maximum_average_full_rate_macro_f1_drop"]),
        "reliability_vs_confidence": all(value > 0.0 for value in reliability_reductions)
        if gates["require_positive_anchored_aurc_reduction_vs_confidence_both"]
        else True,
        "reliability_vs_v1": float(np.mean(anchored_deltas))
        <= float(gates["maximum_average_anchored_aurc_delta_vs_v1"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "average_unseen_macro_f1_delta_vs_v1": float(np.mean(unseen_deltas)),
        "minimum_dataset_unseen_macro_f1_delta_vs_v1": float(np.min(unseen_deltas)),
        "average_full_rate_macro_f1_delta_vs_v1": float(np.mean(full_deltas)),
        "average_anchored_aurc_delta_vs_v1": float(np.mean(anchored_deltas)),
    }


def _run_v3_core_micro(
    project_root: str | Path,
    resume: bool,
    *,
    config_filename: str,
    run_namespace: str,
    report_stem: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    micro_path = root / "configs" / "experiments" / config_filename
    micro_config = yaml.safe_load(micro_path.read_text(encoding="utf-8"))
    if tuple(micro_config["datasets"]) != V3_CORE_DATASETS:
        raise ValueError("v3 core micro datasets changed from the frozen pair")
    if int(micro_config["seed"]) != V3_CORE_SEED:
        raise ValueError("v3 core micro seed changed")
    base_path = root / micro_config["base_config"]
    reliability_path = root / micro_config["reliability_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    reliability_config = yaml.safe_load(reliability_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(micro_path, base_path, reliability_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / run_namespace / f"{run_namespace}__seed17__{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    deadline = started + float(micro_config["wall_time_budget_seconds"])
    pilot_root = _latest_completed_pilot(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    for dataset_id in V3_CORE_DATASETS:
        dataset = load_prepared_dataset(
            root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
        )
        candidate, history = _train_core(
            dataset,
            base_config,
            micro_config,
            protocol_hash,
            run_root / dataset_id,
            seed=V3_CORE_SEED,
            resume=resume,
            deadline=deadline,
        )
        candidate.eval()
        control = _deep_model(dataset, base_config, "nyquistguard", device)
        control_path = (
            pilot_root
            / f"{dataset_id}__nyquistguard__seed{V3_CORE_SEED}"
            / "checkpoint_best.pt"
        )
        control.load_state_dict(
            torch.load(control_path, map_location=device, weights_only=True), strict=True
        )
        control.eval()
        results[dataset_id] = {
            "epochs_completed": len(history),
            "candidate_classification": _evaluate_model(
                candidate, dataset, base_config, device
            ),
            "v1_control_classification": _evaluate_model(
                control, dataset, base_config, device
            ),
            "candidate_reliability": _anchored_reliability(
                candidate, dataset, base_config, reliability_config, V3_CORE_SEED
            ),
            "v1_control_reliability": _anchored_reliability(
                control, dataset, base_config, reliability_config, V3_CORE_SEED
            ),
            "v1_control_checkpoint": str(control_path),
        }
        atomic_write_json(run_root / dataset_id / "metrics.json", results[dataset_id])
        del candidate, control, dataset
    decision = _decision(results, micro_config["development_gates"])
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "status": "completed",
        "protocol_version": micro_config["protocol_version"],
        "protocol_hash": protocol_hash,
        "seed": V3_CORE_SEED,
        "epochs": int(micro_config["epochs"]),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "primary_split": "validation",
        "test_role": "exploratory_appendix",
        "pilot_started": False,
        "full_started": False,
        "results": results,
        "development_gates": micro_config["development_gates"],
        "decision": decision,
        "run_root": str(run_root),
        "finished_at_utc": utc_now(),
    }
    lines = [
        f"# NyquistGuard-TSC {micro_config['stage']}",
        "",
        "- 核心：physical filterbank + Nyquist gate + classifier；无 learned selector、无 CBE。",
        "- 可靠性：训练后拟合 v3.3 confidence-anchored calibrator。",
        f"- BasicMotions/PAMAP2，seed17，固定 {micro_config['epochs']} epochs；{elapsed:.1f} 秒。",
        "- 主判据：validation；test 仅研发附录。",
        f"- 冻结开发门：{'PASS' if decision['passed'] else 'FAIL'}；不授权 Pilot/Full。",
        "",
        "| 数据集 | unseen F1 Δ vs v1 | full F1 Δ vs v1 | anchored AURC reduction vs confidence | anchored AURC Δ vs v1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset_id, row in results.items():
        candidate_class = row["candidate_classification"]["validation"]
        control_class = row["v1_control_classification"]["validation"]
        candidate_rel = row["candidate_reliability"]["validation"]
        control_rel = row["v1_control_reliability"]["validation"]
        lines.append(
            f"| {dataset_id} | "
            f"{candidate_class['mean_unseen_macro_f1'] - control_class['mean_unseen_macro_f1']:+.4f} | "
            f"{candidate_class['full_rate_macro_f1'] - control_class['full_rate_macro_f1']:+.4f} | "
            f"{candidate_rel['pooled_aurc_relative_reduction'] * 100:+.2f}% | "
            f"{candidate_rel['pooled_calibrated_aurc'] - control_rel['pooled_calibrated_aurc']:+.4f} |"
        )
    lines.extend(["", "## 决策检查", ""])
    for name, passed in decision["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    markdown = "\n".join(lines) + "\n"
    atomic_write_json(run_root / f"{report_stem}.json", report)
    _atomic_write_text(run_root / f"{report_stem}.md", markdown)
    atomic_write_json(root / "reports" / f"{report_stem}.json", report)
    _atomic_write_text(root / "reports" / f"{report_stem}.md", markdown)
    return report


def run_v3_core_micro(project_root: str | Path, resume: bool = True) -> dict[str, Any]:
    return _run_v3_core_micro(
        project_root,
        resume,
        config_filename="v3_core_micro.yaml",
        run_namespace="v3_core_micro",
        report_stem="v3_core_micro_report",
    )


def run_v3_core_refinement(project_root: str | Path, resume: bool = True) -> dict[str, Any]:
    return _run_v3_core_micro(
        project_root,
        resume,
        config_filename="v3_core_refinement.yaml",
        run_namespace="v3_core_refinement",
        report_stem="v3_core_refinement_report",
    )
