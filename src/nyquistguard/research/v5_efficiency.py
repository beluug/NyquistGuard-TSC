"""Sequential inference-efficiency audit of frozen V4.1 and V5.1 checkpoints."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.data.v5_independent_datasets import (
    V5_INDEPENDENT_DATASETS,
    prepare_v5_independent_dataset,
)
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.progress import atomic_write_json, utc_now
from nyquistguard.research.v4_observe_only_micro import _new_model
from nyquistguard.research.v5_independent_confirmation import _assert_no_active_other_stage, _view


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flops(model: torch.nn.Module, x: torch.Tensor, rate: float, device: torch.device) -> int | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode(), torch.profiler.profile(
            activities=activities, with_flops=True
        ) as profile:
            model(x[:1], rate)
        return int(sum(event.flops for event in profile.key_averages() if event.flops))
    except (RuntimeError, AssertionError):
        return None


def _profile(
    model: torch.nn.Module,
    x: torch.Tensor,
    rate: float,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, rate)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repeats):
            _synchronize(device)
            started = time.perf_counter()
            model(x, rate)
            _synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        )
    return {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )),
        "flops_per_sample": _flops(model, x, rate, device),
        "batch_size": int(x.shape[0]),
        "latency_ms_batch_mean": float(statistics.mean(samples)),
        "latency_ms_batch_median": float(statistics.median(samples)),
        "latency_ms_per_sample_mean": float(statistics.mean(samples) / x.shape[0]),
        "throughput_samples_per_second": float(x.shape[0] * 1000.0 / statistics.mean(samples)),
        "peak_cuda_memory_bytes": peak,
        "timed_iterations": repeats,
    }


def run_v5_1_efficiency(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    # Reuse the same resource exclusion rule; temporarily mark this read-only stage as allowed.
    status_path = root / "runs" / "dashboard_status.json"
    if status_path.exists():
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if state.get("status") == "running":
            raise RuntimeError(f"refusing efficiency audit while {state.get('stage')} is running")
    started = time.monotonic()
    config = yaml.safe_load(
        (root / "configs/experiments/v5_1_efficiency.yaml").read_text(encoding="utf-8")
    )
    source = json.loads((root / config["source_report"]).read_text(encoding="utf-8"))
    if source.get("protocol_hash") != config["required_source_protocol_hash"]:
        raise ValueError("frozen V5.1 source protocol hash changed")
    if source.get("status") != "completed" or not source.get("decision", {}).get("passed"):
        raise ValueError("frozen V5.1 source is not a completed PASS")
    if tuple(config["datasets"]) != V5_INDEPENDENT_DATASETS:
        raise ValueError("efficiency dataset panel changed")
    if config["sequential_only"] is not True or config["training_forbidden"] is not True:
        raise ValueError("efficiency audit must remain sequential and inference-only")
    base_template = yaml.safe_load((root / config["base_config"]).read_text(encoding="utf-8"))
    run_root = Path(source["run_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    seed = int(config["seed"])
    for dataset_id in V5_INDEPENDENT_DATASETS:
        dataset = prepare_v5_independent_dataset(root, dataset_id, confirmed_test_access=True)
        base_config = dict(base_template)
        base_config["batch_size"] = int(config["batch_size"])
        batch_size = min(int(config["batch_size"]), len(dataset.test.y))
        x = torch.from_numpy(np.asarray(dataset.test.x[:batch_size])).to(device)
        for role in config["roles"]:
            checkpoint = run_root / f"{dataset_id}__seed{seed}__{role}" / "checkpoint_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = _new_model(_view(dataset, "validation"), base_config, role, device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
            metrics = _profile(
                model, x, dataset.sampling_rate_hz, device,
                int(config["warmup_iterations"]), int(config["timed_iterations"]),
            )
            rows.append({
                "dataset_id": dataset_id, "seed": seed, "role": role,
                "checkpoint_bytes": checkpoint.stat().st_size, **metrics,
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {}
    for role in config["roles"]:
        selected = [row for row in rows if row["role"] == role]
        summary[role] = {
            key: float(np.mean([row[key] for row in selected if row[key] is not None]))
            for key in (
                "parameters", "flops_per_sample", "latency_ms_per_sample_mean",
                "throughput_samples_per_second", "checkpoint_bytes",
            )
            if any(row[key] is not None for row in selected)
        }
    report = {
        "status": "completed", "protocol_version": config["protocol_version"],
        "source_protocol_hash": source["protocol_hash"], "training_performed": False,
        "sequential": True, "device": str(device), "rows": rows, "summary": summary,
        "elapsed_seconds": time.monotonic() - started, "later_stage_started": False,
        "finished_at_utc": utc_now(),
    }
    lines = [
        "# V5.1 sequential efficiency benchmark", "",
        "- Frozen checkpoints only; no training was performed.",
        f"- Device: `{device}`; elapsed: `{report['elapsed_seconds']:.2f} s`.", "",
        "| Role | Parameters | FLOPs/sample | Latency ms/sample | Throughput samples/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, row in summary.items():
        lines.append(
            f"| {role} | {row.get('parameters', float('nan')):.0f} | "
            f"{row.get('flops_per_sample', float('nan')):.0f} | "
            f"{row['latency_ms_per_sample_mean']:.4f} | "
            f"{row['throughput_samples_per_second']:.2f} |"
        )
    lines.extend(["", "No later stage was started automatically.", ""])
    atomic_write_json(root / "reports/v5_1_efficiency_report.json", report)
    _atomic_write_text(root / "reports/v5_1_efficiency_report.md", "\n".join(lines))
    return report
