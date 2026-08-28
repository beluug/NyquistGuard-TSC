"""End-to-end real-data smoke stage with dashboard progress reporting."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from nyquistguard.data import ChannelStandardizer, load_uea_ts, resample_antialiased
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.losses import NyquistGuardObjective
from nyquistguard.models import NyquistGuardTSC
from nyquistguard.training import load_training_checkpoint, save_training_checkpoint


SMOKE_TASKS = [
    "环境与设备自检",
    "模型单元与合成回归",
    "BasicMotions 真实数据读取",
    "防泄漏与抗混叠多率视图",
    "CPU/CUDA 单批训练",
    "Checkpoint / Resume 检查",
    "Smoke 汇总报告",
]


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self._stream.write(line + "\n")

    def close(self) -> None:
        self._stream.close()


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
    tracked = [
        config_path,
        Path(__file__),
        Path(__file__).parents[1] / "data" / "uea_ts.py",
        Path(__file__).parents[1] / "data" / "resampling.py",
        Path(__file__).parents[1] / "training" / "checkpointing.py",
    ]
    for path in tracked:
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_snapshot(project_root: Path) -> dict[str, Any]:
    gpu_name = None
    driver = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",", 2)]
        gpu_name = fields[0]
        driver = fields[1]
        gpu_memory_mib = int(float(fields[2]))
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        gpu_memory_mib = None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None

    return {
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_driver": driver,
        "gpu_memory_mib": gpu_memory_mib,
        "git_commit": commit,
        "determinism": {
            "seeded_python_numpy_torch": True,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "strict_deterministic_algorithms": False,
            "note": "Strict deterministic algorithms are not forced; smoke records finite reproducible execution on this host.",
        },
    }


def _balanced_indices(labels: np.ndarray, class_names: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    for class_name in class_names:
        matches = np.flatnonzero(labels == class_name)
        if not len(matches):
            raise ValueError(f"training split has no sample for class {class_name!r}")
        indices.append(int(matches[0]))
    return indices


def _build_model(config: dict[str, Any], input_channels: int, num_classes: int, device: torch.device) -> NyquistGuardTSC:
    model_config = dict(config["model"])
    model_config.update(input_channels=input_channels, num_classes=num_classes)
    return NyquistGuardTSC(**model_config).to(device)


def _build_objective(config: dict[str, Any], device: torch.device) -> NyquistGuardObjective:
    return NyquistGuardObjective(**dict(config["objective"])).to(device)


def _training_step(
    *,
    config: dict[str, Any],
    high_view: torch.Tensor,
    low_view: torch.Tensor,
    high_rate: float,
    low_rate: float,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[NyquistGuardTSC, torch.optim.Optimizer, object | None, dict[str, Any]]:
    _seed_everything(int(config["seed"]))
    model = _build_model(config, high_view.shape[1], int(targets.max().item()) + 1, device)
    objective = _build_objective(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    use_amp = device.type == "cuda"
    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=float(config["amp_initial_scale"]),
            growth_interval=int(config["amp_growth_interval"]),
        )
        if use_amp
        else None
    )
    high_device = high_view.to(device)
    low_device = low_view.to(device)
    targets_device = targets.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        output_high = model(high_device, high_rate)
        output_low = model(low_device, low_rate)
        losses = objective(output_high, targets_device, output_low, targets_device)
    if scaler is None:
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
    else:
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
    if not torch.isfinite(losses["total"]) or not finite_gradients:
        raise RuntimeError(f"non-finite {device.type} loss or gradients")
    metrics = {
        "device": str(device),
        "amp": use_amp,
        "amp_scale": float(scaler.get_scale()) if scaler is not None else None,
        "elapsed_seconds": elapsed,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "finite_gradients": finite_gradients,
        "losses": {key: float(value.detach().float().cpu()) for key, value in losses.items()},
        "high_logits_shape": list(output_high["logits"].shape),
        "low_logits_shape": list(output_low["logits"].shape),
        "high_gate_mean": float(output_high["nyquist_gate"].detach().float().mean().cpu()),
        "low_gate_mean": float(output_low["nyquist_gate"].detach().float().mean().cpu()),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
    }
    return model, optimizer, scaler, metrics


def _run_pytest(project_root: Path, logger: RunLogger) -> dict[str, Any]:
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "-q"]
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    captured: list[str] = []
    for line in process.stdout:
        clean = line.rstrip()
        captured.append(clean)
        logger.log("pytest | " + clean)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"pytest failed with return code {return_code}")
    return {"command": command, "return_code": return_code, "elapsed_seconds": time.perf_counter() - started, "tail": captured[-10:]}


def _find_completed_run(project_root: Path, protocol_hash: str) -> Path | None:
    smoke_root = project_root / "runs" / "smoke"
    if not smoke_root.exists():
        return None
    for status_path in sorted(smoke_root.glob("*/status.json"), reverse=True):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") == "completed" and status.get("protocol_hash") == protocol_hash:
            return status_path.parent
    return None


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    training = report["checks"]["training"]
    checkpoint = report["checks"]["checkpoint_resume"]
    data = report["checks"]["real_data"]
    lines = [
        "# NyquistGuard-TSC Smoke Report",
        "",
        f"- Status: **{report['status']}**",
        f"- Run ID: `{report['run_id']}`",
        f"- Protocol hash: `{report['protocol_hash']}`",
        f"- Finished UTC: `{report['finished_at_utc']}`",
        "",
        "## Verified",
        "",
        f"- BasicMotions parsed as `{data['train_shape']}` train and `{data['test_shape']}` test.",
        f"- Official split IDs disjoint: `{data['split_ids_disjoint']}`.",
        f"- CPU training finite: `{training['cpu']['finite_gradients']}`.",
        f"- CUDA AMP training finite: `{training['cuda']['finite_gradients']}`.",
        f"- Checkpoint restored with max logit difference `{checkpoint['max_abs_logit_difference']:.3e}`.",
        "",
        "## Boundary",
        "",
        "Smoke validates engineering execution only. It is not a pilot result and makes no real-data superiority claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(project_root: str | Path, *, resume: bool = False) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config_path = project_root / "configs" / "experiments" / "smoke.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path)
    dashboard_path = project_root / "runs" / "dashboard_status.json"

    if resume:
        completed = _find_completed_run(project_root, protocol_hash)
        if completed is not None:
            progress = DashboardProgress(dashboard_path, "smoke", SMOKE_TASKS, completed.name)
            for task in progress.tasks:
                task["status"] = "completed"
            progress.finish()
            report_path = completed / "smoke_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"[{utc_now()}] Resume: matching completed smoke run reused: {completed}", flush=True)
            return report

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"smoke__basicmotions_uea__nyquistguard_tsc__seed{config['seed']}__{timestamp}"
    run_dir = project_root / "runs" / "smoke" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(run_dir / "train.log")
    progress = DashboardProgress(dashboard_path, "smoke", SMOKE_TASKS, run_id)
    status_path = run_dir / "status.json"
    started_at = utc_now()
    run_status: dict[str, Any] = {
        "run_id": run_id,
        "stage": "smoke",
        "status": "running",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "protocol_hash": protocol_hash,
        "error": None,
    }
    atomic_write_json(status_path, run_status)
    checks: dict[str, Any] = {}

    def task(index: int, function: Callable[[], Any]) -> Any:
        progress.start_task(index)
        logger.log(f"TASK {index + 1}/{len(SMOKE_TASKS)} START | {SMOKE_TASKS[index]}")
        try:
            value = function()
        except BaseException as error:
            progress.fail_task(index, error)
            run_status.update(status="failed", finished_at_utc=utc_now(), error=f"{type(error).__name__}: {error}")
            atomic_write_json(status_path, run_status)
            logger.log(f"TASK FAILED | {error}")
            raise
        progress.complete_task(index)
        logger.log(f"TASK {index + 1}/{len(SMOKE_TASKS)} DONE | {SMOKE_TASKS[index]}")
        return value

    try:
        _seed_everything(int(config["seed"]))

        def environment_task() -> dict[str, Any]:
            snapshot = _environment_snapshot(project_root)
            if not snapshot["cuda_available"]:
                raise RuntimeError("CUDA is required by this project's smoke protocol but is unavailable")
            atomic_write_json(run_dir / "environment.json", snapshot)
            logger.log(f"Environment: torch={snapshot['torch']} GPU={snapshot['gpu_name']}")
            return snapshot

        checks["environment"] = task(0, environment_task)
        checks["pytest"] = task(1, lambda: _run_pytest(project_root, logger))

        train_path = project_root / "data" / "raw" / "uea" / "BasicMotions" / "BasicMotions_TRAIN.ts"
        test_path = project_root / "data" / "raw" / "uea" / "BasicMotions" / "BasicMotions_TEST.ts"

        def data_task() -> dict[str, Any]:
            nonlocal train, test, standardizer, batch_indices, targets
            train = load_uea_ts(train_path, split="train")
            test = load_uea_ts(test_path, split="test")
            if train.x.shape != (40, 6, 100) or test.x.shape != (40, 6, 100):
                raise RuntimeError(f"unexpected BasicMotions shapes: {train.x.shape}, {test.x.shape}")
            if not np.isfinite(train.x).all() or not np.isfinite(test.x).all():
                raise RuntimeError("BasicMotions contains non-finite values")
            standardizer = ChannelStandardizer.fit(train)
            batch_indices = _balanced_indices(train.y, train.class_names)
            label_to_index = {label: index for index, label in enumerate(train.class_names)}
            targets = torch.tensor([label_to_index[train.y[index]] for index in batch_indices], dtype=torch.long)
            details = {
                "dataset_id": "basicmotions_uea",
                "source_train": str(train.source_path),
                "source_test": str(test.source_path),
                "train_shape": list(train.x.shape),
                "test_shape": list(test.x.shape),
                "class_names": list(train.class_names),
                "train_class_counts": dict(Counter(train.y.tolist())),
                "test_class_counts": dict(Counter(test.y.tolist())),
                "batch_indices": batch_indices,
                "batch_sample_ids": [train.sample_ids[index] for index in batch_indices],
                "standardizer": standardizer.to_dict(),
            }
            logger.log(f"BasicMotions: train={train.x.shape}, test={test.x.shape}, classes={train.class_names}")
            return details

        train = test = standardizer = batch_indices = targets = None
        checks["real_data"] = task(2, data_task)
        assert train is not None and test is not None and standardizer is not None and batch_indices is not None and targets is not None

        def leakage_task() -> dict[str, Any]:
            nonlocal high_view, low_view, high_rate, low_rate, resampling_metadata
            train_ids = set(train.sample_ids)
            test_ids = set(test.sample_ids)
            disjoint = train_ids.isdisjoint(test_ids)
            if not disjoint:
                raise RuntimeError("official train/test sample IDs overlap")
            normalized = standardizer.transform(train.x[batch_indices])
            high_view = torch.from_numpy(normalized)
            high_view, high_rate, high_meta = resample_antialiased(
                high_view,
                float(config["original_sampling_rate_hz"]),
                float(config["high_rate_ratio"]),
                taps=int(config["antialias_fir_taps"]),
                rolloff=float(config["antialias_rolloff"]),
            )
            low_view, low_rate, low_meta = resample_antialiased(
                torch.from_numpy(normalized),
                float(config["original_sampling_rate_hz"]),
                float(config["low_rate_ratio"]),
                taps=int(config["antialias_fir_taps"]),
                rolloff=float(config["antialias_rolloff"]),
            )
            if high_view.shape != (4, 6, 100) or low_view.shape != (4, 6, 50):
                raise RuntimeError(f"unexpected paired view shapes: {high_view.shape}, {low_view.shape}")
            resampling_metadata = {"high": high_meta, "low": low_meta}
            checks["real_data"].update(
                split_ids_disjoint=disjoint,
                preprocessing_fit_split=standardizer.fitted_split,
                high_view_shape=list(high_view.shape),
                low_view_shape=list(low_view.shape),
                high_rate_hz=high_rate,
                low_rate_hz=low_rate,
            )
            logger.log(f"Leakage check passed; paired views {tuple(high_view.shape)} @ {high_rate:g} Hz and {tuple(low_view.shape)} @ {low_rate:g} Hz")
            return {"split_ids_disjoint": disjoint, "resampling": resampling_metadata}

        high_view = low_view = high_rate = low_rate = resampling_metadata = None
        checks["leakage_and_resampling"] = task(3, leakage_task)
        assert high_view is not None and low_view is not None and high_rate is not None and low_rate is not None

        resolved_config = dict(config)
        resolved_config["model"] = dict(config["model"])
        resolved_config["model"].update(input_channels=6, num_classes=4)
        resolved_config["protocol_hash"] = protocol_hash
        resolved_config["project_root"] = str(project_root)
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")

        data_manifest = {
            "dataset_id": "basicmotions_uea",
            "protocol_version": config["protocol_version"],
            "created_at_utc": utc_now(),
            "raw_root": str(train_path.parent),
            "source_files": [str(train_path), str(test_path)],
            "integrity_status": "user_verified_no_recheck_requested",
            "official_split": True,
            "train_sample_ids": list(train.sample_ids),
            "test_sample_ids": list(test.sample_ids),
            "shapes": {"train": list(train.x.shape), "test": list(test.x.shape)},
            "class_names": list(train.class_names),
            "preprocessing": standardizer.to_dict(),
            "resampling": resampling_metadata,
        }
        data_manifest["manifest_hash"] = _hash_json(data_manifest)
        atomic_write_json(run_dir / "data_manifest.json", data_manifest)

        def training_task() -> dict[str, Any]:
            nonlocal checkpoint_model, checkpoint_optimizer, checkpoint_scaler, checkpoint_device
            _, _, _, cpu_metrics = _training_step(
                config=config,
                high_view=high_view,
                low_view=low_view,
                high_rate=high_rate,
                low_rate=low_rate,
                targets=targets,
                device=torch.device("cpu"),
            )
            checkpoint_device = torch.device("cuda")
            checkpoint_model, checkpoint_optimizer, checkpoint_scaler, cuda_metrics = _training_step(
                config=config,
                high_view=high_view,
                low_view=low_view,
                high_rate=high_rate,
                low_rate=low_rate,
                targets=targets,
                device=checkpoint_device,
            )
            logger.log(
                f"Finite training: CPU loss={cpu_metrics['losses']['total']:.4f}; "
                f"CUDA AMP loss={cuda_metrics['losses']['total']:.4f}; peak={cuda_metrics['peak_vram_mib']:.1f} MiB"
            )
            return {"cpu": cpu_metrics, "cuda": cuda_metrics}

        checkpoint_model = checkpoint_optimizer = checkpoint_scaler = checkpoint_device = None
        checks["training"] = task(4, training_task)
        assert checkpoint_model is not None and checkpoint_optimizer is not None and checkpoint_device is not None

        def checkpoint_task() -> dict[str, Any]:
            model_config = dict(resolved_config["model"])
            last_path = run_dir / "checkpoint_last.pt"
            save_training_checkpoint(
                last_path,
                model=checkpoint_model,
                optimizer=checkpoint_optimizer,
                scaler=checkpoint_scaler,
                step=1,
                protocol_hash=protocol_hash,
                model_config=model_config,
                extra={"data_manifest_hash": data_manifest["manifest_hash"]},
            )
            best_path = run_dir / "checkpoint_best.pt"
            shutil.copy2(last_path, best_path)
            checkpoint_model.eval()
            with torch.inference_mode():
                before = checkpoint_model(high_view.to(checkpoint_device), high_rate)["logits"].float().cpu()
            restored = NyquistGuardTSC(**model_config).to(checkpoint_device)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=float(config["learning_rate"]))
            restored_scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
                init_scale=float(config["amp_initial_scale"]),
                growth_interval=int(config["amp_growth_interval"]),
            )
            payload = load_training_checkpoint(
                last_path,
                model=restored,
                optimizer=restored_optimizer,
                scaler=restored_scaler,
                expected_protocol_hash=protocol_hash,
                map_location=checkpoint_device,
                restore_rng=True,
            )
            restored.eval()
            with torch.inference_mode():
                after = restored(high_view.to(checkpoint_device), high_rate)["logits"].float().cpu()
            maximum_difference = float((before - after).abs().max())
            if maximum_difference > 1e-7 or int(payload["step"]) != 1:
                raise RuntimeError(f"checkpoint restore mismatch: {maximum_difference}")
            logger.log(f"Checkpoint/resume exactness max_abs_diff={maximum_difference:.3e}")
            return {
                "checkpoint_last": str(last_path),
                "checkpoint_best": str(best_path),
                "restored_step": int(payload["step"]),
                "protocol_hash_match": True,
                "max_abs_logit_difference": maximum_difference,
            }

        checks["checkpoint_resume"] = task(5, checkpoint_task)

        report: dict[str, Any] = {}

        def report_task() -> dict[str, Any]:
            nonlocal report
            report = {
                "status": "completed",
                "stage": "smoke",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "protocol_hash": protocol_hash,
                "data_manifest_hash": data_manifest["manifest_hash"],
                "checks": checks,
                "scientific_boundary": "Engineering smoke only; no pilot performance or superiority claim.",
            }
            atomic_write_json(run_dir / "metrics.json", report)
            atomic_write_json(run_dir / "smoke_report.json", report)
            atomic_write_json(project_root / "reports" / "smoke_report.json", report)
            _write_markdown_report(run_dir / "smoke_report.md", report)
            _write_markdown_report(project_root / "reports" / "smoke_report.md", report)
            return {"report_json": str(project_root / "reports" / "smoke_report.json")}

        checks["report"] = task(6, report_task)
        progress.finish()
        run_status.update(status="completed", finished_at_utc=utc_now())
        atomic_write_json(status_path, run_status)
        report["finished_at_utc"] = run_status["finished_at_utc"]
        report["checks"] = checks
        atomic_write_json(run_dir / "metrics.json", report)
        atomic_write_json(run_dir / "smoke_report.json", report)
        atomic_write_json(project_root / "reports" / "smoke_report.json", report)
        _write_markdown_report(run_dir / "smoke_report.md", report)
        _write_markdown_report(project_root / "reports" / "smoke_report.md", report)
        logger.log(f"SMOKE COMPLETED | run_dir={run_dir}")
        return report
    finally:
        logger.close()
