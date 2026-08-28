"""Two-worker orchestration for the frozen Full numerical protocol.

This module intentionally lives outside ``full.py`` and is excluded from the
Full numerical protocol hash.  It changes only scheduling: the two workers own
disjoint matrix indices, reuse the same run directories/checkpoints, and meet
at a barrier after every dataset.  Model, data, seeds, rates and evaluation are
all delegated to the frozen Full implementation.  A documented resource
amendment uses 1,000 MultiROCKET kernels on this 16 GiB host; only superseded
MultiROCKET artifacts are rerun, while all other completed methods are reused.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from nyquistguard.data import FULL_DATASETS, full_cache_path, prepare_full_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.aeon_memory import enable_multirocket_memory_patch
from nyquistguard.experiments.full import (
    FULL_METHODS,
    FULL_SEEDS,
    FullRunSpec,
    _aggregate,
    _find_resume_root,
    _generate_figures,
    _protocol_hash,
    _report_markdown,
    _run_one,
    _validate_confirmation,
    _write_aggregate_csv,
    build_full_matrix,
)
from nyquistguard.experiments.progress import atomic_write_json, utc_now


PARALLEL_WORKERS = 2
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MULTIROCKET_KERNELS = 1_000
MULTIROCKET_AMENDMENT_ID = "multirocket_1000kernels_16gib_resource_amendment_v2"


def partition_full_matrix(
    matrix: Sequence[FullRunSpec], workers: int = PARALLEL_WORKERS
) -> tuple[tuple[FullRunSpec, ...], ...]:
    if workers != 2:
        raise ValueError("the frozen local parallel scheduler requires exactly two workers")
    partitions = tuple(
        tuple(spec for index, spec in enumerate(matrix) if index % workers == worker)
        for worker in range(workers)
    )
    flattened = [spec.run_key for partition in partitions for spec in partition]
    expected = [spec.run_key for spec in matrix]
    if len(flattened) != len(expected) or set(flattened) != set(expected):
        raise RuntimeError("parallel partitions must be disjoint and cover the Full matrix")
    return partitions


def _run_protocol_hash(spec: FullRunSpec, protocol_hash: str) -> str:
    if spec.method != "multirocket":
        return protocol_hash
    payload = f"{protocol_hash}|{MULTIROCKET_AMENDMENT_ID}|{MULTIROCKET_KERNELS}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _completed(run_root: Path, spec: FullRunSpec, protocol_hash: str) -> bool:
    path = run_root / spec.run_key / "metrics.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("protocol_hash") == _run_protocol_hash(spec, protocol_hash)
    )


def _archive_incompatible_multirocket_runs(
    run_root: Path, matrix: Sequence[FullRunSpec], protocol_hash: str
) -> list[str]:
    """Move incompatible MultiROCKET directories into a recoverable audit area."""
    archive_root = run_root / "superseded_resource_amendment"
    moved: list[str] = []
    for spec in matrix:
        if spec.method != "multirocket":
            continue
        run_dir = run_root / spec.run_key
        if not run_dir.exists():
            continue
        expected_hash = _run_protocol_hash(spec, protocol_hash)
        hashes: set[str] = set()
        for name in ("metrics.json", "status.json"):
            path = run_dir / name
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("protocol_hash")
            except (OSError, json.JSONDecodeError):
                value = None
            if value:
                hashes.add(str(value))
        if expected_hash in hashes:
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        prior_hash = sorted(hashes)[0][:12] if hashes else "failed_no_hash"
        destination = archive_root / f"{spec.run_key}__pre_{prior_hash}"
        suffix = 1
        while destination.exists():
            destination = archive_root / f"{spec.run_key}__pre_{prior_hash}_{suffix}"
            suffix += 1
        shutil.move(str(run_dir), str(destination))
        moved.append(spec.run_key)
    return moved


def _load_context(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[FullRunSpec], str]:
    config_path = root / "configs" / "experiments" / "full.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = build_full_matrix(config)
    _validate_confirmation(root, config)
    base_path = root / config["base_config"]
    reliability_path = root / config["reliability_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    reliability_config = yaml.safe_load(reliability_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(root, [config_path, base_path, reliability_path])
    return config, base_config, reliability_config, matrix, protocol_hash


def _worker_status_path(run_root: Path, session_id: str, worker_index: int) -> Path:
    return run_root / ".parallel_state" / session_id / f"worker_{worker_index}.json"


def _write_worker_status(
    run_root: Path,
    session_id: str,
    worker_index: int,
    *,
    status: str,
    current_task: str,
    assigned_tasks: int,
    completed_tasks: int,
    error: str | None = None,
) -> None:
    atomic_write_json(
        _worker_status_path(run_root, session_id, worker_index),
        {
            "worker_index": worker_index,
            "status": status,
            "current_task": current_task,
            "assigned_tasks": assigned_tasks,
            "completed_tasks": completed_tasks,
            "error": error,
            "updated_at_utc": utc_now(),
        },
    )


def _read_worker_statuses(run_root: Path, session_id: str) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for worker_index in range(PARALLEL_WORKERS):
        path = _worker_status_path(run_root, session_id, worker_index)
        if not path.exists():
            statuses.append(
                {
                    "worker_index": worker_index,
                    "status": "starting",
                    "current_task": "worker process starting",
                    "assigned_tasks": 105,
                    "completed_tasks": 0,
                    "error": None,
                }
            )
            continue
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return statuses


def _barrier_path(run_root: Path, session_id: str, dataset_id: str, worker_index: int) -> Path:
    return run_root / ".parallel_state" / session_id / f"{dataset_id}__worker{worker_index}.done.json"


def _memory_ready_path(run_root: Path, session_id: str, dataset_id: str, worker_index: int) -> Path:
    return run_root / ".parallel_state" / session_id / f"{dataset_id}__worker{worker_index}.memory_ready.json"


def _heavy_turn_path(run_root: Path, session_id: str, dataset_id: str) -> Path:
    return run_root / ".parallel_state" / session_id / f"{dataset_id}__worker0_heavy.turn.json"


def run_parallel_worker(
    project_root: str | Path,
    run_root: str | Path,
    protocol_hash: str,
    session_id: str,
    worker_index: int,
) -> int:
    root = Path(project_root).resolve()
    destination = Path(run_root).resolve()
    config, base_config, reliability_config, matrix, current_hash = _load_context(root)
    if current_hash != protocol_hash:
        raise ValueError("worker numerical protocol hash differs from the paused Full run")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("worker cannot attach to a run root with another numerical protocol")
    enable_multirocket_memory_patch()
    partitions = partition_full_matrix(matrix)
    assigned = partitions[worker_index]
    completed_count = sum(_completed(destination, spec, protocol_hash) for spec in assigned)
    _write_worker_status(
        destination,
        session_id,
        worker_index,
        status="running",
        current_task="checking completed runs",
        assigned_tasks=len(assigned),
        completed_tasks=completed_count,
    )
    try:
        for dataset_id in FULL_DATASETS:
            dataset_specs = [spec for spec in assigned if spec.dataset_id == dataset_id]
            pending = [spec for spec in dataset_specs if not _completed(destination, spec, protocol_hash)]
            dataset = prepare_full_dataset(root, dataset_id) if pending else None

            def execute(spec: FullRunSpec) -> None:
                nonlocal completed_count
                if _completed(destination, spec, protocol_hash):
                    return
                assert dataset is not None
                _write_worker_status(
                    destination,
                    session_id,
                    worker_index,
                    status="running",
                    current_task=spec.run_key,
                    assigned_tasks=len(assigned),
                    completed_tasks=completed_count,
                )
                print(f"[worker {worker_index}] Starting {spec.run_key}", flush=True)
                run_config = config
                if spec.method == "multirocket":
                    run_config = copy.deepcopy(config)
                    run_config["classical_baselines"]["n_kernels"] = MULTIROCKET_KERNELS
                _run_one(
                    root,
                    destination,
                    dataset,
                    spec,
                    base_config,
                    run_config,
                    reliability_config,
                    _run_protocol_hash(spec, protocol_hash),
                    True,
                )
                completed_count += 1

            for spec in [item for item in dataset_specs if item.method != "multirocket"]:
                execute(spec)

            atomic_write_json(
                _memory_ready_path(destination, session_id, dataset_id, worker_index),
                {"dataset_id": dataset_id, "worker_index": worker_index, "ready_at_utc": utc_now()},
            )
            _write_worker_status(
                destination,
                session_id,
                worker_index,
                status="waiting_for_exclusive_multirocket",
                current_task=f"memory barrier before MultiROCKET on {dataset_id}",
                assigned_tasks=len(assigned),
                completed_tasks=completed_count,
            )
            while not all(
                _memory_ready_path(destination, session_id, dataset_id, peer).exists()
                for peer in range(PARALLEL_WORKERS)
            ):
                time.sleep(0.5)

            heavy_specs = [item for item in dataset_specs if item.method == "multirocket"]
            if worker_index == 0:
                for spec in heavy_specs:
                    execute(spec)
                atomic_write_json(
                    _heavy_turn_path(destination, session_id, dataset_id),
                    {"dataset_id": dataset_id, "worker_index": 0, "completed_at_utc": utc_now()},
                )
            else:
                while not _heavy_turn_path(destination, session_id, dataset_id).exists():
                    time.sleep(0.5)
                for spec in heavy_specs:
                    execute(spec)

            del dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            atomic_write_json(
                _barrier_path(destination, session_id, dataset_id, worker_index),
                {"dataset_id": dataset_id, "worker_index": worker_index, "completed_at_utc": utc_now()},
            )
            _write_worker_status(
                destination,
                session_id,
                worker_index,
                status="waiting_at_dataset_barrier",
                current_task=f"barrier after {dataset_id}",
                assigned_tasks=len(assigned),
                completed_tasks=completed_count,
            )
            while not all(
                _barrier_path(destination, session_id, dataset_id, peer).exists()
                for peer in range(PARALLEL_WORKERS)
            ):
                time.sleep(0.5)
        _write_worker_status(
            destination,
            session_id,
            worker_index,
            status="completed",
            current_task="all assigned runs completed",
            assigned_tasks=len(assigned),
            completed_tasks=completed_count,
        )
        return 0
    except BaseException as error:
        _write_worker_status(
            destination,
            session_id,
            worker_index,
            status="failed",
            current_task="worker failed",
            assigned_tasks=len(assigned),
            completed_tasks=completed_count,
            error=f"{type(error).__name__}: {error}",
        )
        raise


def _dashboard_payload(
    root: Path,
    run_root: Path,
    matrix: list[FullRunSpec],
    protocol_hash: str,
    session_id: str,
    started: float,
    start_completed: int,
    *,
    status: str = "running",
    error: str | None = None,
    aggregate_completed: bool = False,
) -> dict[str, Any]:
    worker_statuses = _read_worker_statuses(run_root, session_id)
    active = {
        str(row.get("current_task"))
        for row in worker_statuses
        if row.get("status") == "running"
    }
    tasks: list[dict[str, str]] = [
        {"name": f"Prepare/cache {dataset_id}", "status": "completed"}
        for dataset_id in FULL_DATASETS
    ]
    completed_runs = 0
    for spec in matrix:
        if _completed(run_root, spec, protocol_hash):
            task_status = "completed"
            completed_runs += 1
        elif spec.run_key in active:
            task_status = "running"
        else:
            task_status = "pending"
        tasks.append({"name": f"Run {spec.run_key}", "status": task_status})
    tasks.append(
        {
            "name": "Aggregate paired statistics and Full report",
            "status": "completed" if aggregate_completed else "running" if completed_runs == len(matrix) else "pending",
        }
    )
    elapsed = max(0.0, time.monotonic() - started)
    newly_completed = max(0, completed_runs - start_completed)
    eta = None
    if newly_completed and completed_runs < len(matrix):
        eta = elapsed / newly_completed * (len(matrix) - completed_runs)
    current = " | ".join(
        f"W{row.get('worker_index')}: {row.get('current_task')}"
        for row in worker_statuses
        if row.get("status") not in {"completed"}
    ) or "parallel workers completed"
    completed_tasks = len(FULL_DATASETS) + completed_runs + int(aggregate_completed)
    return {
        "stage": "full_parallel",
        "run_id": run_root.name,
        "status": status,
        "current_task": current,
        "completed_tasks": completed_tasks,
        "total_tasks": 221,
        "progress_percent": 100.0 * completed_tasks / 221,
        "eta_seconds": eta,
        "elapsed_seconds": elapsed,
        "updated_at_utc": utc_now(),
        "tasks": tasks,
        "parallel_workers": worker_statuses,
        "error": error,
    }


def _stream_reader(worker_index: int, stream: Any, events: queue.Queue[tuple[int, str]]) -> None:
    for line in stream:
        events.put((worker_index, line.rstrip()))


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    else:
        process.terminate()


def run_full_parallel(
    project_root: str | Path,
    *,
    resume: bool = True,
    confirmed: bool = False,
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("parallel Full requires a separate explicit manual confirmation")
    if not resume:
        raise ValueError("parallel Full is a resume-only scheduler for an existing frozen run")
    root = Path(project_root).resolve()
    config, _base, _reliability, matrix, protocol_hash = _load_context(root)
    run_root = _find_resume_root(root / "runs" / "full", protocol_hash)
    if run_root is None:
        raise FileNotFoundError("no paused Full run with the frozen numerical protocol was found")
    completed_report = run_root / "full_report.json"
    if completed_report.exists():
        report = json.loads(completed_report.read_text(encoding="utf-8"))
        if report.get("status") == "completed":
            return report
    missing_caches = [
        dataset_id for dataset_id in FULL_DATASETS if not full_cache_path(root, dataset_id).exists()
    ]
    if missing_caches:
        raise FileNotFoundError(f"parallel resume requires all ten prepared caches; missing {missing_caches}")

    amendment_path = run_root / "protocol_amendment_multirocket_memory.json"
    prior_superseded: list[str] = []
    if amendment_path.exists():
        try:
            prior_superseded = list(
                json.loads(amendment_path.read_text(encoding="utf-8")).get(
                    "superseded_run_keys", []
                )
            )
        except (OSError, json.JSONDecodeError, TypeError):
            prior_superseded = []
    newly_superseded = _archive_incompatible_multirocket_runs(
        run_root, matrix, protocol_hash
    )
    superseded_multirocket_runs = sorted(set(prior_superseded + newly_superseded))
    amendment_hash = _run_protocol_hash(
        next(spec for spec in matrix if spec.method == "multirocket"), protocol_hash
    )
    atomic_write_json(
        amendment_path,
        {
            "amendment_id": MULTIROCKET_AMENDMENT_ID,
            "reason": (
                "2.5k MultiROCKET exceeded the 16 GiB host RAM on Sleep-EDF: "
                "19,488 transformed features caused an 8.49 GiB RidgeCV work array"
            ),
            "scope": "all MultiROCKET runs only",
            "original_n_kernels": int(config["classical_baselines"]["n_kernels"]),
            "amended_n_kernels": MULTIROCKET_KERNELS,
            "amended_feature_count_approx": 8_064,
            "observed_sleep_edfx_feature_count": 7_392,
            "estimated_sleep_edfx_ridge_work_array_gib": 1.22,
            "base_protocol_hash": protocol_hash,
            "multirocket_protocol_hash": amendment_hash,
            "other_methods_unchanged_and_reused": True,
            "old_multirocket_results_used_in_aggregate": False,
            "superseded_run_keys": superseded_multirocket_runs,
            "recorded_at_utc": utc_now(),
        },
    )
    started = time.monotonic()
    start_completed = sum(_completed(run_root, spec, protocol_hash) for spec in matrix)
    prior_elapsed = 0.0
    dashboard_path = root / "runs" / "dashboard_status.json"
    if dashboard_path.exists():
        try:
            previous = json.loads(dashboard_path.read_text(encoding="utf-8"))
            if previous.get("run_id") == run_root.name:
                prior_elapsed = float(previous.get("elapsed_seconds", 0.0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    session_id = datetime.now(timezone.utc).strftime("parallel_%Y%m%dT%H%M%SZ")
    session_root = run_root / ".parallel_state" / session_id
    session_root.mkdir(parents=True, exist_ok=False)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        status="parallel_running",
        orchestration="two_worker_single_gpu_disjoint_matrix_indices",
        parallel_session_id=session_id,
        parallel_start_completed_runs=start_completed,
        numerical_resource_amendment=MULTIROCKET_AMENDMENT_ID,
        multirocket_n_kernels=MULTIROCKET_KERNELS,
        multirocket_protocol_hash=amendment_hash,
        superseded_multirocket_runs=superseded_multirocket_runs,
        updated_at_utc=utc_now(),
    )
    atomic_write_json(manifest_path, manifest)
    print(
        f"Parallel Full attaches to {run_root.name}; reusing {start_completed}/210 completed runs",
        flush=True,
    )

    processes: list[subprocess.Popen[str]] = []
    events: queue.Queue[tuple[int, str]] = queue.Queue()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    for worker_index in range(PARALLEL_WORKERS):
        command = [
            sys.executable,
            "-m",
            "nyquistguard.experiments.full_parallel",
            "--worker-index",
            str(worker_index),
            "--project-root",
            str(root),
            "--run-root",
            str(run_root),
            "--protocol-hash",
            protocol_hash,
            "--session-id",
            session_id,
        ]
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
        )
        assert process.stdout is not None
        processes.append(process)
        threading.Thread(
            target=_stream_reader,
            args=(worker_index, process.stdout, events),
            daemon=True,
        ).start()

    try:
        while any(process.poll() is None for process in processes):
            try:
                while True:
                    worker_index, line = events.get_nowait()
                    print(f"[W{worker_index}] {line}", flush=True)
            except queue.Empty:
                pass
            atomic_write_json(
                dashboard_path,
                _dashboard_payload(
                    root,
                    run_root,
                    matrix,
                    protocol_hash,
                    session_id,
                    started,
                    start_completed,
                ),
            )
            failed = [process for process in processes if process.poll() not in {None, 0}]
            if failed:
                raise RuntimeError(
                    "parallel worker failed with exit code(s) "
                    + ",".join(str(process.returncode) for process in failed)
                )
            time.sleep(0.5)
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError(f"parallel worker exited with code {process.returncode}")
        try:
            while True:
                worker_index, line = events.get_nowait()
                print(f"[W{worker_index}] {line}", flush=True)
        except queue.Empty:
            pass

        missing = [spec.run_key for spec in matrix if not _completed(run_root, spec, protocol_hash)]
        if missing:
            raise RuntimeError(f"parallel workers ended with {len(missing)} incomplete Full runs")
        atomic_write_json(
            dashboard_path,
            _dashboard_payload(
                root,
                run_root,
                matrix,
                protocol_hash,
                session_id,
                started,
                start_completed,
                status="aggregating",
            ),
        )
        aggregate = _aggregate(run_root, matrix, config)
        parallel_elapsed = time.monotonic() - started
        report: dict[str, Any] = {
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
            "elapsed_seconds": prior_elapsed + parallel_elapsed,
            "prior_sequential_elapsed_seconds": prior_elapsed,
            "parallel_elapsed_seconds": parallel_elapsed,
            "orchestration": "two_worker_single_gpu_disjoint_matrix_indices_with_dataset_barriers",
            "scientific_metric_protocol_unchanged": False,
            "main_model_and_non_multirocket_protocol_unchanged": True,
            "numerical_resource_amendment": MULTIROCKET_AMENDMENT_ID,
            "multirocket_n_kernels": MULTIROCKET_KERNELS,
            "multirocket_protocol_hash": amendment_hash,
            "superseded_multirocket_runs": superseded_multirocket_runs,
            "concurrent_wall_time_valid_for_paper_efficiency_claims": False,
            "run_root": str(run_root),
            "automatic_followup_started": False,
            "finished_at_utc": utc_now(),
        }
        _write_aggregate_csv(run_root / "full_results.csv", aggregate["rows"])
        _write_aggregate_csv(root / "reports" / "full_results.csv", aggregate["rows"])
        report["figure_paths"] = _generate_figures(report, [run_root, root / "reports"])
        note = (
            "\n## Parallel orchestration note\n\n"
            "- Remaining runs were scheduled on two disjoint workers sharing one GPU; the union is the unchanged frozen 210-run matrix.\n"
            "- The main model, seeds, data, splits, rates and evaluation metrics were unchanged.\n"
            f"- Resource amendment: all MultiROCKET runs use {MULTIROCKET_KERNELS:,} kernels (about 8,064 features); the earlier 10,000- and 2,500-kernel settings exceeded the 16 GiB host RAM, and all superseded artifacts were excluded.\n"
            "- MultiROCKET runs execute exclusively while the peer worker waits, and aeon removes only redundant same-value array copies.\n"
            "- Per-run concurrent wall times are resource-contention measurements and must not be used for paper efficiency comparisons; use a separate sequential timing benchmark.\n"
        )
        markdown = _report_markdown(report) + note
        atomic_write_json(run_root / "full_report.json", report)
        _atomic_write_text(run_root / "full_report.md", markdown)
        atomic_write_json(root / "reports" / "full_report.json", report)
        _atomic_write_text(root / "reports" / "full_report.md", markdown)
        manifest.update(
            status="completed",
            parallel_session_id=session_id,
            concurrent_wall_time_valid_for_paper_efficiency_claims=False,
            updated_at_utc=utc_now(),
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            dashboard_path,
            _dashboard_payload(
                root,
                run_root,
                matrix,
                protocol_hash,
                session_id,
                started,
                start_completed,
                status="completed",
                aggregate_completed=True,
            ),
        )
        return report
    except BaseException as error:
        for process in processes:
            _terminate_process(process)
        message = f"{type(error).__name__}: {error}"
        manifest.update(status="parallel_failed", error=message, updated_at_utc=utc_now())
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            dashboard_path,
            _dashboard_payload(
                root,
                run_root,
                matrix,
                protocol_hash,
                session_id,
                started,
                start_completed,
                status="failed",
                error=message,
            ),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="NyquistGuard Full parallel worker")
    parser.add_argument("--worker-index", required=True, type=int, choices=(0, 1))
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--protocol-hash", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    return run_parallel_worker(
        args.project_root,
        args.run_root,
        args.protocol_hash,
        args.session_id,
        args.worker_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
