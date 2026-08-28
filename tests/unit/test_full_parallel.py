from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import yaml

from nyquistguard.experiments.full import build_full_matrix
from nyquistguard.experiments.full_parallel import (
    MULTIROCKET_KERNELS,
    _archive_incompatible_multirocket_runs,
    _completed,
    _dashboard_payload,
    _load_context,
    _run_protocol_hash,
    _write_worker_status,
    partition_full_matrix,
)
from nyquistguard.experiments.progress import atomic_write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_HASH = "test-protocol"


def _matrix():
    config = yaml.safe_load((ROOT / "configs/experiments/full.yaml").read_text(encoding="utf-8"))
    return build_full_matrix(config)


def test_two_worker_partitions_are_disjoint_balanced_and_complete() -> None:
    matrix = _matrix()
    first, second = partition_full_matrix(matrix)
    assert len(first) == 105
    assert len(second) == 105
    assert not ({spec.run_key for spec in first} & {spec.run_key for spec in second})
    assert {spec.run_key for spec in first + second} == {spec.run_key for spec in matrix}
    for dataset_id in {spec.dataset_id for spec in matrix}:
        counts = (
            sum(spec.dataset_id == dataset_id for spec in first),
            sum(spec.dataset_id == dataset_id for spec in second),
        )
        assert sorted(counts) == [10, 11]


def test_completed_requires_matching_numerical_protocol(tmp_path: Path) -> None:
    spec = _matrix()[0]
    path = tmp_path / spec.run_key / "metrics.json"
    atomic_write_json(path, {"status": "completed", "protocol_hash": "different"})
    assert not _completed(tmp_path, spec, PROTOCOL_HASH)
    atomic_write_json(path, {"status": "completed", "protocol_hash": PROTOCOL_HASH})
    assert _completed(tmp_path, spec, PROTOCOL_HASH)

    multirocket = next(item for item in _matrix() if item.method == "multirocket")
    amended = _run_protocol_hash(multirocket, PROTOCOL_HASH)
    assert amended != PROTOCOL_HASH
    atomic_write_json(
        tmp_path / multirocket.run_key / "metrics.json",
        {"status": "completed", "protocol_hash": PROTOCOL_HASH},
    )
    assert not _completed(tmp_path, multirocket, PROTOCOL_HASH)
    atomic_write_json(
        tmp_path / multirocket.run_key / "metrics.json",
        {"status": "completed", "protocol_hash": amended},
    )
    assert _completed(tmp_path, multirocket, PROTOCOL_HASH)


def test_old_multirocket_artifact_is_archived_but_amended_resume_is_kept(tmp_path: Path) -> None:
    specs = [item for item in _matrix() if item.method == "multirocket"][:2]
    atomic_write_json(
        tmp_path / specs[0].run_key / "status.json",
        {"status": "failed", "protocol_hash": PROTOCOL_HASH},
    )
    amended = _run_protocol_hash(specs[1], PROTOCOL_HASH)
    atomic_write_json(
        tmp_path / specs[1].run_key / "status.json",
        {"status": "running", "protocol_hash": amended},
    )
    moved = _archive_incompatible_multirocket_runs(tmp_path, specs, PROTOCOL_HASH)
    assert moved == [specs[0].run_key]
    assert not (tmp_path / specs[0].run_key).exists()
    assert list((tmp_path / "superseded_resource_amendment").iterdir())
    assert (tmp_path / specs[1].run_key).exists()
    assert MULTIROCKET_KERNELS == 1_000


def test_dashboard_merges_two_active_workers_without_duplicate_tasks(tmp_path: Path) -> None:
    matrix = _matrix()
    for spec in matrix[:2]:
        atomic_write_json(
            tmp_path / spec.run_key / "metrics.json",
            {"status": "completed", "protocol_hash": PROTOCOL_HASH},
        )
    session = "test_session"
    _write_worker_status(
        tmp_path,
        session,
        0,
        status="running",
        current_task=matrix[2].run_key,
        assigned_tasks=105,
        completed_tasks=1,
    )
    _write_worker_status(
        tmp_path,
        session,
        1,
        status="running",
        current_task=matrix[3].run_key,
        assigned_tasks=105,
        completed_tasks=1,
    )
    payload = _dashboard_payload(
        ROOT,
        tmp_path,
        matrix,
        PROTOCOL_HASH,
        session,
        0.0,
        2,
    )
    assert payload["stage"] == "full_parallel"
    assert payload["completed_tasks"] == 12
    assert payload["total_tasks"] == 221
    assert len(payload["tasks"]) == 221
    assert sum(task["status"] == "running" for task in payload["tasks"]) == 2
    assert "W0:" in payload["current_task"] and "W1:" in payload["current_task"]


def test_two_worker_process_barriers_finish_on_completed_fixture(tmp_path: Path) -> None:
    _config, _base, _reliability, matrix, protocol_hash = _load_context(ROOT)
    atomic_write_json(
        tmp_path / "manifest.json",
        {"status": "running", "protocol_hash": protocol_hash, "run_root": str(tmp_path)},
    )
    for spec in matrix:
        atomic_write_json(
            tmp_path / spec.run_key / "metrics.json",
            {"status": "completed", "protocol_hash": _run_protocol_hash(spec, protocol_hash)},
        )
    session = "process_barrier_fixture"
    processes = []
    for worker_index in (0, 1):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "nyquistguard.experiments.full_parallel",
                    "--worker-index",
                    str(worker_index),
                    "--project-root",
                    str(ROOT),
                    "--run-root",
                    str(tmp_path),
                    "--protocol-hash",
                    protocol_hash,
                    "--session-id",
                    session,
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    outputs = [process.communicate(timeout=30) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs
    statuses = [
        json.loads((tmp_path / ".parallel_state" / session / f"worker_{index}.json").read_text(encoding="utf-8"))
        for index in (0, 1)
    ]
    assert [status["status"] for status in statuses] == ["completed", "completed"]
    assert len(list((tmp_path / ".parallel_state" / session).glob("*.done.json"))) == 20
