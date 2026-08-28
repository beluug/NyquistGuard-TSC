"""Atomic dashboard and run-status progress reporting."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    delay_seconds = 0.02
    try:
        for attempt in range(8):
            try:
                temporary.write_text(content, encoding="utf-8")
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(delay_seconds)
                delay_seconds = min(0.32, delay_seconds * 2.0)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class DashboardProgress:
    def __init__(self, path: str | Path, stage: str, task_names: list[str], run_id: str) -> None:
        self.path = Path(path)
        self.stage = stage
        self.run_id = run_id
        self.started = time.monotonic()
        self.tasks = [{"name": name, "status": "pending"} for name in task_names]
        self.status = "queued"
        self.current_task = "等待开始"
        self.error: str | None = None
        self.write()

    @property
    def completed_count(self) -> int:
        return sum(task["status"] == "completed" for task in self.tasks)

    def start_task(self, index: int) -> None:
        self.status = "running"
        self.tasks[index]["status"] = "running"
        self.current_task = self.tasks[index]["name"]
        self.write()

    def complete_task(self, index: int) -> None:
        self.tasks[index]["status"] = "completed"
        self.write()

    def fail_task(self, index: int, error: BaseException) -> None:
        self.tasks[index]["status"] = "failed"
        self.status = "failed"
        self.current_task = self.tasks[index]["name"]
        self.error = f"{type(error).__name__}: {error}"
        self.write()

    def finish(self, message: str | None = None) -> None:
        self.status = "completed"
        self.current_task = message or f"{self.stage.capitalize()} 已完成"
        self.write()

    def write(self) -> None:
        total = len(self.tasks)
        completed = self.completed_count
        elapsed = max(0.0, time.monotonic() - self.started)
        eta = None
        if completed and completed < total:
            eta = elapsed / completed * (total - completed)
        payload = {
            "stage": self.stage,
            "run_id": self.run_id,
            "status": self.status,
            "current_task": self.current_task,
            "completed_tasks": completed,
            "total_tasks": total,
            "progress_percent": 100.0 * completed / total if total else 0.0,
            "eta_seconds": eta,
            "elapsed_seconds": elapsed,
            "updated_at_utc": utc_now(),
            "tasks": self.tasks,
            "error": self.error,
        }
        atomic_write_json(self.path, payload)
