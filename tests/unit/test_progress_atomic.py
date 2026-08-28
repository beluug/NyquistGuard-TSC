from __future__ import annotations

import json
import os
from pathlib import Path

from nyquistguard.experiments.progress import atomic_write_json


def test_atomic_write_retries_transient_windows_permission_error(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "dashboard_status.json"
    real_replace = os.replace
    calls = 0

    def flaky_replace(source, target):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(5, "transient file lock")
        return real_replace(source, target)

    monkeypatch.setattr("nyquistguard.experiments.progress.os.replace", flaky_replace)
    atomic_write_json(destination, {"status": "running", "completed": 162})
    assert calls == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "running",
        "completed": 162,
    }
    assert not list(tmp_path.glob("*.tmp"))

