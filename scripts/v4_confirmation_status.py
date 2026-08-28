"""Compact live status plus train/validation-only health summary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nyquistguard.analysis.v4_confirmation_artifacts import (
    audit_confirmation,
    summarize_training_health,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dashboard_path = root / "runs" / "dashboard_status.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    audit = audit_confirmation(root, require_complete=False)
    health = summarize_training_health(root)
    histories = Counter(row["dataset"] for row in health)
    flags = Counter(row["dataset"] for row in health if row["constant_prediction_pattern"])
    print(
        f"{dashboard.get('status')} | {dashboard.get('completed_tasks')}/{dashboard.get('total_tasks')} tasks | "
        f"{audit['completed_metrics']}/{audit['expected_metrics']} metrics"
    )
    print(f"Current: {dashboard.get('current_task')}")
    eta = dashboard.get("eta_seconds")
    print("Dashboard ETA: unavailable" if eta is None else f"Dashboard ETA: {float(eta) / 60:.1f} minutes")
    for dataset in ("CharacterTrajectories", "MotorImagery", "WISDM", "PTB-XL"):
        print(f"{dataset}: histories={histories[dataset]}, constant-pattern flags={flags[dataset]}")
    print("Status/health output uses dashboard, file presence, training loss, and validation only; no test metrics are printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
