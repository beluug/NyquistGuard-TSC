"""Write internal train/validation-only health flags without reading test metrics."""

from __future__ import annotations

import csv
from pathlib import Path

from nyquistguard.analysis.v4_confirmation_artifacts import summarize_training_health


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    rows = summarize_training_health(root)
    if not rows:
        print("No completed training histories are available yet.")
        return 1
    output = root / "reports" / "v4_confirmation_training_health.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(bool(row["constant_prediction_pattern"]) for row in rows)
    print(f"Wrote {output}; histories={len(rows)}, constant-pattern flags={flagged}")
    print("This is internal train/validation QC only; no test metrics were read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
