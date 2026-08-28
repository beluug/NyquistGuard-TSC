"""Build frozen post-run CSV, statistics, figures, and manuscript-neutral text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyquistguard.analysis.v4_confirmation_artifacts import build_confirmation_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace previously generated post-run artifacts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_confirmation_artifacts(root, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
