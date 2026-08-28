"""Read-only completeness/protocol audit for the V4.1 confirmation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyquistguard.analysis.v4_confirmation_artifacts import audit_confirmation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="allow a currently running incomplete matrix")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit_confirmation(root, require_complete=not args.partial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
