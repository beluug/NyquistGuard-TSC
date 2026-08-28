"""One-command post-run audit, artifact generation, and supplement packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyquistguard.analysis.v4_confirmation_artifacts import (
    audit_confirmation,
    build_confirmation_artifacts,
    package_confirmation_supplement,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing generated outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = {
        "audit": audit_confirmation(root, require_complete=True),
        "artifacts": build_confirmation_artifacts(root, force=args.force),
        "supplement": package_confirmation_supplement(root, force=args.force),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
