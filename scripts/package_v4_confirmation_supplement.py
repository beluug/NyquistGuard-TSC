"""Package frozen V4.1 code/results without raw data, arrays, or checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyquistguard.analysis.v4_confirmation_artifacts import package_confirmation_supplement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(package_confirmation_supplement(root, force=args.force), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
