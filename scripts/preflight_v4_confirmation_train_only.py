"""Small train/validation-only parser preflight; never enables formal test access."""

from __future__ import annotations

import argparse
from pathlib import Path

from nyquistguard.data.new_confirmation_datasets import (
    CONFIRMATION_DATASETS,
    prepare_confirmation_development_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIRMATION_DATASETS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = prepare_confirmation_development_dataset(root, args.dataset, force=args.force)
    if hasattr(dataset, "test") or dataset.metadata.get("test_accessed") is not False:
        raise RuntimeError("train-only preflight crossed the frozen test boundary")
    print(
        f"PASS {dataset.dataset_id}: train={dataset.train.x.shape}, "
        f"validation={dataset.validation.x.shape}, classes={len(dataset.class_names)}, test_accessed=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
