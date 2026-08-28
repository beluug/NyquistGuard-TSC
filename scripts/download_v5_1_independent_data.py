"""Download the frozen V5.1 UEA panel without parsing any data file."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "SelfRegulationSCP1": 11206265,
    "HandMovementDirection": 11206224,
    "RacketSports": 11206263,
    "Heartbeat": 11206229,
}


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    print(f"Downloading {destination.name}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    args = parser.parse_args()
    splits = ("TRAIN",) if args.split == "train" else ("TEST",)
    if args.split == "all":
        splits = ("TRAIN", "TEST")
    raw_root = ROOT / "data" / "raw" / "uea_v5_1_independent"
    for name, record in DATASETS.items():
        for split in splits:
            filename = f"{name}_{split}.ts"
            url = f"https://zenodo.org/records/{record}/files/{filename}?download=1"
            _download(url, raw_root / name / filename)
    print("Download complete. No .ts file was parsed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
