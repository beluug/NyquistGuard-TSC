"""Strict reader for equal-length, non-timestamped UEA ``.ts`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TimeSeriesCollection:
    """A classification collection using the canonical [case, channel, time] layout."""

    x: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    split: str
    metadata: dict[str, str]
    source_path: Path

    def __post_init__(self) -> None:
        if self.x.ndim != 3:
            raise ValueError("x must have shape [case, channel, time]")
        if len(self.y) != len(self.x) or len(self.sample_ids) != len(self.x):
            raise ValueError("x, y and sample_ids must contain the same number of cases")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")


def _parse_header(line: str) -> tuple[str, str]:
    body = line[1:].strip()
    key, separator, value = body.partition(" ")
    return key.lower(), value.strip() if separator else ""


def load_uea_ts(path: str | Path, *, split: str) -> TimeSeriesCollection:
    """Load a UEA classification file without silently accepting ragged/timestamp data."""

    source = Path(path).resolve()
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if not source.is_file():
        raise FileNotFoundError(source)

    metadata: dict[str, str] = {}
    rows: list[list[np.ndarray]] = []
    labels: list[str] = []
    in_data = False
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data and line.startswith("@"):
            key, value = _parse_header(line)
            if key == "data":
                in_data = True
            else:
                metadata[key] = value
            continue
        if not in_data:
            raise ValueError(f"unexpected content before @data at {source}:{line_number}")

        fields = line.split(":")
        if len(fields) < 2:
            raise ValueError(f"missing class label at {source}:{line_number}")
        label = fields[-1].strip()
        channels: list[np.ndarray] = []
        for field in fields[:-1]:
            values = np.fromstring(field.replace("?", "nan"), sep=",", dtype=np.float32)
            if values.size == 0:
                raise ValueError(f"empty channel at {source}:{line_number}")
            channels.append(values)
        rows.append(channels)
        labels.append(label)

    if not in_data or not rows:
        raise ValueError(f"no @data rows found in {source}")
    if metadata.get("timestamps", "false").lower() != "false":
        raise ValueError("this reader intentionally supports only non-timestamped UEA files")
    if metadata.get("equallength", "true").lower() != "true":
        raise ValueError("this reader intentionally supports only equal-length UEA files")

    expected_channels = int(metadata.get("dimensions", len(rows[0])))
    expected_length = int(metadata.get("serieslength", len(rows[0][0])))
    for row_index, channels in enumerate(rows):
        if len(channels) != expected_channels:
            raise ValueError(f"row {row_index} has {len(channels)} channels, expected {expected_channels}")
        if any(channel.size != expected_length for channel in channels):
            raise ValueError(f"row {row_index} does not have series length {expected_length}")

    x = np.stack([np.stack(channels, axis=0) for channels in rows], axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=str)
    class_field = metadata.get("classlabel", "")
    class_parts = class_field.split()
    class_names = tuple(class_parts[1:]) if class_parts and class_parts[0].lower() == "true" else tuple(sorted(set(labels)))
    unknown = sorted(set(labels).difference(class_names))
    if unknown:
        raise ValueError(f"data contains labels absent from @classLabel: {unknown}")

    problem_name = metadata.get("problemname", source.stem)
    sample_ids = tuple(f"{problem_name}:{split}:{index:05d}" for index in range(len(x)))
    return TimeSeriesCollection(x, y, sample_ids, class_names, split, metadata, source)


@dataclass(frozen=True)
class ChannelStandardizer:
    """Per-channel z-normalizer whose provenance records the fitting split."""

    mean: np.ndarray
    scale: np.ndarray
    fitted_split: str

    @classmethod
    def fit(cls, collection: TimeSeriesCollection, *, epsilon: float = 1e-6) -> "ChannelStandardizer":
        if collection.split != "train":
            raise ValueError("the standardizer may only be fitted on the training split")
        mean = np.nanmean(collection.x, axis=(0, 2), keepdims=True).astype(np.float32)
        scale = np.nanstd(collection.x, axis=(0, 2), keepdims=True).astype(np.float32)
        scale = np.maximum(scale, np.float32(epsilon))
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("training statistics are not finite")
        return cls(mean=mean, scale=scale, fitted_split=collection.split)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape[1] != self.mean.shape[1]:
            raise ValueError("x must have shape [case, fitted_channels, time]")
        transformed = (x.astype(np.float32, copy=False) - self.mean) / self.scale
        return transformed.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "per_channel_zscore",
            "fitted_split": self.fitted_split,
            "mean": self.mean.reshape(-1).tolist(),
            "scale": self.scale.reshape(-1).tolist(),
        }
