"""Leakage-safe preparation of the four frozen pilot datasets.

All returned collections use the aeon-compatible ``[case, channel, time]``
layout.  Continuous UCI recordings are converted to non-overlapping-label
windows before any split statistics are fitted.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .uea_ts import load_uea_ts


PILOT_DATASETS = ("basicmotions_uea", "epilepsy_uea", "pamap2_uci", "mhealth_uci")


@dataclass(frozen=True)
class SplitData:
    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    sampling_rate_hz: float
    class_names: tuple[str, ...]
    train: SplitData
    validation: SplitData
    test: SplitData
    metadata: dict[str, object]


def _stratified_validation_indices(labels: np.ndarray, seed: int, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        count = min(count, len(indices) - 1) if len(indices) > 1 else 0
        validation_parts.append(indices[:count])
        train_parts.append(indices[count:])
    train = np.sort(np.concatenate(train_parts))
    validation = np.sort(np.concatenate(validation_parts))
    return train, validation


def _encode_labels(label_groups: Iterable[np.ndarray]) -> tuple[list[np.ndarray], tuple[str, ...]]:
    groups = [np.asarray(group).astype(str) for group in label_groups]
    class_names = tuple(sorted({value for group in groups for value in group.tolist()}))
    mapping = {name: index for index, name in enumerate(class_names)}
    return [np.asarray([mapping[value] for value in group], dtype=np.int64) for group in groups], class_names


def _standardize(train: SplitData, validation: SplitData, test: SplitData) -> tuple[SplitData, SplitData, SplitData]:
    mean = np.nanmean(train.x, axis=(0, 2), keepdims=True).astype(np.float32)
    scale = np.nanstd(train.x, axis=(0, 2), keepdims=True).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-6))

    def apply(split: SplitData) -> SplitData:
        values = (split.x.astype(np.float32, copy=False) - mean) / scale
        return SplitData(values.astype(np.float32, copy=False), split.y, split.ids)

    return apply(train), apply(validation), apply(test)


def _require_closed_set(train: SplitData, validation: SplitData, test: SplitData, class_names: tuple[str, ...]) -> None:
    train_labels = set(np.unique(train.y).tolist())
    expected = set(range(len(class_names)))
    if train_labels != expected:
        missing = sorted(expected - train_labels)
        raise ValueError(f"training split does not cover all configured classes: {missing}")
    for name, split in (("validation", validation), ("test", test)):
        unseen = sorted(set(np.unique(split.y).tolist()) - train_labels)
        if unseen:
            raise ValueError(f"{name} contains labels absent from training: {unseen}")


def _prepare_uea(root: Path, dataset_id: str, sampling_rate_hz: float, seed: int) -> PreparedDataset:
    prefix = "BasicMotions" if dataset_id == "basicmotions_uea" else "Epilepsy"
    train_collection = load_uea_ts(root / f"{prefix}_TRAIN.ts", split="train")
    test_collection = load_uea_ts(root / f"{prefix}_TEST.ts", split="test")
    encoded, class_names = _encode_labels([train_collection.y, test_collection.y])
    train_indices, validation_indices = _stratified_validation_indices(encoded[0], seed)
    all_train_ids = np.asarray([f"official_train_{index:04d}" for index in range(len(encoded[0]))])
    test_ids = np.asarray([f"official_test_{index:04d}" for index in range(len(encoded[1]))])
    train = SplitData(train_collection.x[train_indices], encoded[0][train_indices], all_train_ids[train_indices])
    validation = SplitData(
        train_collection.x[validation_indices], encoded[0][validation_indices], all_train_ids[validation_indices]
    )
    test = SplitData(test_collection.x, encoded[1], test_ids)
    train, validation, test = _standardize(train, validation, test)
    return PreparedDataset(
        dataset_id,
        sampling_rate_hz,
        class_names,
        train,
        validation,
        test,
        {
            "source_format": "UEA .ts",
            "split_protocol": "official train/test plus deterministic stratified 20% validation from train",
            "validation_seed": seed,
        },
    )


def _interpolate_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    positions = np.arange(result.shape[0])
    for column in range(result.shape[1]):
        finite = np.isfinite(result[:, column])
        if not finite.any():
            result[:, column] = 0.0
        elif not finite.all():
            result[:, column] = np.interp(positions, positions[finite], result[finite, column])
    return result


def _homogeneous_windows(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    window: int,
    stride: int,
    source_id: str,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    ids: list[str] = []
    for start in range(0, max(0, len(labels) - window + 1), stride):
        label_window = labels[start : start + window]
        label = int(label_window[0])
        if label <= 0 or np.any(label_window != label):
            continue
        sample = values[start : start + window]
        if not np.isfinite(sample).all():
            continue
        windows.append(sample.T.astype(np.float32, copy=False))
        window_labels.append(label)
        ids.append(f"{source_id}:{start}:{start + window}")
    return windows, window_labels, ids


def _subject_number(path: Path) -> int:
    match = re.search(r"subject(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot infer subject number from {path}")
    return int(match.group(1))


def _assemble_continuous(
    records: list[tuple[np.ndarray, int, str, int]],
    *,
    sampling_rate_hz: float,
    window_seconds: float,
    stride_seconds: float,
    split_subjects: dict[str, set[int]],
) -> tuple[SplitData, SplitData, SplitData]:
    window = int(round(window_seconds * sampling_rate_hz))
    stride = int(round(stride_seconds * sampling_rate_hz))
    buckets: dict[str, tuple[list[np.ndarray], list[int], list[str]]] = {
        name: ([], [], []) for name in split_subjects
    }
    for values, subject, source_id, label_column in records:
        split = next((name for name, subjects in split_subjects.items() if subject in subjects), None)
        if split is None:
            continue
        labels = values[:, label_column].astype(np.int64)
        features = np.delete(values, label_column, axis=1)
        x_items, y_items, ids = _homogeneous_windows(
            features, labels, window=window, stride=stride, source_id=source_id
        )
        buckets[split][0].extend(x_items)
        buckets[split][1].extend(y_items)
        buckets[split][2].extend(ids)

    output: list[SplitData] = []
    for name in ("train", "validation", "test"):
        x_items, y_items, ids = buckets[name]
        if not x_items:
            raise ValueError(f"continuous preparation produced no {name} windows")
        output.append(
            SplitData(
                np.stack(x_items).astype(np.float32),
                np.asarray(y_items, dtype=np.int64),
                np.asarray(ids),
            )
        )
    return output[0], output[1], output[2]


def _prepare_pamap2(root: Path) -> PreparedDataset:
    # Per each 17-column IMU block: discard temperature and orientation,
    # retain two accelerometers, gyroscope and magnetometer (12 x 3 = 36).
    selected = list(range(4, 16)) + list(range(21, 33)) + list(range(38, 50))
    records: list[tuple[np.ndarray, int, str, int]] = []
    # The main PAMAP2 benchmark uses the 12 Protocol activities. Optional
    # activities are not consistently observed across subject splits (activity
    # 20, for example, occurs only in the former test subjects), so mixing the
    # two folders creates an invalid closed-set classification task.
    files = sorted(
        path
        for path in root.rglob("subject*.dat")
        if path.is_file() and path.parent.name.casefold() == "protocol"
    )
    if not files:
        raise FileNotFoundError(f"no PAMAP2 subject .dat files found below {root}")
    for path in files:
        raw = np.loadtxt(path, dtype=np.float32)
        features = _interpolate_columns(raw[:, selected])
        labels = raw[:, 1:2]
        source_id = f"{path.parent.name}/{path.stem}"
        records.append((np.concatenate([features, labels], axis=1), _subject_number(path), source_id, 36))
    train, validation, test = _assemble_continuous(
        records,
        sampling_rate_hz=100.0,
        window_seconds=5.0,
        stride_seconds=2.5,
        split_subjects={"train": {101, 102, 103, 104, 105}, "validation": {106, 107}, "test": {108, 109}},
    )
    raw_labels = sorted(set(np.concatenate([train.y, validation.y, test.y]).tolist()))
    label_mapping = {label: index for index, label in enumerate(raw_labels)}
    class_names = tuple(str(label) for label in raw_labels)

    def remap(split: SplitData) -> SplitData:
        return SplitData(split.x, np.asarray([label_mapping[int(y)] for y in split.y], dtype=np.int64), split.ids)

    train, validation, test = remap(train), remap(validation), remap(test)
    _require_closed_set(train, validation, test, class_names)
    train, validation, test = _standardize(train, validation, test)
    return PreparedDataset(
        "pamap2_uci", 100.0, class_names, train, validation, test,
        {
            "source_format": "PAMAP2 Protocol .dat only",
            "dataset_protocol_id": "pamap2_protocol_12class_v2",
            "activity_ids": raw_labels,
            "optional_activity_files_excluded": True,
            "channels": "36 motion channels; HR, temperature and orientation excluded",
            "window_seconds": 5.0,
            "stride_seconds": 2.5,
            "split_protocol": "subjects 101-105 train, 106-107 validation, 108-109 test",
        },
    )


def _prepare_mhealth(root: Path) -> PreparedDataset:
    records: list[tuple[np.ndarray, int, str, int]] = []
    files = sorted(root.rglob("mHealth_subject*.log"), key=_subject_number)
    if not files:
        raise FileNotFoundError(f"no MHEALTH subject logs found below {root}")
    motion_columns = list(range(0, 3)) + list(range(5, 23))
    for path in files:
        raw = np.loadtxt(path, dtype=np.float32)
        features = _interpolate_columns(raw[:, motion_columns])
        labels = raw[:, 23:24]
        records.append((np.concatenate([features, labels], axis=1), _subject_number(path), path.stem, 21))
    train, validation, test = _assemble_continuous(
        records,
        sampling_rate_hz=50.0,
        window_seconds=5.0,
        stride_seconds=2.5,
        split_subjects={"train": set(range(1, 7)), "validation": {7, 8}, "test": {9, 10}},
    )
    raw_labels = sorted(set(np.concatenate([train.y, validation.y, test.y]).tolist()))
    label_mapping = {label: index for index, label in enumerate(raw_labels)}
    class_names = tuple(str(label) for label in raw_labels)

    def remap(split: SplitData) -> SplitData:
        return SplitData(split.x, np.asarray([label_mapping[int(y)] for y in split.y], dtype=np.int64), split.ids)

    train, validation, test = remap(train), remap(validation), remap(test)
    _require_closed_set(train, validation, test, class_names)
    train, validation, test = _standardize(train, validation, test)
    return PreparedDataset(
        "mhealth_uci", 50.0, class_names, train, validation, test,
        {
            "source_format": "MHEALTH subject .log",
            "channels": "21 motion channels; two ECG channels excluded",
            "window_seconds": 5.0,
            "stride_seconds": 2.5,
            "split_protocol": "subjects 1-6 train, 7-8 validation, 9-10 test",
        },
    )


def _cache_path(project_root: Path, dataset_id: str) -> Path:
    return project_root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"


def save_prepared_dataset(path: Path, dataset: PreparedDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload: dict[str, object] = {
        "dataset_id": np.asarray(dataset.dataset_id),
        "sampling_rate_hz": np.asarray(dataset.sampling_rate_hz, dtype=np.float64),
        "class_names": np.asarray(dataset.class_names),
        "metadata_json": np.asarray(json.dumps(dataset.metadata, ensure_ascii=False)),
    }
    for name, split in (("train", dataset.train), ("validation", dataset.validation), ("test", dataset.test)):
        payload[f"{name}_x"] = split.x
        payload[f"{name}_y"] = split.y
        payload[f"{name}_ids"] = split.ids
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def load_prepared_dataset(path: str | Path) -> PreparedDataset:
    with np.load(Path(path), allow_pickle=False) as payload:
        splits = {
            name: SplitData(payload[f"{name}_x"], payload[f"{name}_y"], payload[f"{name}_ids"])
            for name in ("train", "validation", "test")
        }
        return PreparedDataset(
            str(payload["dataset_id"].item()),
            float(payload["sampling_rate_hz"].item()),
            tuple(payload["class_names"].astype(str).tolist()),
            splits["train"],
            splits["validation"],
            splits["test"],
            json.loads(str(payload["metadata_json"].item())),
        )


def prepare_pilot_dataset(project_root: str | Path, dataset_id: str, *, force: bool = False, seed: int = 17) -> PreparedDataset:
    root = Path(project_root)
    if dataset_id not in PILOT_DATASETS:
        raise ValueError(f"unknown pilot dataset: {dataset_id}")
    cache = _cache_path(root, dataset_id)
    if cache.exists() and not force:
        return load_prepared_dataset(cache)
    if dataset_id == "basicmotions_uea":
        dataset = _prepare_uea(root / "data" / "raw" / "uea" / "BasicMotions", dataset_id, 10.0, seed)
    elif dataset_id == "epilepsy_uea":
        dataset = _prepare_uea(root / "data" / "raw" / "uea" / "Epilepsy", dataset_id, 16.0, seed)
    elif dataset_id == "pamap2_uci":
        dataset = _prepare_pamap2(root / "data" / "raw" / "uci" / "PAMAP2")
    else:
        dataset = _prepare_mhealth(root / "data" / "raw" / "uci" / "MHEALTH")
    save_prepared_dataset(cache, dataset)
    manifest = {
        "dataset_id": dataset.dataset_id,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "class_names": dataset.class_names,
        "shapes": {
            "train": list(dataset.train.x.shape),
            "validation": list(dataset.validation.x.shape),
            "test": list(dataset.test.x.shape),
        },
        "split_id_overlap": {
            "train_validation": bool(set(dataset.train.ids) & set(dataset.validation.ids)),
            "train_test": bool(set(dataset.train.ids) & set(dataset.test.ids)),
            "validation_test": bool(set(dataset.validation.ids) & set(dataset.test.ids)),
        },
        "metadata": dataset.metadata,
    }
    cache.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset
