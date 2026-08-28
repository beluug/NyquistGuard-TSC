"""Leakage-locked preparation for the four V4.1 confirmation datasets.

Development preparation materializes only train/validation arrays.  Test files
are opened only by :func:`prepare_confirmation_dataset` after the large formal
runner has received its explicit manual-start token.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from aeon.datasets import load_from_ts_file

from .pilot_datasets import PreparedDataset, SplitData, _stratified_validation_indices


CONFIRMATION_DATASETS = (
    "character_trajectories_uea",
    "motor_imagery_uea",
    "wisdm_activity_uci",
    "ptbxl_physionet",
)
PTBXL_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
WISDM_CLASSES = tuple("ABCDEFGHIJKLMOPQRS")


@dataclass(frozen=True)
class ConfirmationDevelopmentDataset:
    """A structural train/validation-only view with no test attribute."""

    dataset_id: str
    sampling_rate_hz: float
    class_names: tuple[str, ...]
    train: SplitData
    validation: SplitData
    metadata: dict[str, Any]


def _selection_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "experiments" / "v4_new_dataset_confirmation_selection.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def confirmation_cache_path(project_root: str | Path, dataset_id: str, *, development: bool) -> Path:
    suffix = "__development.npz" if development else ".npz"
    return Path(project_root) / "data" / "processed" / "v4_confirmation_v1" / f"{dataset_id}{suffix}"


def _encode_from_training(
    train_labels: Sequence[str] | np.ndarray,
    other_groups: Iterable[Sequence[str] | np.ndarray],
    *,
    expected_classes: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], tuple[str, ...]]:
    train_text = np.asarray(train_labels).astype(str)
    if expected_classes is None:
        class_names = tuple(sorted(set(train_text.tolist())))
    else:
        class_names = expected_classes
        missing = sorted(set(class_names) - set(train_text.tolist()))
        unexpected = sorted(set(train_text.tolist()) - set(class_names))
        if missing or unexpected:
            raise ValueError(f"training classes differ from the frozen task; missing={missing}, unexpected={unexpected}")
    mapping = {name: index for index, name in enumerate(class_names)}

    def encode(values: Sequence[str] | np.ndarray, split_name: str) -> np.ndarray:
        text = np.asarray(values).astype(str)
        unseen = sorted(set(text.tolist()) - set(mapping))
        if unseen:
            raise ValueError(f"{split_name} contains labels absent from frozen training classes: {unseen}")
        return np.asarray([mapping[value] for value in text], dtype=np.int64)

    encoded_other = [encode(group, f"split_{index}") for index, group in enumerate(other_groups)]
    return encode(train_text, "train"), encoded_other, class_names


def _standardize_splits(
    train: SplitData,
    validation: SplitData,
    test: SplitData | None,
) -> tuple[SplitData, SplitData, SplitData | None, dict[str, list[float]]]:
    mean = np.mean(train.x, axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    scale = np.std(train.x, axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-6))
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("non-finite training standardization statistics")

    def apply(split: SplitData | None) -> SplitData | None:
        if split is None:
            return None
        values = (split.x.astype(np.float32, copy=False) - mean) / scale
        if not np.isfinite(values).all():
            raise ValueError("non-finite standardized values")
        return SplitData(values.astype(np.float32, copy=False), split.y, split.ids)

    return (
        apply(train),
        apply(validation),
        apply(test),
        {"mean": mean.reshape(-1).tolist(), "scale": scale.reshape(-1).tolist()},
    )  # type: ignore[return-value]


def _load_ts(path: Path) -> tuple[np.ndarray | list[np.ndarray], np.ndarray]:
    loaded = load_from_ts_file(str(path), return_type="auto")
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise ValueError(f"aeon did not return data and labels for {path}")
    x, y = loaded[0], np.asarray(loaded[1]).astype(str)
    if isinstance(x, np.ndarray):
        x = x.astype(np.float32, copy=False)
    else:
        x = [np.asarray(case, dtype=np.float32) for case in x]
    return x, y


def _ragged_stats(cases: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not cases:
        raise ValueError("cannot fit ragged statistics on an empty training split")
    channels = int(cases[0].shape[0])
    sums = np.zeros(channels, dtype=np.float64)
    squares = np.zeros(channels, dtype=np.float64)
    counts = np.zeros(channels, dtype=np.int64)
    for case in cases:
        if case.ndim != 2 or case.shape[0] != channels or not np.isfinite(case).all():
            raise ValueError("ragged cases must be finite [channel,time] arrays with fixed channels")
        sums += case.sum(axis=1, dtype=np.float64)
        squares += np.square(case, dtype=np.float64).sum(axis=1, dtype=np.float64)
        counts += case.shape[1]
    mean = sums / counts
    variance = np.maximum(squares / counts - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _ragged_to_fixed(
    cases: Sequence[np.ndarray], fixed_length: int, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    output = np.zeros((len(cases), len(mean), fixed_length), dtype=np.float32)
    for index, case in enumerate(cases):
        length = min(fixed_length, case.shape[1])
        output[index, :, :length] = (case[:, :length] - mean[:, None]) / scale[:, None]
    return output


def _prepare_character(project_root: Path, *, include_test: bool) -> ConfirmationDevelopmentDataset | PreparedDataset:
    base = project_root / "data" / "raw" / "uea" / "CharacterTrajectories"
    train_all, train_labels_all = _load_ts(base / "CharacterTrajectories_TRAIN.ts")
    if isinstance(train_all, np.ndarray):
        train_cases = [case for case in train_all]
    else:
        train_cases = train_all
    train_indices, validation_indices = _stratified_validation_indices(train_labels_all, 16180, 0.2)
    fit_cases = [train_cases[index] for index in train_indices]
    validation_cases = [train_cases[index] for index in validation_indices]
    fixed_length = max(case.shape[1] for case in fit_cases)
    mean, scale = _ragged_stats(fit_cases)
    train_y_text = train_labels_all[train_indices]
    validation_y_text = train_labels_all[validation_indices]
    test_cases: list[np.ndarray] | None = None
    test_y_text: np.ndarray | None = None
    if include_test:
        test_loaded, test_y_text = _load_ts(base / "CharacterTrajectories_TEST.ts")
        test_cases = [case for case in test_loaded] if isinstance(test_loaded, np.ndarray) else test_loaded
    encoded_train, encoded_other, class_names = _encode_from_training(
        train_y_text,
        [validation_y_text] + ([test_y_text] if test_y_text is not None else []),
    )
    train = SplitData(
        _ragged_to_fixed(fit_cases, fixed_length, mean, scale),
        encoded_train,
        np.asarray([f"official_train_{index:04d}" for index in train_indices]),
    )
    validation = SplitData(
        _ragged_to_fixed(validation_cases, fixed_length, mean, scale),
        encoded_other[0],
        np.asarray([f"official_train_{index:04d}" for index in validation_indices]),
    )
    metadata = {
        "source_format": "UEA unequal-length .ts via aeon",
        "split_protocol": "official test; stratified 20% validation from official train seed16180",
        "fixed_length": fixed_length,
        "fixed_length_rule": "maximum actual-training length; validation/test cropped or mean-padded",
        "normalization": "per-channel training-valid-points zscore; padding is zero after normalization",
        "test_accessed": include_test,
    }
    if not include_test:
        return ConfirmationDevelopmentDataset(
            "character_trajectories_uea", 200.0, class_names, train, validation, metadata
        )
    assert test_cases is not None and test_y_text is not None
    test = SplitData(
        _ragged_to_fixed(test_cases, fixed_length, mean, scale),
        encoded_other[1],
        np.asarray([f"official_test_{index:04d}" for index in range(len(test_cases))]),
    )
    return PreparedDataset(
        "character_trajectories_uea", 200.0, class_names, train, validation, test, metadata
    )


def _prepare_motor_imagery(
    project_root: Path, *, include_test: bool
) -> ConfirmationDevelopmentDataset | PreparedDataset:
    base = project_root / "data" / "raw" / "uea" / "MotorImagery"
    train_all, train_labels_all = _load_ts(base / "MotorImagery_TRAIN.ts")
    if not isinstance(train_all, np.ndarray) or train_all.ndim != 3:
        raise ValueError("MotorImagery train data must be an equal-length 3-D collection")
    train_indices, validation_indices = _stratified_validation_indices(train_labels_all, 16180, 0.2)
    test_x: np.ndarray | None = None
    test_y_text: np.ndarray | None = None
    if include_test:
        loaded_test, test_y_text = _load_ts(base / "MotorImagery_TEST.ts")
        if not isinstance(loaded_test, np.ndarray) or loaded_test.ndim != 3:
            raise ValueError("MotorImagery test data must be an equal-length 3-D collection")
        test_x = loaded_test
    train_y_text = train_labels_all[train_indices]
    validation_y_text = train_labels_all[validation_indices]
    encoded_train, encoded_other, class_names = _encode_from_training(
        train_y_text,
        [validation_y_text] + ([test_y_text] if test_y_text is not None else []),
    )
    train = SplitData(
        train_all[train_indices], encoded_train,
        np.asarray([f"session1_{index:04d}" for index in train_indices]),
    )
    validation = SplitData(
        train_all[validation_indices], encoded_other[0],
        np.asarray([f"session1_{index:04d}" for index in validation_indices]),
    )
    test = None
    if test_x is not None:
        test = SplitData(
            test_x, encoded_other[1],
            np.asarray([f"session2_{index:04d}" for index in range(len(test_x))]),
        )
    train, validation, test, stats = _standardize_splits(train, validation, test)
    metadata = {
        "source_format": "UEA equal-length .ts via aeon",
        "split_protocol": "session1 official train with 20% stratified validation seed16180; session2 official test",
        "normalization": "per-channel actual-training zscore",
        "standardization": stats,
        "test_accessed": include_test,
    }
    if not include_test:
        return ConfirmationDevelopmentDataset(
            "motor_imagery_uea", 1000.0, class_names, train, validation, metadata
        )
    assert test is not None
    return PreparedDataset("motor_imagery_uea", 1000.0, class_names, train, validation, test, metadata)


def _wisdm_subject(path: Path) -> int:
    match = re.fullmatch(r"data_(\d+)_accel_phone\.txt", path.name)
    if match is None:
        raise ValueError(f"unexpected WISDM phone accelerometer filename: {path.name}")
    return int(match.group(1))


def _read_wisdm_subject(path: Path, subject: int, *, window: int, stride: int) -> tuple[list[np.ndarray], list[str], list[str]]:
    labels: list[str] = []
    values: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.rstrip(";").split(",")
            if len(fields) != 6:
                raise ValueError(f"malformed WISDM row at {path}:{line_number}")
            if int(fields[0]) != subject:
                raise ValueError(f"subject mismatch at {path}:{line_number}")
            labels.append(fields[1])
            values.append((float(fields[3]), float(fields[4]), float(fields[5])))
    matrix = np.asarray(values, dtype=np.float32)
    label_array = np.asarray(labels).astype(str)
    windows: list[np.ndarray] = []
    window_labels: list[str] = []
    ids: list[str] = []
    for start in range(0, max(0, len(matrix) - window + 1), stride):
        section_labels = label_array[start : start + window]
        if len(section_labels) != window or np.any(section_labels != section_labels[0]):
            continue
        section = matrix[start : start + window]
        if not np.isfinite(section).all():
            continue
        windows.append(section.T.astype(np.float32, copy=False))
        window_labels.append(str(section_labels[0]))
        ids.append(f"subject{subject}:{start}:{start + window}")
    return windows, window_labels, ids


def _prepare_wisdm(project_root: Path, *, include_test: bool) -> ConfirmationDevelopmentDataset | PreparedDataset:
    selection = _selection_config(project_root)["datasets"]["wisdm_activity_uci"]
    split_subjects = {
        "train": tuple(int(value) for value in selection["train_subjects"]),
        "validation": tuple(int(value) for value in selection["validation_subjects"]),
        "test": tuple(int(value) for value in selection["test_subjects"]),
    }
    all_subjects = [value for group in split_subjects.values() for value in group]
    if len(all_subjects) != 51 or len(set(all_subjects)) != 51 or set(all_subjects) != set(range(1600, 1651)):
        raise ValueError("frozen WISDM subject partition must cover 1600-1650 exactly once")
    raw_dir = (
        project_root / "data" / "raw" / "uci" / "WISDM" / "wisdm-dataset" /
        "wisdm-dataset" / "raw" / "phone" / "accel"
    )
    paths = {
        _wisdm_subject(path): path
        for path in sorted(raw_dir.glob("data_*_accel_phone.txt"))
    }
    if set(paths) != set(range(1600, 1651)):
        raise ValueError("WISDM raw phone accelerometer files do not cover subjects 1600-1650")
    window = int(round(float(selection["window_seconds"]) * float(selection["sampling_rate_hz"])))
    stride = int(round(float(selection["stride_seconds"]) * float(selection["sampling_rate_hz"])))
    split_payload: dict[str, tuple[list[np.ndarray], list[str], list[str]]] = {}
    for split_name in ("train", "validation", "test"):
        x_items: list[np.ndarray] = []
        y_items: list[str] = []
        ids: list[str] = []
        if split_name == "test" and not include_test:
            continue
        for subject in split_subjects[split_name]:
            subject_x, subject_y, subject_ids = _read_wisdm_subject(
                paths[subject], subject, window=window, stride=stride
            )
            x_items.extend(subject_x)
            y_items.extend(subject_y)
            ids.extend(subject_ids)
        if not x_items:
            raise ValueError(f"WISDM produced no {split_name} windows")
        split_payload[split_name] = (x_items, y_items, ids)
    train_text = np.asarray(split_payload["train"][1])
    validation_text = np.asarray(split_payload["validation"][1])
    test_text = np.asarray(split_payload["test"][1]) if include_test else None
    encoded_train, encoded_other, class_names = _encode_from_training(
        train_text,
        [validation_text] + ([test_text] if test_text is not None else []),
        expected_classes=WISDM_CLASSES,
    )
    train = SplitData(
        np.stack(split_payload["train"][0]), encoded_train, np.asarray(split_payload["train"][2])
    )
    validation = SplitData(
        np.stack(split_payload["validation"][0]), encoded_other[0], np.asarray(split_payload["validation"][2])
    )
    test = None
    if include_test:
        test = SplitData(
            np.stack(split_payload["test"][0]), encoded_other[1], np.asarray(split_payload["test"][2])
        )
    train, validation, test, stats = _standardize_splits(train, validation, test)
    metadata = {
        "source_format": "WISDM raw smartphone accelerometer text",
        "selected_sensor": "phone accelerometer x/y/z only",
        "window_samples": window,
        "stride_samples": stride,
        "split_protocol": "frozen subject groups seed16180",
        "split_subjects": split_subjects,
        "standardization": stats,
        "test_accessed": include_test,
    }
    if not include_test:
        return ConfirmationDevelopmentDataset(
            "wisdm_activity_uci", 20.0, class_names, train, validation, metadata
        )
    assert test is not None
    return PreparedDataset("wisdm_activity_uci", 20.0, class_names, train, validation, test, metadata)


def _find_ptbxl_root(project_root: Path) -> Path:
    candidates = list((project_root / "data" / "raw" / "physionet" / "PTBXL").rglob("ptbxl_database.csv"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one PTB-XL metadata file, found {len(candidates)}")
    return candidates[0].parent


def _ptbxl_statement_map(base: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with (base / "scp_statements.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row.get("diagnostic", "0") or 0) == 1.0 and row.get("diagnostic_class") in PTBXL_CLASSES:
                mapping[str(row[""])] = str(row["diagnostic_class"])
    if set(mapping.values()) != set(PTBXL_CLASSES):
        raise ValueError("PTB-XL statement mapping does not cover all frozen superclasses")
    return mapping


def _ptbxl_records(base: Path, *, include_test: bool) -> dict[str, list[dict[str, str]]]:
    mapping = _ptbxl_statement_map(base)
    buckets: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    with (base / "ptbxl_database.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["strat_fold"])
            split_name = "train" if fold <= 8 else "validation" if fold == 9 else "test"
            if split_name == "test" and not include_test:
                continue
            codes = ast.literal_eval(row["scp_codes"])
            classes = sorted({mapping[code] for code in codes if code in mapping})
            if len(classes) != 1:
                continue
            buckets[split_name].append(
                {
                    "ecg_id": row["ecg_id"],
                    "patient_id": row["patient_id"],
                    "filename_lr": row["filename_lr"],
                    "label": classes[0],
                }
            )
    for split_name in ("train", "validation") + (("test",) if include_test else ()):
        if not buckets[split_name]:
            raise ValueError(f"PTB-XL produced no {split_name} single-superclass records")
    return buckets


def _load_ptbxl_waveforms(base: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import wfdb

    signals: list[np.ndarray] = []
    labels: list[str] = []
    ids: list[str] = []
    for row in rows:
        signal, metadata = wfdb.rdsamp(str(base / row["filename_lr"]))
        if int(metadata["fs"]) != 100 or signal.shape != (1000, 12):
            raise ValueError(f"unexpected PTB-XL records100 layout for ecg_id={row['ecg_id']}")
        values = signal.T.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite PTB-XL waveform for ecg_id={row['ecg_id']}")
        signals.append(values)
        labels.append(row["label"])
        ids.append(f"ecg{row['ecg_id']}:patient{row['patient_id']}")
    return np.stack(signals), np.asarray(labels), np.asarray(ids)


def _prepare_ptbxl(project_root: Path, *, include_test: bool) -> ConfirmationDevelopmentDataset | PreparedDataset:
    base = _find_ptbxl_root(project_root)
    records = _ptbxl_records(base, include_test=include_test)
    train_x, train_text, train_ids = _load_ptbxl_waveforms(base, records["train"])
    validation_x, validation_text, validation_ids = _load_ptbxl_waveforms(base, records["validation"])
    test_payload: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if include_test:
        test_payload = _load_ptbxl_waveforms(base, records["test"])
    encoded_train, encoded_other, class_names = _encode_from_training(
        train_text,
        [validation_text] + ([test_payload[1]] if test_payload is not None else []),
        expected_classes=PTBXL_CLASSES,
    )
    train = SplitData(train_x, encoded_train, train_ids)
    validation = SplitData(validation_x, encoded_other[0], validation_ids)
    test = None
    if test_payload is not None:
        test = SplitData(test_payload[0], encoded_other[1], test_payload[2])
    train, validation, test, stats = _standardize_splits(train, validation, test)
    metadata = {
        "source_format": "PTB-XL v1.0.3 WFDB records100",
        "task": "exactly one diagnostic superclass",
        "split_protocol": "strat_fold 1-8 train, 9 validation, 10 test",
        "record_counts": {name: len(rows) for name, rows in records.items() if rows},
        "standardization": stats,
        "test_accessed": include_test,
        "fold10_scp_codes_parsed": include_test,
        "fold10_waveforms_opened": include_test,
    }
    if not include_test:
        return ConfirmationDevelopmentDataset(
            "ptbxl_physionet", 100.0, class_names, train, validation, metadata
        )
    assert test is not None
    return PreparedDataset("ptbxl_physionet", 100.0, class_names, train, validation, test, metadata)


def _prepare(project_root: Path, dataset_id: str, *, include_test: bool) -> ConfirmationDevelopmentDataset | PreparedDataset:
    if dataset_id == "character_trajectories_uea":
        return _prepare_character(project_root, include_test=include_test)
    if dataset_id == "motor_imagery_uea":
        return _prepare_motor_imagery(project_root, include_test=include_test)
    if dataset_id == "wisdm_activity_uci":
        return _prepare_wisdm(project_root, include_test=include_test)
    if dataset_id == "ptbxl_physionet":
        return _prepare_ptbxl(project_root, include_test=include_test)
    raise ValueError(f"unknown V4 confirmation dataset: {dataset_id}")


def _atomic_npz(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _base_payload(dataset: ConfirmationDevelopmentDataset | PreparedDataset) -> dict[str, object]:
    return {
        "dataset_id": np.asarray(dataset.dataset_id),
        "sampling_rate_hz": np.asarray(dataset.sampling_rate_hz, dtype=np.float64),
        "class_names": np.asarray(dataset.class_names),
        "metadata_json": np.asarray(json.dumps(dataset.metadata, ensure_ascii=False)),
        "train_x": dataset.train.x,
        "train_y": dataset.train.y,
        "train_ids": dataset.train.ids,
        "validation_x": dataset.validation.x,
        "validation_y": dataset.validation.y,
        "validation_ids": dataset.validation.ids,
    }


def _write_manifest(path: Path, dataset: ConfirmationDevelopmentDataset | PreparedDataset) -> None:
    splits = {"train": dataset.train, "validation": dataset.validation}
    if isinstance(dataset, PreparedDataset):
        splits["test"] = dataset.test
    ids = {name: set(split.ids.astype(str).tolist()) for name, split in splits.items()}
    overlap = {
        f"{left}_{right}": bool(ids[left] & ids[right])
        for index, left in enumerate(ids)
        for right in list(ids)[index + 1 :]
    }
    manifest = {
        "dataset_id": dataset.dataset_id,
        "test_accessed": isinstance(dataset, PreparedDataset),
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "class_names": dataset.class_names,
        "shapes": {name: list(split.x.shape) for name, split in splits.items()},
        "class_counts": {
            name: np.bincount(split.y, minlength=len(dataset.class_names)).tolist()
            for name, split in splits.items()
        },
        "split_id_overlap": overlap,
        "metadata": dataset.metadata,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_development(path: Path, dataset: ConfirmationDevelopmentDataset) -> None:
    payload = _base_payload(dataset)
    if any(key.startswith("test_") for key in payload):
        raise RuntimeError("development cache payload must not contain test keys")
    _atomic_npz(path, payload)
    _write_manifest(path.with_suffix(".manifest.json"), dataset)


def _save_full(path: Path, dataset: PreparedDataset) -> None:
    payload = _base_payload(dataset)
    payload.update(test_x=dataset.test.x, test_y=dataset.test.y, test_ids=dataset.test.ids)
    _atomic_npz(path, payload)
    _write_manifest(path.with_suffix(".manifest.json"), dataset)


def load_confirmation_development_cache(path: str | Path) -> ConfirmationDevelopmentDataset:
    with np.load(Path(path), allow_pickle=False) as payload:
        if any(name.startswith("test_") for name in payload.files):
            raise ValueError("development cache unexpectedly contains test arrays")
        required = {
            "dataset_id", "sampling_rate_hz", "class_names", "metadata_json",
            "train_x", "train_y", "train_ids", "validation_x", "validation_y", "validation_ids",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"development cache is missing keys: {missing}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("test_accessed") is not False:
            raise ValueError("development cache is not test-locked")
        return ConfirmationDevelopmentDataset(
            str(payload["dataset_id"].item()),
            float(payload["sampling_rate_hz"].item()),
            tuple(payload["class_names"].astype(str).tolist()),
            SplitData(payload["train_x"], payload["train_y"], payload["train_ids"]),
            SplitData(payload["validation_x"], payload["validation_y"], payload["validation_ids"]),
            metadata,
        )


def load_confirmation_cache(path: str | Path) -> PreparedDataset:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("test_accessed") is not True:
            raise ValueError("formal confirmation cache must record test access")
        return PreparedDataset(
            str(payload["dataset_id"].item()),
            float(payload["sampling_rate_hz"].item()),
            tuple(payload["class_names"].astype(str).tolist()),
            SplitData(payload["train_x"], payload["train_y"], payload["train_ids"]),
            SplitData(payload["validation_x"], payload["validation_y"], payload["validation_ids"]),
            SplitData(payload["test_x"], payload["test_y"], payload["test_ids"]),
            metadata,
        )


def prepare_confirmation_development_dataset(
    project_root: str | Path, dataset_id: str, *, force: bool = False
) -> ConfirmationDevelopmentDataset:
    if dataset_id not in CONFIRMATION_DATASETS:
        raise ValueError(f"unknown V4 confirmation dataset: {dataset_id}")
    root = Path(project_root).resolve()
    cache = confirmation_cache_path(root, dataset_id, development=True)
    if cache.exists() and not force:
        return load_confirmation_development_cache(cache)
    dataset = _prepare(root, dataset_id, include_test=False)
    if not isinstance(dataset, ConfirmationDevelopmentDataset) or hasattr(dataset, "test"):
        raise RuntimeError("development preparation materialized a test split")
    _save_development(cache, dataset)
    return dataset


def prepare_confirmation_dataset(
    project_root: str | Path,
    dataset_id: str,
    *,
    force: bool = False,
    confirmed_test_access: bool = False,
) -> PreparedDataset:
    if not confirmed_test_access:
        raise PermissionError("formal confirmation test access requires explicit manual confirmation")
    if dataset_id not in CONFIRMATION_DATASETS:
        raise ValueError(f"unknown V4 confirmation dataset: {dataset_id}")
    root = Path(project_root).resolve()
    cache = confirmation_cache_path(root, dataset_id, development=False)
    if cache.exists() and not force:
        return load_confirmation_cache(cache)
    dataset = _prepare(root, dataset_id, include_test=True)
    if not isinstance(dataset, PreparedDataset):
        raise RuntimeError("formal confirmation preparation did not materialize a test split")
    _save_full(cache, dataset)
    return dataset

