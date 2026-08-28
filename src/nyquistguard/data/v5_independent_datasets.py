"""Leakage-locked data preparation for the frozen V5.1 confirmation panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nyquistguard.data.new_confirmation_datasets import (
    ConfirmationDevelopmentDataset,
    _encode_from_training,
    _load_ts,
    _save_development,
    _save_full,
    _standardize_splits,
    load_confirmation_cache,
    load_confirmation_development_cache,
)
from nyquistguard.data.pilot_datasets import (
    PreparedDataset,
    SplitData,
    _stratified_validation_indices,
)


V5_INDEPENDENT_DATASETS = (
    "self_regulation_scp1_uea",
    "hand_movement_direction_uea",
    "racket_sports_uea",
    "heartbeat_uea",
)


def _selection(project_root: Path) -> dict[str, Any]:
    path = (
        project_root
        / "configs"
        / "experiments"
        / "v5_1_independent_confirmation_selection.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def independent_cache_path(
    project_root: str | Path, dataset_id: str, *, development: bool
) -> Path:
    suffix = "__development.npz" if development else ".npz"
    return (
        Path(project_root)
        / "data"
        / "processed"
        / "v5_1_independent_v1"
        / f"{dataset_id}{suffix}"
    )


def raw_split_path(project_root: str | Path, dataset_id: str, split: str) -> Path:
    spec = _selection(Path(project_root).resolve())["datasets"][dataset_id]
    archive_name = str(spec["archive_name"])
    return (
        Path(project_root)
        / "data"
        / "raw"
        / "uea_v5_1_independent"
        / archive_name
        / f"{archive_name}_{split.upper()}.ts"
    )


def _equal_length(values: np.ndarray | list[np.ndarray], path: Path) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 3:
        raise ValueError(f"frozen V5.1 panel requires equal-length 3-D data: {path}")
    output = values.astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError(f"non-finite values in {path}")
    return output


def _assert_expected_shape(
    x: np.ndarray, y: np.ndarray, spec: dict[str, Any], split: str
) -> None:
    expected_cases = int(spec[f"expected_{split.lower()}_cases"])
    expected = (
        expected_cases,
        int(spec["expected_channels"]),
        int(spec["expected_length"]),
    )
    if tuple(x.shape) != expected or y.shape != (expected_cases,):
        raise ValueError(
            f"unexpected {spec['archive_name']} {split} layout: "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}, expected={expected}"
        )


def _prepare(
    project_root: Path, dataset_id: str, *, include_test: bool
) -> ConfirmationDevelopmentDataset | PreparedDataset:
    if dataset_id not in V5_INDEPENDENT_DATASETS:
        raise ValueError(f"unknown V5.1 independent dataset: {dataset_id}")
    spec = _selection(project_root)["datasets"][dataset_id]
    train_path = raw_split_path(project_root, dataset_id, "TRAIN")
    if not train_path.exists():
        raise FileNotFoundError(f"missing frozen TRAIN file: {train_path}")
    train_loaded, labels = _load_ts(train_path)
    train_all = _equal_length(train_loaded, train_path)
    _assert_expected_shape(train_all, labels, spec, "train")
    train_indices, validation_indices = _stratified_validation_indices(
        labels, 17042, 0.2
    )
    train_text = labels[train_indices]
    validation_text = labels[validation_indices]
    test_x: np.ndarray | None = None
    test_text: np.ndarray | None = None
    if include_test:
        test_path = raw_split_path(project_root, dataset_id, "TEST")
        if not test_path.exists():
            raise FileNotFoundError(f"missing frozen TEST file: {test_path}")
        test_loaded, test_text = _load_ts(test_path)
        test_x = _equal_length(test_loaded, test_path)
        _assert_expected_shape(test_x, test_text, spec, "test")
    train_y, encoded, class_names = _encode_from_training(
        train_text,
        [validation_text] + ([test_text] if test_text is not None else []),
    )
    if len(class_names) != int(spec["expected_classes"]):
        raise ValueError(
            f"unexpected class count for {dataset_id}: {len(class_names)}"
        )
    train = SplitData(
        train_all[train_indices],
        train_y,
        np.asarray([f"official_train_{index:05d}" for index in train_indices]),
    )
    validation = SplitData(
        train_all[validation_indices],
        encoded[0],
        np.asarray([f"official_train_{index:05d}" for index in validation_indices]),
    )
    test = None
    if test_x is not None:
        test = SplitData(
            test_x,
            encoded[1],
            np.asarray([f"official_test_{index:05d}" for index in range(len(test_x))]),
        )
    train, validation, test, statistics = _standardize_splits(train, validation, test)
    metadata = {
        "archive": str(spec["archive_name"]),
        "domain": str(spec["domain"]),
        "source_format": "official UEA .ts split via aeon",
        "split_protocol": (
            "official test retained; deterministic stratified 20% validation "
            "from official train seed17042"
        ),
        "normalization": "per-channel zscore fitted on final training subset only",
        "normalization_statistics": statistics,
        "test_accessed": include_test,
    }
    sampling_rate = float(spec["sampling_rate_hz"])
    if not include_test:
        return ConfirmationDevelopmentDataset(
            dataset_id, sampling_rate, class_names, train, validation, metadata
        )
    assert test is not None
    return PreparedDataset(
        dataset_id, sampling_rate, class_names, train, validation, test, metadata
    )


def prepare_v5_independent_development_dataset(
    project_root: str | Path, dataset_id: str, *, force: bool = False
) -> ConfirmationDevelopmentDataset:
    if dataset_id not in V5_INDEPENDENT_DATASETS:
        raise ValueError(f"unknown V5.1 independent dataset: {dataset_id}")
    root = Path(project_root).resolve()
    cache = independent_cache_path(root, dataset_id, development=True)
    if cache.exists() and not force:
        return load_confirmation_development_cache(cache)
    dataset = _prepare(root, dataset_id, include_test=False)
    if not isinstance(dataset, ConfirmationDevelopmentDataset) or hasattr(dataset, "test"):
        raise RuntimeError("development preparation exposed a test split")
    _save_development(cache, dataset)
    return dataset


def prepare_v5_independent_dataset(
    project_root: str | Path,
    dataset_id: str,
    *,
    force: bool = False,
    confirmed_test_access: bool = False,
) -> PreparedDataset:
    if not confirmed_test_access:
        raise PermissionError(
            "V5.1 formal TEST access requires explicit dashboard manual confirmation"
        )
    if dataset_id not in V5_INDEPENDENT_DATASETS:
        raise ValueError(f"unknown V5.1 independent dataset: {dataset_id}")
    root = Path(project_root).resolve()
    cache = independent_cache_path(root, dataset_id, development=False)
    if cache.exists() and not force:
        return load_confirmation_cache(cache)
    dataset = _prepare(root, dataset_id, include_test=True)
    if not isinstance(dataset, PreparedDataset):
        raise RuntimeError("formal preparation failed to materialize TEST")
    _save_full(cache, dataset)
    return dataset

