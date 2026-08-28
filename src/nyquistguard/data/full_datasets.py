"""Leakage-safe preparation for the frozen ten-dataset Full benchmark.

The four Pilot datasets reuse their already materialized Pilot caches.  The six
Full-only datasets are converted once to ``[case, channel, time]`` arrays and
cached below ``data/processed/full_v1``.  All split statistics are fitted from
training data only; patient/subject/night boundaries never cross splits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from nyquistguard.experiments.progress import atomic_write_json

from .pilot_datasets import (
    PILOT_DATASETS,
    PreparedDataset,
    SplitData,
    _require_closed_set,
    _standardize,
    load_prepared_dataset,
    prepare_pilot_dataset,
    save_prepared_dataset,
)


FULL_DATASETS = PILOT_DATASETS + (
    "hapt_uci",
    "daily_sports_uci",
    "hydraulic_uci",
    "sleep_edfx_physionet",
    "eegmmi_physionet",
    "mitbih_arrhythmia_physionet",
)

FULL_ONLY_DATASETS = FULL_DATASETS[len(PILOT_DATASETS) :]


def full_cache_path(project_root: str | Path, dataset_id: str) -> Path:
    root = Path(project_root)
    if dataset_id in PILOT_DATASETS:
        return root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
    return root / "data" / "processed" / "full_v1" / f"{dataset_id}.npz"


def _split(values: list[np.ndarray], labels: list[int], ids: list[str]) -> SplitData:
    if not values:
        raise ValueError("dataset preparation produced an empty split")
    return SplitData(
        np.stack(values).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        np.asarray(ids),
    )


def _finalize(
    dataset_id: str,
    sampling_rate_hz: float,
    class_names: tuple[str, ...],
    buckets: dict[str, tuple[list[np.ndarray], list[int], list[str]]],
    metadata: dict[str, object],
) -> PreparedDataset:
    train, validation, test = (
        _split(*buckets[name]) for name in ("train", "validation", "test")
    )
    _require_closed_set(train, validation, test, class_names)
    train, validation, test = _standardize(train, validation, test)
    return PreparedDataset(
        dataset_id,
        float(sampling_rate_hz),
        class_names,
        train,
        validation,
        test,
        metadata,
    )


def _fixed_segment(values: np.ndarray, length: int) -> np.ndarray:
    """Interpolate a short labelled segment, or center-crop a long one."""

    if len(values) == length:
        return values.T.astype(np.float32, copy=False)
    if len(values) > length:
        start = (len(values) - length) // 2
        return values[start : start + length].T.astype(np.float32, copy=False)
    old = np.linspace(0.0, 1.0, num=len(values), dtype=np.float64)
    new = np.linspace(0.0, 1.0, num=length, dtype=np.float64)
    result = np.stack([np.interp(new, old, values[:, c]) for c in range(values.shape[1])])
    return result.astype(np.float32)


def _prepare_hapt(root: Path) -> PreparedDataset:
    raw_root = root / "RawData"
    labels = np.loadtxt(raw_root / "labels.txt", dtype=np.int64)
    official_train = set(
        np.loadtxt(root / "Train" / "subject_id_train.txt", dtype=np.int64).tolist()
    )
    official_test = set(
        np.loadtxt(root / "Test" / "subject_id_test.txt", dtype=np.int64).tolist()
    )
    # Frozen group validation: the last four official training subjects.
    validation_subjects = set(sorted(official_train)[-4:])
    training_subjects = official_train - validation_subjects
    buckets = {name: ([], [], []) for name in ("train", "validation", "test")}
    cache: dict[int, np.ndarray] = {}
    for experiment, subject, activity, start_1, end_1 in labels:
        experiment = int(experiment)
        subject = int(subject)
        if experiment not in cache:
            acc = np.loadtxt(raw_root / f"acc_exp{experiment:02d}_user{subject:02d}.txt", dtype=np.float32)
            gyro = np.loadtxt(raw_root / f"gyro_exp{experiment:02d}_user{subject:02d}.txt", dtype=np.float32)
            cache[experiment] = np.concatenate([acc, gyro], axis=1)
        segment = cache[experiment][int(start_1) - 1 : int(end_1)]
        split_name = (
            "validation"
            if subject in validation_subjects
            else "train"
            if subject in training_subjects
            else "test"
            if subject in official_test
            else None
        )
        if split_name is None or len(segment) < 2:
            continue
        # Long activities contribute fixed 2.56 s windows; short transition
        # segments contribute exactly one interpolated window.
        starts = list(range(0, len(segment) - 128 + 1, 64)) if len(segment) >= 128 else [0]
        if not starts:
            starts = [0]
        for ordinal, start in enumerate(starts):
            item = segment[start : start + 128]
            buckets[split_name][0].append(_fixed_segment(item, 128))
            buckets[split_name][1].append(int(activity) - 1)
            buckets[split_name][2].append(
                f"exp{experiment:02d}:subject{subject:02d}:activity{int(activity):02d}:w{ordinal:04d}"
            )
    return _finalize(
        "hapt_uci",
        50.0,
        (
            "walking", "walking_upstairs", "walking_downstairs", "sitting",
            "standing", "laying", "stand_to_sit", "sit_to_stand", "sit_to_lie",
            "lie_to_sit", "stand_to_lie", "lie_to_stand",
        ),
        buckets,
        {
            "dataset_protocol_id": "hapt_raw_12class_subject_grouped_v1",
            "source_format": "HAPT RawData accelerometer+gyroscope",
            "window_samples": 128,
            "window_seconds": 2.56,
            "long_segment_stride_samples": 64,
            "short_transition_policy": "one linearly interpolated 128-sample window",
            "training_subjects": sorted(training_subjects),
            "validation_subjects": sorted(validation_subjects),
            "test_subjects": sorted(official_test),
        },
    )


def _prepare_daily_sports(root: Path) -> PreparedDataset:
    data_root = root / "data"
    subjects = {"train": {1, 2, 3, 4, 5}, "validation": {6}, "test": {7, 8}}
    buckets = {name: ([], [], []) for name in subjects}
    files = sorted(data_root.glob("a*/p*/s*.txt"))
    if not files:
        raise FileNotFoundError(f"no DailySports segments found below {data_root}")
    for path in files:
        activity_match = re.fullmatch(r"a(\d+)", path.parent.parent.name)
        subject_match = re.fullmatch(r"p(\d+)", path.parent.name)
        if not activity_match or not subject_match:
            continue
        activity = int(activity_match.group(1))
        subject = int(subject_match.group(1))
        split_name = next(name for name, members in subjects.items() if subject in members)
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if values.shape != (125, 45):
            raise ValueError(f"unexpected DailySports shape {values.shape} in {path}")
        buckets[split_name][0].append(values.T)
        buckets[split_name][1].append(activity - 1)
        buckets[split_name][2].append(f"a{activity:02d}:p{subject}:{path.stem}")
    return _finalize(
        "daily_sports_uci",
        25.0,
        tuple(f"activity_{index:02d}" for index in range(1, 20)),
        buckets,
        {
            "dataset_protocol_id": "daily_sports_19class_subject_grouped_v1",
            "source_format": "45-channel 5-second segments",
            "split_protocol": "subjects 1-5 train, 6 validation, 7-8 test",
        },
    )


def _prepare_hydraulic(root: Path) -> PreparedDataset:
    sensors = ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1")
    arrays = [np.loadtxt(root / f"{sensor}.txt", dtype=np.float32) for sensor in sensors]
    if len({array.shape for array in arrays}) != 1 or arrays[0].shape != (2205, 6000):
        raise ValueError("hydraulic 100 Hz sensor matrices do not share the expected 2205x6000 shape")
    values = np.stack(arrays, axis=1)
    raw_labels = np.loadtxt(root / "profile.txt", dtype=np.int64)[:, 1]
    classes = (73, 80, 90, 100)
    mapping = {value: index for index, value in enumerate(classes)}
    labels = np.asarray([mapping[int(value)] for value in raw_labels], dtype=np.int64)
    ranges = {"train": (0, 1323), "validation": (1323, 1764), "test": (1764, 2205)}
    buckets = {name: ([], [], []) for name in ranges}
    for name, (start, end) in ranges.items():
        buckets[name][0].extend(values[start:end])
        buckets[name][1].extend(labels[start:end].tolist())
        buckets[name][2].extend(f"cycle_{index + 1:04d}" for index in range(start, end))
    return _finalize(
        "hydraulic_uci",
        100.0,
        tuple(f"valve_{value}" for value in classes),
        buckets,
        {
            "dataset_protocol_id": "hydraulic_valve_4class_blocked_v1",
            "channels": list(sensors),
            "target": "profile column 2 valve condition",
            "split_protocol": "first 60% train, next 20% validation, final 20% test",
            "cycle_seconds": 60,
        },
    )


def _group_partition(groups: list[str], seed: int = 17) -> dict[str, set[str]]:
    ordered = np.asarray(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(ordered)
    train_end = max(1, int(round(0.70 * len(ordered))))
    validation_end = max(train_end + 1, int(round(0.85 * len(ordered))))
    return {
        "train": set(ordered[:train_end].tolist()),
        "validation": set(ordered[train_end:validation_end].tolist()),
        "test": set(ordered[validation_end:].tolist()),
    }


def _edf_reader():
    try:
        import pyedflib  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Full EDF datasets require pyedflib; run .venv\\Scripts\\python.exe -m pip install -r requirements.lock"
        ) from error
    return pyedflib


def _normalize_eegmmi_trial(sample: np.ndarray, source_rate: float) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float32)
    expected_source_samples = int(round(4.0 * float(source_rate)))
    if values.shape != (64, expected_source_samples):
        raise ValueError(
            f"EEGMMI trial must have shape (64,{expected_source_samples}) at {source_rate:g} Hz"
        )
    if source_rate == 128.0:
        values = resample_poly(values, up=5, down=4, axis=-1).astype(np.float32)
    elif source_rate != 160.0:
        raise ValueError(f"unsupported EEGMMI source rate: {source_rate:g} Hz")
    if values.shape != (64, 640):
        raise ValueError(f"unexpected normalized EEGMMI trial shape: {values.shape}")
    return values


def _prepare_sleep_edfx(root: Path) -> PreparedDataset:
    pyedflib = _edf_reader()
    folder = root / "sleep-cassette"
    psg_paths = sorted(folder.glob("SC4*-PSG.edf"))
    # Sleep-EDF-20 is the standard resource-bounded benchmark subset.
    psg_paths = [path for path in psg_paths if int(path.name[3:5]) < 20]
    pair_map = {path.name[:6]: path for path in folder.glob("SC4*-Hypnogram.edf")}
    subjects = [path.name[:5] for path in psg_paths]
    partitions = _group_partition(subjects)
    buckets = {name: ([], [], []) for name in partitions}
    label_map = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }
    for psg_path in psg_paths:
        hyp_path = pair_map.get(psg_path.name[:6])
        if hyp_path is None:
            raise FileNotFoundError(f"missing hypnogram paired with {psg_path.name}")
        subject = psg_path.name[:5]
        split_name = next(name for name, members in partitions.items() if subject in members)
        with pyedflib.EdfReader(str(psg_path)) as psg:
            signal_labels = [str(value).strip() for value in psg.getSignalLabels()]
            channel_indices = []
            for wanted in ("EEG Fpz-Cz", "EEG Pz-Oz"):
                matches = [index for index, value in enumerate(signal_labels) if value == wanted]
                if not matches:
                    raise ValueError(f"{wanted} absent from {psg_path.name}")
                channel_indices.append(matches[0])
            rates = [float(psg.getSampleFrequency(index)) for index in channel_indices]
            if any(abs(rate - 100.0) > 1e-6 for rate in rates):
                raise ValueError(f"unexpected Sleep-EDF EEG sampling rate in {psg_path.name}: {rates}")
            signals = np.stack([psg.readSignal(index).astype(np.float32) for index in channel_indices])
        with pyedflib.EdfReader(str(hyp_path)) as hyp:
            onsets, durations, descriptions = hyp.readAnnotations()
        epochs: list[tuple[int, int]] = []
        for onset, duration, description in zip(onsets, durations, descriptions):
            label = label_map.get(str(description).strip())
            if label is None:
                continue
            for offset in np.arange(float(onset), float(onset) + float(duration) - 29.999, 30.0):
                epochs.append((int(round(offset * 100.0)), label))
        non_wake = [index for index, (_, label) in enumerate(epochs) if label != 0]
        if not non_wake:
            continue
        lower = max(0, non_wake[0] - 60)
        upper = min(len(epochs), non_wake[-1] + 61)
        for ordinal, (start, label) in enumerate(epochs[lower:upper], start=lower):
            sample = signals[:, start : start + 3000]
            if sample.shape != (2, 3000) or not np.isfinite(sample).all():
                continue
            buckets[split_name][0].append(sample)
            buckets[split_name][1].append(label)
            buckets[split_name][2].append(f"{psg_path.name[:6]}:epoch{ordinal:04d}")
    return _finalize(
        "sleep_edfx_physionet",
        100.0,
        ("wake", "n1", "n2", "n3_n4", "rem"),
        buckets,
        {
            "dataset_protocol_id": "sleep_edf20_sc_5class_grouped_v1",
            "subset": "Sleep-EDF-20: Sleep Cassette subjects 00-19",
            "channels": ["EEG Fpz-Cz", "EEG Pz-Oz"],
            "epoch_seconds": 30,
            "wake_trimming": "30 minutes before first and after last non-wake epoch",
            "split_protocol": "seed17 70/15/15 subject grouped; both nights together",
            "split_groups": {name: sorted(values) for name, values in partitions.items()},
        },
    )


def _prepare_eegmmi(root: Path) -> PreparedDataset:
    pyedflib = _edf_reader()
    paths = sorted(root.glob("S*/S*R*.edf"))
    paths = [path for path in paths if int(re.search(r"R(\d+)\.edf$", path.name).group(1)) in {4, 8, 12}]
    subjects = [path.parent.name for path in paths]
    partitions = _group_partition(subjects)
    buckets = {name: ([], [], []) for name in partitions}
    for path in paths:
        subject = path.parent.name
        split_name = next(name for name, members in partitions.items() if subject in members)
        with pyedflib.EdfReader(str(path)) as reader:
            rates = [float(reader.getSampleFrequency(index)) for index in range(reader.signals_in_file)]
            unique_rates = sorted(set(rates))
            if (
                reader.signals_in_file != 64
                or len(unique_rates) != 1
                or unique_rates[0] not in {128.0, 160.0}
            ):
                raise ValueError(f"unexpected EEGMMI channel/rate layout in {path}")
            source_rate = unique_rates[0]
            signals = np.stack(
                [reader.readSignal(index).astype(np.float32) for index in range(64)]
            )
            onsets, _durations, descriptions = reader.readAnnotations()
        trial = 0
        for onset, description in zip(onsets, descriptions):
            token = str(description).strip()
            if token not in {"T1", "T2"}:
                continue
            start = int(round(float(onset) * source_rate))
            source_samples = int(round(4.0 * source_rate))
            sample = signals[:, start : start + source_samples]
            if sample.shape != (64, source_samples) or not np.isfinite(sample).all():
                continue
            # Three official subjects (S088/S092/S100) are stored at 128 Hz.
            # Preserve them and unify the aeon collection at the declared
            # 160 Hz reference via exact 5/4 polyphase resampling.
            sample = _normalize_eegmmi_trial(sample, source_rate)
            buckets[split_name][0].append(sample)
            buckets[split_name][1].append(0 if token == "T1" else 1)
            buckets[split_name][2].append(f"{path.stem}:trial{trial:02d}:{token}")
            trial += 1
    return _finalize(
        "eegmmi_physionet",
        160.0,
        ("left_fist_imagery", "right_fist_imagery"),
        buckets,
        {
            "dataset_protocol_id": "eegmmi_runs_4_8_12_binary_mi_grouped_v1",
            "runs": [4, 8, 12],
            "channels": 64,
            "trial_seconds": 4.0,
            "native_sampling_rates_hz": [128.0, 160.0],
            "rate_normalization": "S088/S092/S100 128 Hz trials resampled to 160 Hz with scipy.signal.resample_poly(up=5,down=4)",
            "rest_annotations_excluded": True,
            "split_protocol": "seed17 70/15/15 subject grouped",
            "split_groups": {name: sorted(values) for name, values in partitions.items()},
        },
    )


def _prepare_mitbih(root: Path) -> PreparedDataset:
    try:
        import wfdb  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "MIT-BIH preparation requires wfdb; run .venv\\Scripts\\python.exe -m pip install -r requirements.lock"
        ) from error
    development = {
        "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
        "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
    }
    test_records = {
        "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
        "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
    }
    validation = {"106", "114", "208", "223"}
    training = development - validation
    groups = {"train": training, "validation": validation, "test": test_records}
    mapping = {
        **{symbol: 0 for symbol in "NLRenj"},
        **{symbol: 1 for symbol in "AaJS"},
        **{symbol: 2 for symbol in "VE"},
        "F": 3,
        **{symbol: 4 for symbol in "/fQ?"},
    }
    buckets = {name: ([], [], []) for name in groups}
    for split_name, records in groups.items():
        for record_name in sorted(records):
            record_path = root / record_name
            record = wfdb.rdrecord(str(record_path), channels=[0])
            annotation = wfdb.rdann(str(record_path), "atr")
            signal = np.asarray(record.p_signal[:, 0], dtype=np.float32)
            for ordinal, (sample_index, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
                label = mapping.get(str(symbol))
                start = int(sample_index) - 90
                end = int(sample_index) + 166
                if label is None or start < 0 or end > len(signal):
                    continue
                buckets[split_name][0].append(signal[start:end][None, :])
                buckets[split_name][1].append(label)
                buckets[split_name][2].append(f"record{record_name}:beat{ordinal:05d}")
    return _finalize(
        "mitbih_arrhythmia_physionet",
        360.0,
        ("N", "S", "V", "F", "Q"),
        buckets,
        {
            "dataset_protocol_id": "mitbih_aami5_interpatient_v1",
            "lead_policy": "first physical lead",
            "beat_window_samples": [90, 166],
            "excluded_paced_records": ["102", "104", "107", "217"],
            "split_protocol": "canonical DS1/DS2 inter-patient; four frozen DS1 records held out for validation",
            "split_groups": {name: sorted(values) for name, values in groups.items()},
        },
    )


PREPARERS: dict[str, tuple[Path, Callable[[Path], PreparedDataset]]] = {
    "hapt_uci": (Path("data/raw/uci/HAPT"), _prepare_hapt),
    "daily_sports_uci": (Path("data/raw/uci/DailySports"), _prepare_daily_sports),
    "hydraulic_uci": (Path("data/raw/uci/HydraulicSystems"), _prepare_hydraulic),
    "sleep_edfx_physionet": (Path("data/raw/physionet/SleepEDFX"), _prepare_sleep_edfx),
    "eegmmi_physionet": (Path("data/raw/physionet/EEGMMI"), _prepare_eegmmi),
    "mitbih_arrhythmia_physionet": (Path("data/raw/physionet/MITBIHArrhythmia"), _prepare_mitbih),
}


def prepare_full_dataset(
    project_root: str | Path,
    dataset_id: str,
    *,
    force: bool = False,
) -> PreparedDataset:
    root = Path(project_root).resolve()
    if dataset_id not in FULL_DATASETS:
        raise ValueError(f"unknown Full dataset: {dataset_id}")
    if dataset_id in PILOT_DATASETS:
        return prepare_pilot_dataset(root, dataset_id, force=force)
    cache = full_cache_path(root, dataset_id)
    if cache.exists() and not force:
        return load_prepared_dataset(cache)
    relative_root, preparer = PREPARERS[dataset_id]
    dataset = preparer(root / relative_root)
    save_prepared_dataset(cache, dataset)
    manifest = {
        "dataset_id": dataset.dataset_id,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "class_names": list(dataset.class_names),
        "shapes": {
            name: list(getattr(dataset, name).x.shape)
            for name in ("train", "validation", "test")
        },
        "class_counts": {
            name: np.bincount(getattr(dataset, name).y, minlength=len(dataset.class_names)).tolist()
            for name in ("train", "validation", "test")
        },
        "split_id_overlap": {
            "train_validation": bool(set(dataset.train.ids) & set(dataset.validation.ids)),
            "train_test": bool(set(dataset.train.ids) & set(dataset.test.ids)),
            "validation_test": bool(set(dataset.validation.ids) & set(dataset.test.ids)),
        },
        "metadata": dataset.metadata,
    }
    atomic_write_json(cache.with_suffix(".manifest.json"), manifest)
    return dataset
