"""Read-only audit, statistics, tables, and figures for V4.1 confirmation.

This module never loads raw arrays or model checkpoints.  It consumes only
frozen JSON metrics and training histories already written by the formal
runner.  Dataset is always the primary replication unit; rates and seeds are
never treated as independent datasets.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nyquistguard.data.new_confirmation_datasets import CONFIRMATION_DATASETS
from nyquistguard.research.v4_new_dataset_confirmation import (
    CONFIRMATION_ROLES,
    CONFIRMATION_SEEDS,
)


RATE_IDS = ("r1000", "r0900", "r0600", "r0400", "r0300")
UNSEEN_RATE_IDS = ("r0900", "r0600", "r0400", "r0300")
RATE_RATIOS = {"r1000": 1.0, "r0900": 0.9, "r0600": 0.6, "r0400": 0.4, "r0300": 0.3}
DISPLAY_NAMES = {
    "character_trajectories_uea": "CharacterTrajectories",
    "motor_imagery_uea": "MotorImagery",
    "wisdm_activity_uci": "WISDM",
    "ptbxl_physionet": "PTB-XL",
}


class ConfirmationArtifactError(RuntimeError):
    """Raised when frozen confirmation artifacts are absent or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmationArtifactError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfirmationArtifactError(f"expected JSON object in {path}")
    return value


def _latest_run_root(project_root: Path) -> Path:
    parent = project_root / "runs" / "v4_new_dataset_confirmation"
    candidates = sorted(parent.glob("v4_1_confirmation__4datasets__3seeds__*"), reverse=True)
    if not candidates:
        raise ConfirmationArtifactError("no V4.1 confirmation run directory exists")
    return candidates[0]


def _expected_keys() -> tuple[str, ...]:
    return tuple(
        f"{dataset_id}__seed{seed}__{role}"
        for dataset_id in CONFIRMATION_DATASETS
        for seed in CONFIRMATION_SEEDS
        for role in CONFIRMATION_ROLES
    )


def audit_confirmation(project_root: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Audit the formal matrix without loading checkpoints or raw/test arrays."""

    root = Path(project_root).resolve()
    run_root = _latest_run_root(root)
    manifest = _read_json(run_root / "manifest.json")
    expected = _expected_keys()
    discovered: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for key in expected:
        path = run_root / key / "metrics.json"
        if not path.exists():
            continue
        try:
            metric = _read_json(path)
        except ConfirmationArtifactError as error:
            errors.append(str(error))
            continue
        discovered[key] = metric
        dataset_id, seed_text, role = key.split("__", 2)
        expected_seed = int(seed_text.removeprefix("seed"))
        if metric.get("dataset_id") != dataset_id or int(metric.get("seed", -1)) != expected_seed:
            errors.append(f"identity mismatch in {path}")
        if metric.get("role") != role or metric.get("status") != "completed":
            errors.append(f"role/status mismatch in {path}")
        if metric.get("protocol_hash") != manifest.get("protocol_hash"):
            errors.append(f"protocol hash mismatch in {path}")
        if metric.get("test_accessed") is not True:
            errors.append(f"formal metrics lack test-access marker in {path}")
        if metric.get("test_used_for_checkpoint_or_threshold_selection") is not False:
            errors.append(f"test selection boundary violated in {path}")
        for split_name in ("validation", "test"):
            evaluation = metric.get(split_name, {})
            if tuple(evaluation.get("per_rate", {}).keys()) != RATE_IDS:
                errors.append(f"rate grid/order mismatch in {path}:{split_name}")
            for rate_id in RATE_IDS:
                row = evaluation.get("per_rate", {}).get(rate_id, {})
                if not math.isfinite(float(row.get("macro_f1", math.nan))):
                    errors.append(f"non-finite macro-F1 in {path}:{split_name}:{rate_id}")
        if role == "v4_1_residual_gate":
            validation_mode = metric.get("validation", {}).get("reliability_mode")
            test_mode = metric.get("test", {}).get("reliability_mode_selected_on_validation")
            if validation_mode not in {"observability", "confidence_fallback"} or test_mode != validation_mode:
                errors.append(f"reliability mode was not carried from validation in {path}")
            floors = metric.get("validation", {}).get("learned_gate_floor", [])
            if not floors or not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in floors):
                errors.append(f"invalid learned gate floors in {path}")
    missing = sorted(set(expected) - set(discovered))
    report_path = run_root / "report.json"
    report = _read_json(report_path) if report_path.exists() else None
    if require_complete:
        if missing:
            errors.append(f"missing {len(missing)} of 24 metrics files")
        if manifest.get("status") != "completed":
            errors.append("manifest is not completed")
        if report is None or report.get("status") != "completed":
            errors.append("completed report.json is absent")
        elif set(report.get("role_results", {})) != set(expected):
            errors.append("report role_results matrix is incomplete")
    result = {
        "status": "pass" if not errors else "fail",
        "require_complete": require_complete,
        "run_root": str(run_root),
        "manifest_status": manifest.get("status"),
        "protocol_hash": manifest.get("protocol_hash"),
        "completed_metrics": len(discovered),
        "expected_metrics": len(expected),
        "missing_keys": missing,
        "errors": errors,
        "raw_arrays_loaded": False,
        "checkpoints_loaded": False,
    }
    if require_complete and errors:
        raise ConfirmationArtifactError("; ".join(errors))
    return result


def exact_sign_flip_test(values: Iterable[float]) -> dict[str, Any]:
    """Exact one-sided sign-flip sensitivity test for a paired mean effect."""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("sign-flip values must be a finite non-empty vector")
    observed = float(np.mean(array))
    permuted = np.asarray([
        np.mean(array * np.asarray(signs, dtype=np.float64))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ])
    p_greater = float(np.mean(permuted >= observed - 1e-15))
    return {
        "n_primary_units": int(len(array)),
        "observed_mean": observed,
        "one_sided_p_greater": p_greater,
        "permutation_count": int(len(permuted)),
        "minimum_attainable_one_sided_p": float(1.0 / len(permuted)),
        "primary_unit": "dataset",
        "inferential_role": "descriptive_sensitivity_only_not_frozen_gate",
    }


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 20260828, draws: int = 20000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def _completed_report(project_root: Path) -> tuple[Path, dict[str, Any]]:
    run_root = _latest_run_root(project_root)
    report = _read_json(run_root / "report.json")
    if report.get("status") != "completed":
        raise ConfirmationArtifactError("formal report exists but is not completed")
    return run_root, report


def _paired_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    role_results = report["role_results"]
    rows: list[dict[str, Any]] = []
    for dataset_id in CONFIRMATION_DATASETS:
        for seed in CONFIRMATION_SEEDS:
            pair = f"{dataset_id}__seed{seed}"
            hard = role_results[f"{pair}__v3_10_hard_gate"]["test"]
            candidate = role_results[f"{pair}__v4_1_residual_gate"]["test"]
            hard_unseen = float(np.mean([hard["per_rate"][key]["macro_f1"] for key in UNSEEN_RATE_IDS]))
            candidate_unseen = float(np.mean([candidate["per_rate"][key]["macro_f1"] for key in UNSEEN_RATE_IDS]))
            rows.append({
                "dataset_id": dataset_id,
                "dataset": DISPLAY_NAMES[dataset_id],
                "seed": seed,
                "hard_full_rate_macro_f1": float(hard["per_rate"]["r1000"]["macro_f1"]),
                "candidate_full_rate_macro_f1": float(candidate["per_rate"]["r1000"]["macro_f1"]),
                "full_rate_macro_f1_delta": float(
                    candidate["per_rate"]["r1000"]["macro_f1"] - hard["per_rate"]["r1000"]["macro_f1"]
                ),
                "hard_mean_unseen_macro_f1": hard_unseen,
                "candidate_mean_unseen_macro_f1": candidate_unseen,
                "mean_unseen_macro_f1_delta": candidate_unseen - hard_unseen,
                "candidate_confidence_aurc": float(candidate["pooled_confidence_aurc"]),
                "candidate_selected_aurc": float(candidate["selected_pooled_aurc"]),
                "selected_aurc_delta_vs_confidence": float(
                    candidate["selected_pooled_aurc"] - candidate["pooled_confidence_aurc"]
                ),
                "reliability_mode_selected_on_validation": candidate["reliability_mode_selected_on_validation"],
            })
    return rows


def _rate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in CONFIRMATION_DATASETS:
        for seed in CONFIRMATION_SEEDS:
            for role in CONFIRMATION_ROLES:
                result = report["role_results"][f"{dataset_id}__seed{seed}__{role}"]
                for split_name in ("validation", "test"):
                    for rate_id in RATE_IDS:
                        rate = result[split_name]["per_rate"][rate_id]
                        rows.append({
                            "dataset_id": dataset_id,
                            "dataset": DISPLAY_NAMES[dataset_id],
                            "seed": seed,
                            "role": role,
                            "split": split_name,
                            "rate_id": rate_id,
                            "rate_ratio": RATE_RATIOS[rate_id],
                            "macro_f1": float(rate["macro_f1"]),
                            "confidence_aurc": float(rate["confidence_aurc"]),
                            "observability_aurc": float(rate["observability_aurc"]),
                            "relative_gate_mass": float(rate["relative_gate_mass"]),
                        })
    return rows


def _dataset_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset_id in CONFIRMATION_DATASETS:
        subset = [row for row in paired_rows if row["dataset_id"] == dataset_id]
        unseen = np.asarray([row["mean_unseen_macro_f1_delta"] for row in subset])
        full = np.asarray([row["full_rate_macro_f1_delta"] for row in subset])
        reliability = np.asarray([row["selected_aurc_delta_vs_confidence"] for row in subset])
        output.append({
            "dataset_id": dataset_id,
            "dataset": DISPLAY_NAMES[dataset_id],
            "seed_count": len(subset),
            "mean_unseen_macro_f1_delta": float(unseen.mean()),
            "sd_unseen_macro_f1_delta_across_seeds": float(unseen.std(ddof=1)),
            "minimum_seed_unseen_macro_f1_delta": float(unseen.min()),
            "maximum_seed_unseen_macro_f1_delta": float(unseen.max()),
            "positive_seed_count": int(np.sum(unseen > 0.0)),
            "mean_full_rate_macro_f1_delta": float(full.mean()),
            "mean_selected_aurc_delta_vs_confidence": float(reliability.mean()),
        })
    return output


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ConfirmationArtifactError(f"cannot write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _planned_outputs(root: Path) -> list[Path]:
    return [
        root / "reports" / "v4_confirmation_paired_seed_results.csv",
        root / "reports" / "v4_confirmation_rate_results.csv",
        root / "reports" / "v4_confirmation_dataset_summary.csv",
        root / "reports" / "v4_confirmation_statistical_summary.json",
        root / "reports" / "v4_confirmation_statistical_summary.md",
        root / "reports" / "v4_confirmation_dataset_deltas.png",
        root / "reports" / "v4_confirmation_dataset_deltas.pdf",
        root / "reports" / "v4_confirmation_rate_curves.png",
        root / "reports" / "v4_confirmation_rate_curves.pdf",
        root / "reports" / "v4_confirmation_figure_manifest.json",
        root / "reports" / "v4_confirmation_manuscript_block.md",
    ]


def _save_figure(fig: Any, path: Path, *, dpi: int = 450) -> None:
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, format=path.suffix.lstrip("."), dpi=dpi, facecolor="white")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _figures(root: Path, paired: list[dict[str, Any]], rate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    blue, orange = "#0072B2", "#D55E00"
    fig, ax = plt.subplots(figsize=(180 / 25.4, 82 / 25.4), layout="constrained")
    offsets = (-0.09, 0.0, 0.09)
    all_values: list[float] = [0.0]
    for index, dataset_id in enumerate(CONFIRMATION_DATASETS):
        subset = [row for row in paired if row["dataset_id"] == dataset_id]
        values = [float(row["mean_unseen_macro_f1_delta"]) for row in subset]
        all_values.extend(values)
        for offset, row in zip(offsets, subset):
            ax.scatter(index + offset, row["mean_unseen_macro_f1_delta"], s=28, color=orange,
                       marker="o", edgecolor="black", linewidth=0.35, zorder=3)
        ax.scatter(index, float(np.mean(values)), s=55, color="black", marker="D", zorder=4)
    padding = max(0.02, 0.12 * (max(all_values) - min(all_values) or 0.1))
    ax.set_ylim(min(all_values) - padding, max(all_values) + padding)
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(len(CONFIRMATION_DATASETS)), [DISPLAY_NAMES[key] for key in CONFIRMATION_DATASETS])
    ax.set_ylabel("V4.1 − hard gate: mean unseen-rate macro-F1")
    ax.set_title("Dataset-level confirmation effects (circles: seeds; diamonds: dataset means)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    delta_png = root / "reports" / "v4_confirmation_dataset_deltas.png"
    delta_pdf = root / "reports" / "v4_confirmation_dataset_deltas.pdf"
    _save_figure(fig, delta_png)
    _save_figure(fig, delta_pdf)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 135 / 25.4), sharex=True, sharey=True, layout="constrained")
    ordered_rates = (0.3, 0.4, 0.6, 0.9, 1.0)
    for ax, dataset_id in zip(axes.flat, CONFIRMATION_DATASETS):
        for role, color, style, marker, label in (
            ("v3_10_hard_gate", blue, "--", "s", "Hard gate"),
            ("v4_1_residual_gate", orange, "-", "o", "V4.1 residual gate"),
        ):
            seed_series = []
            for seed in CONFIRMATION_SEEDS:
                series = [next(
                    row["macro_f1"] for row in rate_rows
                    if row["dataset_id"] == dataset_id and row["seed"] == seed
                    and row["role"] == role and row["split"] == "test"
                    and row["rate_ratio"] == ratio
                ) for ratio in ordered_rates]
                seed_series.append(series)
                ax.plot(ordered_rates, series, color=color, linestyle=style, linewidth=0.65, alpha=0.28)
            mean = np.mean(np.asarray(seed_series), axis=0)
            ax.plot(ordered_rates, mean, color=color, linestyle=style, marker=marker,
                    linewidth=1.8, markersize=4.2, label=label)
        ax.set_title(DISPLAY_NAMES[dataset_id])
        ax.set_ylim(0.0, 1.0)
        ax.grid(color="#E1E1E1", linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("Sampling-rate ratio")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test macro-F1")
    axes[0, 0].legend(frameon=False, loc="best")
    rate_png = root / "reports" / "v4_confirmation_rate_curves.png"
    rate_pdf = root / "reports" / "v4_confirmation_rate_curves.pdf"
    _save_figure(fig, rate_png)
    _save_figure(fig, rate_pdf)
    plt.close(fig)
    return {
        "matplotlib_version": matplotlib.__version__,
        "palette": {"hard_gate": blue, "v4_1": orange},
        "redundant_encoding": "color + line style + marker",
        "dataset_delta_alt_text": (
            "Dot plot of V4.1 minus hard-gate mean unseen-rate macro-F1 for four datasets. "
            "Three circles show seeds and a black diamond shows each dataset mean; a dashed line marks zero."
        ),
        "rate_curve_alt_text": (
            "Four-panel line chart of test macro-F1 across sampling-rate ratios for hard gate and V4.1. "
            "Thin lines are seeds and thick marked lines are seed means; all panels use a common zero-to-one scale."
        ),
        "figure_width_mm": 180,
        "png_dpi": 450,
        "publisher_profile": "provisional_general_not_journal_specific",
    }


def build_confirmation_artifacts(project_root: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Build manuscript-neutral tables/figures only after the formal report completes."""

    root = Path(project_root).resolve()
    audit = audit_confirmation(root, require_complete=True)
    run_root, report = _completed_report(root)
    outputs = _planned_outputs(root)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError("refusing to overwrite existing artifacts; pass force=True: " + ", ".join(map(str, existing)))
    paired = _paired_rows(report)
    rates = _rate_rows(report)
    datasets = _dataset_rows(paired)
    primary_values = np.asarray([row["mean_unseen_macro_f1_delta"] for row in datasets])
    ci_lower, ci_upper = _bootstrap_mean_ci(primary_values)
    sign_flip = exact_sign_flip_test(primary_values)
    statistics = {
        "status": "completed",
        "frozen_decision": report["decision"],
        "primary_unit": "dataset",
        "n_primary_units": 4,
        "seed_role": "within-dataset matched repetitions; not independent primary units",
        "rate_role": "repeated conditions; not independent primary units",
        "mean_dataset_unseen_macro_f1_delta": float(primary_values.mean()),
        "median_dataset_unseen_macro_f1_delta": float(np.median(primary_values)),
        "minimum_dataset_unseen_macro_f1_delta": float(primary_values.min()),
        "maximum_dataset_unseen_macro_f1_delta": float(primary_values.max()),
        "positive_dataset_count": int(np.sum(primary_values > 0.0)),
        "descriptive_dataset_bootstrap_95_percentile_ci": [ci_lower, ci_upper],
        "bootstrap_seed": 20260828,
        "bootstrap_draws": 20000,
        "exact_sign_flip_sensitivity": sign_flip,
        "warning": (
            "With four primary datasets the minimum attainable one-sided exact p-value is 0.0625; "
            "this sensitivity analysis cannot yield p<0.05 and does not replace the frozen gates."
        ),
    }
    _write_csv(outputs[0], paired)
    _write_csv(outputs[1], rates)
    _write_csv(outputs[2], datasets)
    _atomic_json(outputs[3], statistics)
    decision_text = "PASS" if report["decision"]["passed"] else "FAIL"
    markdown = "\n".join([
        "# V4.1 four-dataset confirmation: statistical summary", "",
        f"- Frozen protocol decision: **{decision_text}**.",
        "- Primary unit: dataset (n=4); three seeds are matched within-dataset repetitions.",
        f"- Mean dataset unseen-rate macro-F1 delta: `{primary_values.mean():+.4f}`.",
        f"- Dataset range: `{primary_values.min():+.4f}` to `{primary_values.max():+.4f}`; positive datasets: `{int(np.sum(primary_values > 0))}/4`.",
        f"- Descriptive dataset-bootstrap 95% percentile interval: `[{ci_lower:+.4f}, {ci_upper:+.4f}]` (20,000 draws; seed 20260828).",
        f"- Exact one-sided sign-flip sensitivity p: `{sign_flip['one_sided_p_greater']:.4f}` over all 16 sign assignments.",
        "- Important: with n=4 datasets, the minimum attainable exact one-sided p is 0.0625. The p-value is descriptive sensitivity evidence, not the frozen decision rule.",
        "- Rates and seeds were not counted as independent samples.", "",
    ])
    _atomic_text(outputs[4], markdown)
    figure_meta = _figures(root, paired, rates)
    source_bytes = (run_root / "report.json").read_bytes()
    figure_manifest = {
        "source_report": str(run_root / "report.json"),
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_tables": [str(outputs[0]), str(outputs[1]), str(outputs[2])],
        "transformations": [
            "seed-level paired candidate-minus-hard deltas",
            "dataset mean across three frozen seeds",
            "rate curves show individual seeds and arithmetic seed mean",
        ],
        "missing_data": "none allowed; build aborts on incomplete 24-run matrix",
        "uncertainty": "raw three-seed lines/points; no inferential seed-level error bars",
        **figure_meta,
    }
    _atomic_json(outputs[9], figure_manifest)
    if report["decision"]["passed"]:
        manuscript = "\n".join([
            "# Provisional V4.1 confirmation manuscript block", "",
            "The frozen four-dataset confirmation supported the V4.1 residual-gate candidate under the preregistered directional gates. "
            f"Across the four primary dataset units, the mean change in macro-F1 over unseen sampling-rate ratios was {primary_values.mean():+.4f} relative to the matched hard-gate control "
            f"({int(np.sum(primary_values > 0))}/4 datasets positive; range {primary_values.min():+.4f} to {primary_values.max():+.4f}). "
            "Each dataset estimate averaged three matched seeds; neither seeds nor rate conditions were treated as independent datasets. "
            "The exact sign-flip analysis is reported only as a small-n sensitivity analysis because four datasets permit a minimum one-sided p value of 0.0625.", "",
            "Author review is required before insertion; journal-specific wording and figure requirements remain pending.", "",
        ])
    else:
        manuscript = "\n".join([
            "# V4.1 confirmation manuscript status", "",
            "The frozen four-dataset confirmation did not pass every preregistered gate. "
            "No positive-performance manuscript paragraph was generated automatically. "
            "Use the complete tables and figures for internal evidence review without tuning on test results.", "",
        ])
    _atomic_text(outputs[10], manuscript)
    return {
        "status": "completed", "audit": audit, "statistics": statistics,
        "outputs": [str(path) for path in outputs],
        "model_or_checkpoint_modified": False, "raw_or_test_arrays_loaded": False,
    }


def _constant_prediction_macro_f1(class_counts: Iterable[int]) -> float:
    counts = np.asarray(tuple(class_counts), dtype=np.float64)
    total = float(counts.sum())
    if total <= 0 or len(counts) == 0:
        return math.nan
    scores = (2.0 * counts / (total + counts)) / len(counts)
    return float(scores.max())


def summarize_training_health(project_root: str | Path) -> list[dict[str, Any]]:
    """Summarize train/validation histories only; never inspect test metrics."""

    root = Path(project_root).resolve()
    run_root = _latest_run_root(root)
    rows: list[dict[str, Any]] = []
    for key in _expected_keys():
        history_path = run_root / key / "training_history.json"
        if not history_path.exists():
            continue
        payload = _read_json(history_path)
        history = payload.get("history", [])
        if not history:
            continue
        dataset_id, seed_text, role = key.split("__", 2)
        manifest_path = root / "data" / "processed" / "v4_confirmation_v1" / f"{dataset_id}__development.manifest.json"
        data_manifest = _read_json(manifest_path)
        validation_counts = data_manifest["class_counts"]["validation"]
        constant_floor = _constant_prediction_macro_f1(validation_counts)
        val_scores = np.asarray([float(row["validation_selection_score"]) for row in history])
        losses = np.asarray([float(row["train_loss"]) for row in history])
        best_val = float(np.max(val_scores))
        class_count = len(validation_counts)
        rows.append({
            "dataset_id": dataset_id,
            "dataset": DISPLAY_NAMES[dataset_id],
            "seed": int(seed_text.removeprefix("seed")),
            "role": role,
            "epochs_completed": len(history),
            "initial_train_loss": float(losses[0]),
            "final_train_loss": float(losses[-1]),
            "best_validation_selection_score": best_val,
            "unique_validation_scores": int(len(np.unique(np.round(val_scores, 12)))),
            "best_constant_prediction_macro_f1": constant_floor,
            "chance_cross_entropy_log_classes": float(math.log(class_count)),
            "constant_prediction_pattern": bool(abs(best_val - constant_floor) <= 1e-6),
            "note": "internal training QC; not a confirmatory endpoint and never uses test metrics",
        })
    return rows


def package_confirmation_supplement(project_root: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Create a code/results reproducibility ZIP without raw data or checkpoints."""

    root = Path(project_root).resolve()
    audit = audit_confirmation(root, require_complete=True)
    required = [
        "configs/experiments/v4_new_dataset_confirmation.yaml",
        "configs/experiments/v4_new_dataset_confirmation_selection.yaml",
        "docs/V4_NEW_DATASET_CONFIRMATION_SELECTION.md",
        "src/nyquistguard/data/new_confirmation_datasets.py",
        "src/nyquistguard/research/v4_residual_gate.py",
        "src/nyquistguard/research/v4_new_dataset_confirmation.py",
        "src/nyquistguard/analysis/v4_confirmation_artifacts.py",
        "scripts/audit_v4_confirmation.py",
        "scripts/build_v4_confirmation_artifacts.py",
        "scripts/summarize_v4_training_health.py",
        "scripts/v4_confirmation_status.py",
        "scripts/package_v4_confirmation_supplement.py",
        "scripts/run_v4_confirmation_postrun.py",
        "reports/v4_new_dataset_confirmation_report.json",
        "reports/v4_new_dataset_confirmation_report.md",
        "reports/v4_confirmation_paired_seed_results.csv",
        "reports/v4_confirmation_rate_results.csv",
        "reports/v4_confirmation_dataset_summary.csv",
        "reports/v4_confirmation_statistical_summary.json",
        "reports/v4_confirmation_statistical_summary.md",
        "reports/v4_confirmation_dataset_deltas.png",
        "reports/v4_confirmation_dataset_deltas.pdf",
        "reports/v4_confirmation_rate_curves.png",
        "reports/v4_confirmation_rate_curves.pdf",
        "reports/v4_confirmation_figure_manifest.json",
        "reports/v4_confirmation_manuscript_block.md",
    ]
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise ConfirmationArtifactError(
            "post-run artifacts must be built before packaging; missing: " + ", ".join(missing)
        )
    destination = root / "reports" / "v4_confirmation_reproducibility_bundle.zip"
    if destination.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {destination}; pass force=True")
    file_manifest = {
        relative: {
            "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in required
    }
    readme = "\n".join([
        "# NyquistGuard-TSC V4.1 confirmation reproducibility bundle", "",
        "This bundle contains frozen protocol files, analysis code, aggregate metrics, tables, and figures.",
        "It intentionally excludes raw datasets, processed arrays, predictions, checkpoints, and patient-level records.",
        "Dataset is the primary statistical unit; seeds and sampling-rate conditions are repeated measurements.",
        "See SHA256SUMS.json for the exact included-file manifest.", "",
    ])
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative in required:
                archive.write(root / relative, arcname=relative.replace("\\", "/"))
            archive.writestr("SUPPLEMENT_README.md", readme)
            archive.writestr(
                "SHA256SUMS.json",
                json.dumps({"files": file_manifest, "audit": audit}, ensure_ascii=False, indent=2),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "completed", "path": str(destination),
        "included_file_count": len(required), "raw_data_included": False,
        "processed_arrays_included": False, "checkpoints_included": False,
    }
