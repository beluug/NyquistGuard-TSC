"""Read-only V5.1 multi-seed consensus reliability development.

The classifier and all of its predictions remain frozen. Only the post-hoc
score used to rank predictions for selective risk is chosen.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.progress import atomic_write_json, utc_now


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_consensus_mode(
    validation_gains: list[float],
    *,
    minimum_seed_gain: float,
    minimum_mean_gain: float,
    required_positive_fraction: float,
) -> dict[str, Any]:
    """Select observability using validation gains from all matched seeds."""

    values = np.asarray(validation_gains, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("validation gains must be a finite non-empty vector")
    fraction = float(np.mean(values > float(minimum_seed_gain)))
    mean_gain = float(np.mean(values))
    use_observability = (
        fraction >= float(required_positive_fraction)
        and mean_gain > float(minimum_mean_gain)
    )
    return {
        "mode": "observability" if use_observability else "confidence_fallback",
        "validation_gains": values.tolist(),
        "minimum_validation_gain": float(np.min(values)),
        "mean_validation_gain": mean_gain,
        "positive_seed_fraction": fraction,
    }


def _validate_source(source: dict[str, Any], config: dict[str, Any]) -> None:
    expected = config["source"]
    if source.get("status") != expected["required_status"]:
        raise ValueError("V5 benchmark source is not completed")
    if source.get("protocol_hash") != expected["required_protocol_hash"]:
        raise ValueError("V5 benchmark source protocol hash changed")
    if source.get("retrospective_benchmark_only") is not True:
        raise ValueError("V5 source lost its retrospective boundary")
    datasets = tuple(str(value) for value in expected["datasets"])
    seeds = tuple(int(value) for value in expected["seeds"])
    if tuple(source.get("datasets", ())) != datasets:
        raise ValueError("V5 source dataset panel changed")
    if tuple(int(value) for value in source.get("seeds", ())) != seeds:
        raise ValueError("V5 source seed panel changed")
    role = str(expected["required_candidate_role"])
    candidates = source.get("candidate_results", {})
    required = {
        f"{dataset_id}__seed{seed}__{role}"
        for dataset_id in datasets
        for seed in seeds
    }
    if set(candidates) != required:
        raise ValueError("V5 source candidate matrix is incomplete or changed")


def _source_artifacts(root: Path, source: dict[str, Any], source_path: Path) -> list[Path]:
    paths = [source_path]
    run_root = Path(source["run_root"])
    if not run_root.is_absolute():
        run_root = root / run_root
    for key in sorted(source["candidate_results"]):
        role_root = run_root / key
        for name in ("metrics.json", "checkpoint_best.pt"):
            path = role_root / name
            if not path.exists():
                raise FileNotFoundError(f"missing frozen V5 artifact: {path}")
            paths.append(path)
    return paths


def _classification_checks_passed(source: dict[str, Any]) -> bool:
    checks = source.get("decision", {}).get("checks", {})
    required = (
        "average_dataset_unseen_gain",
        "positive_dataset_count",
        "single_dataset_unseen_floor",
        "average_dataset_full_rate_floor",
        "no_constant_prediction",
        "finite_metrics",
    )
    return all(checks.get(name) is True for name in required)


def derive_safe_reliability(
    source: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze modes from validation, then evaluate the already stored test AURC."""

    datasets = tuple(str(value) for value in config["source"]["datasets"])
    seeds = tuple(int(value) for value in config["source"]["seeds"])
    role = str(config["source"]["required_candidate_role"])
    controller = config["controller"]
    results: dict[str, Any] = {}

    # Phase 1: only validation values are passed into the selector.
    modes: dict[str, dict[str, Any]] = {}
    for dataset_id in datasets:
        gains = []
        for seed in seeds:
            candidate = source["candidate_results"][
                f"{dataset_id}__seed{seed}__{role}"
            ]
            validation = candidate["validation"]
            gains.append(
                float(validation["pooled_confidence_aurc"])
                - float(validation["pooled_observability_aurc"])
            )
        modes[dataset_id] = select_consensus_mode(
            gains,
            minimum_seed_gain=float(controller["minimum_seed_validation_aurc_gain"]),
            minimum_mean_gain=float(
                controller["minimum_dataset_mean_validation_aurc_gain"]
            ),
            required_positive_fraction=float(
                controller["required_positive_seed_fraction"]
            ),
        )

    # Phase 2: apply the already frozen dataset mode to stored test summaries.
    for dataset_id in datasets:
        mode_row = modes[dataset_id]
        mode = mode_row["mode"]
        seed_rows = []
        for seed in seeds:
            candidate = source["candidate_results"][
                f"{dataset_id}__seed{seed}__{role}"
            ]
            validation = candidate["validation"]
            test = candidate["test"]
            selected_key = (
                "pooled_observability_aurc"
                if mode == "observability"
                else "pooled_confidence_aurc"
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "mode_selected_from_validation_consensus": mode,
                    "validation_selected_aurc": float(validation[selected_key]),
                    "validation_confidence_aurc": float(
                        validation["pooled_confidence_aurc"]
                    ),
                    "test_selected_aurc": float(test[selected_key]),
                    "test_confidence_aurc": float(test["pooled_confidence_aurc"]),
                    "test_selected_aurc_delta_vs_confidence": float(
                        test[selected_key] - test["pooled_confidence_aurc"]
                    ),
                    "legacy_test_selected_aurc_delta_vs_confidence": float(
                        test["selected_pooled_aurc"]
                        - test["pooled_confidence_aurc"]
                    ),
                }
            )
        results[dataset_id] = {
            "dataset_id": dataset_id,
            **mode_row,
            "seed_rows": seed_rows,
            "mean_test_selected_aurc_delta_vs_confidence": float(
                np.mean(
                    [row["test_selected_aurc_delta_vs_confidence"] for row in seed_rows]
                )
            ),
            "legacy_mean_test_selected_aurc_delta_vs_confidence": float(
                np.mean(
                    [
                        row["legacy_test_selected_aurc_delta_vs_confidence"]
                        for row in seed_rows
                    ]
                )
            ),
        }

    gates = config["development_gates"]
    deltas = [
        float(row["mean_test_selected_aurc_delta_vs_confidence"])
        for row in results.values()
    ]
    finite = all(
        math.isfinite(float(value))
        for row in results.values()
        for value in (
            row["minimum_validation_gain"],
            row["mean_validation_gain"],
            row["positive_seed_fraction"],
            row["mean_test_selected_aurc_delta_vs_confidence"],
        )
    )
    checks = {
        "source_classification": _classification_checks_passed(source)
        if gates["require_source_classification_checks_passed"]
        else True,
        "each_dataset_reliability_safety": all(value <= 1e-12 for value in deltas)
        if gates["require_each_dataset_selected_test_aurc_nonworse_than_confidence"]
        else True,
        "average_dataset_reliability_safety": float(np.mean(deltas))
        <= float(
            gates["maximum_average_dataset_selected_test_aurc_delta_vs_confidence"]
        ),
        "finite_metrics": finite if gates["require_finite_metrics"] else True,
    }
    decision = {
        "passed_before_artifact_recheck": all(checks.values()),
        "checks": checks,
        "activated_dataset_count": sum(
            row["mode"] == "observability" for row in results.values()
        ),
        "average_dataset_selected_test_aurc_delta_vs_confidence": float(
            np.mean(deltas)
        ),
        "legacy_average_dataset_selected_test_aurc_delta_vs_confidence": float(
            np.mean(
                [
                    row["legacy_mean_test_selected_aurc_delta_vs_confidence"]
                    for row in results.values()
                ]
            )
        ),
    }
    return results, decision


def run_v5_safe_reliability_development(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v5_safe_reliability_development.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = root / config["source"]["report"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _validate_source(source, config)

    artifacts = _source_artifacts(root, source, source_path)
    hashes_before = {str(path): _sha256(path) for path in artifacts}
    results, decision = derive_safe_reliability(source, config)
    hashes_after = {str(path): _sha256(path) for path in artifacts}
    unchanged = hashes_before == hashes_after
    decision["checks"]["classification_artifacts_byte_unchanged"] = (
        unchanged
        if config["development_gates"][
            "require_classification_artifacts_byte_unchanged"
        ]
        else True
    )
    decision["passed"] = all(decision["checks"].values())

    elapsed = time.monotonic() - started
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / "v5_safe_reliability_development" / (
        f"v5_1_safe_reliability__{stamp}"
    )
    report = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "source_report": str(source_path),
        "source_protocol_hash": source["protocol_hash"],
        "primary_selection_split": "validation_only_across_matched_seeds",
        "test_role": "retrospective_development_evaluation_after_mode_freeze",
        "classification_model_trained": False,
        "classification_logits_or_predictions_changed": False,
        "checkpoint_written": False,
        "source_artifacts_byte_unchanged": unchanged,
        "source_artifact_hashes": hashes_after,
        "retrospective_only": True,
        "independent_confirmation_claim_allowed": False,
        "new_untouched_datasets_still_required": int(
            config["scientific_boundary"]["new_untouched_datasets_still_required"]
        ),
        "elapsed_seconds": elapsed,
        "controller": config["controller"],
        "dataset_results": results,
        "decision": decision,
        "run_root": str(run_root),
        "later_stage_started": False,
        "finished_at_utc": utc_now(),
    }
    lines = [
        "# V5.1 multi-seed consensus reliability development",
        "",
        f"- Retrospective development decision: **{'PASS' if decision['passed'] else 'FAIL'}**.",
        "- V5 checkpoints, logits, predicted classes, and classification metrics were not changed.",
        "- Reliability mode was selected from validation gains across all three matched seeds.",
        "- Test was evaluated only after each dataset mode was fixed; this is not independent confirmation.",
        f"- Read-only artifact verification: **{'PASS' if unchanged else 'FAIL'}**.",
        f"- Elapsed: `{elapsed:.2f} s`.",
        "",
        "| Dataset | Mode | Min validation gain | Mean validation gain | Positive seeds | Safe test AURC delta | Legacy delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, row in results.items():
        lines.append(
            f"| {dataset_id} | {row['mode']} | {row['minimum_validation_gain']:+.4f} | "
            f"{row['mean_validation_gain']:+.4f} | {row['positive_seed_fraction'] * 3:.0f}/3 | "
            f"{row['mean_test_selected_aurc_delta_vs_confidence']:+.4f} | "
            f"{row['legacy_mean_test_selected_aurc_delta_vs_confidence']:+.4f} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in decision["checks"].items()
    )
    lines.extend(
        [
            "",
            "This pass freezes a safer controller for future testing; it does not convert the reused datasets into independent evidence.",
            "No later experiment was started automatically.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    atomic_write_json(run_root / "report.json", report)
    _atomic_write_text(run_root / "report.md", markdown)
    atomic_write_json(root / "reports" / "v5_safe_reliability_development_report.json", report)
    _atomic_write_text(root / "reports" / "v5_safe_reliability_development_report.md", markdown)
    return report
