"""Resumable 30-run V5.1 extension of the frozen ten-dataset Full benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nyquistguard.data.full_datasets import FULL_DATASETS, prepare_full_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.full import FULL_METHODS, FULL_SEEDS, _holm
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.research.v4_observe_only_micro import (
    _evaluate_validation,
    _new_model,
    _train_variant,
)
from nyquistguard.research.v5_dual_path_micro import _paired_initial_states
from nyquistguard.research.v5_independent_confirmation import _view
from nyquistguard.research.v5_safe_reliability import select_consensus_mode


CANDIDATE = "v5_1_safe_dual_path"


def v5_full_extension_tasks() -> list[str]:
    tasks = ["Validate frozen V5.1 extension and reused 210-run Full source"]
    tasks.extend(
        f"Train/validate {dataset_id} seed{seed} V5.1"
        for dataset_id in FULL_DATASETS
        for seed in FULL_SEEDS
    )
    tasks.append("Freeze ten reliability modes from validation only")
    tasks.extend(
        f"Evaluate TEST {dataset_id} seed{seed} V5.1"
        for dataset_id in FULL_DATASETS
        for seed in FULL_SEEDS
    )
    tasks.append("Aggregate V5.1 with reused Full baselines and write report")
    return tasks


def _assert_no_active_other_stage(root: Path) -> None:
    path = root / "runs" / "dashboard_status.json"
    if not path.exists():
        return
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if state.get("status") == "running" and state.get("stage") != "v5_1_full_extension":
        raise RuntimeError(f"refusing to start while {state.get('stage', 'another stage')} is running")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_hash(root: Path, config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        config_path,
        base_path,
        root / "configs" / "experiments" / "full.yaml",
        root / "src" / "nyquistguard" / "data" / "full_datasets.py",
        root / "src" / "nyquistguard" / "research" / "v5_dual_path.py",
        root / "src" / "nyquistguard" / "research" / "v5_dual_path_micro.py",
        root / "src" / "nyquistguard" / "research" / "v5_safe_reliability.py",
        root / "src" / "nyquistguard" / "research" / "v4_observe_only_micro.py",
        Path(__file__),
    ):
        digest.update(path.read_bytes())
    digest.update(b"v5.1-frozen-full-extension-30-candidate-runs-v1")
    return digest.hexdigest()


def _validate_sources(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    design = config["design"]
    if tuple(design["datasets"]) != FULL_DATASETS:
        raise ValueError("V5.1 extension datasets differ from frozen Full")
    if tuple(int(value) for value in design["seeds"]) != FULL_SEEDS:
        raise ValueError("V5.1 extension seeds differ from frozen Full")
    if int(config["scientific_boundaries"]["train_only_candidate_runs"]) != 30:
        raise ValueError("V5.1 extension must train exactly 30 candidate runs")
    source = json.loads((root / config["source"]["full_report"]).read_text(encoding="utf-8"))
    if source.get("status") != "completed":
        raise ValueError("source Full is not completed")
    if source.get("protocol_hash") != config["source"]["required_full_protocol_hash"]:
        raise ValueError("source Full protocol hash changed")
    if int(source.get("completed_runs", -1)) != int(config["source"]["required_full_completed_runs"]):
        raise ValueError("source Full must contain 210 completed runs")
    rows = source.get("aggregate", {}).get("rows", [])
    keys = {(row["dataset_id"], row["method"], int(row["seed"])) for row in rows}
    expected = {(dataset, method, seed) for dataset in FULL_DATASETS for method in FULL_METHODS for seed in FULL_SEEDS}
    if keys != expected:
        raise ValueError("source Full aggregate matrix is incomplete or changed")
    independent = json.loads(
        (root / config["source"]["independent_confirmation_report"]).read_text(encoding="utf-8")
    )
    if independent.get("protocol_hash") != config["source"]["required_independent_protocol_hash"]:
        raise ValueError("V5.1 independent confirmation protocol hash changed")
    if bool(independent.get("decision", {}).get("passed")) is not bool(
        config["source"]["require_independent_pass"]
    ):
        raise ValueError("V5.1 independent confirmation did not pass")
    return source


def _find_compatible_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for path in sorted(parent.glob("v5_1_full_extension__10datasets__3seeds__*"), reverse=True):
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("protocol_hash") == protocol_hash:
            return path
    return None


def _freeze_modes(
    run_root: Path,
    protocol_hash: str,
    validations: dict[str, dict[str, Any]],
    controller: dict[str, Any],
) -> dict[str, Any]:
    path = run_root / "reliability_modes_frozen.json"
    hashes = {
        str(item.relative_to(run_root)): _sha256(item)
        for item in sorted(run_root.glob("*__seed*/validation_metrics.json"))
        + sorted(run_root.glob("*__seed*/checkpoint_best.pt"))
    }
    if len(hashes) != 60:
        raise RuntimeError(f"expected 60 frozen validation artifacts, found {len(hashes)}")
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("protocol_hash") != protocol_hash or frozen.get("source_hashes") != hashes:
            raise RuntimeError("frozen reliability modes or validation artifacts changed")
        return frozen
    modes = {}
    for dataset_id in FULL_DATASETS:
        gains = []
        for seed in FULL_SEEDS:
            validation = validations[f"{dataset_id}__seed{seed}"]["validation"]
            gains.append(
                float(validation["pooled_confidence_aurc"])
                - float(validation["pooled_observability_aurc"])
            )
        modes[dataset_id] = select_consensus_mode(
            gains,
            minimum_seed_gain=float(controller["minimum_seed_validation_aurc_gain"]),
            minimum_mean_gain=float(controller["minimum_dataset_mean_validation_aurc_gain"]),
            required_positive_fraction=float(controller["required_positive_seed_fraction"]),
        )
    frozen = {
        "status": "frozen_from_validation_before_extension_test_evaluation",
        "protocol_hash": protocol_hash,
        "modes": modes,
        "source_hashes": hashes,
        "frozen_at_utc": utc_now(),
    }
    atomic_write_json(path, frozen)
    return frozen


def _candidate_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset_id in FULL_DATASETS:
        for seed in FULL_SEEDS:
            payload = results[f"{dataset_id}__seed{seed}"]
            test = payload["test"]
            rows.append({
                "dataset_id": dataset_id,
                "method": CANDIDATE,
                "seed": seed,
                "mean_unseen_macro_f1": float(test["mean_unseen_macro_f1"]),
                "worst_unseen_macro_f1": float(test["worst_unseen_macro_f1"]),
                "full_rate_macro_f1": float(test["full_rate_macro_f1"]),
                "selected_pooled_aurc": float(test["selected_pooled_aurc"]),
                "selected_aurc_delta_vs_confidence": float(
                    test["selected_pooled_aurc"] - test["pooled_confidence_aurc"]
                ),
                "minimum_prediction_class_count": min(
                    int(row["prediction_class_count"]) for row in test["per_rate"].values()
                ),
                "duration_seconds": float(payload["duration_seconds"]),
            })
    return rows


def _paired_statistics(
    candidate_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    by_candidate = {(row["dataset_id"], int(row["seed"])): row for row in candidate_rows}
    by_source = {
        (row["dataset_id"], row["method"], int(row["seed"])): row for row in source_rows
    }
    raw_p: dict[str, float] = {}
    payload: dict[str, Any] = {}
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    draws = int(config["statistics"]["bootstrap_resamples"])
    for baseline in FULL_METHODS:
        dataset_deltas = []
        for dataset_id in FULL_DATASETS:
            values = [
                float(by_candidate[(dataset_id, seed)]["mean_unseen_macro_f1"])
                - float(by_source[(dataset_id, baseline, seed)]["mean_unseen_macro_f1"])
                for seed in FULL_SEEDS
            ]
            dataset_deltas.append(float(np.mean(values)))
        raw_p[baseline] = 1.0 if np.allclose(dataset_deltas, 0.0) else float(
            wilcoxon(dataset_deltas, alternative="two-sided", method="auto").pvalue
        )
        bootstrap = np.mean(
            rng.choice(dataset_deltas, size=(draws, len(dataset_deltas)), replace=True), axis=1
        )
        payload[baseline] = {
            "mean_delta": float(np.mean(dataset_deltas)),
            "median_delta": float(np.median(dataset_deltas)),
            "dataset_deltas": dict(zip(FULL_DATASETS, dataset_deltas)),
            "dataset_clustered_bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))
            ],
            "wilcoxon_p": raw_p[baseline],
            "positive_dataset_count": int(sum(value > 0 for value in dataset_deltas)),
            "dataset_count": len(dataset_deltas),
        }
    adjusted = _holm(raw_p)
    for baseline in FULL_METHODS:
        payload[baseline]["holm_adjusted_p"] = adjusted[baseline]
    return payload


def _decision(
    candidate_rows: list[dict[str, Any]], statistics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    comparison = str(gates["comparison_method"])
    row = statistics[comparison]
    reliability = [float(item["selected_aurc_delta_vs_confidence"]) for item in candidate_rows]
    counts = [int(item["minimum_prediction_class_count"]) for item in candidate_rows]
    values = [float(item["mean_unseen_macro_f1"]) for item in candidate_rows] + reliability
    checks = {
        "average_unseen_gain_vs_v3_10": float(row["mean_delta"])
        > float(gates["minimum_average_dataset_unseen_macro_f1_delta"]),
        "positive_dataset_count_vs_v3_10": int(row["positive_dataset_count"])
        >= int(gates["minimum_positive_dataset_count"]),
        "single_dataset_unseen_floor_vs_v3_10": min(row["dataset_deltas"].values())
        >= -float(gates["maximum_single_dataset_unseen_macro_f1_drop"]),
        "reliability_nonworse": float(np.mean(reliability))
        <= float(gates["maximum_average_dataset_selected_aurc_delta_vs_confidence"]),
        "no_constant_prediction": all(value > 1 for value in counts)
        if gates["require_no_constant_prediction_at_any_test_rate"] else True,
        "finite_metrics": all(math.isfinite(value) for value in values)
        if gates["require_finite_metrics"] else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "comparison_method": comparison,
        "mean_unseen_macro_f1_delta": float(row["mean_delta"]),
        "positive_dataset_count": int(row["positive_dataset_count"]),
        "minimum_dataset_delta": float(min(row["dataset_deltas"].values())),
        "mean_selected_aurc_delta_vs_confidence": float(np.mean(reliability)),
    }


def run_v5_1_full_extension(
    project_root: str | Path, *, resume: bool = True, confirmed: bool = False
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("V5.1 Full extension requires manual confirmation")
    root = Path(project_root).resolve()
    _assert_no_active_other_stage(root)
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v5_1_full_extension.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = _validate_sources(root, config)
    design = config["design"]
    base_path = root / design["base_config"]
    base_template = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(root, config_path, base_path)
    parent = root / "runs" / "v5_1_full_extension"
    run_root = _find_compatible_root(parent, protocol_hash) if resume else None
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v5_1_full_extension__10datasets__3seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
    report_path = run_root / "report.json"
    if resume and report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return cached
    manifest_path = run_root / "manifest.json"
    previous_elapsed = 0.0
    if resume and manifest_path.exists():
        try:
            previous_elapsed = float(json.loads(manifest_path.read_text(encoding="utf-8")).get("cumulative_elapsed_seconds", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    manifest = {
        "stage": config["stage"], "status": "running", "phase": "validation_training",
        "manual_confirmation": True, "protocol_hash": protocol_hash,
        "reused_full_runs": 210, "new_candidate_runs": 30,
        "cumulative_elapsed_seconds": previous_elapsed, "updated_at_utc": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    tasks = v5_full_extension_tasks()
    progress = DashboardProgress(root / "runs" / "dashboard_status.json", config["stage"], tasks, run_root.name)
    deadline = started + float(design["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_task = 0
    validations: dict[str, dict[str, Any]] = {}
    try:
        progress.start_task(current_task)
        datasets = {dataset_id: prepare_full_dataset(root, dataset_id) for dataset_id in FULL_DATASETS}
        progress.complete_task(current_task)
        current_task += 1
        for dataset_id in FULL_DATASETS:
            dataset = datasets[dataset_id]
            base_config = dict(base_template)
            base_config["batch_size"] = int(design["dataset_batch_sizes"][dataset_id])
            for seed in FULL_SEEDS:
                progress.start_task(current_task)
                key = f"{dataset_id}__seed{seed}"
                run_dir = run_root / key
                metrics_path = run_dir / "validation_metrics.json"
                result = None
                if resume and metrics_path.exists():
                    cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if cached.get("protocol_hash") == protocol_hash:
                        result = cached
                if result is None:
                    run_started = time.monotonic()
                    _, candidate_state, exact = _paired_initial_states(dataset, base_config, seed)
                    if not exact:
                        raise RuntimeError(f"shared initialization mismatch for {dataset_id} seed{seed}")

                    def update_epoch(epoch: int, total: int, row: dict[str, Any]) -> None:
                        progress.current_task = (
                            f"{dataset_id} seed{seed}: epoch {epoch}/{total}, "
                            f"val={row['validation_selection_score']:.4f}"
                        )
                        progress.write()

                    model, history = _train_variant(
                        dataset, base_config, design, "v5_dual_path", candidate_state,
                        protocol_hash, run_dir, device, deadline, resume, seed=seed,
                        epoch_callback=update_epoch,
                    )
                    validation = _evaluate_validation(
                        model, _view(dataset, "validation"), base_config,
                        tuple(float(value) for value in design["validation_rate_ratios"]), device,
                    )
                    validation["learned_gate_floor"] = model.gate_floor.detach().cpu().tolist()
                    result = {
                        "status": "validation_completed", "protocol_hash": protocol_hash,
                        "dataset_id": dataset_id, "seed": seed, "epochs_completed": len(history),
                        "duration_seconds": time.monotonic() - run_started,
                        "validation": validation,
                    }
                    atomic_write_json(metrics_path, result)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                validations[key] = result
                print(f"Validation completed {key}", flush=True)
                progress.complete_task(current_task)
                current_task += 1
        progress.start_task(current_task)
        frozen = _freeze_modes(run_root, protocol_hash, validations, config["reliability_controller"])
        manifest.update(phase="test_evaluation", reliability_modes_frozen=True, updated_at_utc=utc_now())
        atomic_write_json(manifest_path, manifest)
        progress.complete_task(current_task)
        current_task += 1
        results: dict[str, dict[str, Any]] = {}
        for dataset_id in FULL_DATASETS:
            dataset = datasets[dataset_id]
            base_config = dict(base_template)
            base_config["batch_size"] = int(design["dataset_batch_sizes"][dataset_id])
            for seed in FULL_SEEDS:
                progress.start_task(current_task)
                key = f"{dataset_id}__seed{seed}"
                run_dir = run_root / key
                metrics_path = run_dir / "metrics.json"
                result = None
                if resume and metrics_path.exists():
                    cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if cached.get("protocol_hash") == protocol_hash:
                        result = cached
                if result is None:
                    model = _new_model(_view(dataset, "validation"), base_config, "v5_dual_path", device)
                    model.load_state_dict(
                        torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=True), strict=True
                    )
                    test = _evaluate_validation(
                        model, _view(dataset, "test"), base_config,
                        tuple(float(value) for value in design["test_rate_ratios"]), device,
                    )
                    mode = frozen["modes"][dataset_id]["mode"]
                    selected_key = "pooled_observability_aurc" if mode == "observability" else "pooled_confidence_aurc"
                    test["reliability_mode_selected_on_validation_consensus"] = mode
                    test["selected_pooled_aurc"] = float(test[selected_key])
                    result = {
                        **validations[key], "status": "completed", "test": test,
                        "test_used_for_any_selection": False,
                    }
                    atomic_write_json(metrics_path, result)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                results[key] = result
                progress.complete_task(current_task)
                current_task += 1
        progress.start_task(current_task)
        candidate_rows = _candidate_rows(results)
        statistics = _paired_statistics(candidate_rows, source["aggregate"]["rows"], config)
        decision = _decision(candidate_rows, statistics, config["confirmation_gates"])
        method_summary = dict(source["aggregate"]["method_summary"])
        method_summary[CANDIDATE] = {
            key: float(np.mean([row[key] for row in candidate_rows]))
            for key in ("mean_unseen_macro_f1", "worst_unseen_macro_f1", "full_rate_macro_f1")
        }
        elapsed = previous_elapsed + (time.monotonic() - started)
        report = {
            "status": "completed", "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash, "manual_confirmation": True,
            "retrospective_full_extension": True, "independent_confirmation": False,
            "datasets": list(FULL_DATASETS), "seeds": list(FULL_SEEDS),
            "new_candidate_runs": 30, "reused_full_runs": 210,
            "source_full_protocol_hash": source["protocol_hash"],
            "candidate_rows": candidate_rows, "method_summary": method_summary,
            "paired_statistics": statistics, "reliability_modes": frozen,
            "decision": decision, "elapsed_seconds": elapsed, "device": str(device),
            "run_root": str(run_root), "later_stage_started": False,
            "finished_at_utc": utc_now(),
        }
        lines = [
            "# V5.1 ten-dataset Full extension", "",
            f"- Frozen decision: **{'PASS' if decision['passed'] else 'FAIL'}**.",
            "- Newly trained: 30 V5.1 runs. Reused without retraining: 210 frozen Full runs.",
            "- This is a retrospective extension on previously accessed Full tests, not independent confirmation.",
            f"- Device / cumulative elapsed: `{device}` / `{elapsed:.1f} s`.", "",
            "| Baseline | Mean unseen F1 delta | 95% CI | Holm p | Positive datasets |",
            "|---|---:|---:|---:|---:|",
        ]
        for baseline, row in statistics.items():
            ci = row["dataset_clustered_bootstrap_95_ci"]
            lines.append(
                f"| {baseline} | {row['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{row['holm_adjusted_p']:.4g} | {row['positive_dataset_count']}/10 |"
            )
        lines.extend(["", "## Frozen checks", ""])
        lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["checks"].items())
        lines.extend(["", "No later stage was started automatically.", ""])
        markdown = "\n".join(lines)
        atomic_write_json(report_path, report)
        _atomic_write_text(run_root / "report.md", markdown)
        atomic_write_json(root / "reports" / "v5_1_full_extension_report.json", report)
        _atomic_write_text(root / "reports" / "v5_1_full_extension_report.md", markdown)
        progress.complete_task(current_task)
        progress.finish(f"V5.1 Full extension {'PASS' if decision['passed'] else 'FAIL'}")
        manifest.update(
            status="completed", phase="completed", decision=decision["passed"],
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(manifest_path, manifest)
        return report
    except BaseException as error:
        elapsed = previous_elapsed + (time.monotonic() - started)
        progress.fail_task(min(current_task, len(tasks) - 1), error)
        manifest.update(
            status="failed", error=f"{type(error).__name__}: {error}",
            cumulative_elapsed_seconds=elapsed, updated_at_utc=utc_now(),
        )
        atomic_write_json(manifest_path, manifest)
        raise
