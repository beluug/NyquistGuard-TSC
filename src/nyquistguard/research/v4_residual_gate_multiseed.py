"""Manual, resumable V4.1 multi-seed validation stability experiment."""

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

from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.pilot import _seed_everything
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.research.v4_observe_only_micro import (
    _evaluate_validation,
    _new_model,
    _train_variant,
    load_development_dataset,
)


V4_STABILITY_DATASETS = ("basicmotions_uea", "pamap2_uci")
V4_STABILITY_SEEDS = (42, 2026)
V4_STABILITY_ROLES = ("v3_10_hard_gate", "v4_1_residual_gate")
V4_STABILITY_TASKS = [
    "冻结协议与train/validation零test访问预检",
    "BasicMotions seed42：硬gate同预算控制",
    "BasicMotions seed42：V4.1残余gate",
    "PAMAP2 seed2026：硬gate同预算控制",
    "PAMAP2 seed2026：V4.1残余gate",
    "BasicMotions seed2026：硬gate同预算控制",
    "BasicMotions seed2026：V4.1残余gate",
    "PAMAP2 seed42：硬gate同预算控制",
    "PAMAP2 seed42：V4.1残余gate",
    "汇总四组配对稳定性门与报告",
]


def _protocol_hash(config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update((Path(__file__).parent / "v4_residual_gate.py").read_bytes())
    digest.update((Path(__file__).parent / "v4_observe_only_micro.py").read_bytes())
    digest.update(b"v4.1-residual-gate-multiseed-stability-v1")
    return digest.hexdigest()


def validate_stability_matrix(config: dict[str, Any]) -> list[tuple[str, int, str]]:
    design = config["design"]
    datasets = tuple(str(value) for value in design["datasets"])
    seeds = tuple(int(value) for value in design["stability_seeds"])
    roles = tuple(str(value) for value in design["roles"])
    matrix = [(str(d), int(s), str(r)) for d, s, r in design["run_order"]]
    if datasets != V4_STABILITY_DATASETS or seeds != V4_STABILITY_SEEDS or roles != V4_STABILITY_ROLES:
        raise ValueError("V4.1 stability datasets, seeds, or roles changed after freeze")
    if int(design["development_seed_excluded_from_primary_gate"]) in seeds:
        raise ValueError("development seed17 must remain excluded from the stability gate")
    if float(design["initial_gate_floor"]) != 0.5:
        raise ValueError("V4.1 initial gate floor changed after freeze")
    expected = {(dataset, seed, role) for dataset in datasets for seed in seeds for role in roles}
    if len(matrix) != len(expected) or set(matrix) != expected:
        raise ValueError("run_order must contain every dataset x seed x role exactly once")
    for index in range(0, len(matrix), 2):
        first, second = matrix[index], matrix[index + 1]
        if first[:2] != second[:2] or first[2:] != ("v3_10_hard_gate",) or second[2:] != ("v4_1_residual_gate",):
            raise ValueError("each matched pair must run hard gate before V4.1")
    return matrix


def _validate_source(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = json.loads((root / config["source_matched_report"]).read_text(encoding="utf-8"))
    if source.get("status") != "completed_matched_candidate_pass":
        raise ValueError("V4.1 matched seed17 development must pass before stability")
    if source.get("test_accessed") is not False:
        raise ValueError("V4.1 source must be leakage-locked")
    if source.get("protocol_hash") != config["source_matched_protocol_hash"]:
        raise ValueError("V4.1 matched source protocol hash changed")
    if source.get("matched_epoch_cap") != 30 or source.get("matched_early_stopping_patience") != 10:
        raise ValueError("V4.1 matched training budget changed")
    return source


def _find_compatible_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for path in sorted(parent.glob("v4_1_stability__2datasets__2seeds__*"), reverse=True):
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


def _paired_initial_states(
    dataset: Any, base_config: dict[str, Any], seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], bool]:
    _seed_everything(seed)
    hard = _new_model(dataset, base_config, "v3_10_hard_gate", torch.device("cpu"))
    _seed_everything(seed)
    candidate = _new_model(dataset, base_config, "v4_1_residual_gate", torch.device("cpu"))
    hard_state = {key: value.detach().clone() for key, value in hard.state_dict().items()}
    candidate_state = {key: value.detach().clone() for key, value in candidate.state_dict().items()}
    common = set(hard_state) & set(candidate_state)
    exact = bool(common) and all(torch.equal(hard_state[key], candidate_state[key]) for key in common)
    return hard_state, candidate_state, exact


def stability_decision(pair_rows: dict[str, dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    rows = list(pair_rows.values())
    unseen = [float(row["unseen_macro_f1_delta_vs_hard_gate"]) for row in rows]
    full = [float(row["full_rate_macro_f1_delta_vs_hard_gate"]) for row in rows]
    reliability = [float(row["selected_aurc_delta_vs_confidence"]) for row in rows]
    seed_means = {
        seed: float(np.mean([row["unseen_macro_f1_delta_vs_hard_gate"] for row in rows if row["seed"] == seed]))
        for seed in V4_STABILITY_SEEDS
    }
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in (
            row["unseen_macro_f1_delta_vs_hard_gate"],
            row["full_rate_macro_f1_delta_vs_hard_gate"],
            row["selected_aurc_delta_vs_confidence"],
            row["minimum_gate_floor"],
            row["maximum_gate_floor"],
        )
    ) and all(0.0 <= row["minimum_gate_floor"] <= row["maximum_gate_floor"] <= 1.0 for row in rows)
    checks = {
        "average_unseen_gain": float(np.mean(unseen)) >= float(gates["minimum_average_unseen_macro_f1_delta_vs_hard_gate"]),
        "positive_unseen_pairs": sum(value > 0.0 for value in unseen) >= int(gates["minimum_positive_unseen_macro_f1_pair_count"]),
        "single_pair_unseen_floor": float(np.min(unseen)) >= -float(gates["maximum_single_pair_unseen_macro_f1_drop"]),
        "positive_each_seed": all(value > 0.0 for value in seed_means.values())
        if gates["require_positive_mean_unseen_delta_each_seed"] else True,
        "average_full_rate_floor": float(np.mean(full)) >= -float(gates["maximum_average_full_rate_macro_f1_drop"]),
        "selected_reliability_safety": all(value <= 1e-12 for value in reliability)
        if gates["require_selected_reliability_nonworse_than_confidence_all_pairs"] else True,
        "finite_rates_and_floors": finite if gates["require_finite_all_rates_and_gate_floors"] else True,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "average_unseen_macro_f1_delta_vs_hard_gate": float(np.mean(unseen)),
        "positive_unseen_macro_f1_pair_count": int(sum(value > 0.0 for value in unseen)),
        "minimum_pair_unseen_macro_f1_delta_vs_hard_gate": float(np.min(unseen)),
        "mean_unseen_macro_f1_delta_by_seed": {str(key): value for key, value in seed_means.items()},
        "average_full_rate_macro_f1_delta_vs_hard_gate": float(np.mean(full)),
        "average_selected_aurc_delta_vs_confidence": float(np.mean(reliability)),
    }


def _pair_row(dataset_id: str, seed: int, hard: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    hard_v = hard["validation"]
    candidate_v = candidate["validation"]
    floors = [float(value) for value in candidate_v["learned_gate_floor"]]
    return {
        "dataset_id": dataset_id, "seed": seed,
        "unseen_macro_f1_delta_vs_hard_gate": float(candidate_v["mean_unseen_macro_f1"] - hard_v["mean_unseen_macro_f1"]),
        "full_rate_macro_f1_delta_vs_hard_gate": float(candidate_v["full_rate_macro_f1"] - hard_v["full_rate_macro_f1"]),
        "worst_unseen_macro_f1_delta_vs_hard_gate": float(candidate_v["worst_unseen_macro_f1"] - hard_v["worst_unseen_macro_f1"]),
        "reliability_mode": candidate_v["reliability_mode"],
        "selected_aurc_delta_vs_confidence": float(candidate_v["selected_pooled_aurc"] - candidate_v["pooled_confidence_aurc"]),
        "minimum_gate_floor": min(floors), "maximum_gate_floor": max(floors),
    }


def run_v4_residual_gate_multiseed(
    project_root: str | Path, *, resume: bool = True, confirmed: bool = False
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("V4.1 multi-seed stability requires manual confirmation")
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v4_residual_gate_multiseed_stability.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = validate_stability_matrix(config)
    source = _validate_source(root, config)
    design = config["design"]
    base_path = root / design["base_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path, base_path)
    parent = root / "runs" / "v4_residual_gate_multiseed_stability"
    run_root = _find_compatible_root(parent, protocol_hash) if resume else None
    if run_root is not None:
        completed_report = run_root / "report.json"
        if completed_report.exists():
            cached = json.loads(completed_report.read_text(encoding="utf-8"))
            if cached.get("status") == "completed":
                progress = DashboardProgress(root / "runs" / "dashboard_status.json", config["stage"], V4_STABILITY_TASKS, run_root.name)
                for index in range(len(V4_STABILITY_TASKS)):
                    progress.start_task(index); progress.complete_task(index)
                progress.finish("V4.1 multi-seed stability reused completed report")
                return cached
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v4_1_stability__2datasets__2seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
    manifest = {
        "stage": config["stage"], "status": "running", "manual_confirmation": True,
        "protocol_hash": protocol_hash, "run_root": str(run_root), "test_accessed": False,
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "manifest.json", manifest)
    progress = DashboardProgress(root / "runs" / "dashboard_status.json", config["stage"], V4_STABILITY_TASKS, run_root.name)
    deadline = started + float(design["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    role_results: dict[str, dict[str, Any]] = {}
    current_task = 0
    try:
        progress.start_task(current_task)
        for dataset_id in V4_STABILITY_DATASETS:
            dataset = load_development_dataset(root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz")
            assert not hasattr(dataset, "test")
            del dataset
        progress.complete_task(current_task)
        current_task += 1
        state_cache: dict[tuple[str, int], tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}
        for matrix_index, (dataset_id, seed, role) in enumerate(matrix):
            progress.start_task(current_task)
            print(f"[{matrix_index + 1:02d}/08] Starting {dataset_id} seed{seed} {role}", flush=True)
            dataset = load_development_dataset(root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz")
            pair_key = (dataset_id, seed)
            if pair_key not in state_cache:
                hard_state, candidate_state, exact = _paired_initial_states(dataset, base_config, seed)
                if not exact:
                    raise RuntimeError(f"paired shared initialization mismatch for {dataset_id} seed{seed}")
                state_cache[pair_key] = (hard_state, candidate_state)
            initial_state = state_cache[pair_key][0 if role == "v3_10_hard_gate" else 1]
            role_dir = run_root / f"{dataset_id}__seed{seed}__{role}"
            metrics_path = role_dir / "metrics.json"
            result: dict[str, Any] | None = None
            if resume and metrics_path.exists():
                candidate_cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                if candidate_cached.get("protocol_hash") == protocol_hash and candidate_cached.get("test_accessed") is False:
                    result = candidate_cached
            if result is None:
                role_started = time.monotonic()
                def update_epoch(epoch: int, total: int, row: dict[str, Any]) -> None:
                    progress.current_task = (
                        f"{dataset_id} seed{seed} {role}: epoch {epoch}/{total}, "
                        f"val={row['validation_selection_score']:.4f}"
                    )
                    progress.write()
                model, history = _train_variant(
                    dataset, base_config, design, role, initial_state, protocol_hash,
                    role_dir, device, deadline, resume, seed=seed, epoch_callback=update_epoch,
                )
                validation = _evaluate_validation(
                    model, dataset, base_config,
                    tuple(float(value) for value in design["validation_rate_ratios"]), device,
                )
                if role == "v4_1_residual_gate":
                    use_observability = validation["pooled_observability_aurc"] <= validation["pooled_confidence_aurc"]
                    validation["reliability_mode"] = "observability" if use_observability else "confidence_fallback"
                    validation["selected_pooled_aurc"] = validation[
                        "pooled_observability_aurc" if use_observability else "pooled_confidence_aurc"
                    ]
                    validation["learned_gate_floor"] = model.gate_floor.detach().cpu().tolist()
                result = {
                    "status": "completed", "protocol_hash": protocol_hash,
                    "dataset_id": dataset_id, "seed": seed, "role": role,
                    "epochs_completed": len(history), "duration_seconds": time.monotonic() - role_started,
                    "validation": validation, "test_accessed": False,
                }
                atomic_write_json(metrics_path, result)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            role_results[f"{dataset_id}__seed{seed}__{role}"] = result
            del dataset
            progress.complete_task(current_task)
            current_task += 1
        progress.start_task(current_task)
        pair_results: dict[str, Any] = {}
        pair_rows: dict[str, Any] = {}
        for dataset_id in V4_STABILITY_DATASETS:
            for seed in V4_STABILITY_SEEDS:
                pair_id = f"{dataset_id}__seed{seed}"
                hard = role_results[f"{pair_id}__v3_10_hard_gate"]
                candidate = role_results[f"{pair_id}__v4_1_residual_gate"]
                row = _pair_row(dataset_id, seed, hard, candidate)
                pair_rows[pair_id] = row
                pair_results[pair_id] = {"hard_gate": hard, "v4_1": candidate, "summary": row}
                atomic_write_json(run_root / f"{pair_id}__paired_metrics.json", pair_results[pair_id])
        decision = stability_decision(pair_rows, config["stability_gates"])
        elapsed = time.monotonic() - started
        report = {
            "status": "completed", "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash, "primary_units": "dataset_x_seed_matched_pairs",
            "stability_seeds": list(V4_STABILITY_SEEDS), "development_seed17_in_primary_gate": False,
            "primary_split": "validation_only", "test_accessed": False,
            "manual_confirmation": True, "independent_confirmation_claim_allowed": False,
            "new_untouched_datasets_still_required": int(config["data_boundary"]["minimum_new_confirmation_datasets_after_pass"]),
            "source_matched_report_hash": source["protocol_hash"],
            "elapsed_seconds": elapsed, "device": str(device), "results": pair_results,
            "stability_gates": config["stability_gates"], "decision": decision,
            "run_root": str(run_root), "pilot_started": False, "full_started": False,
            "finished_at_utc": utc_now(),
        }
        lines = [
            "# V4.1 multi-seed validation stability", "",
            f"- Frozen decision: **{'PASS' if decision['passed'] else 'FAIL'}**.",
            "- Primary paired units: BasicMotions/PAMAP2 x seeds 42/2026; seed17 excluded.",
            "- Existing test arrays were not loaded or scored; this is not independent confirmation.",
            f"- Device / session elapsed: `{device}` / {elapsed:.1f} s.", "",
            "| Pair | Unseen F1 delta | Full-rate F1 delta | Worst-unseen delta | Reliability mode | Floor range |",
            "|---|---:|---:|---:|---|---|",
        ]
        for pair_id, row in pair_rows.items():
            lines.append(
                f"| {pair_id} | {row['unseen_macro_f1_delta_vs_hard_gate']:+.4f} | "
                f"{row['full_rate_macro_f1_delta_vs_hard_gate']:+.4f} | "
                f"{row['worst_unseen_macro_f1_delta_vs_hard_gate']:+.4f} | "
                f"{row['reliability_mode']} | {row['minimum_gate_floor']:.3f}-{row['maximum_gate_floor']:.3f} |"
            )
        lines.extend(["", "## Frozen checks", ""])
        for name, passed in decision["checks"].items():
            lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
        lines.extend(["", "Passing only authorizes freezing V4.1 for a separate confirmation on at least four new untouched datasets.", ""])
        markdown = "\n".join(lines)
        atomic_write_json(run_root / "report.json", report)
        _atomic_write_text(run_root / "report.md", markdown)
        atomic_write_json(root / "reports" / "v4_residual_gate_multiseed_report.json", report)
        _atomic_write_text(root / "reports" / "v4_residual_gate_multiseed_report.md", markdown)
        progress.complete_task(current_task)
        progress.finish(f"V4.1 multi-seed stability {'PASS' if decision['passed'] else 'FAIL'}; no later stage auto-started")
        manifest.update(status="completed", decision=decision["passed"], updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        return report
    except BaseException as error:
        progress.fail_task(min(current_task, len(V4_STABILITY_TASKS) - 1), error)
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        raise
