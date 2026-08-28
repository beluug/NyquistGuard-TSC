"""Manual, resumable matched-control confirmation for the frozen v3.10 candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.pilot import (
    PilotRunSpec,
    _deep_model,
    _seed_everything,
    _train_deep,
)
from nyquistguard.experiments.progress import DashboardProgress, atomic_write_json, utc_now
from nyquistguard.experiments.v2_micro_pilot import _evaluate_model
from nyquistguard.experiments.v3_core_micro import _anchored_reliability, _train_core
from nyquistguard.experiments.v3_guarded_reliability import _guarded_evaluation
from nyquistguard.experiments.v3_multiseed_confirmation import _decision, _run_row


ROLES = ("v1_control", "v3_10_candidate")


def _protocol_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(b"v3.10-independent-matched-confirmation-v1")
    return digest.hexdigest()


def validate_confirmation_matrix(config: dict[str, Any]) -> list[tuple[str, int, str]]:
    datasets = tuple(str(value) for value in config["datasets"])
    seeds = tuple(int(value) for value in config["confirmation_seeds"])
    excluded = {int(value) for value in config["development_seeds_excluded"]}
    matrix = [
        (str(dataset), int(seed), str(role))
        for dataset, seed, role in config["run_order"]
    ]
    if set(seeds) & excluded:
        raise ValueError("development seeds must remain excluded from independent confirmation")
    if seeds != (31415, 27182):
        raise ValueError("v3.10 confirmation seeds changed after protocol freeze")
    expected = {
        (dataset, seed, role)
        for dataset in datasets
        for seed in seeds
        for role in ROLES
    }
    if set(matrix) != expected or len(matrix) != len(expected):
        raise ValueError("confirmation run_order must contain every dataset x seed x role once")
    return matrix


def _validate_source_report(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = json.loads(
        (root / config["source_development_report"]).read_text(encoding="utf-8")
    )
    if source.get("protocol_version") != "v3_continuous_rate_development_v1":
        raise ValueError("independent confirmation requires the frozen v3.10 development report")
    if not source.get("decision", {}).get("passed", False):
        raise ValueError("v3.10 development gate must pass before confirmation")
    return source


def _find_resume_root(parent: Path, protocol_hash: str) -> Path | None:
    if not parent.exists():
        return None
    for path in sorted(parent.glob("v3_10_confirmation__2datasets__2seeds__*"), reverse=True):
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("protocol_hash") == protocol_hash and manifest.get("status") != "completed":
            return path
    return None


def _load_role_model(
    dataset: Any,
    base_config: dict[str, Any],
    run_root: Path,
    dataset_id: str,
    seed: int,
    role: str,
    device: torch.device,
) -> torch.nn.Module:
    variant = "nyquistguard" if role == "v1_control" else "no_selective_head"
    model = _deep_model(dataset, base_config, variant, device)
    checkpoint = run_root / f"{dataset_id}__seed{seed}__{role}" / "checkpoint_best.pt"
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval()
    return model


def run_v3_10_independent_confirmation(
    project_root: str | Path,
    *,
    resume: bool = True,
    confirmed: bool = False,
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("v3.10 independent confirmation requires manual confirmation")
    root = Path(project_root).resolve()
    started = time.monotonic()
    config_path = root / "configs" / "experiments" / "v3_10_independent_confirmation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = validate_confirmation_matrix(config)
    _validate_source_report(root, config)
    base_path = root / config["base_config"]
    reliability_path = root / config["reliability_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    reliability_config = yaml.safe_load(reliability_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(config_path, base_path, reliability_path)

    parent = root / "runs" / "v3_10_independent_confirmation"
    run_root = _find_resume_root(parent, protocol_hash) if resume else None
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v3_10_confirmation__2datasets__2seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")

    manifest = {
        "stage": "v3_10_independent_confirmation",
        "status": "running",
        "manual_confirmation": True,
        "protocol_hash": protocol_hash,
        "run_root": str(run_root),
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "manifest.json", manifest)

    pair_order = [
        (dataset_id, seed)
        for dataset_id in config["datasets"]
        for seed in config["confirmation_seeds"]
    ]
    task_names = [f"Train {dataset} seed{seed} {role}" for dataset, seed, role in matrix]
    task_names.extend(f"Evaluate matched pair {dataset} seed{seed}" for dataset, seed in pair_order)
    task_names.append("Aggregate frozen confirmation gates")
    progress = DashboardProgress(
        root / "runs" / "dashboard_status.json",
        "v3_10_independent_confirmation",
        task_names,
        run_root.name,
    )
    deadline = started + float(config["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_task = 0
    try:
        for index, (dataset_id, seed, role) in enumerate(matrix):
            progress.start_task(current_task)
            print(f"[{index + 1:02d}/08] Starting {dataset_id} seed{seed} {role}", flush=True)
            dataset = load_prepared_dataset(
                root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
            )
            _seed_everything(seed)
            role_dir = run_root / f"{dataset_id}__seed{seed}__{role}"
            if role == "v3_10_candidate":
                candidate, _ = _train_core(
                    dataset,
                    base_config,
                    config,
                    protocol_hash,
                    role_dir,
                    seed=seed,
                    resume=resume,
                    deadline=deadline,
                )
                del candidate
            else:
                control_spec = PilotRunSpec(dataset_id, "nyquistguard", seed)
                control = _train_deep(
                    dataset,
                    control_spec,
                    base_config,
                    protocol_hash,
                    role_dir,
                    resume,
                )
                del control
            del dataset
            progress.complete_task(current_task)
            current_task += 1

        guarded_config = yaml.safe_load(
            (root / "configs" / "experiments" / "v3_guarded_reliability.yaml").read_text(
                encoding="utf-8"
            )
        )
        minimum_gain = float(
            guarded_config["controller"]["minimum_absolute_aurc_gain_to_enable_calibrator"]
        )
        results: dict[str, Any] = {}
        rows: dict[str, dict[str, Any]] = {}
        for dataset_id, seed in pair_order:
            progress.start_task(current_task)
            run_key = f"{dataset_id}__seed{seed}"
            print(f"Evaluating matched pair {run_key}", flush=True)
            dataset = load_prepared_dataset(
                root / "data" / "processed" / "pilot_v1" / f"{dataset_id}.npz"
            )
            candidate = _load_role_model(
                dataset, base_config, run_root, dataset_id, seed, "v3_10_candidate", device
            )
            control = _load_role_model(
                dataset, base_config, run_root, dataset_id, seed, "v1_control", device
            )
            candidate_classification = _evaluate_model(candidate, dataset, base_config, device)
            control_classification = _evaluate_model(control, dataset, base_config, device)
            candidate_reliability = _guarded_evaluation(
                candidate,
                dataset,
                base_config,
                reliability_config,
                seed,
                minimum_gain,
            )
            control_reliability = _anchored_reliability(
                control, dataset, base_config, reliability_config, seed
            )
            row = _run_row(
                candidate_classification,
                control_classification,
                candidate_reliability,
                control_reliability,
            )
            rows[run_key] = row
            results[run_key] = {
                "dataset_id": dataset_id,
                "seed": seed,
                "candidate_classification": candidate_classification,
                "v1_control_classification": control_classification,
                "candidate_reliability": candidate_reliability,
                "v1_control_reliability": control_reliability,
                "summary": row,
            }
            atomic_write_json(run_root / f"{run_key}__paired_metrics.json", results[run_key])
            del candidate, control, dataset
            progress.complete_task(current_task)
            current_task += 1

        progress.start_task(current_task)
        decision = _decision(rows, config["confirmation_gates"])
        elapsed = time.monotonic() - started
        report: dict[str, Any] = {
            "status": "completed",
            "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash,
            "primary_units": "2_datasets_x_2_fresh_seeds_with_matched_v1_controls",
            "confirmation_seeds": config["confirmation_seeds"],
            "excluded_development_seeds": config["development_seeds_excluded"],
            "primary_split": "validation",
            "test_role": "exploratory_appendix",
            "manual_confirmation": True,
            "elapsed_seconds": elapsed,
            "device": str(device),
            "pilot_started": False,
            "full_started": False,
            "results": results,
            "confirmation_gates": config["confirmation_gates"],
            "decision": decision,
            "run_root": str(run_root),
            "finished_at_utc": utc_now(),
        }
        lines = [
            "# NyquistGuard-TSC v3.10 Independent Matched-Control Confirmation",
            "",
            "- Primary units: BasicMotions/PAMAP2 x fresh seeds 31415/27182.",
            "- Every candidate is paired with a newly trained same-seed v1 control.",
            "- Seeds 17/42/2026 are excluded; validation is primary; test is exploratory.",
            f"- Elapsed: {elapsed:.1f} seconds.",
            f"- Frozen confirmation decision: {'PASS' if decision['passed'] else 'FAIL'}.",
            "- Pilot and Full were not started.",
            "",
            "| pair | unseen F1 delta | full F1 delta | reliability mode | selected AURC delta vs v1 | target-risk delta |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for run_key, row in rows.items():
            lines.append(
                f"| {run_key} | {row['unseen_macro_f1_delta_vs_v1']:+.4f} | "
                f"{row['full_rate_macro_f1_delta_vs_v1']:+.4f} | {row['selected_mode']} | "
                f"{row['selected_aurc_delta_vs_v1']:+.4f} | "
                f"{row['target_risk_delta_vs_confidence']:+.4f} |"
            )
        lines.extend(["", "## Frozen checks", ""])
        for name, passed in decision["checks"].items():
            lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
        markdown = "\n".join(lines) + "\n"
        atomic_write_json(run_root / "v3_10_independent_confirmation_report.json", report)
        _atomic_write_text(run_root / "v3_10_independent_confirmation_report.md", markdown)
        atomic_write_json(root / "reports" / "v3_10_independent_confirmation_report.json", report)
        _atomic_write_text(root / "reports" / "v3_10_independent_confirmation_report.md", markdown)
        progress.complete_task(current_task)
        progress.finish(
            f"v3.10 independent confirmation {'PASS' if decision['passed'] else 'FAIL'}; Full remains locked"
        )
        manifest.update(status="completed", decision=decision["passed"], updated_at_utc=utc_now())
        atomic_write_json(run_root / "manifest.json", manifest)
        return report
    except BaseException as error:
        progress.fail_task(current_task, error)
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            updated_at_utc=utc_now(),
        )
        atomic_write_json(run_root / "manifest.json", manifest)
        raise
