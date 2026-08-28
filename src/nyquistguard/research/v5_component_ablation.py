"""Retrained three-component ablation for the frozen V5.1 architecture."""

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
from torch import Tensor, nn
import yaml

from nyquistguard.data.v5_independent_datasets import prepare_v5_independent_dataset
from nyquistguard.experiments.diagnosis import _atomic_write_text
from nyquistguard.experiments.pilot import _resolved_model_config, _seed_everything
from nyquistguard.experiments.progress import atomic_write_json, utc_now
import nyquistguard.research.v4_observe_only_micro as training
from nyquistguard.research.v5_dual_path import DualPathNyquistGuardTSC
from nyquistguard.research.v5_independent_confirmation import _view


ABLATION_DATASETS = ("self_regulation_scp1_uea", "racket_sports_uea")
ABLATION_SEEDS = (314159, 271828, 161803)
ABLATION_VARIANTS = (
    "v5_full",
    "no_signed_spatial_path",
    "mean_only_temporal_summary",
    "fixed_equal_fusion",
)


class MeanOnlyTemporalSummary(nn.Module):
    """Matched hidden-width temporal mean used only by the ablation model."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return self.projection(sequence.mean(dim=-1))


class ComponentAblatedDualPath(DualPathNyquistGuardTSC):
    """V5 with exactly one classification component disabled and then retrained."""

    def __init__(self, *args: Any, ablation: str, **kwargs: Any) -> None:
        if ablation not in ABLATION_VARIANTS[1:]:
            raise ValueError(f"unknown V5 component ablation: {ablation}")
        self.ablation = ablation
        super().__init__(*args, **kwargs)
        if ablation == "mean_only_temporal_summary":
            hidden_dim = int(self.classifier.in_features)
            dropout = float(self.encoder.blocks[0].dropout.p)
            self.physical_summary = MeanOnlyTemporalSummary(hidden_dim, dropout)
            self.spatial_summary = MeanOnlyTemporalSummary(hidden_dim, dropout)

    def _replace_embedding(self, output: dict[str, Any], embedding: Tensor) -> dict[str, Any]:
        logits = self.classifier(embedding)
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(self.num_classes)
        if self.selective_head is None:
            accept_logit = embedding.new_zeros((embedding.shape[0],))
        else:
            accept_logit = self.selective_head(
                embedding, output["nyquist_gate"],
                output["aux"]["sampling_rate_hz"], normalized_entropy.to(embedding.dtype),
            )
        output.update(
            logits=logits,
            accept_logit=accept_logit,
            accept_probability=torch.sigmoid(accept_logit),
            embedding=embedding,
        )
        output["aux"] = dict(output["aux"])
        output["aux"]["normalized_prediction_entropy"] = normalized_entropy
        output["aux"]["component_ablation"] = self.ablation
        return output

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        output = super().forward(*args, **kwargs)
        if self.ablation == "no_signed_spatial_path":
            embedding = self.fusion_norm(output["physical_embedding"])
            return self._replace_embedding(output, embedding)
        if self.ablation == "fixed_equal_fusion":
            embedding = self.fusion_norm(
                0.5 * output["physical_embedding"] + 0.5 * output["spatial_embedding"]
            )
            return self._replace_embedding(output, embedding)
        output["aux"] = dict(output["aux"])
        output["aux"]["component_ablation"] = self.ablation
        return output


def _make_model(dataset: Any, base_config: dict[str, Any], variant: str, device: torch.device) -> nn.Module:
    kwargs = _resolved_model_config(dataset, base_config, "no_selective_head")
    if variant == "v5_full":
        model = DualPathNyquistGuardTSC(**kwargs, initial_gate_floor=0.5, spatial_channels=24)
    else:
        model = ComponentAblatedDualPath(
            **kwargs, initial_gate_floor=0.5, spatial_channels=24, ablation=variant
        )
    if model.selective_head is not None:
        raise RuntimeError("component ablation must not train a selective head")
    return model.to(device)


def _protocol_hash(root: Path, config_path: Path, base_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        config_path, base_path, Path(__file__),
        root / "src/nyquistguard/research/v5_dual_path.py",
        root / "src/nyquistguard/research/v4_observe_only_micro.py",
    ):
        digest.update(path.read_bytes())
    digest.update(b"v5.1-retrained-three-component-ablation-v1")
    return digest.hexdigest()


def _validate(config: dict[str, Any], source: dict[str, Any]) -> None:
    if tuple(config["datasets"]) != ABLATION_DATASETS:
        raise ValueError("component ablation datasets changed")
    if tuple(int(value) for value in config["seeds"]) != ABLATION_SEEDS:
        raise ValueError("component ablation seeds changed")
    if tuple(config["variants"]) != ABLATION_VARIANTS:
        raise ValueError("component ablation variants changed")
    if source.get("protocol_hash") != config["required_source_protocol_hash"]:
        raise ValueError("V5.1 source hash changed")
    if source.get("status") != "completed" or not source.get("decision", {}).get("passed"):
        raise ValueError("V5.1 source is not a completed PASS")
    if config["scientific_boundary"]["retrain_every_variant"] is not True:
        raise ValueError("all component variants must be retrained")


def _assert_idle(root: Path) -> None:
    path = root / "runs/dashboard_status.json"
    if not path.exists():
        return
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if state.get("status") == "running":
        raise RuntimeError(f"refusing ablation while {state.get('stage')} is running")


def _find_root(parent: Path, protocol_hash: str) -> Path | None:
    for path in sorted(parent.glob("v5_1_component_ablation__*"), reverse=True):
        manifest = path / "manifest.json"
        if manifest.exists():
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("protocol_hash") == protocol_hash:
                    return path
            except (OSError, json.JSONDecodeError):
                pass
    return None


def _train(
    dataset: Any, base_config: dict[str, Any], config: dict[str, Any], variant: str,
    state: dict[str, Tensor], protocol_hash: str, run_dir: Path, device: torch.device,
    deadline: float, resume: bool, seed: int,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    original_factory = training._new_model
    training._new_model = _make_model
    try:
        return training._train_variant(
            dataset, base_config, config, variant, state, protocol_hash, run_dir,
            device, deadline, resume, seed=seed,
        )
    finally:
        training._new_model = original_factory


def run_v5_1_component_ablation(
    project_root: str | Path, *, resume: bool = True, confirmed: bool = False
) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("V5.1 component ablation requires manual confirmation")
    root = Path(project_root).resolve()
    _assert_idle(root)
    started = time.monotonic()
    config_path = root / "configs/experiments/v5_1_component_ablation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = json.loads((root / config["source_report"]).read_text(encoding="utf-8"))
    _validate(config, source)
    base_path = root / config["base_config"]
    base_template = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_hash = _protocol_hash(root, config_path, base_path)
    parent = root / "runs/v5_1_component_ablation"
    run_root = _find_root(parent, protocol_hash) if resume else None
    if run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = parent / f"v5_1_component_ablation__2datasets__3seeds__{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_root / "config_frozen.yaml")
    report_path = run_root / "report.json"
    if resume and report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return cached
    manifest_path = run_root / "manifest.json"
    atomic_write_json(manifest_path, {
        "stage": config["stage"], "status": "running", "protocol_hash": protocol_hash,
        "manual_confirmation": True, "updated_at_utc": utc_now(),
    })
    deadline = started + float(config["wall_time_budget_seconds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    try:
        for dataset_id in ABLATION_DATASETS:
            dataset = prepare_v5_independent_dataset(root, dataset_id, confirmed_test_access=True)
            base_config = dict(base_template)
            base_config["batch_size"] = int(config["dataset_batch_sizes"][dataset_id])
            for seed in ABLATION_SEEDS:
                for variant in ABLATION_VARIANTS:
                    key = f"{dataset_id}__seed{seed}__{variant}"
                    run_dir = run_root / key
                    metrics_path = run_dir / "metrics.json"
                    payload = None
                    if resume and metrics_path.exists():
                        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if cached.get("protocol_hash") == protocol_hash:
                            payload = cached
                    if payload is None:
                        _seed_everything(seed)
                        initial = _make_model(dataset, base_config, variant, torch.device("cpu"))
                        state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
                        del initial
                        run_started = time.monotonic()
                        model, history = _train(
                            dataset, base_config, config, variant, state, protocol_hash,
                            run_dir, device, deadline, resume, seed,
                        )
                        test = training._evaluate_validation(
                            model, _view(dataset, "test"), base_config,
                            tuple(float(value) for value in config["test_rate_ratios"]), device,
                        )
                        payload = {
                            "status": "completed", "protocol_hash": protocol_hash,
                            "dataset_id": dataset_id, "seed": seed, "variant": variant,
                            "epochs_completed": len(history), "test": test,
                            "duration_seconds": time.monotonic() - run_started,
                            "test_used_for_selection": False,
                        }
                        atomic_write_json(metrics_path, payload)
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    results[key] = payload
        rows = []
        for dataset_id in ABLATION_DATASETS:
            for variant in ABLATION_VARIANTS:
                values = [
                    results[f"{dataset_id}__seed{seed}__{variant}"]["test"]
                    for seed in ABLATION_SEEDS
                ]
                rows.append({
                    "dataset_id": dataset_id, "variant": variant,
                    "mean_unseen_macro_f1": float(np.mean([v["mean_unseen_macro_f1"] for v in values])),
                    "full_rate_macro_f1": float(np.mean([v["full_rate_macro_f1"] for v in values])),
                })
        by_key = {(row["dataset_id"], row["variant"]): row for row in rows}
        for row in rows:
            full = by_key[(row["dataset_id"], "v5_full")]
            row["unseen_delta_vs_v5_full"] = float(row["mean_unseen_macro_f1"] - full["mean_unseen_macro_f1"])
            row["full_rate_delta_vs_v5_full"] = float(row["full_rate_macro_f1"] - full["full_rate_macro_f1"])
        report = {
            "status": "completed", "protocol_version": config["protocol_version"],
            "protocol_hash": protocol_hash, "manual_confirmation": True,
            "retrained_variants": True, "independent_confirmation": False,
            "datasets": list(ABLATION_DATASETS), "seeds": list(ABLATION_SEEDS),
            "variants": list(ABLATION_VARIANTS), "completed_runs": len(results),
            "rows": rows, "run_results": results, "device": str(device),
            "elapsed_seconds": time.monotonic() - started, "run_root": str(run_root),
            "later_stage_started": False, "finished_at_utc": utc_now(),
        }
        lines = [
            "# V5.1 retrained component ablation", "",
            "- All four variants were independently retrained; this is not inference-time masking.",
            "- This retrospective ablation is explanatory and is not a new independent confirmation.", "",
            "| Dataset | Variant | Mean unseen F1 | Delta vs full V5 | Full-rate F1 |",
            "|---|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['dataset_id']} | {row['variant']} | {row['mean_unseen_macro_f1']:.4f} | "
                f"{row['unseen_delta_vs_v5_full']:+.4f} | {row['full_rate_macro_f1']:.4f} |"
            )
        lines.extend(["", "No later stage was started automatically.", ""])
        markdown = "\n".join(lines)
        atomic_write_json(report_path, report)
        _atomic_write_text(run_root / "report.md", markdown)
        atomic_write_json(root / "reports/v5_1_component_ablation_report.json", report)
        _atomic_write_text(root / "reports/v5_1_component_ablation_report.md", markdown)
        atomic_write_json(manifest_path, {
            "stage": config["stage"], "status": "completed", "protocol_hash": protocol_hash,
            "completed_runs": len(results), "updated_at_utc": utc_now(),
        })
        return report
    except BaseException as error:
        atomic_write_json(manifest_path, {
            "stage": config["stage"], "status": "failed", "protocol_hash": protocol_hash,
            "error": f"{type(error).__name__}: {error}", "updated_at_utc": utc_now(),
        })
        raise
