from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from experiment_dashboard import STAGE_TASKS
from nyquistguard.data.full_datasets import FULL_DATASETS
from nyquistguard.experiments.full import FULL_METHODS, FULL_SEEDS
from nyquistguard.research.v5_full_extension import (
    _paired_statistics,
    _validate_sources,
    run_v5_1_full_extension,
    v5_full_extension_tasks,
)
from nyquistguard.research.v5_component_ablation import (
    ABLATION_DATASETS,
    ABLATION_SEEDS,
    ABLATION_VARIANTS,
    ComponentAblatedDualPath,
    run_v5_1_component_ablation,
)


ROOT = Path(__file__).resolve().parents[2]


def test_manual_confirmation_is_required_before_io(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="manual confirmation"):
        run_v5_1_full_extension(tmp_path)


def test_matrix_and_dashboard_have_63_unique_tasks() -> None:
    tasks = v5_full_extension_tasks()
    assert len(tasks) == 63
    assert len(set(tasks)) == 63
    assert STAGE_TASKS["v5_1_full_extension"] == tasks


def test_frozen_sources_are_complete_and_reused() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v5_1_full_extension.yaml").read_text(encoding="utf-8")
    )
    source = _validate_sources(ROOT, config)
    assert tuple(config["design"]["datasets"]) == FULL_DATASETS
    assert tuple(config["design"]["seeds"]) == FULL_SEEDS
    assert len(source["aggregate"]["rows"]) == 210
    assert config["scientific_boundaries"]["train_only_candidate_runs"] == 30
    assert config["scientific_boundaries"]["reuse_original_full_baseline_runs"] == 210


def test_paired_statistics_uses_dataset_means() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v5_1_full_extension.yaml").read_text(encoding="utf-8")
    )
    source = json.loads((ROOT / "reports/full_report.json").read_text(encoding="utf-8"))
    source_rows = source["aggregate"]["rows"]
    by_v3 = {
        (row["dataset_id"], int(row["seed"])): row
        for row in source_rows if row["method"] == "v3_10"
    }
    candidate_rows = [
        {
            "dataset_id": dataset_id,
            "seed": seed,
            "mean_unseen_macro_f1": by_v3[(dataset_id, seed)]["mean_unseen_macro_f1"] + 0.01,
        }
        for dataset_id in FULL_DATASETS
        for seed in FULL_SEEDS
    ]
    stats = _paired_statistics(candidate_rows, source_rows, config)
    assert set(stats) == set(FULL_METHODS)
    assert stats["v3_10"]["mean_delta"] == pytest.approx(0.01)
    assert stats["v3_10"]["positive_dataset_count"] == 10


def test_efficiency_protocol_is_sequential_and_inference_only() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v5_1_efficiency.yaml").read_text(encoding="utf-8")
    )
    assert config["sequential_only"] is True
    assert config["training_forbidden"] is True
    assert tuple(config["datasets"]) == (
        "self_regulation_scp1_uea",
        "hand_movement_direction_uea",
        "racket_sports_uea",
        "heartbeat_uea",
    )
    assert config["roles"] == ["v4_1_residual_gate", "v5_dual_path"]


def test_component_ablation_requires_manual_confirmation(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="manual confirmation"):
        run_v5_1_component_ablation(tmp_path)


def test_component_ablation_matrix_is_retrained_and_frozen() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v5_1_component_ablation.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(config["datasets"]) == ABLATION_DATASETS
    assert tuple(config["seeds"]) == ABLATION_SEEDS
    assert tuple(config["variants"]) == ABLATION_VARIANTS
    assert len(ABLATION_DATASETS) * len(ABLATION_SEEDS) * len(ABLATION_VARIANTS) == 24
    assert config["scientific_boundary"]["retrain_every_variant"] is True


def test_each_component_ablation_has_a_valid_forward() -> None:
    kwargs = dict(
        input_channels=3, num_classes=4, num_bands=8, pooled_positions=16,
        hidden_dim=16, encoder_depth=1, encoder_kernel_size=3, dropout=0.0,
        min_center_hz=0.2, min_sigma_seconds=0.015, max_sigma_seconds=0.5,
        kernel_support_sigmas=4.0, max_kernel_seconds=1.0,
        filterbank_type="physical", discrete_kernel_size=15,
        use_nyquist_gate=True, use_selective_head=False, selective_hidden_dim=16,
        timestamp_relative_tolerance=0.05, initial_gate_floor=0.5,
        spatial_channels=8,
    )
    x = torch.randn(2, 3, 64)
    for variant in ABLATION_VARIANTS[1:]:
        output = ComponentAblatedDualPath(**kwargs, ablation=variant)(x, 50.0)
        assert output["logits"].shape == (2, 4)
        assert output["aux"]["component_ablation"] == variant
