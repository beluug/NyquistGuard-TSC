from pathlib import Path

import yaml

from nyquistguard.data import load_prepared_dataset
from nyquistguard.experiments.v3_core_micro import v3_core_objective_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v3_core_protocol_is_frozen_and_bounded() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "v3_core_micro.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["datasets"] == ["basicmotions_uea", "pamap2_uci"]
    assert config["seed"] == 17
    assert config["epochs"] == 18
    assert config["wall_time_budget_seconds"] == 540


def test_v3_core_refinement_matches_v1_budget_and_keeps_gates() -> None:
    first = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "v3_core_micro.yaml").read_text(
            encoding="utf-8"
        )
    )
    refined = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "v3_core_refinement.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert refined["epochs"] == 30
    assert refined["checkpoint_selection"]["split"] == "validation"
    assert refined["checkpoint_selection"]["early_stopping_patience"] == 7
    assert refined["development_gates"] == first["development_gates"]


def test_v3_core_objective_removes_old_selector_and_cbe_losses() -> None:
    base = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    dataset = load_prepared_dataset(
        PROJECT_ROOT / "data" / "processed" / "pilot_v1" / "basicmotions_uea.npz"
    )
    objective = v3_core_objective_config(dataset, base)
    assert objective["lambda_cbe"] == 0.0
    assert objective["lambda_selective"] == 0.0
    assert objective["lambda_monotonicity"] == 0.0
    assert objective["lambda_filter_regularization"] > 0.0
