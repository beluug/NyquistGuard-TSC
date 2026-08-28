from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nyquistguard.experiments.v3_10_independent_confirmation import (
    validate_confirmation_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "v3_10_independent_confirmation.yaml"
        ).read_text(encoding="utf-8")
    )


def test_confirmation_matrix_is_complete_paired_and_fresh() -> None:
    config = _config()
    matrix = validate_confirmation_matrix(config)
    assert len(matrix) == 8
    assert {seed for _, seed, _ in matrix} == {31415, 27182}
    assert not ({seed for _, seed, _ in matrix} & {17, 42, 2026})
    for dataset in config["datasets"]:
        for seed in config["confirmation_seeds"]:
            assert (dataset, seed, "v1_control") in matrix
            assert (dataset, seed, "v3_10_candidate") in matrix


def test_confirmation_matrix_rejects_a_development_seed() -> None:
    config = _config()
    config["confirmation_seeds"] = [17, 27182]
    with pytest.raises(ValueError, match="development seeds"):
        validate_confirmation_matrix(config)
