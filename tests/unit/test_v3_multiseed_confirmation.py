from pathlib import Path

import yaml

from nyquistguard.experiments.v3_multiseed_confirmation import _validate_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_confirmation_matrix_is_complete_unique_and_excludes_development_seed() -> None:
    config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "v3_multiseed_confirmation.yaml"
        ).read_text(encoding="utf-8")
    )
    matrix = _validate_matrix(config)
    assert len(matrix) == 4
    assert set(matrix) == {
        ("basicmotions_uea", 42),
        ("basicmotions_uea", 2026),
        ("pamap2_uci", 42),
        ("pamap2_uci", 2026),
    }
    assert all(seed != 17 for _, seed in matrix)
    assert config["wall_time_budget_seconds"] == 540
