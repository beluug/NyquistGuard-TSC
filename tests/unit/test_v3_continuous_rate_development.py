from __future__ import annotations

import pytest

from nyquistguard.experiments.v3_core_micro import secondary_training_ratio


FROZEN = {
    "secondary_rate_sampling": {
        "mode": "identity_plus_continuous_uniform",
        "period_batches": 3,
        "identity_slots": 1,
        "uniform_min": 0.3,
        "uniform_max": 0.75,
    }
}


def test_continuous_rate_schedule_is_deterministic_and_bounded() -> None:
    values = [secondary_training_ratio(FROZEN, (1.0, 0.75, 0.5), 42, 2, i) for i in range(6)]
    repeated = [secondary_training_ratio(FROZEN, (1.0, 0.75, 0.5), 42, 2, i) for i in range(6)]
    assert values == repeated
    assert sum(value == 1.0 for value in values) == 2
    assert all(value == 1.0 or 0.3 <= value <= 0.75 for value in values)


def test_continuous_rate_schedule_rejects_changed_bounds() -> None:
    changed = {"secondary_rate_sampling": {**FROZEN["secondary_rate_sampling"], "uniform_min": 0.4}}
    with pytest.raises(ValueError, match="frozen design"):
        secondary_training_ratio(changed, (1.0, 0.75, 0.5), 42, 0, 1)
