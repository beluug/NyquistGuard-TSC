from __future__ import annotations

import pytest

from nyquistguard.experiments.v3_low_rate_development import resolve_secondary_ratios


def test_v3_low_rate_schedule_is_frozen() -> None:
    assert resolve_secondary_ratios(
        {"secondary_train_rate_ratios": [0.75, 0.5, 0.3]}
    ) == [0.75, 0.5, 0.3]


def test_v3_low_rate_schedule_rejects_posthoc_change() -> None:
    with pytest.raises(ValueError, match="must remain"):
        resolve_secondary_ratios(
            {"secondary_train_rate_ratios": [0.75, 0.5, 0.4]}
        )
