import pytest

from nyquistguard.experiments.v3_core_micro import rate_robust_selection_score


def test_rate_robust_selection_balances_full_and_unseen_rates() -> None:
    assert rate_robust_selection_score(0.8, 0.6, 0.5, 0.5) == pytest.approx(0.7)
    assert rate_robust_selection_score(0.8, 0.6, 1.0, 3.0) == pytest.approx(0.65)


def test_rate_robust_selection_rejects_empty_weighting() -> None:
    with pytest.raises(ValueError):
        rate_robust_selection_score(0.8, 0.6, 0.0, 0.0)
