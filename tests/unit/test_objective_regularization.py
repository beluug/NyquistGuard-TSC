import torch

from nyquistguard.losses.filter_regularization import FilterBankRegularization
from nyquistguard.losses.monotonicity import AcceptanceMonotonicityLoss
from nyquistguard.models.nyquistguard_tsc import NyquistGuardTSC


def _regularizer():
    return FilterBankRegularization(
        min_center_hz=1.0,
        max_center_hz=9.0,
        min_sigma_seconds=0.03,
        max_sigma_seconds=0.20,
        spacing_weight=1.0,
        coverage_weight=1.0,
        gate_degeneracy_weight=1.0,
        minimum_gate_softness=0.02,
    )


def test_collapsed_centers_are_penalized_more_than_covered_centers():
    sigmas = torch.full((5,), 0.10)
    collapsed = _regularizer()(torch.full((5,), 5.0), sigmas)
    covered = _regularizer()(torch.linspace(1.0, 9.0, 5), sigmas)
    assert collapsed["spacing"] > covered["spacing"]
    assert collapsed["coverage"] > covered["coverage"]
    assert collapsed["total"] > covered["total"]


def test_all_open_or_closed_gates_receive_degeneracy_penalty():
    centers = torch.linspace(1.0, 9.0, 5)
    sigmas = torch.full((5,), 0.10)
    hard = _regularizer()(centers, sigmas, torch.ones(3, 5))
    soft = _regularizer()(centers, sigmas, torch.full((3, 5), 0.5))
    assert hard["gate_degeneracy"] > soft["gate_degeneracy"]


def test_monotonicity_only_penalizes_unjustified_low_rate_increase():
    loss_fn = AcceptanceMonotonicityLoss(margin=0.05)
    q_high = torch.tensor([0.8, 0.7])
    ordered_low = torch.tensor([0.4, 0.65])
    violating_low = torch.tensor([0.9, 0.9])
    assert loss_fn(ordered_low, q_high).item() == 0.0
    assert loss_fn(violating_low, q_high).item() > 0.0


def test_discrete_ablation_has_comparable_total_capacity():
    common = dict(
        input_channels=6,
        num_classes=4,
        num_bands=16,
        pooled_positions=16,
        hidden_dim=64,
        encoder_depth=3,
        max_center_hz=20.0,
    )
    physical = NyquistGuardTSC(**common, filterbank_type="physical")
    discrete = NyquistGuardTSC(**common, filterbank_type="discrete")
    physical_count = sum(parameter.numel() for parameter in physical.parameters())
    discrete_count = sum(parameter.numel() for parameter in discrete.parameters())
    ratio = max(physical_count, discrete_count) / min(physical_count, discrete_count)
    assert ratio < 1.35

