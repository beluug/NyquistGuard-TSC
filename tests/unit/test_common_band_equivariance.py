import torch

from nyquistguard.losses.common_band_equivariance import CommonBandEquivarianceLoss


def test_identical_views_have_zero_loss():
    features = torch.randn(2, 4, 7)
    gates = torch.rand(2, 4)
    loss = CommonBandEquivarianceLoss()(features, features, gates, gates)
    assert loss.item() == 0.0


def test_common_band_masks_unobservable_high_frequency_perturbation():
    base = torch.zeros(1, 3, 5)
    changed_high = base.clone()
    changed_high[:, 2] = 10.0
    high_rate_gate = torch.ones(1, 3)
    low_rate_gate = torch.tensor([[1.0, 0.95, 0.002]])
    cbe = CommonBandEquivarianceLoss(mode="common_band_equivariance")(
        base, changed_high, high_rate_gate, low_rate_gate
    )
    full = CommonBandEquivarianceLoss(mode="full_band_equivariance")(
        base, changed_high, high_rate_gate, low_rate_gate
    )
    assert cbe < 0.01 * full


def test_perturbing_common_band_increases_cbe():
    base = torch.zeros(1, 3, 5)
    high_only = base.clone()
    high_only[:, 2] = 2.0
    common_changed = high_only.clone()
    common_changed[:, 0] = 2.0
    high_gate = torch.ones(1, 3)
    low_gate = torch.tensor([[1.0, 0.9, 0.001]])
    loss_fn = CommonBandEquivarianceLoss()
    high_loss = loss_fn(base, high_only, high_gate, low_gate)
    common_loss = loss_fn(base, common_changed, high_gate, low_gate)
    assert common_loss > 100.0 * high_loss


def test_no_equivariance_is_differentiable_zero():
    a = torch.randn(2, 3, 4, requires_grad=True)
    b = torch.randn(2, 3, 4, requires_grad=True)
    gates = torch.rand(2, 3, requires_grad=True)
    loss = CommonBandEquivarianceLoss(mode="no_equivariance")(a, b, gates, gates)
    loss.backward()
    assert loss.item() == 0.0
    assert a.grad is not None and b.grad is not None and gates.grad is not None

