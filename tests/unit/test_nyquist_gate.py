import torch

from nyquistguard.models.nyquist_gate import (
    NyquistObservabilityGate,
    numerical_gabor_energy_ratio,
)


def test_gate_is_bounded_and_monotonic_in_sampling_rate():
    gate = NyquistObservabilityGate()
    centers = torch.tensor([2.0, 8.0, 18.0, 35.0])
    sigmas = torch.tensor([0.12, 0.10, 0.08, 0.06])
    values = gate(torch.tensor([20.0, 50.0, 100.0]), centers, sigmas)
    assert values.shape == (3, 4)
    assert torch.all((0.0 <= values) & (values <= 1.0))
    assert torch.all(values[0] <= values[1] + 1e-7)
    assert torch.all(values[1] <= values[2] + 1e-7)
    assert values[0, -1] < 0.05
    assert values[-1, -1] > 0.90


def test_gate_matches_high_resolution_numerical_integral():
    gate = NyquistObservabilityGate()
    rates = torch.tensor([20.0, 50.0, 100.0])
    centers = torch.tensor([2.0, 11.0, 32.0])
    sigmas = torch.tensor([0.15, 0.08, 0.04])
    analytic = gate(rates, centers, sigmas)
    numerical = numerical_gabor_energy_ratio(
        rates,
        centers,
        sigmas,
        frequency_upper_hz=160.0,
        grid_points=32769,
    )
    assert torch.max(torch.abs(analytic - numerical)) < 7e-4


def test_gate_gradients_reach_center_and_time_scale():
    centers = torch.tensor([5.0, 16.0, 28.0], requires_grad=True)
    sigmas = torch.tensor([0.09, 0.06, 0.04], requires_grad=True)
    values = NyquistObservabilityGate()(torch.tensor([30.0, 55.0]), centers, sigmas)
    values.sum().backward()
    for gradient in (centers.grad, sigmas.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

