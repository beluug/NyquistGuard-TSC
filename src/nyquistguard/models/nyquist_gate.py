"""Nyquist observability from the full ideal complex-Gabor response energy."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .physical_filterbank import normalize_sampling_rates


class NyquistObservabilityGate(nn.Module):
    """Analytic one-sided Gaussian-response energy ratio.

    For ``|H(f)|^2 ∝ exp(-4*pi^2*sigma^2*(f-fc)^2)``, the common scale
    factor cancels from the finite-to-total integral ratio.  The resulting
    expression is monotonic in the Nyquist frequency and differentiable with
    respect to both center frequency and time scale.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(
        self,
        sampling_rate_hz: float | Tensor,
        center_frequencies_hz: Tensor,
        time_scales_seconds: Tensor,
        *,
        batch_size: int | None = None,
    ) -> Tensor:
        if center_frequencies_hz.ndim != 1 or time_scales_seconds.ndim != 1:
            raise ValueError("center frequencies and time scales must have shape [K]")
        if center_frequencies_hz.shape != time_scales_seconds.shape:
            raise ValueError("center frequencies and time scales must have matching shapes")
        if batch_size is None:
            rate_tensor = torch.as_tensor(sampling_rate_hz)
            batch_size = 1 if rate_tensor.ndim == 0 else int(rate_tensor.numel())
        rates = normalize_sampling_rates(
            sampling_rate_hz,
            batch_size,
            device=center_frequencies_hz.device,
        )
        if torch.any(center_frequencies_hz < 0) or torch.any(time_scales_seconds <= 0):
            raise ValueError("centers must be nonnegative and time scales positive")

        centers = center_frequencies_hz.float().unsqueeze(0)
        sigmas = time_scales_seconds.float().unsqueeze(0)
        nyquist = (rates.float() / 2.0).unsqueeze(1)
        scale = 2.0 * math.pi * sigmas
        lower_cdf_term = torch.erf(-scale * centers)
        upper_cdf_term = torch.erf(scale * (nyquist - centers))
        numerator = upper_cdf_term - lower_cdf_term
        denominator = (1.0 - lower_cdf_term).clamp_min(self.epsilon)
        gate = numerator / denominator
        return gate.clamp(0.0, 1.0).to(dtype=center_frequencies_hz.dtype)


def numerical_gabor_energy_ratio(
    sampling_rate_hz: Tensor,
    center_frequencies_hz: Tensor,
    time_scales_seconds: Tensor,
    *,
    frequency_upper_hz: float,
    grid_points: int = 65537,
) -> Tensor:
    """Reference trapezoidal integral used only for validation/diagnostics."""

    if grid_points < 1025:
        raise ValueError("grid_points must be at least 1025 for a stable reference")
    device = center_frequencies_hz.device
    frequency = torch.linspace(
        0.0,
        float(frequency_upper_hz),
        grid_points,
        device=device,
        dtype=torch.float64,
    )
    centers = center_frequencies_hz.double().view(-1, 1)
    sigmas = time_scales_seconds.double().view(-1, 1)
    response = torch.exp(
        -4.0 * math.pi**2 * sigmas.square() * (frequency.view(1, -1) - centers).square()
    )
    total = torch.trapezoid(response, frequency, dim=1)
    ratios = []
    for rate in sampling_rate_hz.reshape(-1):
        boundary = float(rate.detach().cpu().item()) / 2.0
        below = (frequency <= boundary).to(response.dtype)
        observable = torch.trapezoid(response * below, frequency, dim=1)
        ratios.append(observable / total.clamp_min(1e-15))
    return torch.stack(ratios).to(dtype=center_frequencies_hz.dtype)

