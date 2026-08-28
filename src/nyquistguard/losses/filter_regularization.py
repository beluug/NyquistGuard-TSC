"""Anti-collapse regularization and diagnostics for physical filter bands."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FilterBankRegularization(nn.Module):
    def __init__(
        self,
        *,
        min_center_hz: float,
        max_center_hz: float,
        min_sigma_seconds: float,
        max_sigma_seconds: float,
        minimum_spacing_fraction: float = 0.35,
        spacing_weight: float = 1.0,
        coverage_weight: float = 0.1,
        bounds_weight: float = 1.0,
        gate_degeneracy_weight: float = 0.1,
        minimum_gate_softness: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0 <= minimum_spacing_fraction <= 1:
            raise ValueError("minimum_spacing_fraction must be in [0,1]")
        self.min_center_hz = float(min_center_hz)
        self.max_center_hz = float(max_center_hz)
        self.min_sigma_seconds = float(min_sigma_seconds)
        self.max_sigma_seconds = float(max_sigma_seconds)
        self.minimum_spacing_fraction = float(minimum_spacing_fraction)
        self.spacing_weight = float(spacing_weight)
        self.coverage_weight = float(coverage_weight)
        self.bounds_weight = float(bounds_weight)
        self.gate_degeneracy_weight = float(gate_degeneracy_weight)
        self.minimum_gate_softness = float(minimum_gate_softness)

    def forward(
        self,
        center_frequencies_hz: Tensor,
        time_scales_seconds: Tensor,
        gates: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if center_frequencies_hz.ndim != 1 or center_frequencies_hz.numel() < 1:
            raise ValueError("center_frequencies_hz must have shape [K]")
        if time_scales_seconds.shape != center_frequencies_hz.shape:
            raise ValueError("time scales must match center frequencies")
        sorted_centers = torch.sort(center_frequencies_hz).values
        span = max(self.max_center_hz - self.min_center_hz, 1e-8)
        if sorted_centers.numel() > 1:
            ideal_spacing = span / (sorted_centers.numel() - 1)
            minimum_spacing = self.minimum_spacing_fraction * ideal_spacing
            spacing = torch.relu(minimum_spacing - torch.diff(sorted_centers))
            spacing_penalty = spacing.square().mean() / (span**2)
            anchors = torch.linspace(
                self.min_center_hz,
                self.max_center_hz,
                sorted_centers.numel(),
                device=sorted_centers.device,
                dtype=sorted_centers.dtype,
            )
            coverage_penalty = ((sorted_centers - anchors) / span).square().mean()
        else:
            spacing_penalty = sorted_centers.sum() * 0.0
            midpoint = 0.5 * (self.min_center_hz + self.max_center_hz)
            coverage_penalty = ((sorted_centers - midpoint) / span).square().mean()

        center_bounds = (
            torch.relu(self.min_center_hz - center_frequencies_hz).square()
            + torch.relu(center_frequencies_hz - self.max_center_hz).square()
        ).mean() / (span**2)
        sigma_span = max(self.max_sigma_seconds - self.min_sigma_seconds, 1e-8)
        sigma_bounds = (
            torch.relu(self.min_sigma_seconds - time_scales_seconds).square()
            + torch.relu(time_scales_seconds - self.max_sigma_seconds).square()
        ).mean() / (sigma_span**2)
        bounds_penalty = center_bounds + sigma_bounds

        if gates is None:
            gate_penalty = center_frequencies_hz.sum() * 0.0
        else:
            if gates.ndim != 2 or gates.shape[1] != center_frequencies_hz.numel():
                raise ValueError("gates must have shape [B,K]")
            softness = (gates * (1.0 - gates)).mean()
            gate_penalty = torch.relu(
                gates.new_tensor(self.minimum_gate_softness) - softness
            ).square()

        total = (
            self.spacing_weight * spacing_penalty
            + self.coverage_weight * coverage_penalty
            + self.bounds_weight * bounds_penalty
            + self.gate_degeneracy_weight * gate_penalty
        )
        return {
            "total": total,
            "spacing": spacing_penalty,
            "coverage": coverage_penalty,
            "bounds": bounds_penalty,
            "gate_degeneracy": gate_penalty,
        }

