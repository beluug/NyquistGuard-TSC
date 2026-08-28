"""Common observable-band equivariance (CBE) loss."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


EquivarianceMode = Literal[
    "no_equivariance",
    "full_band_equivariance",
    "common_band_equivariance",
]
MaskMode = Literal["exact_min", "soft_min"]


class CommonBandEquivarianceLoss(nn.Module):
    """Align only frequency bands observable in both rate views."""

    def __init__(
        self,
        mode: EquivarianceMode = "common_band_equivariance",
        *,
        mask_mode: MaskMode = "exact_min",
        softmin_temperature: float = 0.05,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        valid_modes = {
            "no_equivariance",
            "full_band_equivariance",
            "common_band_equivariance",
        }
        if mode not in valid_modes:
            raise ValueError(f"unknown equivariance mode: {mode}")
        if mask_mode not in {"exact_min", "soft_min"}:
            raise ValueError(f"unknown common-band mask mode: {mask_mode}")
        if softmin_temperature <= 0:
            raise ValueError("softmin_temperature must be positive")
        self.mode = mode
        self.mask_mode = mask_mode
        self.softmin_temperature = float(softmin_temperature)
        self.epsilon = float(epsilon)

    def common_mask(self, gate_a: Tensor, gate_b: Tensor) -> Tensor:
        if gate_a.shape != gate_b.shape or gate_a.ndim != 2:
            raise ValueError("gate_a and gate_b must have matching shape [B,K]")
        if self.mask_mode == "exact_min":
            return torch.minimum(gate_a, gate_b)
        stacked = torch.stack((-gate_a, -gate_b), dim=0)
        mask = -self.softmin_temperature * torch.logsumexp(
            stacked / self.softmin_temperature, dim=0
        )
        return mask.clamp(0.0, 1.0)

    def forward(
        self,
        features_a: Tensor,
        features_b: Tensor,
        gate_a: Tensor,
        gate_b: Tensor,
    ) -> Tensor:
        if features_a.shape != features_b.shape or features_a.ndim != 3:
            raise ValueError("features must have matching shape [B,K,P]")
        if gate_a.shape != features_a.shape[:2] or gate_b.shape != features_a.shape[:2]:
            raise ValueError("gates must match feature shape [B,K]")
        if self.mode == "no_equivariance":
            return (features_a.sum() + features_b.sum() + gate_a.sum() + gate_b.sum()) * 0.0

        band_error = (features_a - features_b).square().mean(dim=-1)
        if self.mode == "full_band_equivariance":
            weights = torch.ones_like(gate_a)
        else:
            weights = self.common_mask(gate_a, gate_b)
        per_sample = (weights * band_error).sum(dim=1) / (
            weights.sum(dim=1) + self.epsilon
        )
        return per_sample.mean()

