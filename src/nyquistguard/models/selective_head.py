"""Sampling-sufficiency and selective acceptance head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SelectiveHead(nn.Module):
    """Predict acceptance from representation, observability, rate and entropy."""

    def __init__(
        self,
        embedding_dim: int,
        num_bands: int,
        *,
        hidden_dim: int = 64,
        reference_sampling_rate_hz: float = 100.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if reference_sampling_rate_hz <= 0:
            raise ValueError("reference_sampling_rate_hz must be positive")
        self.reference_sampling_rate_hz = float(reference_sampling_rate_hz)
        input_dim = embedding_dim + num_bands + 5
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embedding: Tensor,
        nyquist_gate: Tensor,
        sampling_rate_hz: Tensor,
        normalized_prediction_entropy: Tensor,
    ) -> Tensor:
        if embedding.ndim != 2 or nyquist_gate.ndim != 2:
            raise ValueError("embedding and nyquist_gate must have shapes [B,D] and [B,K]")
        batch_size = embedding.shape[0]
        if nyquist_gate.shape[0] != batch_size:
            raise ValueError("embedding and gate batch dimensions must match")
        if sampling_rate_hz.shape != (batch_size,):
            raise ValueError("sampling_rate_hz must have shape [B]")
        if normalized_prediction_entropy.shape != (batch_size,):
            raise ValueError("normalized_prediction_entropy must have shape [B]")
        rate_feature = torch.log1p(sampling_rate_hz.to(embedding.dtype)) / math.log1p(
            self.reference_sampling_rate_hz
        )
        gate_mean = nyquist_gate.mean(dim=1)
        gate_min = nyquist_gate.min(dim=1).values
        gate_std = nyquist_gate.std(dim=1, unbiased=False)
        statistics = torch.stack(
            [
                gate_mean,
                gate_min,
                gate_std,
                rate_feature,
                normalized_prediction_entropy,
            ],
            dim=1,
        ).to(embedding.dtype)
        inputs = torch.cat([embedding, nyquist_gate.to(embedding.dtype), statistics], dim=1)
        return self.network(inputs).squeeze(-1)

