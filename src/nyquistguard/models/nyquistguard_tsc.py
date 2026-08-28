"""End-to-end NyquistGuard-TSC model."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .nyquist_gate import NyquistObservabilityGate
from .physical_filterbank import (
    DiscreteConvFilterBank,
    PhysicalGaborFilterBank,
    normalize_sampling_rates,
    validate_padding_mask,
)
from .selective_head import SelectiveHead
from .temporal_encoder import NormalizedPhysicalTimePool, TemporalEncoder


class NyquistGuardTSC(nn.Module):
    """Alias-aware sampling-rate-equivariant selective classifier."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        num_bands: int = 16,
        pooled_positions: int = 32,
        hidden_dim: int = 64,
        encoder_depth: int = 3,
        encoder_kernel_size: int = 5,
        dropout: float = 0.05,
        min_center_hz: float = 0.5,
        max_center_hz: float = 45.0,
        min_sigma_seconds: float = 0.015,
        max_sigma_seconds: float = 0.30,
        kernel_support_sigmas: float = 4.0,
        max_kernel_seconds: float = 1.0,
        filterbank_type: str = "physical",
        discrete_kernel_size: int = 31,
        use_nyquist_gate: bool = True,
        use_selective_head: bool = True,
        selective_hidden_dim: int = 64,
        reference_sampling_rate_hz: float = 100.0,
        timestamp_relative_tolerance: float = 0.05,
    ) -> None:
        super().__init__()
        if input_channels < 1 or num_classes < 2:
            raise ValueError("input_channels must be positive and num_classes at least 2")
        if filterbank_type not in {"physical", "discrete"}:
            raise ValueError("filterbank_type must be 'physical' or 'discrete'")
        if timestamp_relative_tolerance < 0:
            raise ValueError("timestamp_relative_tolerance must be nonnegative")
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.num_bands = int(num_bands)
        self.filterbank_type = filterbank_type
        self.use_nyquist_gate = bool(use_nyquist_gate)
        self.use_selective_head = bool(use_selective_head)
        self.timestamp_relative_tolerance = float(timestamp_relative_tolerance)

        common_filter_args = dict(
            input_channels=input_channels,
            num_bands=num_bands,
            min_center_hz=min_center_hz,
            max_center_hz=max_center_hz,
            min_sigma_seconds=min_sigma_seconds,
            max_sigma_seconds=max_sigma_seconds,
        )
        if filterbank_type == "physical":
            self.filterbank = PhysicalGaborFilterBank(
                **common_filter_args,
                kernel_support_sigmas=kernel_support_sigmas,
                max_kernel_seconds=max_kernel_seconds,
            )
        else:
            self.filterbank = DiscreteConvFilterBank(
                **common_filter_args,
                kernel_size=discrete_kernel_size,
            )
        self.nyquist_gate = NyquistObservabilityGate()
        self.time_pool = NormalizedPhysicalTimePool(pooled_positions)
        self.encoder = TemporalEncoder(
            num_bands,
            hidden_dim,
            encoder_depth,
            kernel_size=encoder_kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        if use_selective_head:
            self.selective_head = SelectiveHead(
                hidden_dim,
                num_bands,
                hidden_dim=selective_hidden_dim,
                reference_sampling_rate_hz=reference_sampling_rate_hz,
                dropout=dropout,
            )
        else:
            self.selective_head = None

    def _validate_timestamps(
        self,
        timestamps: Optional[Tensor],
        padding_mask: Tensor,
        sampling_rates: Tensor,
    ) -> Optional[Tensor]:
        if timestamps is None:
            return None
        if timestamps.shape != padding_mask.shape or not timestamps.is_floating_point():
            raise ValueError("timestamps must be floating Tensor[B,T]")
        timestamps = timestamps.to(device=padding_mask.device)
        for index in range(timestamps.shape[0]):
            valid_length = int((~padding_mask[index]).sum().detach().cpu().item())
            valid_times = timestamps[index, :valid_length]
            if not torch.isfinite(valid_times).all():
                raise ValueError("valid timestamps must be finite")
            if valid_length > 1:
                differences = torch.diff(valid_times)
                if torch.any(differences <= 0):
                    raise ValueError("valid timestamps must be strictly increasing")
                expected = sampling_rates[index].reciprocal().to(differences.dtype)
                relative_error = torch.max(torch.abs(differences - expected)) / expected
                if relative_error > self.timestamp_relative_tolerance:
                    raise ValueError(
                        "v1 physical convolution requires approximately uniform timestamps "
                        "consistent with sampling_rate_hz"
                    )
        return timestamps

    def forward(
        self,
        x: Tensor,
        sampling_rate_hz: float | Tensor,
        padding_mask: Optional[Tensor] = None,
        timestamps: Optional[Tensor] = None,
    ) -> dict[str, Tensor | dict]:
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"x must have shape [B,{self.input_channels},T]")
        rates = normalize_sampling_rates(
            sampling_rate_hz, x.shape[0], device=x.device
        )
        resolved_mask = validate_padding_mask(x, padding_mask)
        timestamps = self._validate_timestamps(timestamps, resolved_mask, rates)
        unpooled = self.filterbank(x, rates, resolved_mask)
        band_features = self.time_pool(unpooled, resolved_mask, timestamps)

        if self.use_nyquist_gate:
            gate = self.nyquist_gate(
                rates,
                self.filterbank.center_frequencies_hz,
                self.filterbank.time_scales_seconds,
                batch_size=x.shape[0],
            )
        else:
            gate = band_features.new_ones((x.shape[0], self.num_bands))
        gated_features = band_features * gate.to(band_features.dtype).unsqueeze(-1)
        encoded_sequence, embedding = self.encoder(gated_features)
        logits = self.classifier(embedding)
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(self.num_classes)

        if self.selective_head is None:
            accept_logit = embedding.new_zeros((x.shape[0],))
        else:
            accept_logit = self.selective_head(
                embedding,
                gate,
                rates,
                normalized_entropy.to(embedding.dtype),
            )
        accept_probability = torch.sigmoid(accept_logit)
        valid_lengths = (~resolved_mask).sum(dim=1)
        return {
            "logits": logits,
            "accept_logit": accept_logit,
            "accept_probability": accept_probability,
            "band_features": band_features,
            "gated_band_features": gated_features,
            "nyquist_gate": gate,
            "embedding": embedding,
            "aux": {
                "sampling_rate_hz": rates,
                "center_frequencies_hz": self.filterbank.center_frequencies_hz,
                "time_scales_seconds": self.filterbank.time_scales_seconds,
                "bandwidth_std_hz": self.filterbank.bandwidth_std_hz,
                "valid_lengths": valid_lengths,
                "encoded_sequence": encoded_sequence,
                "normalized_prediction_entropy": normalized_entropy,
                "filterbank_type": self.filterbank_type,
                "nyquist_gate_enabled": self.use_nyquist_gate,
                "selective_head_enabled": self.use_selective_head,
            },
        }

