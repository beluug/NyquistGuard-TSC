"""V5 rate-conditioned dual-path NyquistGuard research candidate.

This module is deliberately isolated from every frozen Pilot/Full/V4 runner.
It preserves the physical Nyquist-aware path while adding a signed,
channel-aware waveform path before any across-channel magnitude reduction.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nyquistguard.models.physical_filterbank import (
    normalize_sampling_rates,
    validate_padding_mask,
)
from nyquistguard.models.temporal_encoder import TemporalEncoder, _group_count
from nyquistguard.research.v4_residual_gate import ResidualGateNyquistGuardTSC


def _masked_channel_standardize(x: Tensor, padding_mask: Tensor, epsilon: float) -> Tensor:
    """Per-case, per-channel standardization using valid time points only."""

    valid = (~padding_mask).unsqueeze(1).to(dtype=x.dtype)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(dim=-1, keepdim=True) / count
    centered = (x - mean) * valid
    variance = centered.square().sum(dim=-1, keepdim=True) / count
    standardized = centered * torch.rsqrt(variance + epsilon)
    return standardized.masked_fill(padding_mask.unsqueeze(1), 0.0)


class TemporalStatisticsSummary(nn.Module):
    """Summarize a fixed physical-time sequence without relying on one mean."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        mean = sequence.mean(dim=-1)
        standard_deviation = torch.sqrt(
            (sequence - mean.unsqueeze(-1)).square().mean(dim=-1).clamp_min(1e-8)
        )
        maximum = sequence.amax(dim=-1)
        return self.projection(torch.cat((mean, standard_deviation, maximum), dim=1))


class DualPathNyquistGuardTSC(ResidualGateNyquistGuardTSC):
    """Physical/signed-waveform dual-path candidate for V5 development.

    The physical path retains the analytic gate and learnable residual floor.
    The waveform path retains signed cross-channel structure that is otherwise
    lost when quadrature magnitudes are mixed across channels.  A bounded
    controller uses sampling rate and analytic observability only to fuse the
    two embeddings; it never deletes either path.
    """

    def __init__(
        self,
        *args,
        spatial_channels: int = 24,
        input_standardization_epsilon: float = 1e-5,
        reference_sampling_rate_hz: float = 100.0,
        **kwargs,
    ) -> None:
        if spatial_channels < 2:
            raise ValueError("spatial_channels must be at least 2")
        if input_standardization_epsilon <= 0:
            raise ValueError("input_standardization_epsilon must be positive")
        super().__init__(
            *args,
            reference_sampling_rate_hz=reference_sampling_rate_hz,
            **kwargs,
        )
        hidden_dim = int(self.classifier.in_features)
        encoder_depth = len(self.encoder.blocks)
        first_block = self.encoder.blocks[0]
        kernel_size = int(first_block.depthwise.kernel_size[0])
        dropout = float(first_block.dropout.p)

        self.spatial_channels = int(spatial_channels)
        self.input_standardization_epsilon = float(input_standardization_epsilon)
        self.reference_sampling_rate_hz = float(reference_sampling_rate_hz)
        self.spatial_projection = nn.Sequential(
            nn.Conv1d(self.input_channels, self.spatial_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(self.spatial_channels), self.spatial_channels),
            nn.SiLU(),
        )
        self.spatial_encoder = TemporalEncoder(
            self.spatial_channels,
            hidden_dim,
            encoder_depth,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.physical_summary = TemporalStatisticsSummary(hidden_dim, dropout)
        self.spatial_summary = TemporalStatisticsSummary(hidden_dim, dropout)
        controller_hidden = max(8, hidden_dim // 4)
        self.fusion_controller = nn.Sequential(
            nn.Linear(3, controller_hidden),
            nn.SiLU(),
            nn.Linear(controller_hidden, 1),
        )
        self.cross_path_residual = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Begin as an equal convex blend with no cross-path residual.  This is
        # stable and still lets gradients reach both paths on the first step.
        nn.init.zeros_(self.fusion_controller[-1].weight)
        nn.init.zeros_(self.fusion_controller[-1].bias)
        nn.init.zeros_(self.cross_path_residual[-1].weight)
        nn.init.zeros_(self.cross_path_residual[-1].bias)
        self.nyquist_gate_mode = "v5_dual_path_residual"

    def forward(
        self,
        x: Tensor,
        sampling_rate_hz: float | Tensor,
        padding_mask: Optional[Tensor] = None,
        timestamps: Optional[Tensor] = None,
    ) -> dict[str, Tensor | dict]:
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"x must have shape [B,{self.input_channels},T]")
        rates = normalize_sampling_rates(sampling_rate_hz, x.shape[0], device=x.device)
        resolved_mask = validate_padding_mask(x, padding_mask)
        timestamps = self._validate_timestamps(timestamps, resolved_mask, rates)

        unpooled = self.filterbank(x, rates, resolved_mask)
        band_features = self.time_pool(unpooled, resolved_mask, timestamps)
        gate = self.nyquist_gate(
            rates,
            self.filterbank.center_frequencies_hz,
            self.filterbank.time_scales_seconds,
            batch_size=x.shape[0],
        )
        floor = self.gate_floor.to(band_features.dtype)
        multiplier = floor.unsqueeze(0) + (1.0 - floor.unsqueeze(0)) * gate.to(
            band_features.dtype
        )
        classifier_features = band_features * multiplier.unsqueeze(-1)
        physical_sequence, _ = self.encoder(classifier_features)
        physical_embedding = self.physical_summary(physical_sequence)

        standardized = _masked_channel_standardize(
            x, resolved_mask, self.input_standardization_epsilon
        )
        spatial_unpooled = self.spatial_projection(standardized)
        spatial_features = self.time_pool(spatial_unpooled, resolved_mask, timestamps)
        spatial_sequence, _ = self.spatial_encoder(spatial_features)
        spatial_embedding = self.spatial_summary(spatial_sequence)

        rate_feature = torch.log2(
            rates.float().clamp_min(1e-6) / self.reference_sampling_rate_hz
        ).clamp(-4.0, 4.0)
        controller_features = torch.stack(
            (rate_feature, gate.float().mean(dim=1), gate.float().amin(dim=1)), dim=1
        ).to(dtype=physical_embedding.dtype)
        physical_weight = torch.sigmoid(self.fusion_controller(controller_features))
        convex_embedding = (
            physical_weight * physical_embedding
            + (1.0 - physical_weight) * spatial_embedding
        )
        cross_residual = self.cross_path_residual(
            torch.cat((physical_embedding, spatial_embedding), dim=1)
        )
        embedding = self.fusion_norm(convex_embedding + cross_residual)

        logits = self.classifier(embedding)
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(self.num_classes)
        if self.selective_head is None:
            accept_logit = embedding.new_zeros((x.shape[0],))
        else:
            accept_logit = self.selective_head(
                embedding, gate, rates, normalized_entropy.to(embedding.dtype)
            )
        return {
            "logits": logits,
            "accept_logit": accept_logit,
            "accept_probability": torch.sigmoid(accept_logit),
            "band_features": band_features,
            "gated_band_features": classifier_features,
            "nyquist_gate": gate,
            "effective_gate_multiplier": multiplier,
            "gate_floor": floor,
            "spatial_features": spatial_features,
            "physical_embedding": physical_embedding,
            "spatial_embedding": spatial_embedding,
            "fusion_physical_weight": physical_weight.squeeze(1),
            "embedding": embedding,
            "aux": {
                "sampling_rate_hz": rates,
                "center_frequencies_hz": self.filterbank.center_frequencies_hz,
                "time_scales_seconds": self.filterbank.time_scales_seconds,
                "bandwidth_std_hz": self.filterbank.bandwidth_std_hz,
                "valid_lengths": (~resolved_mask).sum(dim=1),
                "encoded_sequence": physical_sequence,
                "spatial_encoded_sequence": spatial_sequence,
                "normalized_prediction_entropy": normalized_entropy,
                "filterbank_type": self.filterbank_type,
                "nyquist_gate_enabled": True,
                "nyquist_gate_mode": self.nyquist_gate_mode,
                "selective_head_enabled": self.use_selective_head,
                "classifier_uses_observability_attenuation": True,
                "classifier_observability_attenuation_is_residual": True,
                "signed_spatial_bypass_enabled": True,
                "temporal_statistics_summary": ("mean", "standard_deviation", "maximum"),
            },
        }
