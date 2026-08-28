"""Normalized physical-time pooling and lightweight residual encoder."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class NormalizedPhysicalTimePool(nn.Module):
    """Map every valid time window to the same ``P`` normalized positions."""

    def __init__(self, output_positions: int) -> None:
        super().__init__()
        if output_positions < 2:
            raise ValueError("output_positions must be at least 2")
        self.output_positions = int(output_positions)

    def forward(
        self,
        features: Tensor,
        padding_mask: Tensor,
        timestamps: Optional[Tensor] = None,
    ) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,K,T]")
        batch_size, _, length = features.shape
        if padding_mask.shape != (batch_size, length) or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be BoolTensor[B,T]")
        if timestamps is not None and timestamps.shape != (batch_size, length):
            raise ValueError("timestamps must have shape [B,T]")

        pooled = []
        valid_lengths = (~padding_mask).sum(dim=1)
        for index in range(batch_size):
            valid_length = int(valid_lengths[index].detach().cpu().item())
            sequence = features[index : index + 1, :, :valid_length]
            if valid_length == 1:
                pooled.append(sequence.expand(-1, -1, self.output_positions))
                continue
            if timestamps is None:
                pooled.append(
                    F.interpolate(
                        sequence,
                        size=self.output_positions,
                        mode="linear",
                        align_corners=True,
                    )
                )
                continue

            times = timestamps[index, :valid_length]
            if not torch.isfinite(times).all() or torch.any(torch.diff(times) <= 0):
                raise ValueError("valid timestamps must be finite and strictly increasing")
            query = torch.linspace(
                0.0,
                1.0,
                self.output_positions,
                device=features.device,
                dtype=features.dtype,
            )
            query = times[0].to(features.dtype) + query * (
                times[-1] - times[0]
            ).to(features.dtype)
            right = torch.searchsorted(times.contiguous(), query.to(times.dtype), right=False)
            right = right.clamp(1, valid_length - 1)
            left = right - 1
            t_left = times.index_select(0, left).to(features.dtype)
            t_right = times.index_select(0, right).to(features.dtype)
            weight = ((query - t_left) / (t_right - t_left).clamp_min(1e-12)).view(1, 1, -1)
            left_values = sequence.index_select(-1, left)
            right_values = sequence.index_select(-1, right)
            pooled.append(left_values + weight * (right_values - left_values))
        return torch.cat(pooled, dim=0)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise_expand = nn.Conv1d(channels, 2 * channels, 1)
        self.pointwise_project = nn.Conv1d(2 * channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.depthwise(x)
        x = F.silu(self.pointwise_expand(x))
        x = self.dropout(self.pointwise_project(x))
        return residual + x


class TemporalEncoder(nn.Module):
    """Small depthwise-separable residual TCN."""

    def __init__(
        self,
        input_bands: int,
        hidden_dim: int = 64,
        depth: int = 3,
        *,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.input_projection = nn.Conv1d(input_bands, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** (index % 3),
                    dropout=dropout,
                )
                for index in range(depth)
            ]
        )
        self.output_norm = nn.GroupNorm(_group_count(hidden_dim), hidden_dim)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        sequence = self.input_projection(features)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = F.silu(self.output_norm(sequence))
        embedding = sequence.mean(dim=-1)
        return sequence, embedding

