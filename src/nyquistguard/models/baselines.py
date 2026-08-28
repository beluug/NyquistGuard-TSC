"""Compact fixed-rate and multirate TCN baselines for the pilot matrix."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class TCNClassifier(nn.Module):
    """Rate-unaware TCN used for both fixed-rate and augmentation baselines."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        depth: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.stem = nn.Sequential(nn.Conv1d(input_channels, hidden_dim, 1), nn.BatchNorm1d(hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(
            *(ResidualTCNBlock(hidden_dim, kernel_size, 2**index, dropout) for index in range(depth))
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor, sampling_rate_hz: float | Tensor | None = None) -> dict[str, Tensor]:
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"x must have shape [B,{self.input_channels},T]")
        embedding = self.blocks(self.stem(x)).mean(dim=-1)
        logits = self.classifier(embedding)
        confidence = torch.softmax(logits.float(), dim=-1).amax(dim=-1)
        return {"logits": logits, "accept_probability": confidence, "embedding": embedding}


__all__ = ["TCNClassifier"]
