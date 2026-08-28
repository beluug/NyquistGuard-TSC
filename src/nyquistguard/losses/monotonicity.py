"""Soft pairwise acceptance monotonicity constraint."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AcceptanceMonotonicityLoss(nn.Module):
    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be nonnegative")
        self.margin = float(margin)

    def forward(
        self,
        q_low: Tensor,
        q_high: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        if q_low.shape != q_high.shape:
            raise ValueError("q_low and q_high must have matching shapes")
        penalties = torch.relu(q_low - q_high - self.margin)
        if pair_mask is not None:
            if pair_mask.shape != penalties.shape or pair_mask.dtype != torch.bool:
                raise ValueError("pair_mask must be BoolTensor matching q_low")
            if not torch.any(pair_mask):
                return (q_low.sum() + q_high.sum()) * 0.0
            penalties = penalties[pair_mask]
        return penalties.mean()

