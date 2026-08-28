"""Coverage-constrained selective classification risk."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SelectiveRiskLoss(nn.Module):
    def __init__(
        self,
        target_coverage: float = 0.8,
        coverage_weight: float = 5.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0,1]")
        if coverage_weight < 0:
            raise ValueError("coverage_weight must be nonnegative")
        self.target_coverage = float(target_coverage)
        self.coverage_weight = float(coverage_weight)
        self.epsilon = float(epsilon)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        accept_probability: Tensor,
    ) -> dict[str, Tensor]:
        if logits.ndim != 2 or targets.shape != logits.shape[:1]:
            raise ValueError("logits and targets must have shape [B,N] and [B]")
        if accept_probability.shape != targets.shape:
            raise ValueError("accept_probability must have shape [B]")
        q = accept_probability.clamp(0.0, 1.0)
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        risk = (q * per_sample_ce).sum() / (q.sum() + self.epsilon)
        coverage = q.mean()
        coverage_penalty = torch.relu(
            q.new_tensor(self.target_coverage) - coverage
        ).square()
        total = risk + self.coverage_weight * coverage_penalty
        return {
            "total": total,
            "risk": risk,
            "coverage_penalty": coverage_penalty,
            "coverage": coverage,
        }

