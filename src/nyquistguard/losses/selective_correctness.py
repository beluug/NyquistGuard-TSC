"""Detached correctness-calibration objective for the v2 selector probe."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DetachedCorrectnessSelectiveLoss(nn.Module):
    """Fit acceptance to detached true-class probability.

    The target is deliberately detached so the selector cannot improve its loss
    by manipulating classifier logits.  The caller must also detach selector
    inputs when isolation from the shared representation is required.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        accept_logit: Tensor,
    ) -> dict[str, Tensor]:
        if logits.ndim != 2 or targets.shape != logits.shape[:1]:
            raise ValueError("logits and targets must have shape [B,N] and [B]")
        if accept_logit.shape != targets.shape:
            raise ValueError("accept_logit must have shape [B]")
        probabilities = torch.softmax(logits.detach().float(), dim=-1)
        soft_correctness = probabilities.gather(1, targets[:, None]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(
            accept_logit.float(), soft_correctness
        )
        return {
            "total": loss,
            "soft_correctness_mean": soft_correctness.mean(),
            "acceptance_mean": torch.sigmoid(accept_logit.float()).mean(),
        }

