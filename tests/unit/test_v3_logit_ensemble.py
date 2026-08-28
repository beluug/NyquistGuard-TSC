from __future__ import annotations

import pytest
import torch
from torch import nn

from nyquistguard.experiments.v3_logit_ensemble import FixedLogitEnsemble


class _Member(nn.Module):
    def __init__(self, logits: list[float], retained: float) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits))
        self.retained = retained

    def forward(self, x: torch.Tensor, rate: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        band = torch.ones(batch, 1, 2, device=x.device)
        gated = band * self.retained
        return {
            "logits": self.fixed_logits.to(x.device).repeat(batch, 1),
            "accept_probability": torch.full((batch,), 0.5, device=x.device),
            "band_features": band,
            "gated_band_features": gated,
        }


def test_fixed_logit_ensemble_averages_logits_and_retention() -> None:
    ensemble = FixedLogitEnsemble(_Member([2.0, 0.0], 0.8), _Member([0.0, 4.0], 0.2))
    output = ensemble(torch.zeros(3, 1, 4), torch.ones(3))
    assert torch.allclose(output["logits"], torch.tensor([[1.0, 2.0]]).repeat(3, 1))
    assert torch.allclose(output["retained_band_energy"], torch.full((3,), 0.5))


def test_fixed_logit_ensemble_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError, match="cannot both be zero"):
        FixedLogitEnsemble(_Member([1.0], 1.0), _Member([1.0], 1.0), 0.0, 0.0)
