"""V4.1 learnable residual Nyquist modulation for positive development."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from nyquistguard.models.nyquistguard_tsc import NyquistGuardTSC
from nyquistguard.models.physical_filterbank import normalize_sampling_rates, validate_padding_mask


class ResidualGateNyquistGuardTSC(NyquistGuardTSC):
    """Learn a per-band floor between hard gating and no attenuation.

    The effective classification multiplier is ``floor + (1-floor) * gate``.
    The analytic gate itself is unchanged and remains separately available for
    reliability estimation and mechanism audit.
    """

    def __init__(self, *args, initial_gate_floor: float = 0.5, **kwargs) -> None:
        if not 0.0 < initial_gate_floor < 1.0:
            raise ValueError("initial_gate_floor must be strictly between zero and one")
        kwargs["use_nyquist_gate"] = True
        super().__init__(*args, **kwargs)
        initial_logit = math.log(initial_gate_floor / (1.0 - initial_gate_floor))
        self.raw_gate_floor = nn.Parameter(torch.full((self.num_bands,), initial_logit))
        self.nyquist_gate_mode = "learnable_residual"

    @property
    def gate_floor(self) -> Tensor:
        return torch.sigmoid(self.raw_gate_floor)

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
        multiplier = floor.unsqueeze(0) + (1.0 - floor.unsqueeze(0)) * gate.to(band_features.dtype)
        classifier_features = band_features * multiplier.unsqueeze(-1)
        encoded_sequence, embedding = self.encoder(classifier_features)
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
            "embedding": embedding,
            "aux": {
                "sampling_rate_hz": rates,
                "center_frequencies_hz": self.filterbank.center_frequencies_hz,
                "time_scales_seconds": self.filterbank.time_scales_seconds,
                "bandwidth_std_hz": self.filterbank.bandwidth_std_hz,
                "valid_lengths": (~resolved_mask).sum(dim=1),
                "encoded_sequence": encoded_sequence,
                "normalized_prediction_entropy": normalized_entropy,
                "filterbank_type": self.filterbank_type,
                "nyquist_gate_enabled": True,
                "nyquist_gate_mode": self.nyquist_gate_mode,
                "selective_head_enabled": self.use_selective_head,
                "classifier_uses_observability_attenuation": True,
                "classifier_observability_attenuation_is_residual": True,
            },
        }
