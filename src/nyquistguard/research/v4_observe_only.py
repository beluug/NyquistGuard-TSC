"""V4 observe-only Nyquist model, isolated from the frozen v3.10 implementation."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from nyquistguard.models.nyquistguard_tsc import NyquistGuardTSC
from nyquistguard.models.physical_filterbank import (
    normalize_sampling_rates,
    validate_padding_mask,
)


class ObserveOnlyNyquistGuardTSC(NyquistGuardTSC):
    """Decouple classification features from analytic observability evidence.

    The physical filterbank and analytic Nyquist gate are retained.  Unlike v3.10,
    the gate does not multiply the features sent to the classifier.  It remains
    available to the selective/reliability path, while the classifier receives the
    unattenuated physical-band representation.  This class lives outside the frozen
    model package so an in-progress Full run cannot import it accidentally.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_nyquist_gate"] = True
        super().__init__(*args, **kwargs)
        self.nyquist_gate_mode = "observe_only"

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
        gate = self.nyquist_gate(
            rates,
            self.filterbank.center_frequencies_hz,
            self.filterbank.time_scales_seconds,
            batch_size=x.shape[0],
        )

        classifier_features = band_features
        observability_weighted = band_features * gate.to(
            band_features.dtype
        ).unsqueeze(-1)
        encoded_sequence, embedding = self.encoder(classifier_features)
        logits = self.classifier(embedding)
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
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
            "gated_band_features": classifier_features,
            "observability_weighted_band_features": observability_weighted,
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
                "nyquist_gate_enabled": True,
                "nyquist_gate_mode": self.nyquist_gate_mode,
                "selective_head_enabled": self.use_selective_head,
                "classifier_uses_observability_attenuation": False,
            },
        }

