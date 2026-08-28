"""Composed NyquistGuard-TSC training objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common_band_equivariance import CommonBandEquivarianceLoss
from .filter_regularization import FilterBankRegularization
from .monotonicity import AcceptanceMonotonicityLoss
from .selective_risk import SelectiveRiskLoss


class NyquistGuardObjective(nn.Module):
    """Compute all configured objective terms from one or two model outputs."""

    def __init__(
        self,
        *,
        lambda_cbe: float = 1.0,
        lambda_selective: float = 1.0,
        lambda_monotonicity: float = 0.1,
        lambda_filter_regularization: float = 0.01,
        target_coverage: float = 0.8,
        coverage_weight: float = 5.0,
        monotonicity_margin: float = 0.0,
        equivariance_mode: str = "common_band_equivariance",
        common_band_mask: str = "exact_min",
        softmin_temperature: float = 0.05,
        min_center_hz: float = 0.5,
        max_center_hz: float = 45.0,
        min_sigma_seconds: float = 0.015,
        max_sigma_seconds: float = 0.30,
        regularization_minimum_spacing_fraction: float = 0.35,
        regularization_spacing_weight: float = 1.0,
        regularization_coverage_weight: float = 0.1,
        regularization_bounds_weight: float = 1.0,
        regularization_gate_weight: float = 0.1,
        regularization_minimum_gate_softness: float = 0.01,
    ) -> None:
        super().__init__()
        for name, value in {
            "lambda_cbe": lambda_cbe,
            "lambda_selective": lambda_selective,
            "lambda_monotonicity": lambda_monotonicity,
            "lambda_filter_regularization": lambda_filter_regularization,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        self.lambda_cbe = float(lambda_cbe)
        self.lambda_selective = float(lambda_selective)
        self.lambda_monotonicity = float(lambda_monotonicity)
        self.lambda_filter_regularization = float(lambda_filter_regularization)
        self.equivariance = CommonBandEquivarianceLoss(
            mode=equivariance_mode,
            mask_mode=common_band_mask,
            softmin_temperature=softmin_temperature,
        )
        self.selective = SelectiveRiskLoss(target_coverage, coverage_weight)
        self.monotonicity = AcceptanceMonotonicityLoss(monotonicity_margin)
        self.filter_regularization = FilterBankRegularization(
            min_center_hz=min_center_hz,
            max_center_hz=max_center_hz,
            min_sigma_seconds=min_sigma_seconds,
            max_sigma_seconds=max_sigma_seconds,
            minimum_spacing_fraction=regularization_minimum_spacing_fraction,
            spacing_weight=regularization_spacing_weight,
            coverage_weight=regularization_coverage_weight,
            bounds_weight=regularization_bounds_weight,
            gate_degeneracy_weight=regularization_gate_weight,
            minimum_gate_softness=regularization_minimum_gate_softness,
        )

    @staticmethod
    def _required(output: dict, key: str) -> Tensor:
        if key not in output or not isinstance(output[key], Tensor):
            raise ValueError(f"model output is missing tensor field {key!r}")
        return output[key]

    def forward(
        self,
        output_a: dict,
        targets_a: Tensor,
        output_b: dict | None = None,
        targets_b: Tensor | None = None,
    ) -> dict[str, Tensor]:
        logits_a = self._required(output_a, "logits")
        q_a = self._required(output_a, "accept_probability")
        logits_all = [logits_a]
        q_all = [q_a]
        target_all = [targets_a]

        zero = logits_a.sum() * 0.0
        equivariance = zero
        monotonicity = zero
        if output_b is not None:
            if targets_b is None:
                targets_b = targets_a
            logits_b = self._required(output_b, "logits")
            q_b = self._required(output_b, "accept_probability")
            logits_all.append(logits_b)
            q_all.append(q_b)
            target_all.append(targets_b)
            equivariance = self.equivariance(
                self._required(output_a, "band_features"),
                self._required(output_b, "band_features"),
                self._required(output_a, "nyquist_gate"),
                self._required(output_b, "nyquist_gate"),
            )
            rate_a = output_a["aux"]["sampling_rate_hz"]
            rate_b = output_b["aux"]["sampling_rate_hz"]
            low_is_a = rate_a <= rate_b
            q_low = torch.where(low_is_a, q_a, q_b)
            q_high = torch.where(low_is_a, q_b, q_a)
            # Equal-rate pairs do not carry ordering information.
            monotonicity = self.monotonicity(q_low, q_high, rate_a != rate_b)

        joined_logits = torch.cat(logits_all, dim=0)
        joined_q = torch.cat(q_all, dim=0)
        joined_targets = torch.cat(target_all, dim=0)
        classification = F.cross_entropy(joined_logits, joined_targets)
        selective = self.selective(joined_logits, joined_targets, joined_q)

        aux = output_a.get("aux", {})
        centers = aux.get("center_frequencies_hz")
        sigmas = aux.get("time_scales_seconds")
        if isinstance(centers, Tensor) and isinstance(sigmas, Tensor):
            gates = torch.cat(
                [self._required(out, "nyquist_gate") for out in ([output_a] if output_b is None else [output_a, output_b])],
                dim=0,
            )
            regularization = self.filter_regularization(centers, sigmas, gates)
        else:
            regularization = {
                "total": zero,
                "spacing": zero,
                "coverage": zero,
                "bounds": zero,
                "gate_degeneracy": zero,
            }

        total = (
            classification
            + self.lambda_cbe * equivariance
            + self.lambda_selective * selective["total"]
            + self.lambda_monotonicity * monotonicity
            + self.lambda_filter_regularization * regularization["total"]
        )
        return {
            "total": total,
            "classification": classification,
            "equivariance": equivariance,
            "selective": selective["total"],
            "selective_risk": selective["risk"],
            "coverage_penalty": selective["coverage_penalty"],
            "coverage": selective["coverage"],
            "monotonicity": monotonicity,
            "filter_regularization": regularization["total"],
            "filter_spacing": regularization["spacing"],
            "filter_coverage": regularization["coverage"],
            "filter_bounds": regularization["bounds"],
            "gate_degeneracy": regularization["gate_degeneracy"],
        }

