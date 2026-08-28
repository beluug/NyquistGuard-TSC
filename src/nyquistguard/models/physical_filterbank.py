"""Physical-time and discrete ablation filter banks.

The main filter bank synthesizes a pair of real/imaginary Gabor kernels for
every sampling rate present in a batch.  Kernel coordinates are measured in
seconds and convolution weights include the quadrature factor ``1 / fs``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _logit(value: Tensor, eps: float = 1e-5) -> Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def normalize_sampling_rates(
    sampling_rate_hz: float | Tensor,
    batch_size: int,
    *,
    device: torch.device,
) -> Tensor:
    """Return validated per-sample rates as float32 ``[B]``."""

    rates = torch.as_tensor(sampling_rate_hz, device=device, dtype=torch.float32)
    if rates.ndim == 0:
        rates = rates.expand(batch_size)
    elif rates.ndim == 1 and rates.numel() == 1:
        rates = rates.expand(batch_size)
    elif rates.ndim != 1 or rates.numel() != batch_size:
        raise ValueError(
            "sampling_rate_hz must be a scalar or have shape [B]; "
            f"received {tuple(rates.shape)} for B={batch_size}"
        )
    if not torch.isfinite(rates).all() or torch.any(rates <= 0):
        raise ValueError("sampling_rate_hz must contain finite positive Hz values")
    return rates


def validate_padding_mask(x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Validate the right-padding convention and return a boolean mask."""

    batch_size, _, length = x.shape
    if padding_mask is None:
        return torch.zeros((batch_size, length), dtype=torch.bool, device=x.device)
    if padding_mask.dtype != torch.bool or padding_mask.shape != (batch_size, length):
        raise ValueError(
            "padding_mask must be BoolTensor[B,T] with True denoting padding; "
            f"received dtype={padding_mask.dtype}, shape={tuple(padding_mask.shape)}"
        )
    padding_mask = padding_mask.to(device=x.device)
    valid = ~padding_mask
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("every sequence must contain at least one valid sample")
    # A valid sample after any padding position violates the suffix convention.
    seen_padding = padding_mask.cumsum(dim=1) > 0
    if torch.any(valid & seen_padding):
        raise ValueError("padding_mask must describe right padding (valid prefix only)")
    return padding_mask


class _BoundedPhysicalParameters(nn.Module):
    """Shared bounded parameterization used by both front-end variants."""

    def _init_physical_parameters(
        self,
        num_bands: int,
        min_center_hz: float,
        max_center_hz: float,
        min_sigma_seconds: float,
        max_sigma_seconds: float,
    ) -> None:
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        if not 0 <= min_center_hz < max_center_hz:
            raise ValueError("center-frequency bounds must satisfy 0 <= min < max")
        if not 0 < min_sigma_seconds < max_sigma_seconds:
            raise ValueError("time-scale bounds must satisfy 0 < min < max")

        self.num_bands = int(num_bands)
        self.min_center_hz = float(min_center_hz)
        self.max_center_hz = float(max_center_hz)
        self.min_sigma_seconds = float(min_sigma_seconds)
        self.max_sigma_seconds = float(max_sigma_seconds)

        if num_bands == 1:
            center_fraction = torch.tensor([0.5])
        else:
            center_fraction = torch.linspace(0.02, 0.98, num_bands)
        initial_sigma = math.sqrt(min_sigma_seconds * max_sigma_seconds)
        sigma_fraction = (initial_sigma - min_sigma_seconds) / (
            max_sigma_seconds - min_sigma_seconds
        )
        self.raw_center_frequencies = nn.Parameter(_logit(center_fraction))
        self.raw_time_scales = nn.Parameter(
            _logit(torch.full((num_bands,), float(sigma_fraction)))
        )

    @property
    def center_frequencies_hz(self) -> Tensor:
        span = self.max_center_hz - self.min_center_hz
        return self.min_center_hz + span * torch.sigmoid(self.raw_center_frequencies)

    @property
    def time_scales_seconds(self) -> Tensor:
        span = self.max_sigma_seconds - self.min_sigma_seconds
        return self.min_sigma_seconds + span * torch.sigmoid(self.raw_time_scales)

    @property
    def bandwidth_std_hz(self) -> Tensor:
        # |H(f)|^2 is proportional to exp(-4*pi^2*sigma^2*(f-fc)^2),
        # whose equivalent Gaussian standard deviation is 1/(2*sqrt(2)*pi*sigma).
        return 1.0 / (
            2.0 * math.sqrt(2.0) * math.pi * self.time_scales_seconds
        )


class PhysicalGaborFilterBank(_BoundedPhysicalParameters):
    """Differentiable physical-time complex Gabor filter bank."""

    def __init__(
        self,
        input_channels: int,
        num_bands: int,
        *,
        min_center_hz: float = 0.5,
        max_center_hz: float = 45.0,
        min_sigma_seconds: float = 0.015,
        max_sigma_seconds: float = 0.30,
        kernel_support_sigmas: float = 4.0,
        max_kernel_seconds: float = 1.0,
        magnitude_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if kernel_support_sigmas <= 0 or max_kernel_seconds <= 0:
            raise ValueError("kernel support settings must be positive")
        self.input_channels = int(input_channels)
        self.kernel_support_sigmas = float(kernel_support_sigmas)
        self.max_kernel_seconds = float(max_kernel_seconds)
        self.magnitude_epsilon = float(magnitude_epsilon)
        self._init_physical_parameters(
            num_bands,
            min_center_hz,
            max_center_hz,
            min_sigma_seconds,
            max_sigma_seconds,
        )
        # Each band learns a convex mixture across sensor channels.  This keeps
        # the front end interpretable while allowing multivariate specialization.
        self.channel_logits = nn.Parameter(torch.zeros(num_bands, input_channels))

    def kernel_half_width_samples(self, sampling_rate_hz: float) -> int:
        max_half_seconds = min(
            self.kernel_support_sigmas * self.max_sigma_seconds,
            self.max_kernel_seconds / 2.0,
        )
        return max(1, int(math.ceil(max_half_seconds * float(sampling_rate_hz))))

    def synthesize_kernels(self, sampling_rate_hz: float | Tensor) -> tuple[Tensor, Tensor]:
        """Return continuous-energy-normalized cosine/sine kernels ``[K,L]``."""

        rate_tensor = torch.as_tensor(
            sampling_rate_hz,
            device=self.raw_center_frequencies.device,
            dtype=torch.float32,
        )
        if rate_tensor.numel() != 1 or not torch.isfinite(rate_tensor).all() or rate_tensor <= 0:
            raise ValueError("synthesize_kernels expects one finite positive sampling rate")
        rate = float(rate_tensor.detach().cpu().item())
        half_width = self.kernel_half_width_samples(rate)

        # The numerically delicate basis construction is deliberately float32
        # even under AMP.  The resulting response may be cast by later layers.
        offsets = torch.arange(
            -half_width,
            half_width + 1,
            device=self.raw_center_frequencies.device,
            dtype=torch.float32,
        )
        tau = offsets / rate_tensor
        centers = self.center_frequencies_hz.float().unsqueeze(1)
        sigmas = self.time_scales_seconds.float().unsqueeze(1)
        envelope = torch.exp(-0.5 * (tau.unsqueeze(0) / sigmas).square())
        phase = 2.0 * math.pi * centers * tau.unsqueeze(0)

        delta_t = rate_tensor.reciprocal()
        pair_energy = (envelope.square().sum(dim=1, keepdim=True) * delta_t).clamp_min(1e-12)
        normalized_envelope = envelope * pair_energy.rsqrt()
        cosine = normalized_envelope * torch.cos(phase) * delta_t
        sine = normalized_envelope * torch.sin(phase) * delta_t
        return cosine, sine

    def forward(
        self,
        x: Tensor,
        sampling_rate_hz: float | Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return unpooled band magnitudes with shape ``[B,K,T]``."""

        if x.ndim != 3 or not x.is_floating_point():
            raise ValueError("x must be a floating tensor with shape [B,C,T]")
        batch_size, channels, length = x.shape
        if channels != self.input_channels or length < 1:
            raise ValueError(
                f"expected C={self.input_channels} and T>=1, got shape={tuple(x.shape)}"
            )
        rates = normalize_sampling_rates(
            sampling_rate_hz, batch_size, device=x.device
        )
        padding_mask = validate_padding_mask(x, padding_mask)
        valid = (~padding_mask).unsqueeze(1)
        # Gabor basis generation, quadrature convolution and magnitude are kept
        # in float32 under AMP.  This avoids fp16 underflow in narrow Gaussian
        # tails and ensures every rate group writes the same dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_work = x.float().masked_fill(~valid, 0.0)
            result = x_work.new_zeros((batch_size, self.num_bands, length))
            channel_weights = torch.softmax(self.channel_logits.float(), dim=1)
            # Rate is metadata rather than a learnable variable, so grouping on a
            # detached exact value does not interrupt any model-parameter gradient.
            for rate in torch.unique(rates.detach(), sorted=True):
                indices = torch.nonzero(rates == rate, as_tuple=False).flatten()
                cosine, sine = self.synthesize_kernels(rate)
                half_width = cosine.shape[-1] // 2
                selected = x_work.index_select(0, indices)
                selected = selected.reshape(-1, 1, length)
                padded = F.pad(selected, (half_width, half_width))
                real = F.conv1d(padded, cosine.unsqueeze(1))
                imag = F.conv1d(padded, sine.unsqueeze(1))
                magnitude = torch.sqrt(real.square() + imag.square() + self.magnitude_epsilon)
                magnitude = magnitude.view(indices.numel(), channels, self.num_bands, length)
                mixed = torch.einsum("bckt,kc->bkt", magnitude, channel_weights)
                result = result.index_copy(0, indices, mixed)

        result = result.masked_fill(padding_mask.unsqueeze(1), 0.0)
        return result.to(dtype=x.dtype)


class DiscreteConvFilterBank(_BoundedPhysicalParameters):
    """Ordinary fixed-sample Conv1d front end for the physical-time ablation."""

    def __init__(
        self,
        input_channels: int,
        num_bands: int,
        *,
        kernel_size: int = 31,
        min_center_hz: float = 0.5,
        max_center_hz: float = 45.0,
        min_sigma_seconds: float = 0.015,
        max_sigma_seconds: float = 0.30,
        magnitude_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.input_channels = int(input_channels)
        self.magnitude_epsilon = float(magnitude_epsilon)
        self._init_physical_parameters(
            num_bands,
            min_center_hz,
            max_center_hz,
            min_sigma_seconds,
            max_sigma_seconds,
        )
        self.conv = nn.Conv1d(
            input_channels,
            2 * num_bands,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
        sampling_rate_hz: float | Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"x must have shape [B,{self.input_channels},T]")
        normalize_sampling_rates(sampling_rate_hz, x.shape[0], device=x.device)
        padding_mask = validate_padding_mask(x, padding_mask)
        work = x.masked_fill(padding_mask.unsqueeze(1), 0.0)
        paired = self.conv(work)
        real, imag = paired.chunk(2, dim=1)
        magnitude = torch.sqrt(real.square() + imag.square() + self.magnitude_epsilon)
        return magnitude.masked_fill(padding_mask.unsqueeze(1), 0.0)
