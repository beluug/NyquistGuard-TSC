"""Recorded anti-aliased downsampling for sampling-rate views."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def _lowpass_kernel(
    ratio: float,
    *,
    taps: int,
    rolloff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if taps < 3 or taps % 2 == 0:
        raise ValueError("taps must be an odd integer at least 3")
    if not 0.0 < rolloff <= 1.0:
        raise ValueError("rolloff must be in (0, 1]")
    cutoff = 0.5 * ratio * rolloff
    offsets = torch.arange(taps, device=device, dtype=torch.float64) - (taps - 1) / 2
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * offsets)
    window = torch.hamming_window(taps, periodic=False, dtype=torch.float64, device=device)
    kernel = kernel * window
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).eps)
    return kernel.to(dtype=dtype)


def resample_antialiased(
    x: Tensor,
    source_rate_hz: float,
    ratio: float,
    *,
    taps: int = 31,
    rolloff: float = 0.90,
) -> tuple[Tensor, float, dict[str, float | int | str]]:
    """Low-pass then resize a [B,C,T] collection, returning the effective rate."""

    if x.ndim != 3 or not x.is_floating_point():
        raise ValueError("x must be floating Tensor[B,C,T]")
    if not math.isfinite(source_rate_hz) or source_rate_hz <= 0:
        raise ValueError("source_rate_hz must be finite and positive")
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0,1]")
    if ratio == 1.0:
        metadata: dict[str, float | int | str] = {
            "method": "identity",
            "source_rate_hz": float(source_rate_hz),
            "target_rate_hz": float(source_rate_hz),
            "ratio": 1.0,
            "source_length": int(x.shape[-1]),
            "target_length": int(x.shape[-1]),
        }
        return x.clone(), float(source_rate_hz), metadata

    source_length = int(x.shape[-1])
    target_length = max(2, int(round(source_length * ratio)))
    effective_ratio = target_length / source_length
    maximum_taps = max(3, 2 * source_length - 1)
    resolved_taps = min(int(taps), maximum_taps)
    if resolved_taps % 2 == 0:
        resolved_taps -= 1
    work = x.float()
    kernel = _lowpass_kernel(
        effective_ratio,
        taps=resolved_taps,
        rolloff=rolloff,
        device=work.device,
        dtype=work.dtype,
    )
    channels = work.shape[1]
    grouped_kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
    half_width = resolved_taps // 2
    padded = F.pad(work, (half_width, half_width), mode="reflect")
    filtered = F.conv1d(padded, grouped_kernel, groups=channels)
    resized = F.interpolate(filtered, size=target_length, mode="linear", align_corners=False)
    target_rate = float(source_rate_hz * effective_ratio)
    metadata = {
        "method": "windowed_sinc_hamming_then_linear_resize",
        "source_rate_hz": float(source_rate_hz),
        "target_rate_hz": target_rate,
        "ratio": float(effective_ratio),
        "requested_ratio": float(ratio),
        "source_length": source_length,
        "target_length": target_length,
        "fir_taps": resolved_taps,
        "rolloff": float(rolloff),
    }
    return resized.to(dtype=x.dtype), target_rate, metadata
