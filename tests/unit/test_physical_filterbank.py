import math

import pytest
import torch

from nyquistguard.models.physical_filterbank import PhysicalGaborFilterBank


def _sine(rate: float, frequency: float, duration: float = 2.0) -> torch.Tensor:
    time = torch.arange(round(rate * duration), dtype=torch.float32) / rate
    return torch.sin(2.0 * math.pi * frequency * time).view(1, 1, -1)


@pytest.mark.parametrize("batch_size,channels,length", [(1, 1, 7), (3, 2, 127)])
def test_filterbank_shape_and_short_sequences(batch_size, channels, length):
    bank = PhysicalGaborFilterBank(
        channels,
        5,
        min_center_hz=1.0,
        max_center_hz=15.0,
        max_kernel_seconds=1.0,
    )
    x = torch.randn(batch_size, channels, length)
    rates = torch.linspace(20.0, 80.0, batch_size)
    output = bank(x, rates)
    assert output.shape == (batch_size, 5, length)
    assert torch.isfinite(output).all()


def test_filterbank_padding_and_rate_grouping():
    bank = PhysicalGaborFilterBank(2, 4, max_center_hz=12.0)
    x = torch.randn(3, 2, 40)
    mask = torch.zeros(3, 40, dtype=torch.bool)
    mask[1, 27:] = True
    mask[2, 11:] = True
    output = bank(x, torch.tensor([40.0, 80.0, 40.0]), mask)
    assert torch.count_nonzero(output[1, :, 27:]) == 0
    assert torch.count_nonzero(output[2, :, 11:]) == 0


def test_physical_frequency_response_is_rate_consistent():
    bank = PhysicalGaborFilterBank(
        1,
        3,
        min_center_hz=2.0,
        max_center_hz=8.0,
        min_sigma_seconds=0.08,
        max_sigma_seconds=0.20,
        max_kernel_seconds=1.0,
    )
    responses = []
    winning_bands = []
    for rate in (40.0, 80.0, 120.0):
        output = bank(_sine(rate, 5.0), rate)
        # Remove finite-support edge effects before comparing physical response.
        edge = int(0.55 * rate)
        band_response = output[0, :, edge:-edge].mean(dim=-1)
        responses.append(band_response)
        winning_bands.append(int(band_response.argmax()))
    stacked = torch.stack(responses)
    relative_spread = (stacked.max(dim=0).values - stacked.min(dim=0).values) / (
        stacked.mean(dim=0).clamp_min(1e-6)
    )
    assert winning_bands == [1, 1, 1]
    assert float(relative_spread[1].detach()) < 0.08


def test_learnable_physical_parameters_receive_gradients():
    torch.manual_seed(7)
    bank = PhysicalGaborFilterBank(2, 4, min_center_hz=1.0, max_center_hz=18.0)
    x = torch.randn(3, 2, 96)
    output = bank(x, torch.tensor([32.0, 48.0, 64.0]))
    weights = torch.linspace(0.5, 1.5, output.numel()).reshape_as(output)
    (output * weights).mean().backward()
    for parameter in (
        bank.raw_center_frequencies,
        bank.raw_time_scales,
        bank.channel_logits,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_invalid_non_suffix_mask_is_rejected():
    bank = PhysicalGaborFilterBank(1, 2)
    mask = torch.tensor([[False, True, False]])
    with pytest.raises(ValueError, match="right padding"):
        bank(torch.randn(1, 1, 3), 20.0, mask)
