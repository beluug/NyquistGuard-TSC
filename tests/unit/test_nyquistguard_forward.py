import math

import pytest
import torch

from nyquistguard.losses.objective import NyquistGuardObjective
from nyquistguard.models.nyquistguard_tsc import NyquistGuardTSC


def _model(channels=3, filterbank_type="physical"):
    return NyquistGuardTSC(
        input_channels=channels,
        num_classes=4,
        num_bands=6,
        pooled_positions=12,
        hidden_dim=24,
        encoder_depth=2,
        min_center_hz=1.0,
        max_center_hz=18.0,
        min_sigma_seconds=0.03,
        max_sigma_seconds=0.18,
        max_kernel_seconds=0.6,
        filterbank_type=filterbank_type,
        reference_sampling_rate_hz=64.0,
    )


@pytest.mark.parametrize("batch_size,channels,length", [(1, 1, 9), (2, 3, 64), (4, 5, 111)])
def test_forward_shapes_cpu(batch_size, channels, length):
    model = _model(channels)
    x = torch.randn(batch_size, channels, length)
    rates = torch.linspace(32.0, 80.0, batch_size)
    output = model(x, rates)
    assert output["logits"].shape == (batch_size, 4)
    assert output["accept_logit"].shape == (batch_size,)
    assert output["band_features"].shape == (batch_size, 6, 12)
    assert output["gated_band_features"].shape == (batch_size, 6, 12)
    assert output["nyquist_gate"].shape == (batch_size, 6)
    assert output["embedding"].shape == (batch_size, 24)
    for key in ("logits", "accept_logit", "band_features", "nyquist_gate", "embedding"):
        assert torch.isfinite(output[key]).all()


def test_padding_and_timestamps_are_supported():
    model = _model(2)
    x = torch.randn(2, 2, 50)
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[1, 31:] = True
    rates = torch.tensor([50.0, 40.0])
    timestamps = torch.zeros(2, 50)
    timestamps[0] = torch.arange(50) / 50.0
    timestamps[1, :31] = torch.arange(31) / 40.0
    output = model(x, rates, mask, timestamps)
    assert output["aux"]["valid_lengths"].tolist() == [50, 31]
    assert output["band_features"].shape[-1] == 12


def test_timestamp_rate_mismatch_is_rejected():
    model = _model(1)
    x = torch.randn(1, 1, 20)
    timestamps = (torch.arange(20) / 10.0).view(1, -1)
    with pytest.raises(ValueError, match="approximately uniform"):
        model(x, 40.0, timestamps=timestamps)


def test_full_objective_reaches_all_trainable_components():
    torch.manual_seed(11)
    model = _model(2)
    objective = NyquistGuardObjective(
        max_center_hz=18.0,
        min_sigma_seconds=0.03,
        max_sigma_seconds=0.18,
    )
    y = torch.tensor([0, 1, 2])
    out_high = model(torch.randn(3, 2, 80), torch.tensor([64.0, 56.0, 48.0]))
    out_low = model(torch.randn(3, 2, 55), torch.tensor([40.0, 32.0, 28.0]))
    losses = objective(out_high, y, out_low, y)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    required = {
        "filter_center": model.filterbank.raw_center_frequencies,
        "filter_time_scale": model.filterbank.raw_time_scales,
        "channel_mixer": model.filterbank.channel_logits,
        "encoder": model.encoder.input_projection.weight,
        "classifier": model.classifier.weight,
        "selective": model.selective_head.network[-1].weight,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_discrete_front_end_ablation_runs():
    model = _model(3, filterbank_type="discrete")
    output = model(torch.randn(2, 3, 64), torch.tensor([40.0, 64.0]))
    assert output["aux"]["filterbank_type"] == "discrete"
    assert output["logits"].shape == (2, 4)


def test_cpu_autocast_forward_and_backward():
    model = _model(2)
    x = torch.randn(2, 2, 48)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(x, torch.tensor([32.0, 48.0]))
        loss = output["logits"].float().square().mean() + output["accept_logit"].float().mean()
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_forward_and_backward():
    model = _model(3).cuda()
    x = torch.randn(3, 3, 96, device="cuda")
    rates = torch.tensor([40.0, 64.0, 80.0], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x, rates)
        loss = output["logits"].float().square().mean() + output["accept_logit"].float().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
