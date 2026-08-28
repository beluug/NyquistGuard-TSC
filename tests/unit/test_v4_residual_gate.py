from __future__ import annotations

import torch

from nyquistguard.research import ResidualGateNyquistGuardTSC


def _model() -> ResidualGateNyquistGuardTSC:
    return ResidualGateNyquistGuardTSC(
        input_channels=2, num_classes=3, num_bands=5, pooled_positions=8,
        hidden_dim=16, encoder_depth=1, dropout=0.0, min_center_hz=1.0,
        max_center_hz=18.0, min_sigma_seconds=0.03, max_sigma_seconds=0.18,
        max_kernel_seconds=0.6, reference_sampling_rate_hz=64.0,
        use_selective_head=False, initial_gate_floor=0.5,
    )


def test_residual_multiplier_is_bounded_between_gate_and_one() -> None:
    model = _model().eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 2, 80), torch.tensor([20.0, 64.0]))
    gate = output["nyquist_gate"]
    multiplier = output["effective_gate_multiplier"]
    assert torch.all(multiplier >= gate - 1e-7)
    assert torch.all(multiplier <= 1.0 + 1e-7)
    assert output["aux"]["nyquist_gate_mode"] == "learnable_residual"


def test_residual_floor_receives_classification_gradient() -> None:
    model = _model()
    output = model(torch.randn(4, 2, 80), torch.tensor([20.0, 30.0, 40.0, 64.0]))
    torch.nn.functional.cross_entropy(output["logits"], torch.tensor([0, 1, 2, 1])).backward()
    assert model.raw_gate_floor.grad is not None
    assert torch.isfinite(model.raw_gate_floor.grad).all()
    assert torch.count_nonzero(model.raw_gate_floor.grad) > 0
