from __future__ import annotations

import torch

from nyquistguard.research.v5_dual_path import DualPathNyquistGuardTSC


def _model(channels: int = 8) -> DualPathNyquistGuardTSC:
    return DualPathNyquistGuardTSC(
        input_channels=channels,
        num_classes=3,
        num_bands=6,
        pooled_positions=12,
        hidden_dim=24,
        encoder_depth=2,
        encoder_kernel_size=5,
        dropout=0.0,
        min_center_hz=1.0,
        max_center_hz=18.0,
        min_sigma_seconds=0.03,
        max_sigma_seconds=0.18,
        max_kernel_seconds=0.6,
        reference_sampling_rate_hz=64.0,
        use_selective_head=False,
        initial_gate_floor=0.5,
        spatial_channels=10,
    )


def test_v5_dual_path_shapes_and_bounded_fusion() -> None:
    model = _model().eval()
    mask = torch.zeros(3, 80, dtype=torch.bool)
    mask[1, 63:] = True
    with torch.inference_mode():
        output = model(
            torch.randn(3, 8, 80),
            torch.tensor([24.0, 48.0, 64.0]),
            padding_mask=mask,
        )
    assert output["logits"].shape == (3, 3)
    assert output["spatial_features"].shape == (3, 10, 12)
    assert output["physical_embedding"].shape == (3, 24)
    assert output["spatial_embedding"].shape == (3, 24)
    assert output["fusion_physical_weight"].shape == (3,)
    assert torch.all(output["fusion_physical_weight"] >= 0.0)
    assert torch.all(output["fusion_physical_weight"] <= 1.0)
    assert torch.isfinite(output["embedding"]).all()
    assert output["aux"]["signed_spatial_bypass_enabled"] is True


def test_v5_classification_gradient_reaches_both_paths_and_controller() -> None:
    torch.manual_seed(23)
    model = _model()
    output = model(
        torch.randn(5, 8, 96),
        torch.tensor([24.0, 32.0, 40.0, 56.0, 64.0]),
    )
    torch.nn.functional.cross_entropy(
        output["logits"], torch.tensor([0, 1, 2, 1, 0])
    ).backward()
    required = {
        "physical_filter": model.filterbank.raw_center_frequencies,
        "residual_gate": model.raw_gate_floor,
        "physical_encoder": model.encoder.input_projection.weight,
        "spatial_projection": model.spatial_projection[0].weight,
        "spatial_encoder": model.spatial_encoder.input_projection.weight,
        # The final controller layer is zero-initialized to produce an exact
        # 0.5 blend. It learns on step one; preceding layers learn thereafter.
        "fusion_controller": model.fusion_controller[-1].weight,
        "classifier": model.classifier.weight,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_v5_ignores_values_in_padded_tail() -> None:
    torch.manual_seed(31)
    model = _model(4).eval()
    x = torch.randn(2, 4, 72)
    mask = torch.zeros(2, 72, dtype=torch.bool)
    mask[:, 51:] = True
    changed = x.clone()
    changed[:, :, 51:] = 1000.0 * torch.randn_like(changed[:, :, 51:])
    with torch.inference_mode():
        first = model(x, 64.0, padding_mask=mask)["logits"]
        second = model(changed, 64.0, padding_mask=mask)["logits"]
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
