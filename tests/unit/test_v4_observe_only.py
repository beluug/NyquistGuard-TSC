from __future__ import annotations

import torch

from nyquistguard.models import NyquistGuardTSC
from nyquistguard.research import ObserveOnlyNyquistGuardTSC


def _kwargs(selective: bool = False) -> dict:
    return {
        "input_channels": 2,
        "num_classes": 4,
        "num_bands": 6,
        "pooled_positions": 12,
        "hidden_dim": 24,
        "encoder_depth": 2,
        "dropout": 0.0,
        "min_center_hz": 1.0,
        "max_center_hz": 18.0,
        "min_sigma_seconds": 0.03,
        "max_sigma_seconds": 0.18,
        "max_kernel_seconds": 0.6,
        "reference_sampling_rate_hz": 64.0,
        "use_selective_head": selective,
    }


def test_observe_only_matches_no_gate_classifier_but_retains_physical_gate() -> None:
    torch.manual_seed(17)
    model = ObserveOnlyNyquistGuardTSC(**_kwargs()).eval()
    no_gate = NyquistGuardTSC(
        **_kwargs(), use_nyquist_gate=False
    ).eval()
    no_gate.load_state_dict(model.state_dict())
    x = torch.randn(3, 2, 80)
    rates = torch.tensor([24.0, 40.0, 64.0])
    with torch.inference_mode():
        observed = model(x, rates)
        reference = no_gate(x, rates)
    torch.testing.assert_close(observed["logits"], reference["logits"])
    torch.testing.assert_close(observed["embedding"], reference["embedding"])
    torch.testing.assert_close(
        observed["gated_band_features"], observed["band_features"]
    )
    assert torch.any(observed["nyquist_gate"] < 0.99)
    assert torch.all(reference["nyquist_gate"] == 1)
    assert observed["aux"]["nyquist_gate_mode"] == "observe_only"


def test_observability_remains_rate_monotonic_and_separate() -> None:
    torch.manual_seed(23)
    model = ObserveOnlyNyquistGuardTSC(**_kwargs()).eval()
    x = torch.randn(1, 2, 80).expand(2, -1, -1).contiguous()
    with torch.inference_mode():
        output = model(x, torch.tensor([24.0, 64.0]))
    low, high = output["nyquist_gate"]
    assert torch.all(low <= high + 1e-7)
    assert not torch.equal(
        output["observability_weighted_band_features"], output["band_features"]
    )
    assert output["aux"]["classifier_uses_observability_attenuation"] is False


def test_selective_path_uses_gate_without_changing_classifier_logits() -> None:
    torch.manual_seed(31)
    model = ObserveOnlyNyquistGuardTSC(**_kwargs(selective=True)).eval()
    no_gate = NyquistGuardTSC(
        **_kwargs(selective=True), use_nyquist_gate=False
    ).eval()
    no_gate.load_state_dict(model.state_dict())
    x = torch.randn(2, 2, 72)
    rates = torch.tensor([20.0, 64.0])
    with torch.inference_mode():
        observed = model(x, rates)
        reference = no_gate(x, rates)
    torch.testing.assert_close(observed["logits"], reference["logits"])
    assert not torch.equal(observed["accept_logit"], reference["accept_logit"])


def test_observe_only_backward_reaches_classifier_and_filterbank() -> None:
    model = ObserveOnlyNyquistGuardTSC(**_kwargs())
    output = model(torch.randn(3, 2, 64), torch.tensor([24.0, 40.0, 64.0]))
    loss = output["logits"].square().mean()
    loss.backward()
    for parameter in (
        model.classifier.weight,
        model.encoder.input_projection.weight,
        model.filterbank.raw_center_frequencies,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0

