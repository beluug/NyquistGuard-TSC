from pathlib import Path

import pytest
import torch
import yaml

from nyquistguard.experiments.v2_micro_pilot import (
    V2_DATASETS,
    V2_VARIANTS,
    _balanced_cbe_scale,
    _detached_selector_output,
    build_v2_matrix,
)
from nyquistguard.losses import DetachedCorrectnessSelectiveLoss
from nyquistguard.models import NyquistGuardTSC


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tiny_model() -> NyquistGuardTSC:
    return NyquistGuardTSC(
        input_channels=2,
        num_classes=3,
        num_bands=4,
        pooled_positions=8,
        hidden_dim=12,
        encoder_depth=1,
        min_center_hz=0.5,
        max_center_hz=8.0,
        min_sigma_seconds=0.03,
        max_sigma_seconds=0.2,
        max_kernel_seconds=0.5,
        reference_sampling_rate_hz=20.0,
    )


def test_v2_micro_matrix_is_frozen_and_bounded() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "v2_micro.yaml").read_text(
            encoding="utf-8"
        )
    )
    matrix = build_v2_matrix(config)
    assert len(matrix) == 4
    assert {spec.dataset_id for spec in matrix} == set(V2_DATASETS)
    assert {spec.variant for spec in matrix} == set(V2_VARIANTS)
    assert {spec.seed for spec in matrix} == {17}
    assert config["wall_time_budget_seconds"] == 540


def test_correctness_target_does_not_backpropagate_into_logits() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    accept_logit = torch.zeros(2, requires_grad=True)
    result = DetachedCorrectnessSelectiveLoss()(logits, torch.tensor([0, 1]), accept_logit)
    result["total"].backward()
    assert logits.grad is None
    assert accept_logit.grad is not None
    assert torch.isfinite(accept_logit.grad).all()


def test_detached_selector_update_is_isolated_to_selector_head() -> None:
    model = _tiny_model()
    output = model(torch.randn(4, 2, 40), 20.0)
    detached = _detached_selector_output(model, output)
    loss = DetachedCorrectnessSelectiveLoss()(
        detached["logits"], torch.tensor([0, 1, 2, 0]), detached["accept_logit"]
    )["total"]
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.selective_head.parameters())
    assert all(parameter.grad is None for parameter in model.filterbank.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.classifier.parameters())


def test_cbe_balancer_targets_shared_parameter_gradient_ratio() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    classification = (3.0 * parameter).square().sum()
    equivariance = (0.2 * parameter).square().sum()
    result = _balanced_cbe_scale(
        classification,
        equivariance,
        [parameter],
        target_ratio=0.05,
        minimum_scale=0.1,
        maximum_scale=1000.0,
    )
    assert float(result["scale"]) > 1.0
    assert float(result["achieved_filterbank_gradient_ratio"]) == pytest.approx(
        0.05, rel=1e-5
    )
    assert result["scale"].requires_grad is False

