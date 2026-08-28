import torch

from nyquistguard.losses.selective_risk import SelectiveRiskLoss
from nyquistguard.models.selective_head import SelectiveHead


def test_selective_head_shape_range_and_gradients():
    torch.manual_seed(3)
    head = SelectiveHead(embedding_dim=12, num_bands=5, hidden_dim=16)
    embedding = torch.randn(4, 12, requires_grad=True)
    gates = torch.rand(4, 5, requires_grad=True)
    rates = torch.tensor([20.0, 40.0, 60.0, 100.0])
    entropy = torch.rand(4)
    logit = head(embedding, gates, rates, entropy)
    probability = torch.sigmoid(logit)
    assert logit.shape == (4,)
    assert torch.all((0 < probability) & (probability < 1))
    probability.mean().backward()
    assert torch.count_nonzero(embedding.grad) > 0
    assert torch.count_nonzero(gates.grad) > 0
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_coverage_penalty_prevents_all_reject_escape():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1])
    loss_fn = SelectiveRiskLoss(target_coverage=0.8, coverage_weight=10.0)
    low_q = loss_fn(logits, targets, torch.full((2,), 1e-4))
    adequate_q = loss_fn(logits, targets, torch.full((2,), 0.8))
    assert low_q["coverage_penalty"] > adequate_q["coverage_penalty"]
    assert low_q["total"] > adequate_q["total"]

