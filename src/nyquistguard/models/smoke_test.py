"""Real-data-free model and objective smoke test."""

from __future__ import annotations

import json

import torch

from nyquistguard.losses.objective import NyquistGuardObjective
from nyquistguard.models.nyquistguard_tsc import NyquistGuardTSC


def main() -> None:
    torch.manual_seed(17)
    model = NyquistGuardTSC(
        input_channels=3,
        num_classes=4,
        num_bands=8,
        pooled_positions=16,
        hidden_dim=32,
        encoder_depth=2,
        max_center_hz=20.0,
        reference_sampling_rate_hz=64.0,
    )
    objective = NyquistGuardObjective(max_center_hz=20.0)
    x_a = torch.randn(4, 3, 96)
    x_b = torch.randn(4, 3, 64)
    y = torch.tensor([0, 1, 2, 3])
    out_a = model(x_a, torch.tensor([64.0, 48.0, 64.0, 48.0]))
    out_b = model(x_b, torch.tensor([40.0, 32.0, 40.0, 32.0]))
    losses = objective(out_a, y, out_b, y)
    losses["total"].backward()
    summary = {
        "status": "ok",
        "logits_shape": list(out_a["logits"].shape),
        "band_features_shape": list(out_a["band_features"].shape),
        "gate_range": [
            float(out_b["nyquist_gate"].min().detach()),
            float(out_a["nyquist_gate"].max().detach()),
        ],
        "total_loss": float(losses["total"].detach()),
        "finite_gradients": all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ),
    }
    if not summary["finite_gradients"]:
        raise RuntimeError("non-finite gradient detected")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
