import math

import torch

from nyquistguard.models.nyquistguard_tsc import NyquistGuardTSC


def test_tiny_synthetic_dataset_can_be_overfit():
    torch.manual_seed(2026)
    rate = 40.0
    duration = 1.5
    time = torch.arange(round(rate * duration)) / rate
    signals = []
    targets = []
    for class_id, frequency in enumerate((3.0, 7.0)):
        for phase in (0.0, 0.4, 0.8, 1.2):
            signal = torch.sin(2.0 * math.pi * frequency * time + phase)
            signal += 0.10 * torch.cos(2.0 * math.pi * (frequency + 1.0) * time)
            signals.append(signal)
            targets.append(class_id)
    x = torch.stack(signals).unsqueeze(1)
    y = torch.tensor(targets)
    model = NyquistGuardTSC(
        input_channels=1,
        num_classes=2,
        num_bands=6,
        pooled_positions=12,
        hidden_dim=20,
        encoder_depth=2,
        dropout=0.0,
        min_center_hz=1.0,
        max_center_hz=9.0,
        min_sigma_seconds=0.05,
        max_sigma_seconds=0.18,
        max_kernel_seconds=0.8,
        reference_sampling_rate_hz=40.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    final_loss = None
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, rate)
        loss = torch.nn.functional.cross_entropy(output["logits"], y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if final_loss < 0.005:
            break
    with torch.inference_mode():
        output = model(x, rate)
        accuracy = (output["logits"].argmax(dim=1) == y).float().mean()
    print({"tiny_overfit_accuracy": float(accuracy), "final_loss": final_loss})
    assert accuracy >= 0.99
    assert final_loss is not None and final_loss < 0.05

