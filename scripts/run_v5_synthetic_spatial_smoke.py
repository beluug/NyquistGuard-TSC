"""Bounded CPU-only positive-capability smoke for the isolated V5 candidate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nyquistguard.research.v5_dual_path import DualPathNyquistGuardTSC


def _dataset(examples: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    channels, length, rate = 8, 128, 64.0
    labels = torch.arange(examples) % 2
    time = torch.arange(length, dtype=torch.float32) / rate
    phase = 2.0 * math.pi * torch.rand(examples, generator=generator)
    amplitude = 0.8 + 0.4 * torch.rand(examples, generator=generator)
    carrier = amplitude[:, None] * torch.sin(
        2.0 * math.pi * 7.0 * time[None, :] + phase[:, None]
    )
    same_sign = torch.ones(channels)
    alternating_sign = torch.where(torch.arange(channels) % 2 == 0, 1.0, -1.0)
    signs = torch.where(labels[:, None] == 0, same_sign, alternating_sign)
    signals = signs[:, :, None] * carrier[:, None, :]
    noise = 0.12 * torch.randn(examples, channels, length, generator=generator)
    # Per-channel magnitudes are matched by construction; only signed spatial
    # organization separates the classes.
    return signals + noise, labels.long()


def run(steps: int, seed: int) -> dict[str, float | int | bool | str]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_x, train_y = _dataset(96, seed=seed)
    validation_x, validation_y = _dataset(48, seed=seed + 1)
    model = DualPathNyquistGuardTSC(
        input_channels=8,
        num_classes=2,
        num_bands=6,
        pooled_positions=16,
        hidden_dim=32,
        encoder_depth=2,
        encoder_kernel_size=5,
        dropout=0.0,
        min_center_hz=1.0,
        max_center_hz=20.0,
        min_sigma_seconds=0.025,
        max_sigma_seconds=0.16,
        max_kernel_seconds=0.5,
        reference_sampling_rate_hz=64.0,
        use_selective_head=False,
        initial_gate_floor=0.5,
        spatial_channels=12,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 2)
    model.train()
    last_loss = math.inf
    for _ in range(steps):
        indices = torch.randint(0, train_x.shape[0], (24,), generator=generator)
        output = model(train_x[indices], 64.0)
        loss = F.cross_entropy(output["logits"], train_y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    model.eval()
    with torch.inference_mode():
        output = model(validation_x, 64.0)
        predictions = output["logits"].argmax(dim=1)
        accuracy = float((predictions == validation_y).float().mean())
        fusion_mean = float(output["fusion_physical_weight"].mean())
    passed = bool(math.isfinite(last_loss) and accuracy >= 0.90)
    return {
        "stage": "v5_synthetic_signed_spatial_smoke",
        "data_boundary": "synthetic_only_no_test_split",
        "device": "cpu",
        "seed": seed,
        "steps": steps,
        "final_minibatch_loss": last_loss,
        "validation_accuracy": accuracy,
        "mean_physical_fusion_weight": fusion_mean,
        "threshold": 0.90,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=5105)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "v5_synthetic_spatial_smoke.json",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 300:
        parser.error("--steps must be in [1, 300]")
    result = run(args.steps, args.seed)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(result, indent=2))
    print(f"Report: {args.report}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
