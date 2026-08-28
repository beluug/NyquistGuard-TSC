from pathlib import Path

import pytest
import torch

from nyquistguard.training.checkpointing import load_training_checkpoint, save_training_checkpoint


def test_checkpoint_roundtrip_and_protocol_guard(tmp_path: Path):
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3)
    model(x).square().mean().backward()
    optimizer.step()
    expected = model(x).detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        step=1,
        protocol_hash="abc",
        model_config={"in_features": 3, "out_features": 2},
    )

    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_training_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        expected_protocol_hash="abc",
        restore_rng=False,
    )
    assert payload["step"] == 1
    assert torch.equal(expected, restored(x).detach())
    with pytest.raises(ValueError, match="protocol hash"):
        load_training_checkpoint(path, model=restored, expected_protocol_hash="different", restore_rng=False)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_rng_restore_when_loaded_to_cuda(tmp_path: Path):
    model = torch.nn.Linear(2, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "cuda_checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        step=0,
        protocol_hash="cuda",
        model_config={"in_features": 2, "out_features": 2},
    )
    restored = torch.nn.Linear(2, 2).cuda()
    load_training_checkpoint(
        path,
        model=restored,
        expected_protocol_hash="cuda",
        map_location="cuda",
        restore_rng=True,
    )
    assert all(parameter.is_cuda for parameter in restored.parameters())
