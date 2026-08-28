import pytest
import torch

from nyquistguard.experiments.v3_spectral_reliability import (
    sample_retained_band_energy,
)


def test_sample_retained_energy_is_sample_specific_and_bounded() -> None:
    bands = torch.tensor([[[1.0, 1.0], [3.0, 3.0]], [[4.0, 0.0], [1.0, 1.0]]])
    gated = torch.tensor([[[1.0, 1.0], [0.0, 0.0]], [[2.0, 0.0], [1.0, 1.0]]])
    retention = sample_retained_band_energy(bands, gated)
    assert retention.tolist() == pytest.approx([0.25, 4.0 / 6.0])
    assert torch.all((retention >= 0.0) & (retention <= 1.0))


def test_sample_retained_energy_requires_matching_shapes() -> None:
    with pytest.raises(ValueError):
        sample_retained_band_energy(torch.ones(2, 3), torch.ones(2, 4))
