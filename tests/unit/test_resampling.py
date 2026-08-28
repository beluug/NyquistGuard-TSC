import torch

from nyquistguard.data.resampling import resample_antialiased


def test_antialiased_resampling_shape_rate_and_metadata():
    x = torch.randn(3, 2, 100)
    view, rate, metadata = resample_antialiased(x, 10.0, 0.5, taps=31)
    assert view.shape == (3, 2, 50)
    assert rate == 5.0
    assert metadata["method"] == "windowed_sinc_hamming_then_linear_resize"
    assert metadata["fir_taps"] == 31
    assert torch.isfinite(view).all()


def test_antialiased_resampling_preserves_constant_signal():
    x = torch.ones(2, 3, 100)
    view, _, _ = resample_antialiased(x, 20.0, 0.5)
    assert torch.allclose(view, torch.ones_like(view), atol=1e-5)


def test_identity_view_is_a_clone():
    x = torch.randn(1, 2, 17)
    view, rate, metadata = resample_antialiased(x, 12.0, 1.0)
    assert rate == 12.0
    assert metadata["method"] == "identity"
    assert torch.equal(view, x)
    assert view.data_ptr() != x.data_ptr()
