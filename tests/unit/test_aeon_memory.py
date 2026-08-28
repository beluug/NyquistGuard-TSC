from __future__ import annotations

import numpy as np

from nyquistguard.experiments.aeon_memory import enable_multirocket_memory_patch


def test_multirocket_memory_patch_is_numerically_equivalent() -> None:
    from aeon.transformations.collection.convolution_based import MultiRocket

    rng = np.random.default_rng(17)
    train = rng.normal(size=(8, 2, 32)).astype(np.float32)
    test = rng.normal(size=(3, 2, 32)).astype(np.float32)
    transformer = MultiRocket(n_kernels=168, n_jobs=1, random_state=17).fit(train)
    original_method = MultiRocket._transform
    expected = original_method(transformer, test)
    try:
        enable_multirocket_memory_patch()
        actual = transformer.transform(test)
    finally:
        MultiRocket._transform = original_method
    np.testing.assert_array_equal(actual, expected)

