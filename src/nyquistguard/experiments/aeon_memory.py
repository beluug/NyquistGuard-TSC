"""Memory-equivalent compatibility helpers for the pinned aeon 1.3 runtime."""

from __future__ import annotations

import multiprocessing
from typing import Any, Callable

import numpy as np
from numba import get_num_threads, set_num_threads


def _memory_safe_multirocket_transform(self: Any, X: np.ndarray, y: Any = None) -> np.ndarray:
    """Pinned aeon MultiRocket transform without redundant full-array copies.

    This is aeon 1.3's ``MultiRocket._transform`` with only two allocation
    changes: ``astype(..., copy=False)`` and ``nan_to_num(..., copy=False)``.
    Both preserve the returned numerical values because the convolution output
    is a newly allocated working array.
    """
    from aeon.transformations.collection.convolution_based._multirocket import (
        MultiRocket,
        _transform_multi,
        _transform_uni,
    )

    _, n_channels, _n_timepoints = X.shape
    if self.normalise:
        X = (X - X.mean(axis=-1, keepdims=True)) / (
            X.std(axis=-1, keepdims=True) + 1e-8
        )
    previous_threads = get_num_threads()
    if self._n_jobs < 1 or self._n_jobs > multiprocessing.cpu_count():
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = self._n_jobs
    set_num_threads(n_jobs)
    try:
        X = X.astype(np.float32, copy=False)
        if n_channels > 1:
            X1 = np.diff(X, 1)
            transformed = _transform_multi(
                X,
                X1,
                self.parameter,
                self.parameter1,
                self.n_features_per_kernel,
                MultiRocket._indices,
                self.random_state_,
            )
        else:
            X = X.reshape(X.shape[0], X.shape[2])
            X1 = np.diff(X, 1)
            transformed = _transform_uni(
                X,
                X1,
                self.parameter,
                self.parameter1,
                self.n_features_per_kernel,
                MultiRocket._indices,
                self.random_state_,
            )
        return np.nan_to_num(transformed, copy=False)
    finally:
        set_num_threads(previous_threads)


def enable_multirocket_memory_patch() -> Callable[..., np.ndarray]:
    """Install the process-local, numerically equivalent aeon transform patch."""
    from aeon.transformations.collection.convolution_based import MultiRocket

    original = MultiRocket._transform
    MultiRocket._transform = _memory_safe_multirocket_transform
    return original

