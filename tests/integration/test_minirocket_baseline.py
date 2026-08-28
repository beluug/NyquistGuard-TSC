import numpy as np

from nyquistguard.experiments.metrics import align_probability_columns


def test_minirocket_tiny_multivariate_fit() -> None:
    from aeon.classification.convolution_based import MiniRocketClassifier

    rng = np.random.default_rng(17)
    x = rng.normal(size=(12, 2, 32)).astype(np.float32)
    y = np.asarray([0, 1] * 6)
    x[y == 1, 0] += 1.0
    model = MiniRocketClassifier(n_kernels=168, n_jobs=1, random_state=17)
    model.fit(x, y)
    probabilities = model.predict_proba(x[:3])
    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_minirocket_missing_training_class_is_aligned_defensively() -> None:
    from aeon.classification.convolution_based import MiniRocketClassifier

    rng = np.random.default_rng(23)
    x = rng.normal(size=(12, 2, 32)).astype(np.float32)
    y = np.asarray([0, 2] * 6)
    model = MiniRocketClassifier(n_kernels=168, n_jobs=1, random_state=23).fit(x, y)
    local = model.predict_proba(x[:2])
    aligned = align_probability_columns(local, model.classes_, num_classes=3)
    assert aligned.shape == (2, 3)
    assert np.allclose(aligned[:, 1], 0.0)
