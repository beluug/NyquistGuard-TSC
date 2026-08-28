"""Dependency-light classification and selective prediction metrics."""

from __future__ import annotations

import math

import numpy as np


def align_probability_columns(
    probabilities: np.ndarray,
    observed_classes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Map estimator-specific probability columns onto global class indices."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(observed_classes)
    if probabilities.ndim != 2 or classes.ndim != 1:
        raise ValueError("probabilities must be 2-D and observed_classes 1-D")
    if probabilities.shape[1] != len(classes):
        raise ValueError("probability column count does not match observed_classes")
    if not np.issubdtype(classes.dtype, np.integer):
        if not np.equal(classes, np.floor(classes)).all():
            raise ValueError("observed classes must be integer global indices")
        classes = classes.astype(np.int64)
    classes = classes.astype(np.int64, copy=False)
    if len(set(classes.tolist())) != len(classes) or np.any(classes < 0) or np.any(classes >= num_classes):
        raise ValueError("observed classes must be unique indices within the global class range")
    aligned = np.zeros((probabilities.shape[0], int(num_classes)), dtype=np.float64)
    aligned[:, classes] = probabilities
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(aligned)) or np.any(row_sums <= 0):
        raise ValueError("probabilities must be finite with a positive row sum")
    return aligned / row_sums


def _macro_f1(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    scores: list[float] = []
    for label in range(num_classes):
        true_positive = np.sum((targets == label) & (predictions == label))
        false_positive = np.sum((targets != label) & (predictions == label))
        false_negative = np.sum((targets == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def _balanced_accuracy(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    recalls: list[float] = []
    for label in range(num_classes):
        selected = targets == label
        if selected.any():
            recalls.append(float(np.mean(predictions[selected] == label)))
    return float(np.mean(recalls)) if recalls else math.nan


def _ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value if total else math.nan)


def _aurc(errors: np.ndarray, acceptance: np.ndarray) -> float:
    order = np.argsort(-acceptance, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return float(np.mean(cumulative_risk))


def classification_metrics(targets: np.ndarray, logits: np.ndarray, acceptance: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or len(targets) != len(logits):
        raise ValueError("targets/logits must have shapes [N] and [N,num_classes]")
    if len(targets) == 0 or np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("targets contain a class outside the logits class axis")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    threshold_mask = np.asarray(acceptance) >= 0.5
    selective_risk = float(1.0 - correct[threshold_mask].mean()) if threshold_mask.any() else 0.0
    one_hot = np.eye(probabilities.shape[1])[targets]
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": _balanced_accuracy(targets, predictions, probabilities.shape[1]),
        "macro_f1": _macro_f1(targets, predictions, probabilities.shape[1]),
        "nll": float(-np.log(probabilities[np.arange(len(targets)), targets].clip(1e-12)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": _ece(targets, probabilities),
        "coverage_at_0_5": float(threshold_mask.mean()),
        "selective_risk_at_0_5": selective_risk,
        "aurc": _aurc((~correct).astype(np.float64), np.asarray(acceptance, dtype=np.float64)),
    }
