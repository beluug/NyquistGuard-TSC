"""Reproducible, read-only analysis helpers for completed experiments."""

from .v4_confirmation_artifacts import (
    ConfirmationArtifactError,
    audit_confirmation,
    build_confirmation_artifacts,
    exact_sign_flip_test,
    package_confirmation_supplement,
    summarize_training_health,
)

__all__ = [
    "ConfirmationArtifactError",
    "audit_confirmation",
    "build_confirmation_artifacts",
    "exact_sign_flip_test",
    "package_confirmation_supplement",
    "summarize_training_health",
]
