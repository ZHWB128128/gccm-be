"""Godel boundary: three-fold criteria decide undecidability."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GodelBoundary:
    """Outputs undecidable when prediction error, geometric curvature and self-predicted error simultaneously degrade."""

    error_threshold: float = 0.8
    curvature_threshold: float = 0.01
    self_error_threshold: float = 0.5

    def evaluate(
        self,
        prediction_error: float,
        curvature_min_eig: float,
        self_predicted_error: float,
    ) -> dict:
        error_bad = prediction_error > self.error_threshold
        curvature_bad = curvature_min_eig < self.curvature_threshold
        self_bad = self_predicted_error > self.self_error_threshold

        undecidable = error_bad and curvature_bad and self_bad
        return {
            "undecidable": undecidable,
            "error_bad": error_bad,
            "curvature_bad": curvature_bad,
            "self_bad": self_bad,
            "severity": sum([error_bad, curvature_bad, self_bad]),
        }
