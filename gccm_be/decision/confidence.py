"""Confidence evaluator: combine prediction error, curvature and model mismatch into a strategy confidence level."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.curvature import CurvatureAnalysis


@dataclass
class ConfidenceEvaluator:
    """Simple weighted confidence model."""

    error_scale: float = 2.0
    curvature_scale: float = 0.02

    def evaluate(
        self,
        prediction_error: float = 0.0,
        curvature: CurvatureAnalysis | None = None,
        model_mismatch: float = 0.0,
    ) -> float:
        confidence = 1.0
        confidence -= prediction_error / self.error_scale
        if curvature is not None:
            # 负稳定性越强，置信度越低
            confidence -= max(0.0, -curvature.stability) / self.curvature_scale * 0.1
        confidence -= model_mismatch
        return float(np.clip(confidence, 0.0, 1.0))
