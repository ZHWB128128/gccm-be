"""哥德尔边界：三重判据决定不可判定。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GodelBoundary:
    """当预测误差、几何曲率、自指误差模型同时恶化时输出不可判定。"""

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
