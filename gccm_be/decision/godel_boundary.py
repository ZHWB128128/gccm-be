"""Godel boundary: three-fold criteria decide undecidability."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        # curvature_min_eig 可为 None(无曲率分析)→ 视为通过(不恶化)
        curvature_bad = (curvature_min_eig is not None
                         and curvature_min_eig < self.curvature_threshold)
        self_bad = self_predicted_error > self.self_error_threshold

        undecidable = error_bad and curvature_bad and self_bad
        return {
            "undecidable": undecidable,
            "error_bad": error_bad,
            "curvature_bad": curvature_bad,
            "self_bad": self_bad,
            "severity": sum([error_bad, curvature_bad, self_bad]),
        }

    def geometric_indeterminacy_index(
        self,
        prediction_error: float,
        curvature_min_eig: float,
        self_predicted_error: float,
    ) -> dict:
        """连续化 GI 指数（0~1）：三通道加权和，支持分级响应。

        边界二突破：AND 三值门是"最后保险丝"；连续指数让系统在**接近**边界时
        即可感知（正常 <0.4 / 观察 0.4~0.7 / 降级 >0.7）。各通道先用经验尺度
        归一化，权重可调。

        通道归一化：error→/1.0；curvature(λ_min 越小越不确定)→1-clip(λ/0.1)；
        self→/1.0。权重默认等权（后续可按工况自适应）。
        """
        e_n = float(np.clip(prediction_error / 1.0, 0.0, 1.0))
        c_n = float(np.clip(1.0 - curvature_min_eig / 0.1, 0.0, 1.0))
        s_n = float(np.clip(self_predicted_error / 1.0, 0.0, 1.0))
        gi = (e_n + c_n + s_n) / 3.0
        level = "normal" if gi < 0.4 else ("watch" if gi < 0.7 else "degrade")
        return {"gi": gi, "level": level,
                "channels": {"error": e_n, "curvature": c_n, "self_pred": s_n}}

