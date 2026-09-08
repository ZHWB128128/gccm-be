"""Confidence evaluator: combine prediction error, curvature and model mismatch into a strategy confidence level."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.curvature import CurvatureAnalysis


@dataclass
class ConfidenceEvaluator:
    """Simple weighted confidence model, optionally self-calibrated.

    固定尺度（`calibrated=False`，默认）为历史行为；`calibrated=True` 时为
    conformal 式分位数校准：维护已实现预测误差的滚动历史，以 P{quantile} 为
    自适应尺度——当前误差相对"见过的最差 10%"越大，置信度越低。曲率与模型
    失配项的相对权重两种模式一致。
    """

    error_scale: float = 2.0
    curvature_scale: float = 0.02
    calibrated: bool = False
    quantile: float = 0.90
    window: int = 100
    min_history: int = 20      # 样本不足回退固定尺度（冷启动）
    _error_history: list[float] = field(default_factory=list, init=False)

    def observe_error(self, prediction_error: float) -> None:
        """记录一个已实现的预测误差（由诊断层每步喂入）。未校准模式不记录。"""
        if not self.calibrated:
            return
        if prediction_error is not None and np.isfinite(prediction_error):
            self._error_history.append(float(prediction_error))
            if len(self._error_history) > self.window:
                del self._error_history[: len(self._error_history) - self.window]

    def error_scale_current(self) -> float:
        """当前有效误差尺度：校准分位数，或固定尺度（冷启动/未校准）。"""
        if self.calibrated and len(self._error_history) >= self.min_history:
            q = float(np.quantile(self._error_history[-self.window:], self.quantile))
            return max(q, 1e-6)
        return self.error_scale

    def evaluate(
        self,
        prediction_error: float = 0.0,
        curvature: CurvatureAnalysis | None = None,
        model_mismatch: float = 0.0,
    ) -> float:
        scale = self.error_scale_current()
        confidence = 1.0
        confidence -= prediction_error / scale
        if curvature is not None:
            # 负稳定性越强，置信度越低
            confidence -= max(0.0, -curvature.stability) / self.curvature_scale * 0.1
        confidence -= model_mismatch
        return float(np.clip(confidence, 0.0, 1.0))
