"""自指回路：模型监控自身预测误差，并反馈给决策层。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SelfMonitor:
    """轻量自指误差模型。

    每个控制周期接收实际状态与预测状态的残差，
    用滑动平均/AR(1) 预测未来误差是否发散。
    """

    window: int = 24
    divergence_ratio: float = 1.5
    error_threshold: float = 0.5
    residuals: List[float] = field(default_factory=list)

    def update(self, error: float) -> None:
        self.residuals.append(float(error))
        if len(self.residuals) > self.window:
            self.residuals.pop(0)

    def recent_mean(self, n: int = 6) -> float:
        if not self.residuals:
            return 0.0
        return float(np.mean(self.residuals[-n:]))

    def predicted_error(self, horizon: int = 1) -> float:
        """基于最近残差趋势的简单 AR(1) 预测。"""
        if len(self.residuals) < 3:
            return self.recent_mean()
        x = np.array(self.residuals)
        y = x[1:]
        X = np.column_stack([x[:-1], np.ones(len(y))])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coef
            pred = alpha * x[-1] + beta
            return float(max(0.0, pred))
        except np.linalg.LinAlgError:
            return self.recent_mean()

    def is_diverging(self) -> bool:
        if len(self.residuals) < 6:
            return False
        return self.recent_mean(3) > self.divergence_ratio * self.recent_mean(12) + 1e-6

    def confidence_factor(self) -> float:
        """返回 [0,1] 的自置信度系数，误差越大越低。"""
        pred = self.predicted_error()
        return float(np.clip(1.0 - pred / max(self.error_threshold, 1e-6), 0.0, 1.0))

    def feedback(self) -> dict:
        return {
            "predicted_error": self.predicted_error(),
            "diverging": self.is_diverging(),
            "confidence_factor": self.confidence_factor(),
            "recent_mean": self.recent_mean(),
        }
