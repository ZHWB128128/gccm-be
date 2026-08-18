"""Axiom-mutation triggers: monitor abnormal events and trigger mode switches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..geometry.curvature import CurvatureAnalysis
from ..types import ExternalInput, SystemState


@dataclass
class AxiomMutationTrigger:
    """Rule-based trigger conditions; extensible to learned triggers later."""

    error_threshold: float = 0.8
    price_spike_threshold: float = 1.2
    curvature_anomaly_threshold: float = -0.05

    def check(
        self,
        state: SystemState,
        external: ExternalInput,
        prediction_error: float = 0.0,
        curvature: Optional[CurvatureAnalysis] = None,
    ) -> List[str]:
        triggers: List[str] = []
        if prediction_error > self.error_threshold:
            triggers.append("prediction_error")
        if external.price > self.price_spike_threshold:
            triggers.append("price_spike")
        if curvature is not None and curvature.stability < self.curvature_anomaly_threshold:
            triggers.append("curvature_anomaly")
        # 温度越限
        t = state.x[0]
        if t > 28.0 or t < 18.0:
            triggers.append("temperature_limit")
        return triggers

    def suggest_mode(self, triggers: List[str]) -> Optional[str]:
        if "temperature_limit" in triggers:
            return "comfort"
        if "price_spike" in triggers:
            return "demand_response"
        if "curvature_anomaly" in triggers or "prediction_error" in triggers:
            return "balanced"
        return None

    def suggest_landscape_mutation(self, curvature: Optional[CurvatureAnalysis]) -> Optional[dict]:
        """基于曲率几何结构触发能量景观重构。"""
        if curvature is None:
            return None
        if curvature.classification == "saddle" or curvature.stability < 0.0:
            return {
                "type": "relax_comfort_band",
                "comfort_min": None,
                "comfort_max": None,
                "reason": "saddle_or_singular",
            }
        if curvature.classification == "flat":
            return {
                "type": "tighten_margin",
                "reason": "flat_uncertain_region",
            }
        return None
