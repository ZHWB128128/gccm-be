"""Weight mapper: map context labels and modes into energy landscape weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..types import EnergyLandscapeParams
from .context import ContextLabels
from .modes import ModeManager


class WeightMapper:
    """Generates energy landscape weights from mode and context."""

    BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
        "comfort": {"comfort": 1.2, "energy": 0.1, "smooth": 0.3},
        "balanced": {"comfort": 0.8, "energy": 0.5, "smooth": 0.2},
        "energy": {"comfort": 0.4, "energy": 1.2, "smooth": 0.2},
        "demand_response": {"comfort": 0.2, "energy": 2.0, "smooth": 0.3},
    }

    def map(
        self,
        mode: str,
        context: ContextLabels,
        setpoints: Dict[str, float],
    ) -> EnergyLandscapeParams:
        weights = dict(self.BASE_WEIGHTS.get(mode, self.BASE_WEIGHTS["balanced"]))

        # 上下文微调
        if context.has("peak_price"):
            weights["energy"] *= 1.5
        if context.has("too_hot") or context.has("too_cold"):
            weights["comfort"] *= 1.2

        return EnergyLandscapeParams(
            weights=weights,
            setpoints=dict(setpoints),
            mode=mode,
            metadata={"context": context.as_dict()},
        )
