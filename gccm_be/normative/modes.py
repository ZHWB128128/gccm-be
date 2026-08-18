"""Multi-mode manager: predefined operation modes and switching logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .context import ContextLabels


@dataclass
class ModeManager:
    """Mode set and switching."""

    current_mode: str = "balanced"
    allowed_modes: List[str] = field(default_factory=lambda: ["comfort", "balanced", "energy", "demand_response"])

    def switch(self, new_mode: str) -> bool:
        if new_mode in self.allowed_modes:
            self.current_mode = new_mode
            return True
        return False

    def suggest_mode(self, context: ContextLabels, forced: Optional[str] = None) -> str:
        if forced is not None:
            return forced
        if context.has("peak_price") and context.has("too_hot"):
            # 需求响应优先但需保证最低安全
            return "demand_response"
        if context.has("peak_price"):
            return "energy"
        if context.has("too_hot") or context.has("too_cold"):
            return "comfort"
        return "balanced"
