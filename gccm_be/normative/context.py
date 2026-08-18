"""上下文标签生成器：将情境信息抽象为标签。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..types import ExternalInput, SystemState


@dataclass
class ContextLabels:
    labels: List[str] = field(default_factory=list)

    def has(self, label: str) -> bool:
        return label in self.labels

    def as_dict(self) -> dict:
        return {label: True for label in self.labels}


class ContextLabeler:
    """根据天气、电价、用户偏好等生成上下文标签。"""

    def __init__(
        self,
        comfort_min: float = 22.0,
        comfort_max: float = 26.0,
        peak_price_threshold: float = 0.9,
    ) -> None:
        self.comfort_min = comfort_min
        self.comfort_max = comfort_max
        self.peak_price_threshold = peak_price_threshold

    def generate(
        self,
        state: SystemState,
        external: ExternalInput,
        time_h: float,
        user_preference: str = "balanced",
    ) -> ContextLabels:
        labels = [f"pref_{user_preference}"]
        t_air = state.x[0]
        if t_air > self.comfort_max:
            labels.append("too_hot")
        elif t_air < self.comfort_min:
            labels.append("too_cold")

        t_out = external.w[0]
        if t_out >= 28.0:
            labels.append("hot_outside")
        elif t_out <= 10.0:
            labels.append("cold_outside")

        price = external.price
        if price >= self.peak_price_threshold:
            labels.append("peak_price")

        hour = time_h % 24.0
        if 10.0 <= hour <= 15.0:
            labels.append("peak_hours")
        if 22.0 <= hour or hour <= 6.0:
            labels.append("night")

        return ContextLabels(labels)
