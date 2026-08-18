"""外部输入模块：天气预报、电价、室内热源等统一数据接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..types import ExternalInput


class ExternalInputProvider(ABC):
    """外部输入数据源抽象接口。"""

    @abstractmethod
    def get(self, time_h: float, horizon: int = 1) -> List[ExternalInput]:
        """返回从当前时刻开始的未来 horizon 步外部输入。"""
        raise NotImplementedError


@dataclass
class MockExternalInputProvider(ExternalInputProvider):
    """用于演示的确定性外部输入源。"""

    t_out_base: float = 30.0
    t_out_amp: float = 5.0
    solar_base: float = 0.3
    price_base: float = 0.6
    occ_base: float = 0.5
    dt_h: float = 1.0 / 12.0

    labels: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = ["T_out", "solar", "occ", "price"]

    def get(self, time_h: float, horizon: int = 1) -> List[ExternalInput]:
        result = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            t_out = self.t_out_base + self.t_out_amp * np.sin(2 * np.pi * t / 24.0 - 1.0)
            solar = max(0.0, self.solar_base * np.sin(np.pi * ((t - 6.0) / 12.0))) if 6.0 <= t % 24.0 <= 18.0 else 0.0
            price = self.price_base * (1.8 if 10.0 <= t % 24.0 <= 15.0 else 1.0)
            occ = self.occ_base * (1.5 if 8.0 <= t % 24.0 <= 18.0 else 0.3)
            result.append(ExternalInput(np.array([t_out, solar, occ, price]), list(self.labels)))
        return result
