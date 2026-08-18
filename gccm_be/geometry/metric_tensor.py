"""显式度量张量：由规范层权重生成，而非固定常数。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ..types import SystemState

if TYPE_CHECKING:
    from .landscape import EnergyLandscape


@dataclass
class MetricTensor:
    """在状态空间上定义对角度量 g(z)。

    当前实现为对角度量，后续可扩展为低秩或全度量。
    """

    comfort_weight: float = 1.0
    energy_weight: float = 0.5
    smooth_weight: float = 0.2
    temperature_scale: float = 5.0
    coupling: float = 0.0
    state_labels: Optional[list] = None

    @classmethod
    def from_landscape(cls, landscape: EnergyLandscape) -> "MetricTensor":
        return cls(
            comfort_weight=landscape.weights.get("comfort", 1.0),
            energy_weight=landscape.weights.get("energy", 0.5),
            smooth_weight=landscape.weights.get("smooth", 0.2),
            temperature_scale=landscape.manifold.scale.get("T_air", 5.0),
            coupling=getattr(landscape, "metric_coupling", 0.0),
            state_labels=list(landscape.manifold.labels),
        )

    def matrix(self, state: SystemState) -> np.ndarray:
        n = state.dim
        g = np.ones(n)
        for i, label in enumerate(self.state_labels or []):
            if label.startswith("T_air"):
                # 温度方向由舒适权重主导，尺度由物理单位决定
                g[i] = max(self.comfort_weight / (self.temperature_scale ** 2), 1e-6)
            else:
                g[i] = 1.0
        g = np.diag(g)
        # 非对角耦合：空气-墙体之间的几何耦合
        if self.coupling != 0.0 and n >= 2:
            g[0, 1] = self.coupling
            g[1, 0] = self.coupling
        return g

    def kinetic(self, delta_state: np.ndarray, state: SystemState, dt: float = 1.0) -> float:
        g = self.matrix(state)
        return 0.5 * float(delta_state.T @ g @ delta_state) / (dt * dt)
