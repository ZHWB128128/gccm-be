"""State manifold manager: state dimensions, semantics, normalization and coordinate transforms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..types import SystemState


@dataclass
class StateManifold:
    """Maintains semantic information of the state vector."""

    labels: List[str]
    units: Dict[str, str] = field(default_factory=dict)
    bounds: Dict[str, tuple[float, float]] = field(default_factory=dict)
    scale: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in self.labels:
            self.units.setdefault(label, "-")
            self.scale.setdefault(label, 1.0)

    @property
    def dim(self) -> int:
        return len(self.labels)

    def normalize(self, state: SystemState) -> np.ndarray:
        """将物理状态映射到无量纲归一化坐标。"""
        return np.array([
            (value - self.bounds.get(label, (0.0, 1.0))[0]) / self.scale.get(label, 1.0)
            for value, label in zip(state.x, self.labels)
        ])

    def denormalize(self, x_norm: np.ndarray) -> SystemState:
        values = []
        for x, label in zip(x_norm, self.labels):
            low = self.bounds.get(label, (0.0, 1.0))[0]
            values.append(low + x * self.scale.get(label, 1.0))
        return SystemState(np.array(values), list(self.labels))

    def curvature_metric(self) -> np.ndarray:
        """用于几何计算的简单对角度量张量，以物理尺度倒数构造。"""
        return np.diag([1.0 / (self.scale.get(label, 1.0) ** 2) for label in self.labels])
