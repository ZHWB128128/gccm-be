"""Energy landscape constructor: map normative weights and setpoints into a geometric optimization objective."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from ..physics.models import HVACModel
from ..types import ControlInput, ExternalInput, SystemState
from .manifold import StateManifold
from .metric_tensor import MetricTensor


@dataclass
class EnergyLandscape:
    """Energy function E(state, control, external) and its weighted structure."""

    setpoints: Dict[str, float]
    weights: Dict[str, float]
    manifold: StateManifold
    hvac: HVACModel
    comfort_band: float = 1.0
    metric_coupling: float = 0.0
    metric_state_dependence: float = 0.0
    comfort_min: Optional[float] = None
    comfort_max: Optional[float] = None
    below_comfort_penalty: float = 0.0
    peak_price_threshold: float = 1.0
    peak_energy_penalty: float = 1.0
    storage_targets: Dict[str, float] = field(default_factory=dict)
    storage_weight: float = 0.0

    def __post_init__(self) -> None:
        self.weights.setdefault("comfort", 1.0)
        self.weights.setdefault("energy", 0.5)
        self.weights.setdefault("smooth", 0.2)

    def metric(self, state: SystemState) -> np.ndarray:
        """显式对角度量张量 g(z)。

        数值版与 metric_casadi 使用同一公式。
        """
        n = state.dim
        g = np.eye(n)
        for i, label in enumerate(self.manifold.labels):
            if label in self.setpoints:
                scale = self.manifold.scale.get(label, 1.0)
                base = self.weights.get("comfort", 1.0) / (scale * scale)
                if self.metric_state_dependence != 0.0:
                    dev = state.x[i] - self.setpoints[label]
                    base = base * np.exp(self.metric_state_dependence * dev)
                g[i, i] = base
        if self.metric_coupling != 0.0 and n >= 2:
            # 正定性钳制：|coupling| < sqrt(g00*g11)，保证度量矩阵严格正定
            limit = float(np.sqrt(max(g[0, 0] * g[1, 1], 0.0))) * 0.999
            coupling = float(np.clip(self.metric_coupling, -limit, limit)) if limit > 0.0 else 0.0
            g[0, 1] = coupling
            g[1, 0] = coupling
        return g

    def metric_casadi(self, state):
        """CasADi 符号兼容版本，用于自动微分 Christoffel。"""
        import casadi as ca
        n = self.manifold.dim
        diag = []
        for i, label in enumerate(self.manifold.labels):
            if label in self.setpoints:
                scale = self.manifold.scale.get(label, 1.0)
                base = ca.MX(self.weights.get("comfort", 1.0)) / (scale * scale)
                if self.metric_state_dependence != 0.0:
                    dev = state.x[i] - self.setpoints[label]
                    base = base * ca.exp(self.metric_state_dependence * dev)
                diag.append(base)
            else:
                diag.append(ca.MX(1.0))
        G = ca.MX.zeros(n, n)
        for i, val in enumerate(diag):
            G[i, i] = val
        # 非对角耦合（正定性钳制：|coupling| < sqrt(g00*g11)，与数值版 metric() 一致）
        if self.metric_coupling != 0.0 and n >= 2:
            limit = ca.sqrt(G[0, 0] * G[1, 1]) * 0.999
            coupling = ca.fmin(ca.fmax(ca.MX(self.metric_coupling), -limit), limit)
            G[0, 1] = coupling
            G[1, 0] = coupling
        return G

    def kinetic_term(self, delta_state: np.ndarray, dt: float = 1.0) -> float:
        """离散作用量中的动能项 0.5 * dz^T g dz / dt^2。"""
        if dt <= 0:
            dt = 1.0
        dummy = SystemState(np.zeros_like(delta_state), list(self.manifold.labels))
        g = self.metric(dummy)
        return 0.5 * float(delta_state.T @ g @ delta_state) / (dt * dt)

    def running_cost(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        prev_control: Optional[ControlInput] = None,
    ) -> float:
        comfort = 0.0
        for i, label in enumerate(self.manifold.labels):
            if label in self.setpoints:
                scale = self.manifold.scale.get(label, 1.0)
                setpoint = self.setpoints[label]
                value = state.x[i]
                if self.comfort_min is not None and self.comfort_max is not None:
                    if value > self.comfort_max:
                        excess = value - self.comfort_max
                    elif value < self.comfort_min:
                        excess = (self.comfort_min - value) * self.below_comfort_penalty
                    else:
                        excess = 0.0
                else:
                    deviation = abs(value - setpoint)
                    excess = max(0.0, deviation - self.comfort_band)
                comfort += (excess / scale) ** 2

        elec = self.hvac.electrical_power(control)
        price = external.price
        energy_price = price * (self.peak_energy_penalty if price > self.peak_price_threshold else 1.0)
        energy = energy_price * elec

        # 储能目标项：仅惩罚"高于目标温度"（罐偏暖 = 冷量不足），
        # 给 MPC 一个储存冷量的价值信号，避免时域内短视排空
        storage = 0.0
        if self.storage_weight > 0.0 and self.storage_targets:
            for i, label in enumerate(self.manifold.labels):
                if label in self.storage_targets:
                    scale = self.manifold.scale.get(label, 1.0)
                    warm = max(0.0, state.x[i] - self.storage_targets[label])
                    storage += (warm / scale) ** 2

        smooth = 0.0
        if prev_control is not None:
            smooth = float(np.sum((control.u - prev_control.u) ** 2))

        return (
            self.weights.get("comfort", 1.0) * comfort
            + self.weights.get("energy", 0.5) * energy
            + self.weights.get("smooth", 0.2) * smooth
            + self.storage_weight * storage
        )

    def terminal_cost(self, state: SystemState) -> float:
        """终端代价，鼓励末状态接近设定点。"""
        cost = 0.0
        for i, label in enumerate(self.manifold.labels):
            if label in self.setpoints:
                scale = self.manifold.scale.get(label, 1.0)
                setpoint = self.setpoints[label]
                value = state.x[i]
                if self.comfort_min is not None and self.comfort_max is not None:
                    if value > self.comfort_max:
                        excess = value - self.comfort_max
                    elif value < self.comfort_min:
                        excess = (self.comfort_min - value) * self.below_comfort_penalty
                    else:
                        excess = 0.0
                else:
                    deviation = abs(value - setpoint)
                    excess = max(0.0, deviation - self.comfort_band)
                cost += (excess / scale) ** 2
        cost = self.weights.get("comfort", 1.0) * cost
        if self.storage_weight > 0.0 and self.storage_targets:
            for i, label in enumerate(self.manifold.labels):
                if label in self.storage_targets:
                    scale = self.manifold.scale.get(label, 1.0)
                    warm = max(0.0, state.x[i] - self.storage_targets[label])
                    cost += self.storage_weight * (warm / scale) ** 2
        return cost
