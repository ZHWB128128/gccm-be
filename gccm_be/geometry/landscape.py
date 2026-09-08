"""Energy landscape constructor: map normative weights and setpoints into a geometric optimization objective."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics.models import HVACModel
from ..types import ControlInput, ExternalInput, SystemState
from .manifold import StateManifold


@dataclass
class EnergyLandscape:
    """Energy function E(state, control, external) and its weighted structure."""

    setpoints: dict[str, float]
    weights: dict[str, float]
    manifold: StateManifold
    hvac: HVACModel
    comfort_band: float = 1.0
    metric_coupling: float = 0.0
    metric_state_dependence: float = 0.0
    kinetic_weight: float = 1.0
    comfort_min: float | None = None
    comfort_max: float | None = None
    below_comfort_penalty: float = 0.0
    peak_price_threshold: float = 1.0
    peak_energy_penalty: float = 1.0
    storage_targets: dict[str, float] = field(default_factory=dict)
    storage_weight: float = 0.0
    # 每区舒适权重（由重整化序参量相关度驱动）：label -> 乘子。缺省 1.0。
    zone_comfort_weights: dict[str, float] = field(default_factory=dict)
    # 每区独立舒适带：label -> (lo, hi)，覆盖标量 comfort_min/comfort_max。
    # 未给出的区回退到标量边界；两者都缺时用 comfort_band 软带。
    zone_comfort_bounds: dict[str, tuple] = field(default_factory=dict)

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

    def kinetic_term(self, delta_state: np.ndarray, dt: float = 1.0,
                     state: SystemState | None = None) -> float:
        """离散作用量中的动能项 0.5 * w * dz^T g(z) dz / dt^2。

        `kinetic_weight`(w) 控制作用量项相对代价的强度。度量 g(z) 在**当前状态
        `state`** 处取值（而非零点 dummy）——这是关键：当 metric_state_dependence≠0
        时，g 依赖状态，用零点 dummy 会使 exp(state_dep·(0−setpoint)) 塌缩到近零，
        令动能项失效。传入真实状态才能让状态相关度量真正弯曲作用量。
        """
        if dt <= 0:
            dt = 1.0
        ref = state if state is not None else SystemState(
            np.zeros_like(delta_state), list(self.manifold.labels)
        )
        g = self.metric(ref)
        return 0.5 * self.kinetic_weight * float(delta_state.T @ g @ delta_state) / (dt * dt)

    def bounds_for(self, label: str) -> tuple | None:
        """该状态标签的舒适边界 (lo, hi)；无边界时返回 None。

        优先级：每区覆盖 > 标量 comfort_min/comfort_max > None（用 comfort_band 软带）。
        """
        if label in self.zone_comfort_bounds:
            return tuple(self.zone_comfort_bounds[label])
        if self.comfort_min is not None and self.comfort_max is not None:
            return (self.comfort_min, self.comfort_max)
        return None

    def running_cost(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        prev_control: ControlInput | None = None,
    ) -> float:
        comfort = 0.0
        for i, label in enumerate(self.manifold.labels):
            if label in self.setpoints:
                scale = self.manifold.scale.get(label, 1.0)
                setpoint = self.setpoints[label]
                value = state.x[i]
                bounds = self.bounds_for(label)
                if bounds is not None:
                    lo, hi = bounds
                    if value > hi:
                        excess = value - hi
                    elif value < lo:
                        excess = (lo - value) * self.below_comfort_penalty
                    else:
                        excess = 0.0
                else:
                    deviation = abs(value - setpoint)
                    excess = max(0.0, deviation - self.comfort_band)
                # 分区舒适权重：重整化识别出的关键序参量区获得更高惩罚权重，
                # 使 MPC 把有限的制冷努力优先投向真正驱动全楼峰值的区。
                zone_w = self.zone_comfort_weights.get(label, 1.0)
                comfort += zone_w * (excess / scale) ** 2

        elec = self.hvac.electrical_power(control, external)
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
                zone_w = self.zone_comfort_weights.get(label, 1.0)
                cost += zone_w * (excess / scale) ** 2
        cost = self.weights.get("comfort", 1.0) * cost
        if self.storage_weight > 0.0 and self.storage_targets:
            for i, label in enumerate(self.manifold.labels):
                if label in self.storage_targets:
                    scale = self.manifold.scale.get(label, 1.0)
                    warm = max(0.0, state.x[i] - self.storage_targets[label])
                    cost += self.storage_weight * (warm / scale) ** 2
        return cost
