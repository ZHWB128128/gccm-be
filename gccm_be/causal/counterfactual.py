"""可辩护的因果链：通过反事实仿真解释“几何结构 -> 控制 -> 结果”。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..physics.models import Simulator
from .scm import StructuralCausalModel
from ..types import ControlInput, ExternalInput, SystemState


@dataclass
class CounterfactualResult:
    name: str
    total_cost: float
    comfort_violation: float
    peak_power: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "total_cost": self.total_cost,
            "comfort_violation": self.comfort_violation,
            "peak_power": self.peak_power,
        }


class CounterfactualAnalyzer:
    """反事实分析器。

    在相同外部输入和初始状态下，替换控制策略，比较结果差异。
    这可以支撑“规范权重/度量 -> 控制策略 -> 能耗/舒适”的可解释因果链。
    """

    def __init__(
        self,
        simulator: Simulator,
        initial_state: SystemState,
        externals: List[ExternalInput],
        dt: float = 0.25,
        comfort_min: float = 25.0,
        comfort_max: float = 27.0,
        scm: Optional[StructuralCausalModel] = None,
    ) -> None:
        self.simulator = simulator
        self.initial_state = initial_state
        self.externals = externals
        self.dt = dt
        self.comfort_min = comfort_min
        self.comfort_max = comfort_max
        self.scm = scm

    def causal_effect(self, var: str, do_a: dict, do_b: dict) -> float:
        """如果配置了 SCM，则返回 do(do_a) 与 do(do_b) 对 var 的因果效应。"""
        if self.scm is None:
            raise ValueError("SCM not configured")
        return self.scm.effect(var, do_a, do_b)

    def run_policy(self, name: str, policy: Callable[[SystemState, float], ControlInput]) -> CounterfactualResult:
        state = self.initial_state.copy()
        t = 0.0
        temps = []
        powers = []
        prices = []
        for w in self.externals:
            control = policy(state, t)
            state = self.simulator.step(state, control, w, self.dt)
            temps.append(state.x[0] if state.dim > 1 else state.x[0])
            powers.append(self.simulator.hvac.electrical_power(control))
            prices.append(w.w[-1] if w.w.size > 0 else 1.0)
            t += self.dt
        temps = np.array(temps)
        cost = float(np.sum(np.array(powers) * np.array(prices) * self.dt))
        viol = float(np.mean((temps > self.comfort_max) | (temps < self.comfort_min)) * 100.0)
        peak = float(np.max(powers))
        return CounterfactualResult(name, cost, viol, peak)
