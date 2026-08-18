"""结构因果模型（SCM）原型：用于可辩护的因果推断与 do-干预。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np


@dataclass
class StructuralCausalModel:
    """极简 SCM：每个变量由其父变量和噪声决定。

    示例变量：
        mode, metric, control, temperature, cost
    """

    equations: Dict[str, Callable[[Dict[str, float]], float]] = field(default_factory=dict)
    noise: Dict[str, float] = field(default_factory=dict)

    def sample(self) -> Dict[str, float]:
        values: Dict[str, float] = dict(self.noise)
        # 按拓扑顺序简单处理：这里假设 dict 插入顺序即因果顺序
        for var, func in self.equations.items():
            values[var] = func(values)
        return values

    def do(self, intervention: Dict[str, float]) -> Dict[str, float]:
        """do-干预：固定某些变量，切断其父变量影响。"""
        values = dict(self.noise)
        values.update(intervention)
        for var, func in self.equations.items():
            if var in intervention:
                continue
            values[var] = func(values)
        return values

    def effect(self, var: str, do_a: Dict[str, float], do_b: Dict[str, float]) -> float:
        """计算 do(do_a) 与 do(do_b) 对 var 的因果效应。"""
        return float(self.do(do_a)[var] - self.do(do_b)[var])


def build_rc_scm(building, hvac, mode: float = 0.0) -> "StructuralCausalModel":
    """从 RC 物理模型自动生成 SCM。

    变量：
        mode -> comfort_weight -> cooling_power -> temperature -> cost
    """
    c_air = getattr(building, "c_air", 0.6)
    r_air = getattr(building, "r_air", 0.8)
    r_wall = getattr(building, "r_wall", 2.0)
    cop_cooling = getattr(hvac, "cop_cooling", 3.8)

    # 用物理参数设定简化因果系数
    k_cool = 1.0 / (c_air * r_air) * 0.5
    k_cost = 1.0 / cop_cooling

    return StructuralCausalModel(
        equations={
            "comfort_weight": lambda v: 5.0 if v["mode"] == 0.0 else 0.5,
            "cooling_power": lambda v: 1.0 + v["comfort_weight"] * k_cool,
            "temperature": lambda v: 28.0 - v["cooling_power"] * 0.4,
            "cost": lambda v: v["cooling_power"] * k_cost,
        },
        noise={"mode": mode},
    )
