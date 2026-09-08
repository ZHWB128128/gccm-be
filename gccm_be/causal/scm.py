"""Structural causal model (SCM) prototype for defensible causal inference and do-intervention."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class StructuralCausalModel:
    """Minimal SCM: each variable is determined by its parents and noise.

    Example variables:
        mode, metric, control, temperature, cost
    """

    equations: dict[str, Callable[[dict[str, float]], float]] = field(default_factory=dict)
    noise: dict[str, float] = field(default_factory=dict)

    def sample(self) -> dict[str, float]:
        values: dict[str, float] = dict(self.noise)
        # 按拓扑顺序简单处理：这里假设 dict 插入顺序即因果顺序
        for var, func in self.equations.items():
            values[var] = func(values)
        return values

    def do(self, intervention: dict[str, float]) -> dict[str, float]:
        """do-干预：固定某些变量，切断其父变量影响。"""
        values = dict(self.noise)
        values.update(intervention)
        for var, func in self.equations.items():
            if var in intervention:
                continue
            values[var] = func(values)
        return values

    def effect(self, var: str, do_a: dict[str, float], do_b: dict[str, float]) -> float:
        """计算 do(do_a) 与 do(do_b) 对 var 的因果效应。"""
        return float(self.do(do_a)[var] - self.do(do_b)[var])


def build_rc_scm(
    building,
    hvac,
    mode: float = 0.0,
    t_out_nominal: float = 33.0,
    internal_gain: float = 0.7,
    comfort_setpoint: float = 25.5,
    dr_setpoint: float = 27.0,
) -> StructuralCausalModel:
    """从 RC 物理模型自动生成 SCM（物理接地版，已退休硬编码玩具系数）。

    因果链： mode -> comfort_weight -> cooling_power -> temperature -> cost

    所有系数均由 RC 模型的**稳态热平衡**导出，而非示意常数：

    - 稳态温度增益 R_eff = 1 / (1/(r_air+r_wall) + 1/r_wall)：
      令 dT_wall/dt = 0 消去墙体节点后，dT_air/dt = 0 给出
          T_air = T_out + (G + Q) * R_eff
      其中 G = 内部/太阳得热(kW)，Q = HVAC 热功率(kW，制冷为负)。
      因此 ∂T_air/∂Q = R_eff 是真实的物理稳态增益（°C/kW）。
    - 制冷电功率 = |Q| / COP_cooling，直接来自 HVACModel 的能效定义。
    - 各模式的目标温度决定所需制冷量：Q = (T_target - T_out)/R_eff - G，
      模式差异（comfort 更冷 vs demand_response 更暖）由 setpoint 驱动，
      不再使用 comfort_weight 的任意数值。
    """
    r_air = getattr(building, "r_air", 0.8)
    r_wall = getattr(building, "r_wall", 2.0)
    cop_cooling = getattr(hvac, "cop_cooling", 3.8)

    # 稳态温度增益（°C/kW），由 2R RC 热平衡精确导出
    r_eff = 1.0 / (1.0 / (r_air + r_wall) + 1.0 / r_wall)

    def _cooling_for_target(t_target: float) -> float:
        # 达到目标温度所需的净热功率 Q（制冷为负），来自 T_target = T_out + (G+Q)*R_eff
        q = (t_target - t_out_nominal) / r_eff - internal_gain
        return abs(min(q, 0.0))  # 只计制冷；若无需制冷则为 0

    return StructuralCausalModel(
        equations={
            # 目标温度：comfort 模式盯紧较低设定点，demand_response 允许漂移到上限
            "target_temp": lambda v: comfort_setpoint if v["mode"] == 0.0 else dr_setpoint,
            # 达到目标所需的制冷热功率（kW），由稳态增益反解
            "cooling_power": lambda v: _cooling_for_target(v["target_temp"]),
            # 实际稳态温度 = 目标温度（闭环达标），保留为可观测变量
            "temperature": lambda v: v["target_temp"],
            # 制冷电费 ∝ 制冷电功率 = 热功率 / COP
            "cost": lambda v: v["cooling_power"] / cop_cooling,
        },
        noise={"mode": mode},
    )
