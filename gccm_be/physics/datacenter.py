"""数据中心机房冷却模型：冷通道温度 + 蓄冷罐（主动充放冷）。

物理结构（二阶 + 双控制）：
    - 冷通道（cold aisle）：IT 设备产热 Q_it + 室外耦合 + 机组直供冷 + 蓄冷罐放冷
    - 蓄冷罐（thermal storage）：大热容，受主动充冷/放冷指令控制
    - 控制：u0 = 机组直供冷功率（kW，负值=制冷）；u1 = 蓄冷罐流量（正值=充冷，负值=放冷）
    - 电价：峰谷差价驱动 MPC"谷时充冷、峰时放冷"（电价套利 + 峰时削峰）

接口与 Simulator 兼容，可直接接入 GCCMEngine（换模型不换引擎）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..types import ControlInput, ExternalInput, SystemState


@dataclass
class DataCenterCoolingModel:
    """冷通道 + 蓄冷罐热模型（显式 Euler，双控制）。"""

    c_aisle: float = 20.0      # 冷通道热容 kWh/K（约 500m² 机房空气+设备表面）
    c_tank: float = 150.0      # 蓄冷罐热容 kWh/K（约 55m³ 水罐，按峰期 7h×100kW 蓄冷设计）
    r_out: float = 0.20        # 室外耦合热阻 K/kW
    q_disc_max: float = 120.0  # 蓄冷罐最大放冷功率 kW（泵/板换能力）
    q_chg_max: float = 150.0   # 蓄冷罐最大充冷功率 kW
    dt: float = 1.0 / 12.0     # 默认步长 5 分钟

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: Optional[float] = None,
    ) -> SystemState:
        if dt is None:
            dt = self.dt
        T_a, T_s = float(state.x[0]), float(state.x[1])
        u0 = float(control.u[0])           # 机组直供冷（≤0）
        u1 = float(control.u[1])           # 罐流量：>0 充冷，<0 放冷
        q_chill = max(0.0, -u0)            # 直供冷量
        q_chg = max(0.0, u1)               # 充冷（罐降温）
        q_disc = max(0.0, -u1)             # 放冷（冷进冷通道，罐升温）
        Q_it = float(external.w[2])        # IT 负载（内热）
        T_out = float(external.w[0])

        dTa = (
            Q_it
            - q_chill
            - q_disc
            + (T_out - T_a) / self.r_out
        ) / self.c_aisle
        dTs = (q_disc - q_chg) / self.c_tank
        return SystemState(
            np.array([T_a + dTa * dt, T_s + dTs * dt]),
            list(state.labels),
        )


@dataclass
class DataCenterHVAC:
    """制冷机组 + 蓄冷泵：容量、COP（含部分负荷效率）、电功率。"""

    q_min: float = -400.0    # 机组最大制冷 kW（覆盖 IT 峰值）
    q_max: float = 0.0       # 机组无加热
    chg_max: float = 150.0   # 蓄冷罐最大充冷功率 kW
    disc_max: float = 120.0  # 蓄冷罐最大放冷功率 kW
    cop_cooling: float = 5.0
    part_load_penalty: float = 0.15
    n_units: int = 2
    control_labels: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.control_labels is None:
            self.control_labels = ["Q_chiller", "Q_tank"]

    def bounds(self) -> List[tuple[float, float]]:
        return [(self.q_min, self.q_max), (-self.disc_max, self.chg_max)]

    def electrical_power(self, control: ControlInput) -> float:
        """电功率：机组直供冷 |u0|/COP + 充冷 u1/COP（放冷是免费的冷）。"""
        total = 0.0
        u0 = float(control.u[0])
        u1 = float(control.u[1]) if control.u.size > 1 else 0.0
        abs_q = abs(u0) + max(0.0, u1)  # 充冷消耗机组电功
        load_ratio = min(abs_q / max(abs(self.q_min), 1e-9), 1.0)
        cop_eff = self.cop_cooling * (1.0 - self.part_load_penalty * (1.0 - load_ratio) ** 2)
        total += abs_q / max(cop_eff, 1e-6)
        return total


@dataclass
class DataCenterBuilding:
    dt: float = 1.0 / 12.0


class DataCenterSimulator:
    """与 Simulator 兼容的数据中心仿真器（可直接传给 GCCMEngine）。"""

    def __init__(
        self,
        model: Optional[DataCenterCoolingModel] = None,
        hvac: Optional[DataCenterHVAC] = None,
    ) -> None:
        self.model = model or DataCenterCoolingModel()
        self.hvac = hvac or DataCenterHVAC()
        self.building = DataCenterBuilding(dt=self.model.dt)

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: Optional[float] = None,
    ) -> SystemState:
        return self.model.step(state, control, external, dt)

    def rollout(
        self,
        initial_state: SystemState,
        controls: Sequence[ControlInput],
        externals: Sequence[ExternalInput],
        dt: Optional[float] = None,
    ) -> List[SystemState]:
        states = [initial_state.copy()]
        state = initial_state
        for u, w in zip(controls, externals):
            state = self.step(state, u, w, dt)
            states.append(state.copy())
        return states


class DataCenterProvider:
    """数据中心外部输入：室外温度 / 太阳(0) / IT 负载 / 电价。

    布局与 ExternalInput 约定一致：w=[T_out, solar, it_load, price]。
    IT 负载：白天高（200 kW）、夜间低（120 kW）；电价含峰谷尖峰。
    """

    labels: List[str] = ["T_out", "solar", "it_load", "price"]

    def __init__(self, peak_price: float = 3.0, seed: int = 0) -> None:
        self.peak_price = peak_price
        self.rng = np.random.default_rng(seed)

    def get(self, time_h: float, horizon: int = 1) -> List[ExternalInput]:
        result = []
        for k in range(horizon):
            t = time_h + k * (1.0 / 12.0)
            hour = t % 24.0
            t_out = 26.0 + 6.0 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            it_load = 170.0 if 8.0 <= hour <= 22.0 else 110.0
            it_load += self.rng.normal(0.0, 5.0)  # IT 负载波动
            if hour < 8.0 or hour >= 22.0:
                price = 0.4
            elif hour < 11.0 or hour >= 18.0:
                price = 0.9
            else:
                price = self.peak_price  # 峰时电价
            result.append(ExternalInput(np.array([t_out, 0.0, it_load, price]), list(self.labels)))
        return result
