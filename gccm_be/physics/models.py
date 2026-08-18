"""Physical world simulation layer: building thermal and HVAC system models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..types import ControlInput, ExternalInput, SystemState


@dataclass
class RCBuildingModel:
    """Second-order RC building thermal model.

    States:
        T_air  : indoor air temperature (°C)
        T_wall : wall temperature (°C)
    Control:
        Q_hvac : heat injected into the room (kW); positive = heating, negative = cooling
    External:
        T_out  : outdoor temperature (°C)
        solar  : solar equivalent heat power (kW)
        occ    : occupants/equipment heat power (kW)
        price  : electricity price (¥/kWh), does not affect state evolution
    """

    c_air: float = 0.6       # kWh/K
    c_wall: float = 4.0      # kWh/K
    r_air: float = 0.8       # K/kW, 空气-墙体热阻
    r_wall: float = 2.0      # K/kW, 墙体-室外热阻
    solar_gain: float = 0.05  # 太阳辐射进入室内比例
    dt: float = 1.0 / 12.0    # 默认步长: 5 分钟 (小时)

    state_labels: List[str] = field(default_factory=lambda: ["T_air", "T_wall"])
    control_labels: List[str] = field(default_factory=lambda: ["Q_hvac"])
    external_labels: List[str] = field(default_factory=lambda: ["T_out", "solar", "occ", "price"])

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: Optional[float] = None,
    ) -> SystemState:
        if dt is None:
            dt = self.dt
        T_air, T_wall = state.x
        Q_hvac = control.u[0]
        T_out = external.w[0]
        solar = external.w[1]
        occ = external.w[2]

        dT_air = (
            (T_wall - T_air) / self.r_air
            + (T_out - T_air) / self.r_wall
            + self.solar_gain * solar
            + occ
            + Q_hvac
        ) / self.c_air

        dT_wall = (
            (T_air - T_wall) / self.r_air
            + (T_out - T_wall) / self.r_wall
        ) / self.c_wall

        new_state = np.array([T_air + dT_air * dt, T_wall + dT_wall * dt])
        return SystemState(new_state, list(self.state_labels))

    def initial_state(self, t_air: float = 24.0, t_wall: float = 24.0) -> SystemState:
        return SystemState(np.array([t_air, t_wall]), list(self.state_labels))



@dataclass
class TwoZoneRCBuildingModel:
    """Two-zone RC building thermal model with a coupled partition wall.

    States:
        T_air_A, T_wall_A, T_air_B, T_wall_B, T_partition
    Control:
        Q_hvac_A, Q_hvac_B
    External:
        T_out, solar_A, solar_B, occ_A, occ_B, price
    """

    c_air: float = 0.6
    c_wall: float = 4.0
    c_partition: float = 3.0
    r_air: float = 0.8
    r_wall_a: float = 2.0       # A 区外墙热阻
    r_wall_b: float = 1.5       # B 区外墙热阻（保温稍差）
    r_partition: float = 1.0
    solar_gain_a: float = 0.08  # A 区南向，得热大
    solar_gain_b: float = 0.02  # B 区北向，得热小
    dt: float = 1.0 / 12.0

    state_labels: List[str] = field(default_factory=lambda: [
        "T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition",
    ])
    control_labels: List[str] = field(default_factory=lambda: ["Q_hvac_A", "Q_hvac_B"])
    external_labels: List[str] = field(default_factory=lambda: [
        "T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price",
    ])

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: Optional[float] = None,
    ) -> SystemState:
        if dt is None:
            dt = self.dt
        T_air_A, T_wall_A, T_air_B, T_wall_B, T_partition = state.x
        Q_A, Q_B = control.u
        T_out = external.w[0]
        solar_A = external.w[1]
        solar_B = external.w[2]
        occ_A = external.w[3]
        occ_B = external.w[4]

        dT_air_A = (
            (T_wall_A - T_air_A) / self.r_air
            + (T_out - T_air_A) / self.r_wall_a
            + (T_partition - T_air_A) / self.r_partition
            + self.solar_gain_a * solar_A
            + occ_A
            + Q_A
        ) / self.c_air

        dT_wall_A = (
            (T_air_A - T_wall_A) / self.r_air
            + (T_out - T_wall_A) / self.r_wall_a
        ) / self.c_wall

        dT_air_B = (
            (T_wall_B - T_air_B) / self.r_air
            + (T_out - T_air_B) / self.r_wall_b
            + (T_partition - T_air_B) / self.r_partition
            + self.solar_gain_b * solar_B
            + occ_B
            + Q_B
        ) / self.c_air

        dT_wall_B = (
            (T_air_B - T_wall_B) / self.r_air
            + (T_out - T_wall_B) / self.r_wall_b
        ) / self.c_wall

        dT_partition = (
            (T_air_A - T_partition) / self.r_partition
            + (T_air_B - T_partition) / self.r_partition
        ) / self.c_partition

        new_state = state.x + np.array([
            dT_air_A,
            dT_wall_A,
            dT_air_B,
            dT_wall_B,
            dT_partition,
        ]) * dt
        return SystemState(new_state, list(self.state_labels))

    def initial_state(self, t: float = 28.0) -> SystemState:
        return SystemState(np.full(5, t), list(self.state_labels))


@dataclass
class HVACModel:
    """HVAC system model: converts heat power to electric power and provides control bounds.

    Supports multiple independent units/zones with identical per-unit bounds.
    """

    q_min: float = -6.0   # 每台最大制冷热功率 kW
    q_max: float = 6.0    # 每台最大制热热功率 kW
    cop_heating: float = 3.2
    cop_cooling: float = 3.8
    part_load_penalty: float = 0.15
    n_units: int = 1
    control_labels: List[str] = field(default_factory=lambda: ["Q_hvac"])

    def __post_init__(self) -> None:
        if len(self.control_labels) != self.n_units:
            self.control_labels = [f"Q_hvac_{i}" for i in range(self.n_units)]

    def electrical_power(self, control: ControlInput) -> float:
        total = 0.0
        for q in control.u:
            q = float(q)
            if q >= 0:
                cop = self.cop_heating
                abs_q = q
            else:
                cop = self.cop_cooling
                abs_q = -q
            load_ratio = min(abs_q / max(abs(self.q_min), abs(self.q_max), 1e-9), 1.0)
            cop_eff = cop * (1.0 - self.part_load_penalty * (1.0 - load_ratio) ** 2)
            total += abs_q / max(cop_eff, 1e-6)
        return total

    def bounds(self) -> List[tuple[float, float]]:
        return [(self.q_min, self.q_max)] * self.n_units


class Simulator:
    """Digital-twin simulator: wraps model stepping and trajectory rollout."""

    def __init__(
        self,
        building: Optional[RCBuildingModel] = None,
        hvac: Optional[HVACModel] = None,
    ) -> None:
        self.building = building or RCBuildingModel()
        self.hvac = hvac or HVACModel()

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: Optional[float] = None,
    ) -> SystemState:
        return self.building.step(state, control, external, dt)

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
