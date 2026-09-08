"""Physical world simulation layer: building thermal and HVAC system models."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..types import ControlInput, ExternalInput, SystemState


@dataclass
class ThreeRCBuildingModel:
    """Third-order RC building model: air + wall + furniture thermal mass.

    Adds a furniture/internal-mass node (T_furn) that exchanges heat only with
    air — captures solar-driven furniture heat storage, the main driver of
    afternoon peak-load prediction in high-solar buildings.

    States:
        T_air  : indoor air temperature (°C)
        T_wall : wall temperature (°C)
        T_furn : furniture/internal mass temperature (°C)
    Control:
        Q_hvac : heat injected (kW); positive = heating, negative = cooling
    External:
        T_out, solar (W/m2), occ (kW), price
    """

    c_air: float = 0.6       # kWh/K
    c_wall: float = 4.0      # kWh/K
    c_furn: float = 2.0      # kWh/K, furniture/internal mass
    r_air: float = 0.8       # K/kW, air-wall
    r_wall: float = 2.0      # K/kW, wall-outdoor
    r_furn: float = 1.0      # K/kW, furniture-air
    solar_gain: float = 0.05  # solar into air/furniture (fraction of W/m2 scale)
    dt: float = 1.0 / 12.0

    state_labels: list[str] = field(default_factory=lambda: ["T_air", "T_wall", "T_furn"])
    control_labels: list[str] = field(default_factory=lambda: ["Q_hvac"])
    external_labels: list[str] = field(default_factory=lambda: ["T_out", "solar", "occ", "price"])

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: float | None = None,
    ) -> SystemState:
        if dt is None:
            dt = self.dt
        T_air, T_wall, T_furn = state.x
        Q_hvac = control.u[0]
        T_out = external.w[0]
        solar = external.w[1]
        occ = external.w[2]

        dT_air = (
            (T_wall - T_air) / self.r_air
            + (T_furn - T_air) / self.r_furn
            + (T_out - T_air) / self.r_wall
            + self.solar_gain * solar
            + occ
            + Q_hvac
        ) / self.c_air
        dT_wall = (
            (T_air - T_wall) / self.r_air
            + (T_out - T_wall) / self.r_wall
        ) / self.c_wall
        dT_furn = ((T_air - T_furn) / self.r_furn) / self.c_furn

        new_state = np.array([
            T_air + dT_air * dt,
            T_wall + dT_wall * dt,
            T_furn + dT_furn * dt,
        ])
        return SystemState(new_state, list(self.state_labels))

    def initial_state(self, t_air: float = 24.0, t_wall: float = 24.0,
                      t_furn: float | None = None) -> SystemState:
        if t_furn is None:
            t_furn = t_air
        return SystemState(np.array([t_air, t_wall, t_furn]), list(self.state_labels))


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

    state_labels: list[str] = field(default_factory=lambda: ["T_air", "T_wall"])
    control_labels: list[str] = field(default_factory=lambda: ["Q_hvac"])
    external_labels: list[str] = field(default_factory=lambda: ["T_out", "solar", "occ", "price"])

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: float | None = None,
    ) -> SystemState:
        if dt is None:
            dt = self.dt
        T_air, T_wall = state.x
        Q_hvac = control.u[0]
        T_out = external.get("T_out", external.w[0])
        solar = external.get("solar", external.w[1] if external.w.size > 1 else 0.0)
        occ = external.get("occ", external.w[2] if external.w.size > 2 else 0.0)

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


def dew_point_gram(w_g_kg: float, pressure_kpa: float = 101.325) -> float:
    """绝对湿度 (g/kg 干空气) → 露点 (°C)。Magnus 公式反解（工程近似）。"""
    import math

    if w_g_kg <= 0:
        return -50.0
    pv = w_g_kg * pressure_kpa / (622.0 + w_g_kg) * 1000.0  # 水汽分压 Pa（w 为 g/kg 干空气）
    pv = max(pv, 1.0)
    ln = math.log(pv / 610.94)
    return 243.04 * ln / (17.625 - ln)


@dataclass
class NonlinearRCBuildingModel:
    """Second-order RC building with three physically-motivated NONLINEARITIES
    that a linear RC model cannot represent — the regime where geometric
    (Riemannian) control is theoretically expected to matter.

    States:
        T_air  : indoor air temperature (°C)
        T_wall : wall temperature (°C)
        W_air  : indoor absolute humidity (g/kg), driven by latent loads
    Control:
        Q_hvac : sensible heat (kW; <0 cooling)  [u[0]]
        m_dot  : supply-air / fan flow fraction in [0,1] (VAV)  [u[1], optional]
    External:
        T_out, solar, occ, price, W_out (outdoor humidity, optional)

    Nonlinearities
    --------------
    1. Window opening (natural ventilation): when the indoor-outdoor temperature
       difference is favorable AND indoors is warm, a buoyancy-driven vent opens
       with a smooth logistic gate, adding a coupling term
       g_win(T_air,T_out)·(T_out − T_air). This is bilinear in the state and
       switches on/off — a genuine state-dependent conductance.
    2. Dehumidification threshold: cooling only removes latent load once the coil
       is below dew point, i.e. when |Q_hvac| exceeds a sensible threshold. The
       latent removal is a rectified (ReLU-like) nonlinear function of cooling.
    3. Variable-air-volume (VAV) transport: convective coupling scales with the
       fan flow fraction m_dot, so the effective air-node conductance is a
       *product* of a control and a state difference (bilinear control term).
    """

    c_air: float = 0.6
    c_wall: float = 4.0
    r_air: float = 0.8
    r_wall: float = 2.0
    solar_gain: float = 0.05
    # window / natural ventilation
    window_gain: float = 0.6          # max vent conductance (kW/K) when fully open
    window_open_temp: float = 25.5    # indoor temp above which venting is useful
    window_sharpness: float = 2.0     # logistic steepness
    # humidity / latent（P0 修正：除湿判据从"制冷量阈值"改为"盘管温度 vs 露点"）
    c_hum: float = 2.0                # humidity capacitance (kg_da 等效，g/kg·kW⁻¹)
    latent_per_occ: float = 0.8       # latent gain per occupancy unit
    dehum_rate: float = 2.0           # latent removal rate ∝ (T_dew − T_coil) g/kg/h·K
    coil_approach: float = 0.5        # 盘管温度 ≈ 送风温度 + approach（K）
    w_out_default: float = 12.0       # default outdoor humidity (g/kg)
    # VAV
    vav_base: float = 0.3             # baseline transport when fan idle
    dt: float = 1.0 / 12.0

    state_labels: list[str] = field(default_factory=lambda: ["T_air", "T_wall", "W_air"])
    control_labels: list[str] = field(default_factory=lambda: ["Q_hvac", "m_dot"])
    external_labels: list[str] = field(default_factory=lambda: ["T_out", "solar", "occ", "price"])

    def _window_conductance(self, t_air: float, t_out: float) -> float:
        """Logistic buoyancy gate: opens when warm indoors and cooler outdoors."""
        # only useful to vent when it is cooler outside
        if t_out >= t_air:
            return 0.0
        gate = 1.0 / (1.0 + np.exp(-self.window_sharpness * (t_air - self.window_open_temp)))
        return self.window_gain * gate

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: float | None = None,
    ) -> SystemState:
        if dt is None:
            dt = self.dt
        T_air, T_wall, W_air = state.x
        Q_hvac = float(control.u[0])
        m_dot = float(control.u[1]) if control.u.size > 1 else 1.0
        m_dot = float(np.clip(m_dot, 0.0, 1.0))
        T_out = external.get("T_out", external.w[0])
        solar = external.get("solar", external.w[1] if external.w.size > 1 else 0.0)
        occ = external.get("occ", external.w[2] if external.w.size > 2 else 0.0)
        W_out = external.get("W_out", self.w_out_default)

        # (1) window natural ventilation — state-dependent conductance
        g_win = self._window_conductance(T_air, T_out)
        # (3) VAV: air-wall convective coupling scales with fan flow
        vav = self.vav_base + (1.0 - self.vav_base) * m_dot
        g_air = vav / self.r_air

        dT_air = (
            g_air * (T_wall - T_air)
            + (T_out - T_air) / self.r_wall
            + g_win * (T_out - T_air)
            + self.solar_gain * solar
            + occ
            + Q_hvac
        ) / self.c_air

        dT_wall = (
            g_air * (T_air - T_wall)
            + (T_out - T_wall) / self.r_wall
        ) / self.c_wall

        # (2) humidity: latent gain from occupancy + ventilation exchange, minus
        # dehumidification when the coil surface is BELOW the dew point of room
        # air（P0 修正：原 ReLU"制冷量>阈值即除湿"与物理不符——除湿取决于盘管
        # 表面温度是否低于露点，而露点由 W_air/大气压决定）
        cooling = max(0.0, -Q_hvac)
        t_dew = dew_point_gram(W_air)
        # 盘管温度 ≈ 送风温度 + approach；送风温降 ΔT = Q/(ṁ·cp)，ṁ = m_dot×设计风量
        flow_kgs = max(0.05, m_dot) * 0.4     # 设计风量 0.4 kg/s @ m_dot=1
        t_supply = T_air - cooling / (flow_kgs * 1.006)
        t_coil = t_supply + self.coil_approach
        dehum = self.dehum_rate * max(0.0, t_dew - t_coil) if cooling > 1e-9 else 0.0
        latent_in = self.latent_per_occ * occ + g_win * (W_out - W_air) * 0.1
        dW = (latent_in - dehum) / self.c_hum

        new_state = np.array([
            T_air + dT_air * dt,
            T_wall + dT_wall * dt,
            max(0.0, W_air + dW * dt),
        ])
        return SystemState(new_state, list(self.state_labels))

    def initial_state(self, t_air: float = 24.0, t_wall: float = 24.0,
                      w_air: float = 10.0) -> SystemState:
        return SystemState(np.array([t_air, t_wall, w_air]), list(self.state_labels))


@dataclass
class TwoZoneRCBuildingModel:
    """Two-zone RC building thermal model with a coupled partition wall.

    States (default):
        T_air_A, T_wall_A, T_air_B, T_wall_B, T_partition
    States (with_slab=True，地暖/辐射供暖场景):
        ..., T_slab_A, T_slab_B  —— 每区一块地板蓄热层
    Control:
        Q_hvac_A, Q_hvac_B
    External:
        T_out, solar_A, solar_B, occ_A, occ_B, price

    with_slab 语义：Q 不再直接注入空气，而是先加热本区蓄热层（大热容），再经
    r_slab 缓释到空气——捕捉地板辐射供暖的大热惯性（BOPTEST twozone_apartment
    试验台上默认结构失配的根源，见 docs/data/README.md §3）。
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
    with_slab: bool = False     # 启用每区地板蓄热层（7 状态）
    c_slab: float = 15.0        # 蓄热层热容 kWh/K（水泥层，远大于空气）
    r_slab: float = 0.5         # 蓄热层 → 空气热阻 K/kW

    state_labels: list[str] | None = None
    control_labels: list[str] = field(default_factory=lambda: ["Q_hvac_A", "Q_hvac_B"])
    external_labels: list[str] = field(default_factory=lambda: [
        "T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price",
    ])

    def __post_init__(self) -> None:
        if self.state_labels is None:
            self.state_labels = ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]
            if self.with_slab:
                self.state_labels += ["T_slab_A", "T_slab_B"]

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: float | None = None,
    ) -> SystemState:
        if dt is None:
            dt = self.dt
        T_air_A, T_wall_A, T_air_B, T_wall_B, T_partition = state.x[:5]
        Q_A, Q_B = control.u
        T_out = external.get("T_out", external.w[0])
        solar_A = external.solar(zone="A")
        solar_B = external.solar(zone="B")
        occ_A = external.occupancy(zone="A")
        occ_B = external.occupancy(zone="B")

        if self.with_slab:
            T_slab_A, T_slab_B = state.x[5], state.x[6]
            # 制冷/制热先注入蓄热层，再缓释进空气
            heat_A = (T_slab_A - T_air_A) / self.r_slab
            heat_B = (T_slab_B - T_air_B) / self.r_slab
        else:
            heat_A, heat_B = Q_A, Q_B

        dT_air_A = (
            (T_wall_A - T_air_A) / self.r_air
            + (T_out - T_air_A) / self.r_wall_a
            + (T_partition - T_air_A) / self.r_partition
            + self.solar_gain_a * solar_A
            + occ_A
            + heat_A
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
            + heat_B
        ) / self.c_air

        dT_wall_B = (
            (T_air_B - T_wall_B) / self.r_air
            + (T_out - T_wall_B) / self.r_wall_b
        ) / self.c_wall

        dT_partition = (
            (T_air_A - T_partition) / self.r_partition
            + (T_air_B - T_partition) / self.r_partition
        ) / self.c_partition

        deltas = [dT_air_A, dT_wall_A, dT_air_B, dT_wall_B, dT_partition]
        if self.with_slab:
            deltas += [(Q_A - heat_A) / self.c_slab, (Q_B - heat_B) / self.c_slab]

        new_state = state.x + np.array(deltas) * dt
        return SystemState(new_state, list(self.state_labels))

    def initial_state(self, t: float = 28.0) -> SystemState:
        return SystemState(np.full(len(self.state_labels), t), list(self.state_labels))


@dataclass
class OutdoorDependentCOP:
    """室外温度相关 COP 模型（P0 修正：常数 COP 系统性偏向预冷策略）。

    线性降额：室外每升高 1K，COP 下降 `degrade_per_K`（冷水机组典型 0.03~0.08/K，
    风冷热泵更高）。名义 COP 在 `t_out_nominal` 处定义。

    cop(T_out) = cop_nominal + degrade_per_K * (t_out_nominal - T_out)，下限 1.0。

    集成：HVACModel.cop_provider 可选注入；electrical_power() 优先用它，
    external 需含 T_out。缺省行为（None）保持常数 COP 完全向后兼容。
    """

    cop_nominal: float = 3.8
    t_out_nominal: float = 30.0
    degrade_per_K: float = 0.05
    cop_floor: float = 1.0

    def cop(self, t_out: float) -> float:
        return max(self.cop_floor,
                   self.cop_nominal + self.degrade_per_K * (self.t_out_nominal - float(t_out)))


@dataclass
class HVACModel:
    """HVAC system model: converts heat power to electric power and provides control bounds.

    Supports multiple independent units/zones; per-unit bounds available via
    ``unit_bounds`` (e.g. 区 A 一台 8kW、区 B 一台 6kW), falling back to the
    uniform ``q_min``/``q_max`` for all units.
    """

    q_min: float = -6.0   # 每台最大制冷热功率 kW
    q_max: float = 6.0    # 每台最大制热热功率 kW
    cop_heating: float = 3.2
    cop_cooling: float = 3.8
    part_load_penalty: float = 0.15
    n_units: int = 1
    control_labels: list[str] = field(default_factory=lambda: ["Q_hvac"])
    # 每台独立容量 [(q_min_j, q_max_j), ...]：覆盖统一 q_min/q_max。
    # bounds()/electrical_power()/各求解器均按台生效。
    unit_bounds: list[tuple] | None = None
    # 室外温度相关 COP（可选）：注入后 electrical_power 按 external 的 T_out 取 COP
    cop_provider: OutdoorDependentCOP | None = None

    def __post_init__(self) -> None:
        if len(self.control_labels) != self.n_units:
            self.control_labels = [f"Q_hvac_{i}" for i in range(self.n_units)]
        if self.unit_bounds is not None:
            if len(self.unit_bounds) != self.n_units:
                raise ValueError(
                    f"unit_bounds 长度 {len(self.unit_bounds)} != n_units {self.n_units}"
                )
            self.unit_bounds = [(float(lo), float(hi)) for lo, hi in self.unit_bounds]

    def _unit_cap(self, j: int, heating: bool) -> float:
        """第 j 台设备该方向的容量上限（部分负荷系数的额定基准）。

        q_min 为制冷额定（负值），q_max 为制热额定——两者可以不同。
        """
        if self.unit_bounds is not None:
            lo, hi = self.unit_bounds[j]
            return max(hi if heating else abs(lo), 1e-9)
        return max(self.q_max if heating else abs(self.q_min), 1e-9)

    def electrical_power(self, control: ControlInput, external: ExternalInput | None = None) -> float:
        # 工况 COP：注入 cop_provider 且 external 带 T_out 时按室外温度取（默认名义值）
        t_out = None
        if self.cop_provider is not None and external is not None:
            t_out = external.get("T_out")
        cop_c = (self.cop_provider.cop(t_out) if (self.cop_provider and t_out is not None)
                 else self.cop_cooling)
        total = 0.0
        for j, u in enumerate(control.u):
            q = float(u)
            heating = q >= 0
            cop = self.cop_heating if heating else cop_c
            abs_q = q if heating else -q
            load_ratio = min(abs_q / self._unit_cap(j, heating), 1.0)
            cop_eff = cop * (1.0 - self.part_load_penalty * (1.0 - load_ratio) ** 2)
            total += abs_q / max(cop_eff, 1e-6)
        return total

    def bounds(self) -> list[tuple[float, float]]:
        if self.unit_bounds is not None:
            return [(lo, hi) for lo, hi in self.unit_bounds]
        return [(self.q_min, self.q_max)] * self.n_units


class Simulator:
    """Digital-twin simulator: wraps model stepping and trajectory rollout."""

    def __init__(
        self,
        building: RCBuildingModel | None = None,
        hvac: HVACModel | None = None,
    ) -> None:
        self.building = building or RCBuildingModel()
        self.hvac = hvac or HVACModel()

    def step(
        self,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
        dt: float | None = None,
    ) -> SystemState:
        return self.building.step(state, control, external, dt)

    def rollout(
        self,
        initial_state: SystemState,
        controls: Sequence[ControlInput],
        externals: Sequence[ExternalInput],
        dt: float | None = None,
    ) -> list[SystemState]:
        states = [initial_state.copy()]
        state = initial_state
        for u, w in zip(controls, externals):
            state = self.step(state, u, w, dt)
            states.append(state.copy())
        return states
