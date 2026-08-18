"""CasADi + IPOPT 求解器：带硬约束的非线性 MPC。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

import numpy as np

from ..physics.models import RCBuildingModel, TwoZoneRCBuildingModel
from ..types import ControlInput, ExternalInput, SystemState, Trajectory
from .landscape import EnergyLandscape

try:
    import casadi as ca
    HAS_CASADI = True
except Exception:  # pragma: no cover
    ca = None
    HAS_CASADI = False


def _casadi_step(building, x, u, w, dt):
    """返回 CasADi 下一时刻状态表达式。"""
    if isinstance(building, TwoZoneRCBuildingModel):
        T_air_A, T_wall_A, T_air_B, T_wall_B, T_partition = x[0], x[1], x[2], x[3], x[4]
        Q_A, Q_B = u[0], u[1]
        T_out = w[0]
        solar_A = w[1]
        solar_B = w[2]
        occ_A = w[3]
        occ_B = w[4]

        dT_air_A = (
            (T_wall_A - T_air_A) / building.r_air
            + (T_out - T_air_A) / building.r_wall_a
            + (T_partition - T_air_A) / building.r_partition
            + building.solar_gain_a * solar_A
            + occ_A
            + Q_A
        ) / building.c_air
        dT_wall_A = (
            (T_air_A - T_wall_A) / building.r_air
            + (T_out - T_wall_A) / building.r_wall_a
        ) / building.c_wall
        dT_air_B = (
            (T_wall_B - T_air_B) / building.r_air
            + (T_out - T_air_B) / building.r_wall_b
            + (T_partition - T_air_B) / building.r_partition
            + building.solar_gain_b * solar_B
            + occ_B
            + Q_B
        ) / building.c_air
        dT_wall_B = (
            (T_air_B - T_wall_B) / building.r_air
            + (T_out - T_wall_B) / building.r_wall_b
        ) / building.c_wall
        dT_partition = (
            (T_air_A - T_partition) / building.r_partition
            + (T_air_B - T_partition) / building.r_partition
        ) / building.c_partition
        return x + dt * ca.vertcat(dT_air_A, dT_wall_A, dT_air_B, dT_wall_B, dT_partition)
    elif isinstance(building, RCBuildingModel):
        T_air, T_wall = x[0], x[1]
        Q = u[0]
        T_out = w[0]
        solar = w[1]
        occ = w[2]
        dT_air = (
            (T_wall - T_air) / building.r_air
            + (T_out - T_air) / building.r_wall
            + building.solar_gain * solar
            + occ
            + Q
        ) / building.c_air
        dT_wall = (
            (T_air - T_wall) / building.r_air
            + (T_out - T_wall) / building.r_wall
        ) / building.c_wall
        return x + dt * ca.vertcat(dT_air, dT_wall)
    else:
        raise TypeError(f"Unsupported building model: {type(building)}")


@dataclass
class CasadiGeodesicSolver:
    """使用 CasADi + IPOPT 的硬约束测地线求解器。"""

    simulator: object
    landscape: EnergyLandscape
    horizon: int = 12
    dt: Optional[float] = None
    ipopt_options: dict = None  # type: ignore[assignment]
    two_stage: bool = False
    use_kinetic: bool = False

    def __post_init__(self) -> None:
        if not HAS_CASADI:
            raise RuntimeError("CasADi is not installed")
        if self.dt is None:
            self.dt = getattr(self.simulator.building, "dt", 1.0 / 12.0)
        if self.ipopt_options is None:
            self.ipopt_options = {
                "print_time": 0,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 1000,
                "ipopt.tol": 1e-7,
                "ipopt.acceptable_tol": 1e-6,
                "ipopt.acceptable_iter": 100,
            }

    def solve(
        self,
        initial_state: SystemState,
        external_seq: Sequence[ExternalInput],
        prev_control: Optional[ControlInput] = None,
        initial_controls: Optional[Sequence[ControlInput]] = None,
    ) -> Trajectory:
        if len(external_seq) < self.horizon:
            raise ValueError("external sequence too short")

        # 两级求解：先用舒适优先 MPC 生成可行初始点
        if self.two_stage and self.landscape.weights.get("energy", 0.0) > 0.0:
            comfort_weights = dict(self.landscape.weights)
            comfort_weights["energy"] = 0.0
            comfort_landscape = replace(
                self.landscape,
                weights=comfort_weights,
            )
            comfort_solver = CasadiGeodesicSolver(
                simulator=self.simulator,
                landscape=comfort_landscape,
                horizon=self.horizon,
                dt=self.dt,
                ipopt_options=self.ipopt_options,
                two_stage=False,
            )
            comfort_traj = comfort_solver.solve(
                initial_state,
                external_seq,
                prev_control=prev_control,
                initial_controls=None,
            )
            if comfort_traj.controls and len(comfort_traj.controls) == self.horizon:
                initial_controls = comfort_traj.controls

        building = self.simulator.building
        hvac = self.simulator.hvac
        n_x = initial_state.dim
        n_u = len(hvac.bounds())
        bounds = hvac.bounds()
        u_min = np.array([b[0] for b in bounds])
        u_max = np.array([b[1] for b in bounds])

        opti = ca.Opti()
        U = opti.variable(n_u, self.horizon)
        opti.set_initial(U, np.zeros((n_u, self.horizon)))
        if initial_controls is not None and len(initial_controls) == self.horizon:
            init_mat = np.column_stack([c.u for c in initial_controls])
            opti.set_initial(U, init_mat)

        for i in range(n_u):
            opti.subject_to(opti.bounded(u_min[i], U[i, :], u_max[i]))

        X0 = opti.parameter(n_x)
        W = opti.parameter(external_seq[0].dim, self.horizon)
        Uprev = opti.parameter(n_u)
        opti.set_value(X0, initial_state.x)
        opti.set_value(W, np.column_stack([w.w for w in external_seq[:self.horizon]]))
        if prev_control is not None:
            opti.set_value(Uprev, prev_control.u)
        else:
            opti.set_value(Uprev, np.zeros(n_u))

        X = [X0]
        total_cost = 0.0
        weights = self.landscape.weights
        setpoints = self.landscape.setpoints
        comfort_min = self.landscape.comfort_min
        comfort_max = self.landscape.comfort_max
        comfort_band = self.landscape.comfort_band
        below_penalty = self.landscape.below_comfort_penalty
        peak_threshold = self.landscape.peak_price_threshold
        peak_penalty = self.landscape.peak_energy_penalty

        # 根据标签识别空气温度状态索引
        air_indices = [i for i, lab in enumerate(self.landscape.manifold.labels) if lab.startswith("T_air")]
        if not air_indices:
            air_indices = [0]

        metric_diag = None
        if self.use_kinetic:
            dummy_state = SystemState(np.zeros(n_x), list(self.landscape.manifold.labels))
            metric_diag = np.diag(self.landscape.metric(dummy_state))

        def comfort_expr(value: ca.MX, setpoint: float, scale: float) -> ca.MX:
            if comfort_min is not None and comfort_max is not None:
                above = ca.fmax(0.0, value - comfort_max)
                below = ca.fmax(0.0, comfort_min - value) * below_penalty
                excess = above + below
            else:
                deviation = ca.fabs(value - setpoint)
                excess = ca.fmax(0.0, deviation - comfort_band)
            return (excess / scale) ** 2

        # 电价索引：优先按 price 标签定位（两区域 6 维时 price 在最后，w[3] 是 occ_A）
        price_idx = 3
        if external_seq and external_seq[0].labels:
            labs = list(external_seq[0].labels)
            if "price" in labs:
                price_idx = labs.index("price")
            elif len(labs) > 5:
                price_idx = 5

        step_costs: list = []
        for k in range(self.horizon):
            u = U[:, k]
            w = W[:, k]
            prev_u = Uprev if k == 0 else U[:, k - 1]
            xk = X[k]
            xk_next = _casadi_step(building, xk, u, w, self.dt)
            X.append(xk_next)
            step_cost = 0.0

            # 动能项：显式度量张量下的离散测地线作用量
            if self.use_kinetic and metric_diag is not None:
                delta = xk_next - xk
                kinetic = 0.5 * ca.dot(delta, ca.MX(metric_diag) * delta) / (self.dt ** 2)
                step_cost += kinetic

            # 舒适代价
            comfort = 0.0
            for i, lab in enumerate(self.landscape.manifold.labels):
                if lab in setpoints:
                    scale = self.landscape.manifold.scale.get(lab, 1.0)
                    comfort += comfort_expr(xk[i], setpoints[lab], scale)
            step_cost += weights.get("comfort", 1.0) * comfort

            # 电费（与 scipy 后端一致：部分负荷 COP）
            elec = 0.0
            q_nom = max(abs(hvac.q_min), abs(hvac.q_max), 1e-9)
            for j in range(n_u):
                q = u[j]
                cop = ca.if_else(q >= 0, hvac.cop_heating, hvac.cop_cooling)
                load = ca.fabs(q) / q_nom
                cop_eff = cop * (1.0 - hvac.part_load_penalty * (1.0 - load) ** 2)
                elec += ca.fabs(q) / ca.fmax(cop_eff, 1e-6)
            price = w[price_idx] if w.numel() > price_idx else 1.0
            energy_price = price * ca.if_else(price > peak_threshold, peak_penalty, 1.0)
            step_cost += weights.get("energy", 0.5) * energy_price * elec

            smooth = 0.0
            for j in range(n_u):
                smooth += (u[j] - prev_u[j]) ** 2
            step_cost += weights.get("smooth", 0.2) * smooth

            step_costs.append(step_cost)
            total_cost += step_cost

            # 硬舒适约束：每个空气温度状态（仅当上下界已配置时施加，避免 None 约束崩溃）
            if comfort_max is not None:
                for idx in air_indices:
                    opti.subject_to(xk_next[idx] <= comfort_max)
            if comfort_min is not None:
                for idx in air_indices:
                    opti.subject_to(xk_next[idx] >= comfort_min)

        # 终端舒适代价
        terminal = 0.0
        for i, lab in enumerate(self.landscape.manifold.labels):
            if lab in setpoints:
                scale = self.landscape.manifold.scale.get(lab, 1.0)
                terminal += comfort_expr(X[-1][i], setpoints[lab], scale)
        terminal_cost = weights.get("comfort", 1.0) * terminal
        total_cost += terminal_cost

        opti.minimize(total_cost)
        opti.solver("ipopt", self.ipopt_options)

        try:
            sol = opti.solve()
            controls = [
                ControlInput(np.array(sol.value(U[:, k])).flatten(),
                             list(getattr(hvac, "control_labels", [f"u{i}" for i in range(n_u)])))
                for k in range(self.horizon)
            ]
            states = [
                SystemState(np.array(sol.value(X[k])).flatten(),
                            list(self.landscape.manifold.labels))
                for k in range(self.horizon + 1)
            ]
            costs = [float(sol.value(c)) for c in step_costs] + [float(sol.value(terminal_cost))]
            total_value = float(sol.value(total_cost))
            success = True
            msg = ""
        except Exception as exc:  # noqa: BLE001
            controls = []
            states = []
            costs = []
            total_value = 0.0
            success = False
            msg = str(exc)
            try:
                controls = [
                    ControlInput(np.array(opti.debug.value(U[:, k])).flatten(),
                                 list(getattr(hvac, "control_labels", [f"u{i}" for i in range(n_u)])))
                    for k in range(self.horizon)
                ]
                states = [
                    SystemState(np.array(opti.debug.value(X[k])).flatten(),
                                list(self.landscape.manifold.labels))
                    for k in range(self.horizon + 1)
                ]
            except Exception:  # noqa: BLE001
                pass

        return Trajectory(
            controls=controls,
            states=states,
            costs=costs,
            total_cost=total_value,
            success=success,
            message=msg,
        )
