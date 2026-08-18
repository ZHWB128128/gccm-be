"""Geodesic solver: find finite-horizon optimal control paths on the energy landscape."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from ..physics.models import Simulator
from ..types import ControlInput, ExternalInput, SystemState, Trajectory
from .landscape import EnergyLandscape
from .casadi_solver import CasadiGeodesicSolver
from .riemannian import christoffel_symbols, christoffel_symbols_analytic, christoffel_symbols_autodiff, geodesic_step


@dataclass
class GeodesicSolver:
    """Unified solving interface; internally replaceable with iLQR, SQP, interior-point, etc."""

    simulator: Simulator
    landscape: EnergyLandscape
    horizon: int = 12
    dt: Optional[float] = None
    options: dict = field(default_factory=lambda: {"maxiter": 200, "ftol": 1e-8})
    enforce_comfort: bool = False
    comfort_min: Optional[float] = None
    comfort_max: Optional[float] = None
    constraint_options: dict = field(default_factory=lambda: {"maxiter": 30, "ftol": 1e-5})
    use_casadi: bool = False
    two_stage: bool = False
    use_kinetic: bool = False
    use_riemannian: bool = False
    use_riemannian_control: bool = False
    riemannian_control_weight: float = 1.0
    use_autodiff_christoffel: bool = False
    use_analytic_christoffel: bool = True
    riemannian_strength: float = 1.0
    geodesic_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        # 统一时间步长：未显式指定时继承建筑模型默认步长，避免动力学(1/12h)与
        # 动能/黎曼修正(1.0)使用不同时间基准
        if self.dt is None:
            self.dt = getattr(self.simulator.building, "dt", 1.0 / 12.0)

    def _riemannian_corrected_step(self, state: SystemState, control: ControlInput,
                                   external: ExternalInput, dt: float) -> SystemState:
        """在物理步进基础上加入 Christoffel 联络修正。"""
        physical_next = self.simulator.step(state, control, external, dt)
        if not self.use_riemannian:
            return physical_next
        dt = dt or self.dt
        v = (physical_next.x - state.x) / dt
        if self.use_analytic_christoffel:
            try:
                Gamma = christoffel_symbols_analytic(self.landscape, state)
            except Exception:
                Gamma = christoffel_symbols(self.landscape.metric, state)
        elif self.use_autodiff_christoffel:
            try:
                if hasattr(self.landscape, "metric_casadi"):
                    Gamma = christoffel_symbols_autodiff(
                        lambda s: self.landscape.metric_casadi(s), state
                    )
                else:
                    Gamma = christoffel_symbols_autodiff(self.landscape.metric, state)
            except Exception:
                Gamma = christoffel_symbols(self.landscape.metric, state)
        else:
            Gamma = christoffel_symbols(self.landscape.metric, state)
        n = state.dim
        corr = np.zeros(n)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    corr[k] += -0.5 * self.riemannian_strength * Gamma[k, i, j] * v[i] * v[j] * dt * dt
        return SystemState(physical_next.x + corr, list(state.labels))

    def solve(
        self,
        initial_state: SystemState,
        external_seq: Sequence[ExternalInput],
        prev_control: Optional[ControlInput] = None,
        initial_controls: Optional[Sequence[ControlInput]] = None,
    ) -> Trajectory:
        if len(external_seq) < self.horizon:
            raise ValueError(f"外部输入序列长度 {len(external_seq)} 小于预测时域 {self.horizon}")

        if self.use_casadi:
            casadi_solver = CasadiGeodesicSolver(
                simulator=self.simulator,
                landscape=self.landscape,
                horizon=self.horizon,
                dt=self.dt,
                ipopt_options={
                    "print_time": 0,
                    "ipopt.print_level": 0,
                    "ipopt.max_iter": 1000,
                    "ipopt.tol": 1e-7,
                    "ipopt.acceptable_tol": 1e-6,
                    "ipopt.acceptable_iter": 100,
                },
                two_stage=self.two_stage,
                use_kinetic=self.use_kinetic,
            )
            return casadi_solver.solve(
                initial_state,
                external_seq,
                prev_control=prev_control,
                initial_controls=initial_controls,
            )

        bounds = self.simulator.hvac.bounds()
        n_u = len(bounds)
        if n_u == 0:
            n_u = 1
            bounds = [(-np.inf, np.inf)]
        hvac_labels = getattr(self.simulator.hvac, "control_labels", None)
        if hvac_labels and len(hvac_labels) == n_u:
            ctrl_labels = list(hvac_labels)
        else:
            ctrl_labels = [f"u{i}" for i in range(n_u)]

        if initial_controls is not None and len(initial_controls) == self.horizon:
            x0 = np.concatenate([c.u for c in initial_controls])
        else:
            x0 = np.zeros(self.horizon * n_u)

        def unpack(z: np.ndarray) -> List[ControlInput]:
            return [
                ControlInput(z[i * n_u:(i + 1) * n_u], list(ctrl_labels))
                for i in range(self.horizon)
            ]

        def objective(z: np.ndarray) -> float:
            controls = unpack(z)
            state = initial_state.copy()
            total = 0.0
            for k in range(self.horizon):
                u = controls[k]
                w = external_seq[k]
                prev = prev_control if k == 0 else controls[k - 1]
                total += self.landscape.running_cost(state, u, w, prev)
                next_state = self._riemannian_corrected_step(state, u, w, self.dt)
                if self.use_kinetic:
                    total += self.landscape.kinetic_term(next_state.x - state.x, self.dt)
                if self.use_riemannian_control:
                    dt = self.dt
                    delta = next_state.x - state.x
                    # 控制对 T_air 的近似贡献
                    c_air = getattr(self.simulator.building, "c_air", 1.0)
                    b = np.zeros(state.dim)
                    b[0] = dt / c_air
                    deviation = delta - b * float(u.u[0])
                    total += self.riemannian_control_weight * 0.5 * float(deviation.T @ self.landscape.metric(state) @ deviation) / (dt * dt)
                if self.use_riemannian and self.geodesic_penalty_weight > 0.0:
                    dt = self.dt
                    v = (next_state.x - state.x) / dt
                    geo_state, _ = geodesic_step(state, v, self.landscape.metric, dt)
                    total += self.geodesic_penalty_weight * float(np.sum((next_state.x - geo_state.x) ** 2))
                state = next_state
            total += self.landscape.terminal_cost(state)
            return float(total)

        flat_bounds = bounds * self.horizon
        if self.enforce_comfort and self.comfort_min is not None and self.comfort_max is not None:
            def comfort_constraints(z: np.ndarray) -> np.ndarray:
                controls = unpack(z)
                state = initial_state.copy()
                values: List[float] = []
                for k in range(self.horizon):
                    state = self._riemannian_corrected_step(state, controls[k], external_seq[k], self.dt)
                    values.append(self.comfort_max - state.x[0])
                    values.append(state.x[0] - self.comfort_min)
                return np.array(values)

            constraints = [{"type": "ineq", "fun": comfort_constraints}]
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=flat_bounds,
                constraints=constraints,
                options=self.constraint_options,
            )
        else:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=flat_bounds,
                options=self.options,
            )

        controls = unpack(result.x)
        if self.use_riemannian:
            states = [initial_state.copy()]
            st = initial_state
            for k in range(self.horizon):
                st = self._riemannian_corrected_step(st, controls[k], external_seq[k], self.dt)
                states.append(st.copy())
        else:
            states = self.simulator.rollout(initial_state, controls, external_seq[: self.horizon], self.dt)
        costs: List[float] = []
        total = 0.0
        state = initial_state.copy()
        for k in range(self.horizon):
            prev = prev_control if k == 0 else controls[k - 1]
            c = self.landscape.running_cost(state, controls[k], external_seq[k], prev)
            costs.append(float(c))
            total += c
            state = self._riemannian_corrected_step(state, controls[k], external_seq[k], self.dt)
        terminal = self.landscape.terminal_cost(state)
        costs.append(float(terminal))
        total += terminal

        success = bool(result.success or result.nit == 0)
        return Trajectory(
            controls=controls,
            states=states,
            costs=costs,
            total_cost=float(total),
            success=success,
            message=result.message if hasattr(result, "message") else "",
        )


def solve_geodesic(
    simulator: Simulator,
    landscape: EnergyLandscape,
    initial_state: SystemState,
    external_seq: Sequence[ExternalInput],
    horizon: int = 12,
    dt: Optional[float] = None,
    prev_control: Optional[ControlInput] = None,
    initial_controls: Optional[Sequence[ControlInput]] = None,
) -> Trajectory:
    solver = GeodesicSolver(simulator=simulator, landscape=landscape, horizon=horizon, dt=dt)
    return solver.solve(initial_state, external_seq, prev_control, initial_controls)
