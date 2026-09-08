"""Geodesic solver: find finite-horizon optimal control paths on the energy landscape."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..physics.models import Simulator
from ..types import ControlInput, ExternalInput, SystemState, Trajectory
from .casadi_solver import CasadiGeodesicSolver
from .landscape import EnergyLandscape
from .riemannian import (
    christoffel_symbols,
    christoffel_symbols_analytic,
    christoffel_symbols_autodiff,
    geodesic_step,
)


@dataclass
class GeodesicSolver:
    """Unified solving interface; internally replaceable with iLQR, SQP, interior-point, etc.

    Riemannian switch semantics (four *orthogonal* mechanisms)
    ----------------------------------------------------------
    These four flags were historically muddled; here is the authoritative spec.
    They act on different parts of the discrete action  A = ∫(½żᵀg(z)ż + E)dt:

    - ``use_kinetic`` — adds the ACTION KINETIC TERM ½żᵀg(z)ż to the objective.
      This is a *state-velocity* (trajectory-smoothness) regularizer, evaluated
      at the current state's metric g(z). PROVEN to improve temperature-ramp /
      comfort-stability KPIs (see docs/RIEMANNIAN_ABLATION.md §7). This is the
      recommended, product-facing Riemannian mechanism. Strength = kinetic_weight.

    - ``use_riemannian`` — applies a CHRISTOFFEL CONNECTION CORRECTION to the
      *dynamics step* itself (`_riemannian_corrected_step`): the propagated state
      is nudged by −½Γ^k_ij v^i v^j dt². It changes the model rollout, not the
      cost. Strength = riemannian_strength. On near-linear building RC this is a
      near-zero perturbation (Christoffel ≈ 0) — keep OFF unless the metric is
      strongly curved. Independent of use_kinetic.

    - ``use_riemannian_control`` — adds a CONTROL-EFFORT PENALTY measured in the
      metric norm ½ (Δz − b·u)ᵀ g (Δz − b·u)/dt², penalizing state moves that the
      control cannot explain. Strength = riemannian_control_weight. This is a
      *control-tracking* term, distinct from the kinetic (state-smoothness) term.

    - ``geodesic_penalty_weight`` — penalizes the propagated state's deviation
      from the exact one-step GEODESIC (soft "stay-on-geodesic" constraint). Only
      active when use_riemannian is also on. Strength = geodesic_penalty_weight.

    Recommended presets:
      * weighted-MPC baseline : all four OFF (use_kinetic may stay on with
        kinetic_weight=0 → degenerate to plain MPC).
      * smoothness mode       : use_kinetic=True (kinetic_weight>0), others OFF.
      * full Riemannian       : use_kinetic + use_riemannian (+ optional
        geodesic_penalty_weight) with a strongly state-dependent/coupled metric.
    """

    simulator: Simulator
    landscape: EnergyLandscape
    horizon: int = 12
    dt: float | None = None
    options: dict = field(default_factory=lambda: {"maxiter": 200, "ftol": 1e-8})
    enforce_comfort: bool = False
    comfort_min: float | None = None
    comfort_max: float | None = None
    # 每区独立舒适带 label -> (lo, hi)：硬约束逐 T_air* 状态生效，覆盖标量边界
    zone_comfort_bounds: dict[str, tuple] = field(default_factory=dict)
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
        self._validate_riemannian_switches()

    def _validate_riemannian_switches(self) -> None:
        """Enforce the documented switch spec; fail loud on incoherent combos."""
        # geodesic penalty is meaningless without the Christoffel-corrected step
        if self.geodesic_penalty_weight > 0.0 and not self.use_riemannian:
            raise ValueError(
                "geodesic_penalty_weight>0 requires use_riemannian=True "
                "(the penalty measures deviation from the corrected geodesic step)."
            )
        # negative strengths are never valid
        for name, val in (
            ("riemannian_strength", self.riemannian_strength),
            ("riemannian_control_weight", self.riemannian_control_weight),
            ("geodesic_penalty_weight", self.geodesic_penalty_weight),
        ):
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0, got {val}")

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
        prev_control: ControlInput | None = None,
        initial_controls: Sequence[ControlInput] | None = None,
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

        def unpack(z: np.ndarray) -> list[ControlInput]:
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
                    total += self.landscape.kinetic_term(next_state.x - state.x, self.dt, state)
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

        # 硬舒适约束：逐 T_air* 状态生效。每区边界优先取 zone_comfort_bounds，
        # 缺省回退标量 comfort_min/comfort_max；两者都缺则不加硬约束。
        # 兼容旧约定：无 T_air* 标签的模型（如数据中心 T_aisle）硬约束作用于第一个状态。
        has_air_label = any(lab.startswith("T_air") for lab in initial_state.labels)
        constrained_air: list[tuple] = []
        for i, lab in enumerate(initial_state.labels):
            is_primary = lab.startswith("T_air") or (not has_air_label and i == 0)
            if not is_primary:
                continue
            zone = self.zone_comfort_bounds.get(lab)
            if zone is not None:
                constrained_air.append((i, float(zone[0]), float(zone[1])))
            elif self.comfort_min is not None and self.comfort_max is not None:
                constrained_air.append((i, self.comfort_min, self.comfort_max))

        if self.enforce_comfort and constrained_air:
            def comfort_constraints(z: np.ndarray) -> np.ndarray:
                controls = unpack(z)
                state = initial_state.copy()
                values: list[float] = []
                for k in range(self.horizon):
                    state = self._riemannian_corrected_step(state, controls[k], external_seq[k], self.dt)
                    for idx, lo, hi in constrained_air:
                        values.append(hi - state.x[idx])
                        values.append(state.x[idx] - lo)
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
        costs: list[float] = []
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

        # nit==0 仅表示"未迭代"（如初值即最优），不算求解成功——避免掩盖停滞
        success = bool(result.success)
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
    dt: float | None = None,
    prev_control: ControlInput | None = None,
    initial_controls: Sequence[ControlInput] | None = None,
) -> Trajectory:
    solver = GeodesicSolver(simulator=simulator, landscape=landscape, horizon=horizon, dt=dt)
    return solver.solve(initial_state, external_seq, prev_control, initial_controls)
