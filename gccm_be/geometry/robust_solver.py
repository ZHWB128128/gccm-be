"""Lightweight robust MPC: shared control sequence across model scenarios, constraints satisfied for all scenarios."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..physics.models import Simulator
from ..types import ControlInput, ExternalInput, SystemState, Trajectory
from .landscape import EnergyLandscape


@dataclass
class RobustGeodesicSolver:
    """Robust geodesic solver prototype for single-zone RC.

    Rolls out multiple possible building models simultaneously; the control sequence is shared and comfort constraints must hold for all scenarios.
    """

    nominal_sim: Simulator
    landscape: EnergyLandscape
    scenario_sims: list[Simulator]
    horizon: int = 24
    dt: float = 0.25
    comfort_min: float = 25.0
    comfort_max: float = 27.0

    def solve(
        self,
        initial_state: SystemState,
        external_seq: Sequence[ExternalInput],
        prev_control: ControlInput = None,  # type: ignore[assignment]
        initial_controls: Sequence[ControlInput] = None,  # type: ignore[assignment]
    ) -> Trajectory:
        n_u = len(self.nominal_sim.hvac.bounds())
        bounds = self.nominal_sim.hvac.bounds()
        sims = [self.nominal_sim] + self.scenario_sims

        # 逐区舒适边界：每区独立覆盖（landscape.zone_comfort_bounds），缺省回退标量。
        # 兼容旧约定：无 T_air* 标签的模型（如数据中心 T_aisle）硬约束作用于第一个状态。
        has_air_label = any(lab.startswith("T_air") for lab in initial_state.labels)
        air_cons: list[tuple] = []
        for i, lab in enumerate(initial_state.labels):
            is_primary = lab.startswith("T_air") or (not has_air_label and i == 0)
            if not is_primary:
                continue
            zone = self.landscape.zone_comfort_bounds.get(lab)
            if zone is not None:
                air_cons.append((i, float(zone[0]), float(zone[1])))
            else:
                air_cons.append((i, self.comfort_min, self.comfort_max))
        if not air_cons:
            air_cons = [(0, self.comfort_min, self.comfort_max)]

        if initial_controls is not None and len(initial_controls) == self.horizon:
            x0 = np.concatenate([c.u for c in initial_controls])
        else:
            x0 = np.zeros(self.horizon * n_u)

        def unpack(z):
            ctrl_labels = list(getattr(self.nominal_sim.hvac, "control_labels", None)
                               or [f"u{i}" for i in range(n_u)])
            if len(ctrl_labels) != n_u:
                ctrl_labels = [f"u{i}" for i in range(n_u)]
            return [ControlInput(z[i * n_u:(i + 1) * n_u], list(ctrl_labels))
                    for i in range(self.horizon)]

        def obj(z):
            controls = unpack(z)
            total = 0.0
            for sim in sims:
                state = initial_state.copy()
                for k in range(self.horizon):
                    total += self.landscape.running_cost(state, controls[k], external_seq[k], prev_control if k == 0 else controls[k-1])
                    state = sim.step(state, controls[k], external_seq[k], self.dt)
                total += self.landscape.terminal_cost(state)
            return total / len(sims)

        def constraints(z):
            controls = unpack(z)
            vals = []
            for sim in sims:
                state = initial_state.copy()
                for k in range(self.horizon):
                    state = sim.step(state, controls[k], external_seq[k], self.dt)
                    for idx, lo, hi in air_cons:
                        vals.append(hi - state.x[idx])
                        vals.append(state.x[idx] - lo)
            return np.array(vals)

        # 性能注记：obj 与 constraints 各自独立 rollout(N 场景×H 步)，
        # SLSQP 数值梯度下总代价 O(N·H²·n_u)——horizon 48×双场景即分钟级。
        # 中期解法：CasADi 鲁棒路径（解析导数）已在路线图；scipy 后端保留为原型。
        cons = [{"type": "ineq", "fun": constraints}]
        res = minimize(obj, x0, method="SLSQP", bounds=bounds * self.horizon,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-6})
        controls = unpack(res.x)
        states = [initial_state.copy()]
        state = initial_state.copy()
        for k in range(self.horizon):
            state = self.nominal_sim.step(state, controls[k], external_seq[k], self.dt)
            states.append(state.copy())
        # 名义场景回填逐步代价（约定与 GeodesicSolver 一致：costs 为逐步代价，
        # 末项为终端代价；闭环累计电费依赖 total_cost）
        costs: list[float] = []
        total = 0.0
        state = initial_state.copy()
        for k in range(self.horizon):
            c = self.landscape.running_cost(
                state, controls[k], external_seq[k],
                prev_control if k == 0 else controls[k - 1],
            )
            costs.append(float(c))
            total += c
            state = self.nominal_sim.step(state, controls[k], external_seq[k], self.dt)
        terminal = self.landscape.terminal_cost(state)
        costs.append(float(terminal))
        total += terminal
        return Trajectory(controls=controls, states=states, costs=costs,
                          total_cost=float(total),
                          success=bool(res.success), message=str(res.message))
