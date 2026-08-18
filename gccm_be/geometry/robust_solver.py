"""轻量鲁棒 MPC：多模型场景下共享控制序列，约束对所有场景满足。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.optimize import minimize

from ..physics.models import RCBuildingModel, HVACModel, Simulator
from ..types import ControlInput, ExternalInput, SystemState, Trajectory
from .landscape import EnergyLandscape


@dataclass
class RobustGeodesicSolver:
    """面向单区域 RC 的鲁棒测地线求解器原型。

    对多个可能的建筑模型同时 rollout，
    控制序列共享，舒适约束要求所有场景都满足。
    """

    nominal_sim: Simulator
    landscape: EnergyLandscape
    scenario_sims: List[Simulator]
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

        if initial_controls is not None and len(initial_controls) == self.horizon:
            x0 = np.concatenate([c.u for c in initial_controls])
        else:
            x0 = np.zeros(self.horizon * n_u)

        def unpack(z):
            return [ControlInput(z[i*n_u:(i+1)*n_u], ["Q_hvac"]) for i in range(self.horizon)]

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
                    vals.append(self.comfort_max - state.x[0])
                    vals.append(state.x[0] - self.comfort_min)
            return np.array(vals)

        cons = [{"type": "ineq", "fun": constraints}]
        res = minimize(obj, x0, method="SLSQP", bounds=bounds * self.horizon,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-6})
        controls = unpack(res.x)
        states = [initial_state.copy()]
        state = initial_state.copy()
        for k in range(self.horizon):
            state = self.nominal_sim.step(state, controls[k], external_seq[k], self.dt)
            states.append(state.copy())
        return Trajectory(controls=controls, states=states, success=bool(res.success), message=str(res.message))
