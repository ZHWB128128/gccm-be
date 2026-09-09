"""CasADi + IPOPT robust MPC: shared control across multiple model scenarios, hard constraints satisfied for all scenarios."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..physics.models import Simulator
from ..types import ControlInput, ExternalInput, SystemState, Trajectory
from .landscape import EnergyLandscape

try:
    import casadi as ca
    HAS_CASADI = True
except Exception:  # pragma: no cover
    ca = None
    HAS_CASADI = False

from .casadi_solver import _casadi_step


@dataclass
class CasadiRobustGeodesicSolver:
    """CasADi + IPOPT robust geodesic solver (single-zone RC, multiple scenarios)."""

    nominal_sim: Simulator
    landscape: EnergyLandscape
    scenario_sims: list[Simulator]
    horizon: int = 12
    dt: float = 0.25
    comfort_min: float = 25.0
    comfort_max: float = 27.0
    robust_penalty: float = 1000.0
    enforce_lower_bound: bool = True
    heating_cost_factor: float = 1.0
    ipopt_options: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not HAS_CASADI:
            raise RuntimeError("CasADi is not installed")
        if self.ipopt_options is None:
            self.ipopt_options = {
                "print_time": 0,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 2000,
                "ipopt.tol": 1e-8,
                "ipopt.acceptable_tol": 1e-7,
            }

    def solve(
        self,
        initial_state: SystemState,
        external_seq: Sequence[ExternalInput],
        prev_control: ControlInput = None,  # type: ignore[assignment]
        initial_controls: Sequence[ControlInput] = None,  # type: ignore[assignment]
    ) -> Trajectory:
        sims = [self.nominal_sim] + list(self.scenario_sims)
        n_x = initial_state.dim
        n_u = len(self.nominal_sim.hvac.bounds())
        bounds = self.nominal_sim.hvac.bounds()
        u_min = np.array([b[0] for b in bounds])
        u_max = np.array([b[1] for b in bounds])

        opti = ca.Opti()
        U = opti.variable(n_u, self.horizon)
        if initial_controls is not None and len(initial_controls) == self.horizon:
            opti.set_initial(U, np.column_stack([c.u for c in initial_controls]))
        else:
            opti.set_initial(U, np.zeros((n_u, self.horizon)))
        for i in range(n_u):
            opti.subject_to(opti.bounded(u_min[i], U[i, :], u_max[i]))

        W = opti.parameter(external_seq[0].dim, self.horizon)
        opti.set_value(W, np.column_stack([w.w for w in external_seq[:self.horizon]]))
        X0 = opti.parameter(n_x)
        opti.set_value(X0, initial_state.x)
        Uprev = opti.parameter(n_u)
        opti.set_value(Uprev, prev_control.u if prev_control is not None else np.zeros(n_u))

        weights = self.landscape.weights
        setpoints = self.landscape.setpoints
        below_pen = getattr(self.landscape, "below_comfort_penalty", 1.0)
        total_cost = 0.0

        # 电价索引：优先按 price 标签定位（两区域 6 维时 price 在最后，w[3] 是 occ_A）
        price_idx = 3
        if external_seq and external_seq[0].labels:
            labs = list(external_seq[0].labels)
            if "price" in labs:
                price_idx = labs.index("price")
            elif len(labs) > 5:
                price_idx = 5

        # 逐区舒适边界：每区独立覆盖（landscape.zone_comfort_bounds），缺省回退标量。
        # 兼容旧约定：无 T_air* 标签的模型（如数据中心 T_aisle）硬约束作用于第一个状态。
        _labels = list(self.landscape.manifold.labels)
        _has_air = any(lab.startswith("T_air") for lab in _labels)
        air_cons: list = []
        for i, lab in enumerate(_labels):
            is_primary = lab.startswith("T_air") or (not _has_air and i == 0)
            if not is_primary:
                continue
            zone = self.landscape.zone_comfort_bounds.get(lab)
            if zone is not None:
                air_cons.append((i, float(zone[0]), float(zone[1])))
            else:
                air_cons.append((i, self.comfort_min, self.comfort_max))
        if not air_cons:
            air_cons = [(0, self.comfort_min, self.comfort_max)]

        all_X = []
        for sim in sims:
            building = sim.building
            X = [X0]
            for k in range(self.horizon):
                u = U[:, k]
                w = W[:, k]
                prev_u = Uprev if k == 0 else U[:, k - 1]
                x_next = _casadi_step(building, X[k], u, w, self.dt)
                X.append(x_next)
                for ai, lo, hi in air_cons:
                    opti.subject_to(x_next[ai] <= hi)
                    if self.enforce_lower_bound:
                        opti.subject_to(x_next[ai] >= lo)
                    # 软惩罚：即使硬约束未完全满足，也尽量少违规
                    excess = ca.fmax(0.0, x_next[ai] - hi)
                    if self.enforce_lower_bound:
                        excess += ca.fmax(0.0, lo - x_next[ai])
                    total_cost += self.robust_penalty * excess * excess

                # 舒适代价（与名义求解器对齐：每区 bounds_for 优先，软带回退）
                comfort = 0.0
                for i, lab in enumerate(self.landscape.manifold.labels):
                    if lab in setpoints:
                        scale = self.landscape.manifold.scale.get(lab, 1.0)
                        bnds = self.landscape.bounds_for(lab)
                        if bnds is not None:
                            lo_b, hi_b = bnds
                            excess = (ca.fmax(0.0, X[k][i] - hi_b)
                                      + ca.fmax(0.0, lo_b - X[k][i]) * below_pen)
                        else:
                            dev = ca.fabs(X[k][i] - setpoints[lab])
                            excess = ca.fmax(0.0, dev - self.landscape.comfort_band)
                        comfort += (excess / scale) ** 2
                # 电费（与名义求解器对齐：部分负荷 COP 惩罚）
                elec = 0.0
                unit_caps = [max(abs(lo), abs(hi), 1e-9) for lo, hi in sim.hvac.bounds()]
                for j in range(n_u):
                    q = u[j]
                    cop = ca.if_else(q >= 0, sim.hvac.cop_heating, sim.hvac.cop_cooling)
                    load = ca.fabs(q) / (unit_caps[j] if j < len(unit_caps) else 1e-9)
                    cop_eff = cop * (1.0 - sim.hvac.part_load_penalty * (1.0 - load) ** 2)
                    factor = ca.if_else(q >= 0, self.heating_cost_factor, 1.0)
                    elec += factor * ca.fabs(q) / ca.fmax(cop_eff, 1e-6)
                price = w[price_idx] if w.numel() > price_idx else 1.0
                smooth = ca.sumsqr(u - prev_u)
                total_cost += (
                    weights.get("comfort", 1.0) * comfort
                    + weights.get("energy", 0.5) * price * elec
                    + weights.get("smooth", 0.2) * smooth
                )
            all_X.append(X)

        opti.minimize(total_cost / len(sims))
        opti.solver("ipopt", self.ipopt_options)

        try:
            sol = opti.solve()
            controls = [
                ControlInput(np.array(sol.value(U[:, k])).flatten(),
                             list(getattr(self.nominal_sim.hvac, "control_labels", [f"u{i}" for i in range(n_u)])))
                for k in range(self.horizon)
            ]
            X0_sol = all_X[0]
            states = [
                SystemState(np.array(sol.value(X0_sol[k])).flatten(),
                            list(self.landscape.manifold.labels))
                for k in range(self.horizon + 1)
            ]
            total_value = float(sol.value(total_cost / len(sims)))
            return Trajectory(controls=controls, states=states, costs=[], total_cost=total_value,
                              success=True, message="")
        except Exception as exc:
            controls = []
            states = []
            try:
                controls = [
                    ControlInput(np.array(opti.debug.value(U[:, k])).flatten(),
                                 list(getattr(self.nominal_sim.hvac, "control_labels", [f"u{i}" for i in range(n_u)])))
                    for k in range(self.horizon)
                ]
                X0_sol = all_X[0]
                states = [
                    SystemState(np.array(opti.debug.value(X0_sol[k])).flatten(),
                                list(self.landscape.manifold.labels))
                    for k in range(self.horizon + 1)
                ]
            except Exception:
                pass
            return Trajectory(controls=controls, states=states, costs=[], total_cost=0.0,
                              success=False, message=str(exc))
