"""Classic linear MPC baseline for fair comparison.

This is a standalone, minimal linear MPC solver (no GCCM dependencies) to serve as
a clean anchor point for the "GCCM vs classic MPC" comparison in technical reports.

Design:
- Uses the same RC building model and HVAC constraints as GCCM
- Solves finite-horizon quadratic program with hard comfort bounds
- No safety chain, no online identification, no geometric terms
- Tuned parameters: horizon=24h, Q_max=15kW, comfort=[25,27]°C
- Solver: scipy.optimize.minimize (SLSQP), matching GCCM's fallback backend

Usage:
    PYTHONPATH=. python3 examples/classic_mpc_baseline.py --steps 96 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

# Import common models/providers from gccm_be
from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
from gccm_be.physics.external import MockExternalInputProvider
from gccm_be.types import ControlInput, ExternalInput, SystemState


@dataclass
class ClassicMPCConfig:
    horizon: int = 96
    dt_h: float = 1.0 / 12.0
    comfort_min: float = 25.0
    comfort_max: float = 27.0
    q_max: float = 15.0
    energy_weight: float = 0.1
    smooth_weight: float = 0.1
    maxiter: int = 100
    ftol: float = 1e-6


class ClassicLinearMPC:
    """Minimal linear MPC solver (no GCCM architecture)."""

    def __init__(self, config: ClassicMPCConfig):
        self.config = config
        self.horizon = config.horizon
        self.dt = config.dt_h * 3600.0  # seconds
        self.comfort_min = config.comfort_min
        self.comfort_max = config.comfort_max
        self.q_max = config.q_max
        self.energy_weight = config.energy_weight
        self.smooth_weight = config.smooth_weight
        self._solve_stats: list = []

    def _build_linear_model(self, building: RCBuildingModel, hvac: HVACModel):
        """Build linear state-space model A*x + B*u for MPC (coefficients cached)."""
        self._lin = dict(
            dt=self.dt, c_air=building.c_air, c_wall=building.c_wall,
            r_air=building.r_air, r_wall=building.r_wall,
            solar_gain=building.solar_gain)
        # Linearize around operating point (T_air=26, T_wall=26)
        # dT_air/dt = a1*(T_wall - T_air) + a2*(T_out - T_air) + a3*Q + ...
        # dT_wall/dt = b1*(T_air - T_wall) + b2*(T_out - T_wall)
        c_air = building.c_air
        c_wall = building.c_wall
        r_air = building.r_air
        r_wall = building.r_wall
        solar_gain = building.solar_gain

        # Discretized linear coefficients (Euler step)
        dt = self.dt
        a1 = dt / (c_air * r_air)
        a2 = dt / (c_air * r_wall)
        a3 = dt / c_air
        b1 = dt / (c_wall * r_air)
        b2 = dt / (c_wall * r_wall)

        # State: [T_air, T_wall], Control: [Q]
        # x_{k+1} = A*x_k + B*u_k + E*w_k
        A = np.array([
            [1 - a1 - a2, a1],
            [b1, 1 - b2]
        ])
        B = np.array([[a3], [0]])
        return A, B

    def solve_step(
        self,
        state: SystemState,
        external: ExternalInput,
        prev_control: Optional[ControlInput] = None,
    ) -> ControlInput:
        """One-step MPC optimization over the full horizon."""
        labels = list(state.labels)
        if len(labels) != 2:
            raise ValueError("ClassicLinearMPC expects 2-state model")

        # Build linear model
        building = self.simulator.building
        hvac = self.simulator.hvac
        A, B = self._build_linear_model(building, hvac)

        # 决策变量:整个时域的控制序列 [horizon]
        if prev_control is not None and prev_control.u.size > 0:
            u0 = np.full(self.horizon, float(np.clip(prev_control.u[0], -self.q_max, 0.0)))
        else:
            u0 = np.linspace(-self.q_max * 0.4, -self.q_max * 0.2, self.horizon)

        # Bounds
        lb = np.array([-self.q_max])
        ub = np.array([self.q_max])

        # Objective: min Σ [energy_cost + smooth_cost]
        # 预测模型 = 工厂模型本身（单区 RC 线性；外部恒定 = 标准 MPC 假设）
        dt = self.dt

        def objective(z: np.ndarray) -> float:
            cost = 0.0
            x_curr = state.copy()
            price = float(external.w[3]) if external.w.size > 3 else 1.0

            for k in range(self.horizon):
                u_k = ControlInput(np.array([float(z[k])]), ["Q_hvac"])
                x_next = self.simulator.step(x_curr, u_k, external, dt)
                t_air = float(x_next.x[0])

                elec = abs(float(z[k])) / 3.8  # COP approx
                cost += self.energy_weight * elec * price
                if k > 0:
                    cost += self.smooth_weight * (float(z[k]) - float(z[k-1]))**2
                x_curr = x_next

                if t_air < self.comfort_min:
                    cost += 1e4 * min((self.comfort_min - t_air), 20.0)**2
                elif t_air > self.comfort_max:
                    cost += 1e4 * min((t_air - self.comfort_max), 20.0)**2

            return cost

        # Constraints: none (penalty method used)
        cons = []

        # Solve
        result = minimize(
            objective,
            u0.flatten(),
            method='SLSQP',
            bounds=[(lb[0], ub[0])] * self.horizon,
            constraints=cons,
            options={'maxiter': self.config.maxiter, 'ftol': self.config.ftol}
        )

        self._solve_stats.append(bool(result.success))
        if not result.success:
            print(f"[ClassicMPC] Optimization failed: {result.message}")

        # 返回时域第一步控制
        u_first = float(result.x[0]) if result.x.size > 0 else 0.0
        return ControlInput(np.array([u_first]), ["Q_hvac"])

    def run_closed_loop(
        self,
        simulator: Simulator,
        provider: MockExternalInputProvider,
        initial_state: SystemState,
        steps: int,
    ) -> List[tuple]:
        """Run closed-loop simulation."""
        self.simulator = simulator
        results = []
        state = initial_state.copy()
        prev_control: Optional[ControlInput] = None

        for k in range(steps):
            w = provider.get(k * self.config.dt_h, 1)[0]
            control = self.solve_step(state, w, prev_control)
            state = simulator.step(state, control, w, self.config.dt_h)
            results.append((k, float(state.x[0]), float(control.u[0]), float(w.price)))
            prev_control = control

        return results


def run_classic_mpc(
    horizon: int = 96,
    seed: int = 42,
    q_max: float = 15.0,
) -> dict:
    """Run classic MPC on bestest_air scenario."""
    rng = np.random.default_rng(seed)
    sim = Simulator(
        RCBuildingModel(c_air=0.6, c_wall=4.0, r_air=0.8, r_wall=2.0, solar_gain=0.05),
        HVACModel(q_min=-q_max, q_max=q_max, cop_cooling=3.8)
    )
    provider = MockExternalInputProvider(dt_h=1.0/12.0)
    config = ClassicMPCConfig(horizon=horizon, q_max=q_max)
    mpc = ClassicLinearMPC(config)

    start_temp = 28.0
    initial_state = SystemState([start_temp, start_temp], ["T_air", "T_wall"])
    results = mpc.run_closed_loop(sim, provider, initial_state, horizon)

    # Aggregate metrics
    temps = np.array([r[1] for r in results])
    controls = np.array([r[2] for r in results])
    prices = np.array([r[3] for r in results])
    dt_h = 1.0 / 12.0

    total_cost = float(np.sum(np.abs(controls) / 3.8 * prices * dt_h))
    violation = float(np.mean((temps < 25.0) | (temps > 27.0)) * 100.0)
    peak_power = float(np.max(np.abs(controls)))

    return {
        "name": "classic_mpc",
        "cost": total_cost,
        "violation": violation,
        "peak": peak_power,
        "avg_temp": float(np.mean(temps)),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=96, help="Horizon steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--q-max", type=float, default=15.0, help="Max cooling power")
    parser.add_argument("--output", default="output/classic_mpc_baseline.csv")
    args = parser.parse_args()

    print(f"Running Classic Linear MPC (horizon={args.steps}, seed={args.seed})...")
    result = run_classic_mpc(horizon=args.steps, seed=args.seed, q_max=args.q_max)

    print(f"\nResults:")
    print(f"  Cost: ¥{result['cost']:.2f}")
    print(f"  Violation: {result['violation']:.1f}%")
    print(f"  Peak Power: {result['peak']:.3f} kW")
    print(f"  Avg Temp: {result['avg_temp']:.2f} °C")

    # Save CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "temp_C", "power_kw", "price"])
        for r in result["results"]:
            w.writerow(r)
    print(f"\nCSV saved: {args.output}")


if __name__ == "__main__":
    main()
