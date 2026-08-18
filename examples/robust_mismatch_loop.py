"""鲁棒 MPC 在模型失配场景下的闭环测试。

控制器使用名义模型 + 悲观模型，优化时要求所有场景满足舒适约束。
"""
from __future__ import annotations

import time

import numpy as np

from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.geometry.robust_solver import RobustGeodesicSolver
from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
from gccm_be.types import SystemState

from examples.model_mismatch_experiment import (
    COMFORT_MAX,
    COMFORT_MIN,
    CONTROLLER_PARAMS,
    PLANT_PARAMS,
    STEP_H,
    STEPS,
    SummerProvider,
)

Q_MAX = 8.0
HORIZON = 12


def main() -> None:
    provider = SummerProvider()
    plant_sim = Simulator(RCBuildingModel(**PLANT_PARAMS), HVACModel(q_min=-Q_MAX, q_max=Q_MAX))
    nominal_sim = Simulator(RCBuildingModel(**CONTROLLER_PARAMS), HVACModel(q_min=-Q_MAX, q_max=Q_MAX))
    # 悲观模型：比真实更差，保证鲁棒
    pessimistic_sim = Simulator(
        RCBuildingModel(c_air=0.45, c_wall=3.0, r_air=0.45, r_wall=1.0, solar_gain=0.1),
        HVACModel(q_min=-Q_MAX, q_max=Q_MAX),
    )

    manifold = StateManifold(labels=["T_air", "T_wall"], scale={"T_air": 5, "T_wall": 5})
    landscape = EnergyLandscape(
        setpoints={"T_air": 26.0},
        weights={"comfort": 5.0, "energy": 0.1, "smooth": 0.1},
        manifold=manifold,
        hvac=nominal_sim.hvac,
        comfort_min=COMFORT_MIN,
        comfort_max=26.5,
    )
    solver = RobustGeodesicSolver(
        nominal_sim,
        landscape,
        [pessimistic_sim],
        horizon=HORIZON,
        dt=STEP_H,
        comfort_min=COMFORT_MIN,
        comfort_max=26.5,
    )

    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    ts = []
    ps = []
    prices = []
    solve_times = []
    prev_controls = None

    for i in range(STEPS):
        exts = provider.get(t, HORIZON)
        init = None
        if prev_controls is not None and len(prev_controls) == HORIZON:
            init = prev_controls[1:] + [prev_controls[-1]]
        start = time.perf_counter()
        traj = solver.solve(state, exts, initial_controls=init)
        solve_times.append(time.perf_counter() - start)
        w = exts[0]
        if not traj.controls:
            continue
        prev_controls = traj.controls
        state = plant_sim.step(state, traj.controls[0], w, STEP_H)
        ts.append(state.x[0])
        ps.append(plant_sim.hvac.electrical_power(traj.controls[0]))
        prices.append(w.w[3])
        t += STEP_H
        if (i + 1) % 16 == 0:
            print(f"进度 {i+1}/{STEPS}，平均求解 {np.mean(solve_times[-16:]):.2f}s", flush=True)

    cost = float(np.sum(np.array(ps) * np.array(prices) * STEP_H))
    viol = float(np.mean((np.array(ts) > COMFORT_MAX) | (np.array(ts) < COMFORT_MIN)) * 100.0)
    print("\n鲁棒 MPC 模型失配闭环结果")
    print(f"  电费: {cost:.2f} 元")
    print(f"  违规: {viol:.1f}%")
    print(f"  最高温: {max(ts):.2f}°C")
    print(f"  最低温: {min(ts):.2f}°C")
    print(f"  平均求解: {np.mean(solve_times):.3f} s")


if __name__ == "__main__":
    main()
