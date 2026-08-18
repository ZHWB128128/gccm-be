"""GCCMEngine + CasADi 鲁棒 MPC 后端在模型失配场景下的闭环测试。"""
from __future__ import annotations

import time

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
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
    pessimistic_sim = Simulator(
        RCBuildingModel(c_air=0.45, c_wall=3.0, r_air=0.45, r_wall=1.0, solar_gain=0.1),
        HVACModel(q_min=-Q_MAX, q_max=Q_MAX),
    )
    manifold = StateManifold(labels=["T_air", "T_wall"], scale={"T_air": 5, "T_wall": 5})

    engine = GCCMEngine(
        simulator=nominal_sim,
        external_provider=provider,
        manifold=manifold,
        horizon=HORIZON,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=COMFORT_MIN,
        comfort_max=26.5,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        use_casadi_robust=True,
        robust_scenarios=[pessimistic_sim],
    )

    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    prev = None
    ts, ps, prices = [], [], []
    solve_times = []

    for i in range(STEPS):
        start = time.perf_counter()
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort")
        solve_times.append(time.perf_counter() - start)
        w = provider.get(t, 1)[0]
        state = plant_sim.step(state, dec.control, w, STEP_H)
        ts.append(state.x[0])
        ps.append(plant_sim.hvac.electrical_power(dec.control))
        prices.append(w.w[3])
        prev = dec.control
        t += STEP_H
        if (i + 1) % 16 == 0:
            print(f"进度 {i+1}/{STEPS}，平均求解 {np.mean(solve_times[-16:]):.2f}s", flush=True)

    cost = float(np.sum(np.array(ps) * np.array(prices) * STEP_H))
    viol = float(np.mean((np.array(ts) > COMFORT_MAX) | (np.array(ts) < COMFORT_MIN)) * 100.0)
    print("\nGCCMEngine + CasADi 鲁棒 MPC 模型失配闭环结果")
    print(f"  电费: {cost:.2f} 元")
    print(f"  违规: {viol:.1f}%")
    print(f"  最高温: {max(ts):.2f}°C")
    print(f"  最低温: {min(ts):.2f}°C")
    print(f"  平均求解: {np.mean(solve_times):.3f} s")


if __name__ == "__main__":
    main()
