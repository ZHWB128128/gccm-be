"""两区域 CasADi 鲁棒 MPC 闭环测试。"""
from __future__ import annotations

import time

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import SystemState

from examples.two_zone_compare import TwoZoneProvider, make_two_zone_simulator

STEP_H = 0.25
STEPS = 96
HORIZON = 6


def main() -> None:
    provider = TwoZoneProvider()
    plant_sim = make_two_zone_simulator()

    # 悲观两区域模型：保温更差、得热更大
    pessimistic_sim = Simulator(
        TwoZoneRCBuildingModel(
            c_air=0.55, c_wall=3.0, r_air=0.55, r_wall_a=1.5, r_wall_b=1.0,
            r_partition=0.9, solar_gain_a=0.1, solar_gain_b=0.04,
        ),
        HVACModel(q_min=-8, q_max=8, n_units=2, control_labels=["Q_hvac_A", "Q_hvac_B"]),
    )

    manifold = StateManifold(
        labels=["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"],
        units={l: "°C" for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        bounds={l: (15, 40) for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        scale={l: 5 for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
    )
    engine = GCCMEngine(
        simulator=make_two_zone_simulator(),
        external_provider=provider,
        manifold=manifold,
        horizon=HORIZON,
        dt=STEP_H,
        setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
        comfort_min=25.2,
        comfort_max=26.5,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        use_casadi_robust=True,
        robust_scenarios=[pessimistic_sim],
    )

    state = SystemState(np.full(5, 28.0), manifold.labels)
    t = 0.0
    prev = None
    ts_a, ts_b, ps, prices = [], [], [], []
    solve_times = []

    for i in range(STEPS):
        start = time.perf_counter()
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort")
        solve_times.append(time.perf_counter() - start)
        w = provider.get(t, 1)[0]
        state = plant_sim.step(state, dec.control, w, STEP_H)
        ts_a.append(state.x[0])
        ts_b.append(state.x[2])
        ps.append(plant_sim.hvac.electrical_power(dec.control))
        prices.append(w.w[5])
        prev = dec.control
        t += STEP_H
        if (i + 1) % 16 == 0:
            print(f"进度 {i+1}/{STEPS}，平均求解 {np.mean(solve_times[-16:]):.2f}s", flush=True)

    cost = float(np.sum(np.array(ps) * np.array(prices) * STEP_H))
    viol_a = float(np.mean((np.array(ts_a) > 27.0) | (np.array(ts_a) < 25.0)) * 100.0)
    viol_b = float(np.mean((np.array(ts_b) > 27.0) | (np.array(ts_b) < 25.0)) * 100.0)
    print("\n两区域鲁棒 MPC 结果")
    print(f"  电费: {cost:.2f} 元")
    print(f"  A违规: {viol_a:.1f}%, B违规: {viol_b:.1f}%")
    print(f"  A最高温: {max(ts_a):.2f}, A最低温: {min(ts_a):.2f}, B最高温: {max(ts_b):.2f}, B最低温: {min(ts_b):.2f}")
    print(f"  平均求解: {np.mean(solve_times):.3f} s")


if __name__ == "__main__":
    main()
