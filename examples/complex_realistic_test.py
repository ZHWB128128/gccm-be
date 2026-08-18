"""复杂真实环境测试：模型失配 + 预报噪声 + 电价尖峰 + 鲁棒 MPC。"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
from gccm_be.types import SystemState

from examples.hard_benchmark import BenchmarkProvider
from examples.model_mismatch_experiment import (
    COMFORT_MAX,
    COMFORT_MIN,
    CONTROLLER_PARAMS,
    PLANT_PARAMS,
    STEP_H,
    STEPS,
)

Q_MAX = 15.0
HORIZON = 12


@dataclass
class Result:
    seed: int
    cost: float
    violation: float
    max_temp: float
    min_temp: float
    solve: float


def run_seed(seed: int) -> Result:
    # 电价尖峰 + 预报噪声
    provider = BenchmarkProvider(peak_temp=35.0, price_mode="spike", seed=seed, noise_std=0.5)
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
    undecidable = 0
    for i in range(STEPS):
        start = time.perf_counter()
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort")
        solve_times.append(time.perf_counter() - start)
        w = provider.get(t, 1)[0]
        state = plant_sim.step(state, dec.control, w, STEP_H)
        ts.append(state.x[0])
        ps.append(plant_sim.hvac.electrical_power(dec.control))
        prices.append(w.w[3])
        if dec.diagnosis.undecidable:
            undecidable += 1
        prev = dec.control
        t += STEP_H
    cost = float(np.sum(np.array(ps) * np.array(prices) * STEP_H))
    arr = np.array(ts)
    viol = float(np.mean((arr > COMFORT_MAX) | (arr < COMFORT_MIN)) * 100.0)
    high_viol = float(np.mean(arr > COMFORT_MAX) * 100.0)
    low_viol = float(np.mean(arr < COMFORT_MIN) * 100.0)
    print(f"  seed={seed}: cost={cost:.2f}, viol={viol:.1f}%, high={high_viol:.1f}%, low={low_viol:.1f}%", flush=True)
    return Result(seed, cost, viol, float(np.max(ts)), float(np.min(ts)), float(np.mean(solve_times)))


def main() -> None:
    seeds = [0, 1, 2]
    results = []
    for seed in seeds:
        print(f"运行 seed={seed} ...", flush=True)
        results.append(run_seed(seed))
    print("\n" + "=" * 70)
    print(f"{'seed':>5}{'电费':>8}{'违规%':>8}{'最高温':>8}{'最低温':>8}{'求解s':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r.seed:>5}{r.cost:>8.2f}{r.violation:>8.1f}{r.max_temp:>8.2f}{r.min_temp:>8.2f}{r.solve:>8.3f}")
    print("=" * 70)
    print(f"平均违规: {np.mean([r.violation for r in results]):.1f}%")


if __name__ == "__main__":
    main()
