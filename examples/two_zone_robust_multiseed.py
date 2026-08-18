"""两区域鲁棒 MPC 多 seed 验证（带预报噪声）。"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ExternalInput, SystemState

from examples.two_zone_compare import TwoZoneProvider, make_two_zone_simulator

STEP_H = 0.25
STEPS = 96
HORIZON = 6


class NoisyTwoZoneProvider(TwoZoneProvider):
    def __init__(self, seed: int = 0, noise_std: float = 0.3) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std

    def get(self, time_h: float, horizon: int = 1):
        out = super().get(time_h, horizon)
        for w in out:
            w.w[0] += self.rng.normal(0.0, self.noise_std)
        return out


@dataclass
class Result:
    seed: int
    cost: float
    viol_a: float
    viol_b: float
    solve: float


def run_seed(seed: int) -> Result:
    provider = NoisyTwoZoneProvider(seed=seed)
    plant_sim = make_two_zone_simulator()
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
        comfort_min=25.5,
        comfort_max=26.5,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        use_casadi_robust=True,
        robust_heating_cost_factor=0.2,
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
        ts_a.append(state.x[0]); ts_b.append(state.x[2])
        ps.append(plant_sim.hvac.electrical_power(dec.control)); prices.append(w.w[5])
        prev = dec.control; t += STEP_H
    cost = float(np.sum(np.array(ps) * np.array(prices) * STEP_H))
    arr_a = np.array(ts_a); arr_b = np.array(ts_b)
    va = float(np.mean((arr_a > 27) | (arr_a < 25)) * 100)
    vb = float(np.mean((arr_b > 27) | (arr_b < 25)) * 100)
    ha = float(np.mean(arr_a > 27) * 100); la = float(np.mean(arr_a < 25) * 100)
    hb = float(np.mean(arr_b > 27) * 100); lb = float(np.mean(arr_b < 25) * 100)
    print(f"  seed={seed}: violA={va:.1f}% (high={ha:.1f},low={la:.1f}), violB={vb:.1f}% (high={hb:.1f},low={lb:.1f})", flush=True)
    return Result(seed, cost, va, vb, float(np.mean(solve_times)))


def main() -> None:
    results = [run_seed(s) for s in [0, 1]]
    print("\n" + "=" * 60)
    print(f"{'seed':>5}{'电费':>8}{'A违规%':>8}{'B违规%':>8}{'求解s':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r.seed:>5}{r.cost:>8.2f}{r.viol_a:>8.1f}{r.viol_b:>8.1f}{r.solve:>8.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
