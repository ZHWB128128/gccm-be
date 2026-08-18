"""最优 Pareto 点多 seed / 多场景验证。

配置：energy_weight=0.5, comfort_margin=0.3, Q_MAX=15
场景：hot_35, hot_35_spike
种子：0, 1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np

from compare_baselines import (
    COMFORT_MAX,
    COMFORT_MIN,
    make_simulator,
    run_gccm,
)
from fair_compare import run_strict_pid
from hard_benchmark import BenchmarkProvider

ENERGY_WEIGHT = 0.5
COMFORT_MARGIN = 0.3
Q_MAX = 15.0
HORIZON = 24


@dataclass
class Run:
    scenario: str
    seed: int
    pid_cost: float
    pid_viol: float
    gccm_cost: float
    gccm_viol: float
    saving: float
    peak: float
    solve: float


def run_one(scenario: str, seed: int, noise_std: float, noise_adaptive_margin: bool = False, margin: float = COMFORT_MARGIN, energy_weight: float = ENERGY_WEIGHT) -> Run:
    provider = BenchmarkProvider(peak_temp=35.0, price_mode=scenario, seed=seed, noise_std=noise_std)
    sim = make_simulator(q_max=Q_MAX)
    pid = run_strict_pid(sim, provider)
    gccm = run_gccm(
        provider,
        horizon=HORIZON,
        mode="comfort",
        comfort_band=0.0,
        comfort_margin=margin,
        noise_adaptive_margin=noise_adaptive_margin,
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=energy_weight,
        smooth_weight=0.1,
        enforce_comfort=True,
        use_kinetic=True,
        constraint_options={"maxiter": 50, "ftol": 1e-6},
        solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
        verbose=False,
    )
    saving = (pid.total_cost - gccm.total_cost) / pid.total_cost * 100.0
    return Run(
        scenario=scenario,
        seed=seed,
        pid_cost=pid.total_cost,
        pid_viol=pid.comfort_violation * 100.0,
        gccm_cost=gccm.total_cost,
        gccm_viol=gccm.comfort_violation * 100.0,
        saving=saving,
        peak=gccm.peak_power,
        solve=gccm.avg_solve_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=str, default="hot_35,hot_35_spike")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=COMFORT_MARGIN)
    parser.add_argument("--energy-weight", type=float, default=ENERGY_WEIGHT)
    args = parser.parse_args()

    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    print("Pareto 点验证")
    print(f"  energy_weight={args.energy_weight}, comfort_margin={args.margin}, Q_MAX={Q_MAX}")
    print(f"  场景: {scenarios}, seeds: {seeds}, noise_std={args.noise_std}")
    print()

    rows: List[Run] = []
    for scen in scenarios:
        for seed in seeds:
            print(f"运行 {scen} seed={seed} ...", flush=True)
            rows.append(run_one(scen, seed, args.noise_std, args.noise_std > 0.0, args.margin, args.energy_weight))

    print("\n" + "=" * 110)
    print(f"{'场景':<14}{'seed':>5}{'PID电费':>8}{'PID违规%':>9}{'GCCM电费':>10}{'GCCM违规%':>10}{'节省%':>8}{'峰值':>7}{'求解s':>7}")
    print("-" * 110)
    for r in rows:
        print(f"{r.scenario:<14}{r.seed:>5}{r.pid_cost:>8.2f}{r.pid_viol:>9.1f}{r.gccm_cost:>10.2f}"
              f"{r.gccm_viol:>10.1f}{r.saving:>8.1f}{r.peak:>7.2f}{r.solve:>7.3f}")
    print("=" * 110)

    feasible = [r for r in rows if r.gccm_viol <= 5.0]
    if feasible:
        avg_saving = float(np.mean([r.saving for r in feasible]))
        avg_cost = float(np.mean([r.gccm_cost for r in feasible]))
        max_viol = float(np.max([r.gccm_viol for r in feasible]))
        print(f"\n可行运行数: {len(feasible)}/{len(rows)}")
        print(f"平均 GCCM 电费: {avg_cost:.2f} 元")
        print(f"平均节省: {avg_saving:.1f}%")
        print(f"最大 GCCM 违规: {max_viol:.1f}%")


if __name__ == "__main__":
    main()
