"""GCCM 违规-电费 Pareto 扫描。

场景：hot_35，Q_MAX=15（物理可行容量）。
扫描 energy_weight 与 comfort_margin，输出帕累托前沿。

运行：
    PYTHONPATH=. python3 examples/pareto_sweep.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from compare_baselines import (
    COMFORT_MAX,
    COMFORT_MIN,
    make_simulator,
    run_gccm,
)
from hard_benchmark import BenchmarkProvider, run_method
from fair_compare import run_strict_pid

# 物理可行场景
PEAK_TEMP = 35.0
Q_MAX = 15.0
HORIZON = 24


@dataclass
class Point:
    energy_weight: float
    comfort_margin: float
    cost: float
    violation: float
    peak: float
    solve: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-weights", type=str, default="0.1,0.3,0.5,0.8")
    parser.add_argument("--margins", type=str, default="0.3,0.5")
    args = parser.parse_args()

    energy_weights = [float(x) for x in args.energy_weights.split(",")]
    margins = [float(x) for x in args.margins.split(",")]

    provider = BenchmarkProvider(peak_temp=PEAK_TEMP, price_mode="standard")
    sim = make_simulator(q_max=Q_MAX)

    # 参考基线：严格舒适 PID
    pid = run_strict_pid(sim, provider)
    print(f"严格舒适 PID: cost={pid.total_cost:.2f}, violation={pid.comfort_violation*100:.1f}%")
    print()

    print(f"{'energy':>6}{'margin':>7}{'电费':>8}{'违规%':>7}{'峰值':>7}{'求解s':>7}")
    points = []
    for ew in energy_weights:
        for m in margins:
            r = run_gccm(
                provider,
                horizon=HORIZON,
                mode="comfort",
                comfort_band=0.0,
                comfort_margin=m,
                comfort_min=COMFORT_MIN,
                comfort_max=COMFORT_MAX,
                below_comfort_penalty=0.1,
                peak_energy_penalty=1.0,
                comfort_weight=5.0,
                energy_weight=ew,
                smooth_weight=0.1,
                enforce_comfort=True,
                use_kinetic=True,
                constraint_options={"maxiter": 50, "ftol": 1e-6},
                solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
                verbose=False,
            )
            p = Point(ew, m, r.total_cost, r.comfort_violation * 100.0, r.peak_power, r.avg_solve_time)
            points.append(p)
            print(f"{ew:>6.2f}{m:>7.2f}{p.cost:>8.2f}{p.violation:>7.1f}{p.peak:>7.2f}{p.solve:>7.3f}")

    # 简单 Pareto：违规<=5% 且电费最低
    feasible = [p for p in points if p.violation <= 5.0]
    if feasible:
        best = min(feasible, key=lambda p: p.cost)
        print(f"\n推荐配置: energy_weight={best.energy_weight}, comfort_margin={best.comfort_margin}")
        print(f"  电费 {best.cost:.2f} 元，违规 {best.violation:.1f}%，峰值 {best.peak:.2f} kW")
    else:
        print("\n当前扫描中没有违规<=5%的点")


if __name__ == "__main__":
    main()
