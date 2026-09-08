"""GCCM-BE 能量景观参数扫描。

目的：找到“总电费较低且舒适违规 <=5%”的权重组合。
默认使用 48 步预测时域（12 小时），滚动周期 15 分钟，仿真 24 小时。

运行:
    PYTHONPATH=. python3 examples/parameter_scan.py
    PYTHONPATH=. python3 examples/parameter_scan.py --horizon 48
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from compare_baselines import (
    STEPS,
    ScenarioExternalInputProvider,
    run_gccm,
)

# 固定舒适区间与舒适权重
COMFORT_MIN = 25.0
COMFORT_MAX = 27.0
COMFORT_WEIGHT = 5.0
BELOW_PENALTY = 0.0

# 待扫描配置
DEFAULT_CONFIGS: List[Dict] = [
    {"name": "energy0.5_peak1.0", "energy_weight": 0.5, "peak_penalty": 1.0},
    {"name": "energy0.5_peak1.2", "energy_weight": 0.5, "peak_penalty": 1.2},
    {"name": "energy0.5_peak1.5", "energy_weight": 0.5, "peak_penalty": 1.5},
    {"name": "energy0.5_peak2.0", "energy_weight": 0.5, "peak_penalty": 2.0},
    {"name": "energy1.0_peak2.0", "energy_weight": 1.0, "peak_penalty": 2.0},
    {"name": "energy1.5_peak3.0", "energy_weight": 1.5, "peak_penalty": 3.0},
]


@dataclass
class ScanResult:
    name: str
    total_cost: float
    violation: float
    peak_power: float
    avg_solve_time: float
    night_avg_power: float
    peak_avg_power: float


def scan_config(
    provider: ScenarioExternalInputProvider,
    cfg: Dict,
    horizon: int,
    enforce_comfort: bool = True,
    solver_options: Optional[dict] = None,
    constraint_options: Optional[dict] = None,
) -> ScanResult:
    if solver_options is None:
        solver_options = {"maxiter": 60, "ftol": 1e-4, "maxls": 20}

    r = run_gccm(
        provider,
        horizon=horizon,
        mode="comfort",
        comfort_band=0.0,
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        below_comfort_penalty=BELOW_PENALTY,
        peak_energy_penalty=float(cfg["peak_penalty"]),
        comfort_weight=COMFORT_WEIGHT,
        energy_weight=float(cfg["energy_weight"]),
        smooth_weight=0.1,
        enforce_comfort=enforce_comfort,
        solver_options=solver_options,
        constraint_options=constraint_options,
        verbose=False,
    )

    night_mask = r.times_h < 8.0
    peak_mask = (r.times_h >= 11.0) & (r.times_h < 18.0)
    return ScanResult(
        name=cfg["name"],
        total_cost=r.total_cost,
        violation=r.comfort_violation * 100.0,
        peak_power=r.peak_power,
        avg_solve_time=r.avg_solve_time,
        night_avg_power=float(np.mean(r.p_elec[night_mask])) if np.any(night_mask) else 0.0,
        peak_avg_power=float(np.mean(r.p_elec[peak_mask])) if np.any(peak_mask) else 0.0,
    )


def print_scan(results: List[ScanResult]) -> None:
    print("\n" + "=" * 100)
    print(f"{'配置':<24}{'总电费(元)':>10}{'违规(%)':>8}{'峰值功率(kW)':>12}"
          f"{'平均求解(s)':>12}{'谷时均功率(kW)':>14}{'峰时均功率(kW)':>14}")
    print("-" * 100)
    for r in results:
        print(f"{r.name:<24}{r.total_cost:>10.2f}{r.violation:>8.1f}{r.peak_power:>12.2f}"
              f"{r.avg_solve_time:>12.3f}{r.night_avg_power:>14.3f}{r.peak_avg_power:>14.3f}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="GCCM-BE 能量景观参数扫描")
    parser.add_argument("--horizon", type=int, default=48,
                        help="GCCM 预测时域步数，默认 48（12 小时）")
    parser.add_argument("--configs", type=str, default=None,
                        help="逗号分隔的配置名，例如 energy0.5_peak2.0,energy1.0_peak2.0")
    parser.add_argument("--no-enforce-comfort", action="store_true",
                        help="不使用硬约束，改为纯软惩罚")
    parser.add_argument("--constraint-maxiter", type=int, default=80,
                        help="硬约束 SLSQP 最大迭代次数")
    args = parser.parse_args()

    provider = ScenarioExternalInputProvider()
    configs = DEFAULT_CONFIGS
    if args.configs:
        allowed = {c["name"]: c for c in configs}
        names = [x.strip() for x in args.configs.split(",") if x.strip()]
        configs = [allowed[n] for n in names if n in allowed]

    print(f"扫描配置数: {len(configs)}，预测时域: {args.horizon} 步（{args.horizon * 0.25:.1f} 小时）")
    results = []
    enforce_comfort = not args.no_enforce_comfort
    for cfg in configs:
        print(f"运行 {cfg['name']} ...", flush=True)
        results.append(scan_config(
            provider,
            cfg,
            args.horizon,
            enforce_comfort=enforce_comfort,
            constraint_options={"maxiter": args.constraint_maxiter, "ftol": 1e-6},
        ))

    print_scan(results)

    # 简单筛选：违规 <=5% 且电费最低
    feasible = [r for r in results if r.violation <= 5.0]
    if feasible:
        best = min(feasible, key=lambda r: r.total_cost)
        print(f"\n推荐配置: {best.name}")
        print(f"  总电费 {best.total_cost:.2f} 元，违规 {best.violation:.1f}%，"
              f"峰值功率 {best.peak_power:.2f} kW")
    else:
        print("\n当前扫描中没有找到违规 <=5% 的配置，需要继续增大舒适权重或加强峰时惩罚。")


if __name__ == "__main__":
    main()
