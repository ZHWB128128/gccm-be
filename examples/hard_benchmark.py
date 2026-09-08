"""硬基准对比：固定协议、多场景、多基线、统一指标。

方法：
- 严格舒适 PID
- 规则控制
- 经典 MPC（无几何动能项，use_kinetic=False）
- GCCM（use_kinetic=True）

场景：
- mild_30：温和天气 30°C
- hot_35：典型夏季 35°C
- hot_35_spike：35°C + 尖峰电价

运行：
    PYTHONPATH=. python3 examples/hard_benchmark.py --horizon 24 --scenarios mild_30,hot_35
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from compare_baselines import (
    COMFORT_MAX,
    COMFORT_MIN,
    STEP_H,
    STEPS,
    ScenarioExternalInputProvider,
    make_simulator,
    run_gccm,
    run_rule,
)
from fair_compare import run_strict_pid
from gccm_be.types import ExternalInput


class BenchmarkProvider(ScenarioExternalInputProvider):
    def __init__(self, peak_temp: float = 35.0, price_mode: str = "standard", seed: int = 0, noise_std: float = 0.0) -> None:
        super().__init__(STEP_H)
        self.peak_temp = peak_temp
        self.price_mode = price_mode
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def get(self, time_h: float, horizon: int = 1):
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = (self.peak_temp - 5.5) + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            if self.noise_std > 0.0:
                t_out += self.rng.normal(0.0, self.noise_std)
            if 6.0 <= hour <= 18.0:
                solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0))
            else:
                solar = 0.0
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3

            if self.price_mode == "spike":
                if hour < 8.0 or hour >= 22.0:
                    price = 0.2
                elif hour < 11.0 or hour >= 18.0:
                    price = 0.8
                else:
                    price = 2.0
            else:
                if hour < 8.0 or hour >= 22.0:
                    price = 0.3
                elif hour < 11.0 or hour >= 18.0:
                    price = 0.8
                else:
                    price = 1.5
            out.append(ExternalInput(
                np.array([t_out, solar, occ, price]),
                ["T_out", "solar", "occ", "price"],
            ))
        return out


@dataclass
class MethodResult:
    name: str
    cost: float
    violation: float
    mean_abs_err: float
    max_abs_err: float
    peak_power: float
    solve_time: float


def run_method(name: str, provider, sim, horizon: int) -> MethodResult:
    if name == "strict_pid":
        r = run_strict_pid(sim, provider)
        return MethodResult(name, r.total_cost, r.comfort_violation * 100.0,
                            float(np.mean(np.abs(r.t_air - 26.0))),
                            float(np.max(np.abs(r.t_air - 26.0))), r.peak_power, 0.0)
    if name == "rule":
        r = run_rule(sim, provider)
        return MethodResult(name, r.total_cost, r.comfort_violation * 100.0,
                            float(np.mean(np.abs(r.t_air - 26.0))),
                            float(np.max(np.abs(r.t_air - 26.0))), r.peak_power, 0.0)
    if name == "classic_mpc":
        r = run_gccm(provider, horizon=horizon, mode="comfort", comfort_band=0.0,
                     comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX,
                     below_comfort_penalty=0.1, peak_energy_penalty=1.0,
                     comfort_weight=5.0, energy_weight=0.1, smooth_weight=0.1,
                     enforce_comfort=True, use_kinetic=False,
                     simulator=sim,
                     constraint_options={"maxiter": 50, "ftol": 1e-6},
                     solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
                     verbose=False)
        return MethodResult(name, r.total_cost, r.comfort_violation * 100.0,
                            float(np.mean(np.abs(r.t_air - 26.0))),
                            float(np.max(np.abs(r.t_air - 26.0))), r.peak_power, r.avg_solve_time)
    if name == "gccm":
        r = run_gccm(provider, horizon=horizon, mode="comfort", comfort_band=0.0,
                     comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX,
                     below_comfort_penalty=0.1, peak_energy_penalty=1.0,
                     comfort_weight=5.0, energy_weight=0.1, smooth_weight=0.1,
                     enforce_comfort=True, use_kinetic=True,
                     simulator=sim,
                     constraint_options={"maxiter": 50, "ftol": 1e-6},
                     solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
                     verbose=False)
        return MethodResult(name, r.total_cost, r.comfort_violation * 100.0,
                            float(np.mean(np.abs(r.t_air - 26.0))),
                            float(np.max(np.abs(r.t_air - 26.0))), r.peak_power, r.avg_solve_time)
    raise ValueError(name)


SCENARIOS = {
    "mild_30": {"peak": 30.0, "price": "standard"},
    "warm_32": {"peak": 32.0, "price": "standard"},
    "hot_35": {"peak": 35.0, "price": "standard"},
    "hot_35_spike": {"peak": 35.0, "price": "spike"},
}

METHODS = ["strict_pid", "rule", "classic_mpc", "gccm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--scenarios", type=str, default="mild_30,hot_35")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q-max", type=float, default=8.0)
    args = parser.parse_args()

    scenario_names = [x.strip() for x in args.scenarios.split(",") if x.strip() in SCENARIOS]
    print("硬基准实验协议")
    print(f"  场景: {scenario_names}")
    print(f"  预测时域: {args.horizon} 步，滚动周期 15 分钟，仿真 24 小时")
    print(f"  舒适区间: {COMFORT_MIN}~{COMFORT_MAX}°C，设定 26°C，初始 28°C")
    print(f"  seed: {args.seed}")
    print()

    for scen in scenario_names:
        cfg = SCENARIOS[scen]
        provider = BenchmarkProvider(peak_temp=cfg["peak"], price_mode=cfg["price"], seed=args.seed)
        sim = make_simulator(q_max=args.q_max)
        print(f"=== 场景 {scen} (peak={cfg['peak']}°C, price={cfg['price']}) ===")
        print(f"{'方法':<14}{'电费(元)':>10}{'违规%':>8}{'平均绝对误差':>12}{'最大误差':>10}{'峰值kW':>8}{'求解s':>8}")
        results = []
        for m in METHODS:
            r = run_method(m, provider, sim, args.horizon)
            results.append(r)
            print(f"{m:<14}{r.cost:>10.2f}{r.violation:>8.1f}{r.mean_abs_err:>12.3f}{r.max_abs_err:>10.3f}{r.peak_power:>8.2f}{r.solve_time:>8.3f}")
        pid = next(r for r in results if r.name == "strict_pid")
        if pid.cost > 0:
            for r in results:
                if r.name != "strict_pid":
                    print(f"  vs strict_pid 节省: {r.name:<10} {(pid.cost - r.cost) / pid.cost * 100:.1f}%")
        print()


if __name__ == "__main__":
    main()
