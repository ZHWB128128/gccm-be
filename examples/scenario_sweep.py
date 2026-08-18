"""多场景鲁棒性测试：GCCM vs 严格舒适 PID。

场景：
- baseline_35     : 夏季高温 35°C，标准电价
- hot_40          : 极端高温 40°C，空调容量加大到 12 kW
- mild_30         : 温和 30°C，标准电价
- high_price      : 35°C，峰谷价差拉大（峰时 2.0 元/kWh）
- heavy_building  : 35°C，重质墙体/更好保温
- forecast_noise  : 35°C，GCCM 使用含噪声的预报，真实世界用无噪声天气

运行:
    PYTHONPATH=. python3 examples/scenario_sweep.py --scenarios baseline_35,hot_40
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from compare_baselines import (
    COMFORT_MAX,
    COMFORT_MIN,
    STEP_H,
    STEPS,
    ScenarioExternalInputProvider,
    make_simulator,
    run_gccm,
)
from gccm_be.types import ExternalInput
from fair_compare import run_strict_pid

# 标准场景定义
SCENARIOS = {
    "baseline_35": {"peak": 35.0, "price": "standard", "qmax": 8, "heavy": False, "noise": 0.0},
    "hot_40": {"peak": 40.0, "price": "standard", "qmax": 12, "heavy": False, "noise": 0.0},
    "mild_30": {"peak": 30.0, "price": "standard", "qmax": 8, "heavy": False, "noise": 0.0},
    "high_price": {"peak": 35.0, "price": "high_spread", "qmax": 8, "heavy": False, "noise": 0.0},
    "heavy_building": {"peak": 35.0, "price": "standard", "qmax": 8, "heavy": True, "noise": 0.0},
    "forecast_noise": {"peak": 35.0, "price": "standard", "qmax": 8, "heavy": False, "noise": 1.0},
}


class VariantProvider(ScenarioExternalInputProvider):
    """可配置天气/电价/预报噪声的外部输入。"""

    def __init__(
        self,
        peak_temp: float = 35.0,
        price_mode: str = "standard",
        noise_std: float = 0.0,
        seed: int = 0,
        dt_h: float = STEP_H,
    ) -> None:
        super().__init__(dt_h)
        self.peak_temp = peak_temp
        self.price_mode = price_mode
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def get(self, time_h: float, horizon: int = 1) -> List:
        result = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            mean = 29.5 + (self.peak_temp - 35.0)
            t_out = mean + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            if self.noise_std > 0.0:
                t_out = t_out + self.rng.normal(0.0, self.noise_std)
                t_out = max(15.0, min(45.0, t_out))

            if 6.0 <= hour <= 18.0:
                solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0))
            else:
                solar = 0.0

            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3

            if self.price_mode == "high_spread":
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

            result.append(ExternalInput(
                np.array([t_out, solar, occ, price]),
                ["T_out", "solar", "occ", "price"],
            ))
        return result


@dataclass
class ScenarioResult:
    name: str
    pid_cost: float
    pid_viol: float
    gccm_cost: float
    gccm_viol: float
    saving: float
    gccm_peak: float
    gccm_solve: float


def run_one(name: str, cfg: dict, horizon: int, constraint_maxiter: int, comfort_max: float = COMFORT_MAX) -> ScenarioResult:
    clean_provider = VariantProvider(peak_temp=cfg["peak"], price_mode=cfg["price"], noise_std=0.0)
    forecast_provider = VariantProvider(
        peak_temp=cfg["peak"],
        price_mode=cfg["price"],
        noise_std=cfg["noise"],
        seed=1,
    )
    sim = make_simulator(q_max=cfg["qmax"], heavy=cfg["heavy"])

    pid = run_strict_pid(sim, clean_provider)
    gccm = run_gccm(
        forecast_provider,
        horizon=horizon,
        mode="comfort",
        simulator=sim,
        plant_provider=clean_provider,
        comfort_band=0.0,
        comfort_min=COMFORT_MIN,
        comfort_max=comfort_max,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=0.5,
        smooth_weight=0.1,
        enforce_comfort=True,
        constraint_options={"maxiter": constraint_maxiter, "ftol": 1e-6},
        verbose=False,
    )
    saving = (pid.total_cost - gccm.total_cost) / pid.total_cost * 100.0 if pid.total_cost > 0 else 0.0
    print(f"  完成 {name}: PID {pid.total_cost:.2f}元/{pid.comfort_violation*100:.1f}%, "
          f"GCCM {gccm.total_cost:.2f}元/{gccm.comfort_violation*100:.1f}%, 节省 {saving:.1f}%", flush=True)
    return ScenarioResult(
        name=name,
        pid_cost=pid.total_cost,
        pid_viol=pid.comfort_violation * 100.0,
        gccm_cost=gccm.total_cost,
        gccm_viol=gccm.comfort_violation * 100.0,
        saving=saving,
        gccm_peak=gccm.peak_power,
        gccm_solve=gccm.avg_solve_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GCCM-BE 多场景鲁棒性测试")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--constraint-maxiter", type=int, default=50)
    parser.add_argument("--scenarios", type=str, default=None,
                        help="逗号分隔场景名，默认全部")
    parser.add_argument("--comfort-max", type=float, default=COMFORT_MAX,
                        help="GCCM 内部硬约束上限（用于安全裕度），默认 27.0")
    args = parser.parse_args()

    names = list(SCENARIOS.keys())
    if args.scenarios:
        names = [x.strip() for x in args.scenarios.split(",") if x.strip() in SCENARIOS]

    print(f"运行场景: {names}")
    results: List[ScenarioResult] = []
    for name in names:
        print(f"开始 {name} ...", flush=True)
        results.append(run_one(name, SCENARIOS[name], args.horizon, args.constraint_maxiter, args.comfort_max))

    print("\n" + "=" * 110)
    print(f"{'场景':<18}{'PID电费':>8}{'PID违规%':>9}{'GCCM电费':>10}"
          f"{'GCCM违规%':>10}{'节省%':>8}{'GCCM峰值kW':>12}{'GCCM求解s':>10}")
    print("-" * 110)
    for r in results:
        print(f"{r.name:<18}{r.pid_cost:>8.2f}{r.pid_viol:>9.1f}{r.gccm_cost:>10.2f}"
              f"{r.gccm_viol:>10.1f}{r.saving:>8.1f}{r.gccm_peak:>12.2f}{r.gccm_solve:>10.3f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
