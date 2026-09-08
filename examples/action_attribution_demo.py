"""边界三突破：动作级反事实归因（Action-level Counterfactual Attribution）。

把 Shapley 因子归因（policy/weather/price）细化为**动作集归因**：
GCCM 24h 控制序列分解为三个动作集，各自做"去掉该动作集"的反事实仿真：

- 预冷集:谷价时段(-1.0元以下)的超额制冷
- 削峰集:峰价时段(>1.5元)的降载
- 常规集:其余时段的动作

方法：对每个动作集,用 GCCM 原控制替换为"基线控制"(恒温器),其余时段保持
GCCM 原控制,跑闭环仿真,与全 GCCM/全基线对比 → 每个动作集的独立贡献。

用法: PYTHONPATH=. python3 examples/action_attribution_demo.py --steps 96
"""
from __future__ import annotations

import argparse

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState

DT = 1.0 / 12.0
COMFORT_MIN, COMFORT_MAX = 25.0, 27.0
PRICE_PRECOOL = 0.5     # 低于此价 = 预冷时段(谷 0.3)
PRICE_PEAK = 1.5        # 高于此价 = 峰时时段(峰 1.8)


def build_engines():
    m = StateManifold(labels=["T_air", "T_wall"], units={}, bounds={},
                      scale={"T_air": 5.0, "T_wall": 5.0})
    gccm = GCCMEngine(
        simulator=Simulator(RCBuildingModel(c_air=0.6, c_wall=4.0, r_air=0.8,
                                            r_wall=2.0, solar_gain=0.05),
                            HVACModel(q_min=-15, q_max=15, cop_cooling=3.8)),
        manifold=m, horizon=24, dt=DT, setpoints={"T_air": 26.0},
        comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX, comfort_margin=0.3,
        comfort_weight=15.0, energy_weight=1.5, smooth_weight=1e-4,
        enforce_comfort_constraints=True,
    )
    return gccm


def thermostat_control(t_air: float, prev: float) -> float:
    """恒温器基线：>24 全冷，<22 停机，中间保持。"""
    if t_air > 24.0:
        return -15.0
    if t_air < 22.0:
        return 0.0
    return prev


def classify(price: float) -> str:
    if price <= PRICE_PRECOOL:
        return "precool"
    if price >= PRICE_PEAK:
        return "peak"
    return "normal"


def run_closed_loop(gccm, provider, steps, mode: str, gccm_24h=None):
    """跑一次闭环。mode: baseline(全恒温器) / gccm(全 GCCM) / 或动作集名(混合)。

    混合模式：在指定动作集的时段用恒温器，其余时段用 GCCM——反事实"去掉该动作集"。
    """
    m = StateManifold(labels=["T_air", "T_wall"], units={}, bounds={},
                      scale={"T_air": 5.0, "T_wall": 5.0})
    state = SystemState(np.array([28.0, 28.0]), list(m.labels))
    prev_control = None
    prev_therm = 0.0
    rows = []
    for k in range(steps):
        t_h = k * DT
        w = provider.get(t_h, 1)[0]
        t_air = float(state.x[0])
        price = w.price
        zone = classify(price)

        if mode == "baseline" or (mode in ("precool", "peak", "normal") and zone == mode):
            u = thermostat_control(t_air, prev_therm)
            ctrl = ControlInput(np.array([u]), ["Q_hvac"])
        else:
            dec = gccm.optimize(state, t_h, prev_control=prev_control,
                                forced_mode="comfort")
            ctrl = dec.control
            prev_control = dec.control
            u = float(ctrl.u[0])

        state = gccm.simulator.step(state, ctrl, w, DT)
        elec = gccm.simulator.hvac.electrical_power(ctrl, w)
        prev_therm = u
        t_air = float(state.x[0])
        rows.append((t_h, t_air, elec, price, zone))
    cost = sum(r[2] * r[3] * DT for r in rows)
    viol = sum(1 for r in rows if r[1] < COMFORT_MIN or r[1] > COMFORT_MAX) / len(rows) * 100
    return {"cost": cost, "viol": viol, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()

    from gccm_be.physics.external import ExternalInputProvider
    import math

    class TOUPriceProvider(ExternalInputProvider):
        """三段分时电价:谷 0.3(22~8h) / 平 0.8 / 峰 1.8(11~18h)。"""
        def __init__(self, dt_h=DT):
            self.dt_h = dt_h
            self.labels = ["T_out", "solar", "occ", "price"]
        def get(self, time_h, horizon=1):
            out = []
            for k in range(horizon):
                t = time_h + k * self.dt_h
                t_out = 30.0 + 5.0 * math.sin(2 * math.pi * t / 24.0 - 1.0)
                hour = t % 24.0
                if 11.0 <= hour < 18.0:
                    price = 1.8
                elif hour >= 22.0 or hour < 8.0:
                    price = 0.3
                else:
                    price = 0.8
                solar = max(0.0, 0.6 * math.sin(math.pi * (hour - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
                occ = 0.5 if 8.0 <= hour < 19.0 else 0.1
                out.append(ExternalInput(
                    np.array([t_out, solar, occ, price]),
                    list(self.labels)))
            return out

    provider = TOUPriceProvider(dt_h=DT)

    print("== 动作级反事实归因（24h, 尖峰电价）==\n")
    gccm = build_engines()
    base = run_closed_loop(gccm, provider, args.steps, "baseline")
    gccm = build_engines()
    full = run_closed_loop(gccm, provider, args.steps, "gccm")
    print(f"baseline: 电费={base['cost']:.2f}元 违温={base['viol']:.1f}%")
    print(f"GCCM 全开: 电费={full['cost']:.2f}元 违温={full['viol']:.1f}%")
    total_saving = base["cost"] - full["cost"]

    zone_results = {}
    for zone_mode in ("precool", "peak", "normal"):
        g2 = build_engines()
        r = run_closed_loop(g2, provider, args.steps, zone_mode)
        zone_results[zone_mode] = r
        saving = base["cost"] - r["cost"]
        print(f"去掉 {zone_mode:>7} 动作集: 电费={r['cost']:.2f}元 "
              f"(vs 全GCCM 贡献 {full['cost'] - r['cost']:+.2f}元)")

    # 反事实归因:动作集贡献 = 全GCCM收益 - 去掉该集后的收益
    print(f"\n总节省: {total_saving:.2f}元")
    for zone_mode in ("precool", "peak", "normal"):
        r = zone_results[zone_mode]
        contribution = (base["cost"] - r["cost"]) - (base["cost"] - full["cost"])
        print(f"  {zone_mode:>7} 动作集独立贡献: {contribution:+.2f}元 "
              f"(占 GCCM 总节省 {contribution/total_saving*100:.0f}%)" if total_saving != 0 else "")
    csv_path = "output/action_attribution.csv"
    import csv, os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "t_air", "elec_kw", "price", "zone"])
        for r in full["rows"]:
            w.writerow(r)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
