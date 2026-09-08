"""BOPTEST 不确定度压力验证：鲁棒 MPC vs 普通 MPC（weather forecast uncertainty）。

BOPTEST 场景自带 temperature_uncertainty / solar_uncertainty（low/medium/high + seed）：
实际天气会偏离名义预测——量化 GCCM 鲁棒层（robust_scenarios + casadi）的防护价值。

用法（worker 镜像容器内）：
    PYTHONPATH=/opt/boptest python3 /opt/boptest/boptest_uncertainty.py --steps 288
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/boptest")
from boptest_direct import (  # noqa: E402
    BOPTESTProvider, STEP_SEC, fetch_forecast, get_price_trajectory,
    identify_params, occ_from_payload, read_state, u_to_setpoint, cooling_inputs,
)
from gccm_be.physics.models import HVACModel, Simulator, ThreeRCBuildingModel  # noqa: E402
from gccm_be.types import SystemState  # noqa: E402

FMU = "testcases/bestest_air/models/wrapped.fmu"
PEAK_DAY = 289 * 86400
WARMUP = 86400
COMFORT_MIN, COMFORT_MAX = 22.0, 27.0
SETPOINT = 24.0
Q_MAX = 1.5


def make_engine(identified: dict, provider, robust: bool):
    from gccm_be import GCCMEngine
    from gccm_be.geometry.manifold import StateManifold

    manifold = StateManifold(
        labels=["T_air", "T_wall", "T_furn"],
        units={"T_air": "°C", "T_wall": "°C", "T_furn": "°C"},
        bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0), "T_furn": (15.0, 40.0)},
        scale={"T_air": 5.0, "T_wall": 5.0, "T_furn": 5.0},
    )
    model = ThreeRCBuildingModel(c_air=identified.get("c_air", 0.6),
                                 c_wall=identified.get("c_wall", 4.0),
                                 c_furn=identified.get("c_furn", 2.0),
                                 r_air=identified.get("r_air", 0.8),
                                 r_wall=identified.get("r_wall", 2.0),
                                 r_furn=identified.get("r_furn", 1.0),
                                 solar_gain=identified.get("solar_gain", 0.05))
    robust_scenarios = []
    if robust:
        # 悲观场景：保温更差、太阳更强、热容更小（更热的方向）
        pessimistic = Simulator(
            ThreeRCBuildingModel(c_air=identified.get("c_air", 0.6) * 0.8,
                                 c_wall=identified.get("c_wall", 4.0) * 0.9,
                                 c_furn=identified.get("c_furn", 2.0) * 0.8,
                                 r_air=identified.get("r_air", 0.8),
                                 r_wall=identified.get("r_wall", 2.0) * 0.7,
                                 r_furn=identified.get("r_furn", 1.0),
                                 solar_gain=identified.get("solar_gain", 0.05) * 1.5),
            HVACModel(q_min=-Q_MAX, q_max=0.0),
        )
        robust_scenarios = [pessimistic]
    return GCCMEngine(
        simulator=Simulator(model, HVACModel(q_min=-Q_MAX, q_max=0.0)),
        external_provider=provider,
        manifold=manifold,
        horizon=48,
        dt=STEP_SEC / 3600.0,
        setpoints={"T_air": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        comfort_margin=1.0,
        comfort_weight=15.0,
        energy_weight=1.5,
        smooth_weight=1e-5,
        enforce_comfort_constraints=True,
        use_casadi_robust=robust,
        robust_scenarios=robust_scenarios,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )


def run_arm(tc, steps, identified, prices, robust: bool, label: str):
    provider = BOPTESTProvider(tc, STEP_SEC, occ_bias=identified.get("occ_bias", 0.0))
    engine = make_engine(identified, provider, robust)
    labels3 = ["T_air", "T_wall", "T_furn"]
    t_room = SETPOINT
    prev = None
    rows = []
    for i in range(steps):
        state = SystemState([t_room, t_room, t_room], labels3)
        dec = engine.optimize(state, i * STEP_SEC / 3600.0, prev_control=prev, forced_mode="comfort")
        u = float(dec.control.u[0])
        sp = u_to_setpoint(u)
        prev = dec.control
        y = tc.advance(cooling_inputs(sp, cooling=(u < -0.01)))
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            break
        t_room, tout, solar, elec = read_state(yp, t_room, u, sp)
        price = prices[i] if prices else 0.6
        rows.append([float(i * STEP_SEC / 3600.0), float(t_room), float(u), float(sp),
                     float(elec), float(price), label])
        if (i + 1) % 96 == 0:
            print(f"  [{label}] step {i+1}/{steps} t={t_room:.2f}°C outdoor={tout:.1f}°C "
                  f"elec={elec:.3f}kW", flush=True)
    return rows


def summarize(rows, label):
    d = np.array(rows)
    cost = float(np.sum(d[:, 4].astype(float) * d[:, 5].astype(float) * (STEP_SEC / 3600.0)))
    viol = float(np.mean((d[:, 1].astype(float) > COMFORT_MAX) | (d[:, 1].astype(float) < COMFORT_MIN)) * 100.0)
    peak_t = float(np.max(d[:, 1].astype(float)))
    print(f"{label}: 电费={cost:.2f}元 违温={viol:.1f}% 最高室温={peak_t:.2f}°C", flush=True)
    return {"label": label, "cost": cost, "violation": viol, "peak_t": peak_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--output", default="/opt/boptest/output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    results = []

    for level in ["low", "high"]:
        print(f"\n===== 不确定度等级: {level} =====", flush=True)
        tc = TestCase(fmupath=FMU)
        tc.step = STEP_SEC
        tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
        tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                         "temperature_uncertainty": level, "solar_uncertainty": level,
                         "seed": 7})
        identified = identify_params(tc, steps=288)
        prices = get_price_trajectory(tc, args.steps)

        for robust, lbl in [(False, "plain"), (True, "robust")]:
            tc2 = TestCase(fmupath=FMU)
            tc2.step = STEP_SEC
            tc2.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
            tc2.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                              "temperature_uncertainty": level, "solar_uncertainty": level,
                              "seed": 7})
            rows = run_arm(tc2, args.steps, identified, prices, robust, lbl)
            res = summarize(rows, f"[{level}] {lbl}")
            res["level"] = level
            results.append(res)
            all_rows += rows

    csv_path = os.path.join(args.output, "boptest_uncertainty.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "t_room", "u_kw", "setpoint_c", "elec_kw", "price", "arm"])
        w.writerows(all_rows)

    print(f"\n===== 不确定度验证汇总 =====", flush=True)
    for r in results:
        print(f"  [{r['level']}] {r['label']}: 电费={r['cost']:.2f}元 违温={r['violation']:.1f}% "
              f"最高室温={r['peak_t']:.2f}°C", flush=True)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    from testcase import TestCase  # noqa: E402
    main()
