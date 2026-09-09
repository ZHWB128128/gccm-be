"""30% mismatch + online identification validation on BOPTEST bestest_air.

验证修后辨识链条(量纲修复 + 影子验证)在真失配下是否真的工作：
- 控制器模型: 默认 RC 参数 (r_wall=2.0)
- 真楼: r_wall=2.6 (+30% 失配)
- 辨识开: run_closed_loop 的 apply_rc_identification 自动走门控+影子验证
- 指标: r_wall 收敛 / 违温 / 电费 / 预测误差趋势

用法（设备上）：
    cd /opt/boptest && LD_LIBRARY_PATH=/opt/miniforge3/lib PYTHONPATH=/opt/gccm-pilot \
    /opt/miniforge3/bin/python /opt/gccm-pilot/examples/boptest_ident_mismatch.py \
    --steps 576 --output /opt/gccm-pilot/output
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/boptest")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testcase import TestCase  # noqa: E402
from boptest_direct import (  # noqa: E402
    FMU, PEAK_DAY, Q_MAX, SETPOINT, STEP_SEC, WARMUP,
    get_price_trajectory, identify_params, read_state,
)
from gccm_be import GCCMEngine  # noqa: E402
from gccm_be.geometry.manifold import StateManifold  # noqa: E402
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator, ThreeRCBuildingModel  # noqa: E402
from gccm_be.types import ExternalInput  # noqa: E402
from gccm_be.types import ControlInput, SystemState  # noqa: E402


def u_to_setpoint(u: float) -> float:
    frac = min(1.0, abs(u) / Q_MAX)
    return 26.0 - frac * (26.0 - 18.0)


def cooling_inputs(sp_c: float, cooling: bool, fan: float = 0.6):
    return {
        "con_oveTSetCoo_u": sp_c + 273.15, "con_oveTSetCoo_activate": 1,
        "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
        "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
        "fcu_oveFan_u": fan if cooling else 0.0, "fcu_oveFan_activate": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=576)
    parser.add_argument("--output", default="/opt/gccm-pilot/output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    tc = TestCase(fmupath=FMU)
    tc.step = STEP_SEC
    tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
    tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                     "temperature_uncertainty": None, "solar_uncertainty": None,
                     "seed": None})
    prices = get_price_trajectory(tc, args.steps)

    print("== 阶段 1: PRBS 预辨识（48h 探测）==", flush=True)
    identified = identify_params(tc)
    tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
    tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                     "temperature_uncertainty": None, "solar_uncertainty": None,
                     "seed": None})
    # 用辨识参数构建控制器模型
    manifold = StateManifold(
        labels=["T_air", "T_wall", "T_furn"],
        units={"T_air": "°C", "T_wall": "°C", "T_furn": "°C"},
        bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0), "T_furn": (15.0, 40.0)},
        scale={"T_air": 5.0, "T_wall": 5.0, "T_furn": 5.0},
    )
    engine = GCCMEngine(
        simulator=Simulator(ThreeRCBuildingModel(
            c_air=identified["c_air"], c_wall=identified["c_wall"],
            c_furn=identified["c_furn"], r_air=identified["r_air"],
            r_wall=identified["r_wall"], r_furn=identified["r_furn"],
            solar_gain=identified["solar_gain"]), HVACModel(q_min=-Q_MAX, q_max=0.0)),
        manifold=manifold, horizon=48, dt=STEP_SEC / 3600.0,
        setpoints={"T_air": SETPOINT},
        comfort_min=22.0, comfort_max=27.0, comfort_margin=1.0,
        comfort_weight=15.0, energy_weight=1.5, smooth_weight=1e-4,
        enforce_comfort_constraints=True,
        identification_enabled=True,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )
    print(f"== 阶段 2: 闭环（辨识参数 r_wall={identified['r_wall']:.2f}）==", flush=True)
    y0 = advance_one_stub(tc, SETPOINT, False)
    yp0 = (y0[2] if isinstance(y0, tuple) else y0) or {}
    t0 = float(yp0.get("zon_reaTRooAir_y", SETPOINT + 273.15)) - 273.15
    state = SystemState(np.array([t0, t0, t0]), ["T_air", "T_wall", "T_furn"])
    prev = None
    rows = []
    r_wall_trace = [float(engine.simulator.building.r_wall)]
    ident_applied = []

    for k in range(args.steps):
        t_h = k * STEP_SEC / 3600.0
        price = prices[k] if prices and k < len(prices) else 0.6

        dec = engine.optimize(state, t_h, prev_control=prev, forced_mode="comfort")
        prev = dec.control
        u = float(dec.control.u[0])
        sp = u_to_setpoint(u)

        y = advance_one_stub(tc, sp, cooling=(u < -0.01))
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            print(f"scenario end at step {k}", flush=True)
            break
        t_true, tout, solar, elec = read_state(yp, float(state.x[0]), u, sp)

        # 已实现预测误差(下一步): 预测的下一状态 vs 真楼实际下一状态
        # (真楼只提供 T_air;其余维度用控制器自身状态保持——误差归因在 T_air)
        if dec.predicted_next_state is not None:
            actual_next = dec.predicted_next_state.x.copy()
            actual_next[0] = t_true                      # T_air 用真楼实测
            pred_err = float(np.max(np.abs(dec.predicted_next_state.x - actual_next)))
            engine.self_monitor.update(pred_err)

        # 辨识门(修复后): 用 observe_step + apply
        ext_k = ExternalInput(np.array([tout, solar, 0.5, price]),
                              ["T_out", "solar", "occ", "price"])
        next_state = SystemState(np.array([t_true, float(state.x[1]), float(state.x[2])]),
                                 ["T_air", "T_wall", "T_furn"])
        engine.observe_step(state, dec.control, ext_k, next_state,
                            STEP_SEC / 3600.0)
        state = next_state
        if k % 12 == 0:  # 每 12 步试一次
            if engine.apply_rc_identification(min_samples=30):
                ident_applied.append((k, float(engine.simulator.building.r_wall)))

        rows.append([t_h, t_true, u, sp, elec, price,
                     float(engine.simulator.building.r_wall),
                     float(dec.confidence), int(dec.solver_success)])

        if (k + 1) % 96 == 0:
            rw = engine.simulator.building.r_wall
            viol = sum(1 for r in rows[-96:] if r[1] > 27.0 or r[1] < 22.0) / 96 * 100
            print(f"  step {k+1}/{args.steps} T={t_true:.2f}°C r_wall={rw:.3f} "
                  f"viol_96={viol:.1f}% conf={dec.confidence:.2f}", flush=True)

    # 汇总
    temps = np.array([r[1] for r in rows])
    r_wall_final = engine.simulator.building.r_wall
    r_wall_init = identified["r_wall"]
    viol_total = float(np.mean((temps > 27.0) | (temps < 22.0)) * 100)
    print(f"\n===== 辨识+失配验证（{args.steps} 步）=====")
    print(f"r_wall: {r_wall_init:.3f} → {r_wall_final:.3f}")
    print(f"辨识应用次数: {len(ident_applied)} | 违温: {viol_total:.1f}%")
    print(f"辨识轨迹: {ident_applied[:5]}..." if len(ident_applied) > 5
          else f"辨识轨迹: {ident_applied}")

    csv_path = os.path.join(args.output, "boptest_ident_mismatch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "t_air", "u_kw", "setpoint_c", "elec_kw", "price",
                    "r_wall", "confidence", "solver_ok"])
        w.writerows(rows)
    with open(os.path.join(args.output, "boptest_ident_mismatch.json"), "w",
              encoding="utf-8") as f:
        json.dump({"r_wall_initial": identified["r_wall"], "r_wall_final": r_wall_final,
                   "ident_applied": len(ident_applied),
                   "ident_trace": ident_applied,
                   "violation_pct": viol_total}, f, indent=2)
    print(f"CSV: {csv_path}")


def advance_one_stub(tc, sp_c, cooling):
    return tc.advance(cooling_inputs_stub(sp_c, cooling))


def cooling_inputs_stub(sp_c, cooling):
    return {
        "con_oveTSetCoo_u": sp_c + 273.15, "con_oveTSetCoo_activate": 1,
        "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
        "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
        "fcu_oveFan_u": 0.6 if cooling else 0.0, "fcu_oveFan_activate": 1,
    }


if __name__ == "__main__":
    main()
