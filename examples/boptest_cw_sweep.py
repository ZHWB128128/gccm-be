"""comfort_weight 参数扫描：BOPTEST bestest_air 峰值制冷日 24h 窗口。

目的：ABAB 演练显示 GCCM 违温与基线持平（容量受限场景），扫描 comfort_weight
寻找"守温优先"配置。每个权重从同一起点重置 FMU（同一 24h 窗口，公平对比），
另跑一次恒温器基线作参照。

用法（设备上）：
    cd /opt/boptest && LD_LIBRARY_PATH=/opt/miniconda3/lib PYTHONPATH=/opt/gccm-pilot \
    /opt/miniconda3/bin/python /opt/gccm-pilot/examples/boptest_cw_sweep.py \
    --weights 15,50,150,500 --output /opt/gccm-pilot/output
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/boptest")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testcase import TestCase  # noqa: E402
from boptest_direct import (  # noqa: E402
    COMFORT_MAX, COMFORT_MIN, FMU, PEAK_DAY, Q_MAX, SETPOINT, STEP_SEC, WARMUP,
    BOPTESTProvider, get_price_trajectory, identify_params, read_state, u_to_setpoint,
)
from gccm_be.geometry.manifold import StateManifold  # noqa: E402
from gccm_be.physics.models import HVACModel, Simulator, ThreeRCBuildingModel  # noqa: E402
from gccm_be.engine import GCCMEngine  # noqa: E402
from gccm_be.types import SystemState  # noqa: E402


def make_engine_cw(identified: dict, provider, comfort_weight: float,
                   enforce_comfort: bool = True) -> GCCMEngine:
    """与 boptest_direct.make_engine 相同，仅 comfort_weight 可调。"""
    manifold = StateManifold(
        labels=["T_air", "T_wall", "T_furn"],
        units={lab: "°C" for lab in ("T_air", "T_wall", "T_furn")},
        bounds={lab: (15.0, 40.0) for lab in ("T_air", "T_wall", "T_furn")},
        scale={lab: 5.0 for lab in ("T_air", "T_wall", "T_furn")},
    )
    model = ThreeRCBuildingModel(
        c_air=identified.get("c_air", 0.6), c_wall=identified.get("c_wall", 4.0),
        c_furn=identified.get("c_furn", 2.0), r_air=identified.get("r_air", 0.8),
        r_wall=identified.get("r_wall", 2.0), r_furn=identified.get("r_furn", 1.0),
        solar_gain=identified.get("solar_gain", 0.05))
    return GCCMEngine(
        simulator=Simulator(model, HVACModel(q_min=-Q_MAX, q_max=0.0)),
        manifold=manifold, horizon=48, dt=STEP_SEC / 3600.0,
        setpoints={"T_air": SETPOINT},
        comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX, comfort_margin=1.0,
        comfort_weight=comfort_weight, energy_weight=1.5, smooth_weight=1e-5,
        enforce_comfort_constraints=enforce_comfort, external_provider=provider,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )


def init_fmu(tc):
    tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
    tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                     "temperature_uncertainty": None, "solar_uncertainty": None, "seed": None})


def advance_one(tc, sp_c: float, cooling: bool, fan: float = 0.6):
    y = tc.advance({
        "con_oveTSetCoo_u": sp_c + 273.15, "con_oveTSetCoo_activate": 1,
        "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
        "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
        "fcu_oveFan_u": fan if cooling else 0.0, "fcu_oveFan_activate": 1,
    })
    return y[2] if isinstance(y, tuple) else y


def run_window(tc, steps, controller, identified, prices, comfort_weight=None,
               enforce_comfort=True):
    """同一 24h 窗口跑一种控制器，返回 (rows, solve_times)。rows 含 cw 标签。"""
    provider = BOPTESTProvider(tc, STEP_SEC)
    engine = None
    if controller == "gccm":
        engine = make_engine_cw(identified, provider, comfort_weight, enforce_comfort)
    labels3 = ["T_air", "T_wall", "T_furn"]
    y = advance_one(tc, SETPOINT, cooling=False)
    yp = y if y is not None else {}
    t_room = float(yp.get("zon_reaTRooAir_y", SETPOINT + 273.15)) - 273.15
    state = SystemState([t_room, t_room, t_room], labels3)
    prev, prev_u = None, 0.0
    rows, solve_times = [], []

    for i in range(steps):
        if controller == "gccm":
            state = SystemState([t_room, t_room, t_room], labels3)
            t0 = time.perf_counter()
            dec = engine.optimize(state, i * STEP_SEC / 3600.0,
                                  prev_control=prev, forced_mode="comfort")
            solve_times.append(time.perf_counter() - t0)
            u = float(dec.control.u[0])
            prev = dec.control
        else:
            if t_room > 24.0:
                u = -Q_MAX
            elif t_room < 22.0:
                u = 0.0
            else:
                u = prev_u
        sp = u_to_setpoint(u)
        y = tc.advance({
            "con_oveTSetCoo_u": sp + 273.15, "con_oveTSetCoo_activate": 1,
            "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
            "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
            "fcu_oveFan_u": 0.6 if u < -0.01 else 0.0, "fcu_oveFan_activate": 1,
        })
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            print(f"  [{controller}] scenario end at step {i}", flush=True)
            break
        t_room, tout, solar, elec = read_state(yp, t_room, u, sp)
        price = prices[i] if prices and i < len(prices) else 0.6
        rows.append([comfort_weight if comfort_weight is not None else -1.0,
                     float(i * STEP_SEC / 3600.0), float(t_room), float(u), float(sp),
                     float(elec), float(price), float(tout)])
        if (i + 1) % 96 == 0:
            print(f"  [cw={comfort_weight}/{controller}] step {i+1}/{steps} t={t_room:.2f}°C "
                  f"out={tout:.1f}°C u={u:.2f} elec={elec:.3f}kW", flush=True)
    return rows, solve_times


def summarize(rows, label):
    d = np.array([[r[1], r[2], r[5], r[6], r[7]] for r in rows], dtype=float)
    dt_h = STEP_SEC / 3600.0
    time_h, t_room, elec, price, t_out = (d[:, i] for i in range(5))
    cost = float(np.sum(elec * price * dt_h))
    over = float(np.mean(t_room > COMFORT_MAX) * 100.0)
    under = float(np.mean(t_room < COMFORT_MIN) * 100.0)
    viol = over + under
    peak = float(np.max(elec))
    mask = (time_h % 24.0 >= 11.0) & (time_h % 24.0 < 18.0)
    peak_avg = float(np.mean(elec[mask])) if mask.any() else 0.0
    print(f"{label}: 电费={cost:.2f}元 违温={viol:.1f}% (超27={over:.1f}% 低于22={under:.1f}%) "
          f"峰值={peak:.3f}kW 峰时均功率={peak_avg:.3f}kW 室外均温={float(np.mean(t_out)):.2f}°C",
          flush=True)
    return {"label": label, "cost": cost, "violation": viol, "over_27": over,
            "under_22": under, "peak": peak, "peak_avg_power": peak_avg,
            "t_out_mean": float(np.mean(t_out))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="15,50,150,500")
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--output", default="/opt/gccm-pilot/output")
    parser.add_argument("--soft", action="store_true",
                        help="关闭硬舒适约束（观察 comfort_weight 软惩罚的真实作用）")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    weights = [float(x) for x in args.weights.split(",")]

    # 辨识一次（48h 探测），之后每个配置重置 FMU 到同一起点
    tc = TestCase(fmupath=FMU)
    tc.step = STEP_SEC
    init_fmu(tc)
    identified = identify_params(tc)
    prices = get_price_trajectory(tc, args.steps + 1)

    all_rows, summaries = [], []
    # 恒温器基线（参照）
    print("\n== 基线：恒温器 ==", flush=True)
    init_fmu(tc)
    rows, _ = run_window(tc, args.steps, "rule", identified, prices)
    all_rows += rows
    summaries.append(summarize(rows, "rule-baseline"))

    # comfort_weight 扫描
    for cw in weights:
        print(f"\n== GCCM comfort_weight={cw} ==", flush=True)
        init_fmu(tc)
        rows, solve_times = run_window(tc, args.steps, "gccm", identified, prices,
                                       comfort_weight=cw,
                                       enforce_comfort=not args.soft)
        all_rows += rows
        s = summarize(rows, f"cw={cw:g}{'-soft' if args.soft else ''}")
        s["avg_solve_ms"] = float(np.mean(solve_times) * 1000) if solve_times else 0.0
        summaries.append(s)
        print(f"  平均求解 {s['avg_solve_ms']:.0f} ms/步", flush=True)
        # 每个权重跑完立即落盘，防中途异常丢数据
        dump_csv(args.output, all_rows)
        dump_json(args.output, summaries)

    dump_csv(args.output, all_rows)
    dump_json(args.output, summaries)
    print("\n===== 扫描完成 =====", flush=True)


def dump_csv(output, all_rows):
    path = os.path.join(output, "boptest_cw_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comfort_weight", "time_h", "t_room", "u_kw", "setpoint_c",
                    "elec_kw", "price", "t_out"])
        w.writerows(all_rows)


def dump_json(output, summaries):
    with open(os.path.join(output, "boptest_cw_sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
