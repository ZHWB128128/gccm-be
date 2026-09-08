"""ABAB M&V 演练：在 BOPTEST bestest_air 上按 docs/PILOT_PLAN.md §5 协议跑完整四期交替。

设计（试验台压缩版；真楼每期 5~7 天，这里每期 24h=288 步）：
    A1(恒温器基线 24h) → B1(GCCM 24h) → A2(基线 24h) → B2(GCCM 24h)
    同一连续 FMU 仿真交替切换控制器，天气连续演化，配对期对比。

指标（§5.4）：电费 Σ elec×price×dt；违温率（22~27°C 带，与既往仿真口径一致）；
峰时(11~18h)平均功率；每期室外均温（天气可比性）。节省 = 配对期 (A-B)/A。

用法（设备上）：
    cd /opt/boptest && LD_LIBRARY_PATH=/opt/miniconda3/lib PYTHONPATH=/opt/gccm-pilot \
    /opt/miniconda3/bin/python /opt/gccm-pilot/examples/boptest_abab_drill.py \
    --period-steps 288 --output /opt/gccm-pilot/output
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/boptest")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testcase import TestCase  # noqa: E402
from boptest_direct import (  # noqa: E402
    COMFORT_MAX, COMFORT_MIN, FMU, PEAK_DAY, Q_MAX, SETPOINT, STEP_SEC, WARMUP,
    BOPTESTProvider, get_price_trajectory, identify_params, make_engine,
    read_state, u_to_setpoint,
)
from gccm_be.types import SystemState  # noqa: E402

PEAK_HOURS = (11.0, 18.0)


class RateLimiter:
    """§5.3：设定温度每周期变化 ≤1°C，避免压缩机频繁启停。"""

    def __init__(self, max_delta: float = 1.0):
        self.max_delta = max_delta
        self.prev: float | None = None

    def __call__(self, sp: float) -> float:
        if self.prev is None:
            self.prev = sp
            return sp
        sp = float(np.clip(sp, self.prev - self.max_delta, self.prev + self.max_delta))
        self.prev = sp
        return sp


def run_period(tc, period: str, controller: str, steps: int, start_temp: float,
               identified: dict, prices, start_idx: int, rate: RateLimiter | None,
               prev_u_holder: dict) -> list:
    """跑一个期间，返回行列表；期间内控制器固定。"""
    rows = []
    t_room = start_temp
    labels3 = ["T_air", "T_wall", "T_furn"]
    engine = None
    if controller == "gccm":
        provider = BOPTESTProvider(tc, STEP_SEC)
        engine = make_engine(identified, provider)
        prev = None
    solve_times = []

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
            # 原控制 = 恒温器：>24 全冷，<22 停机，中间保持（与既往基线一致）
            if t_room > 24.0:
                u = -Q_MAX
            elif t_room < 22.0:
                u = 0.0
            else:
                u = prev_u_holder.get("u", 0.0)
        prev_u_holder["u"] = u

        sp = u_to_setpoint(u)
        if rate is not None:
            sp = rate(sp)

        y = tc.advance(cooling_inputs_adapted(sp, cooling=(u < -0.01)))
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            print(f"  [{period}] scenario end at step {i}", flush=True)
            break
        t_room, tout, solar, elec = read_state(yp, t_room, u, sp)
        price = prices[start_idx + i] if prices and start_idx + i < len(prices) else 0.6
        rows.append([float((start_idx + i) * STEP_SEC / 3600.0), period, controller,
                     float(t_room), float(u), float(sp), float(elec), float(price),
                     float(tout)])
        if (i + 1) % 96 == 0:
            print(f"  [{period}/{controller}] step {i+1}/{steps} t={t_room:.2f}°C "
                  f"out={tout:.1f}°C u={u:.2f} elec={elec:.3f}kW", flush=True)

    if solve_times:
        print(f"  [{period}] 平均求解 {np.mean(solve_times)*1000:.0f} ms/步 "
              f"(max {max(solve_times)*1000:.0f} ms)", flush=True)
    return rows


def cooling_inputs_adapted(sp_c: float, cooling: bool, fan: float = 0.6):
    # 与 boptest_direct.cooling_inputs 相同的控制输入映射
    return {
        "con_oveTSetCoo_u": sp_c + 273.15, "con_oveTSetCoo_activate": 1,
        "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
        "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
        "fcu_oveFan_u": fan if cooling else 0.0, "fcu_oveFan_activate": 1,
    }


def summarize_period(rows, period):
    # 行结构: [time_h, period, controller, t_room, u_kw, setpoint, elec, price, t_out]
    num = np.array([[r[0], r[3], r[4], r[5], r[6], r[7], r[8]] for r in rows], dtype=float)
    dt_h = STEP_SEC / 3600.0
    time_h, t_room, u, _sp, elec, price, t_out = (num[:, i] for i in range(7))
    cost = float(np.sum(elec * price * dt_h))
    viol = float(np.mean((t_room > COMFORT_MAX) | (t_room < COMFORT_MIN)) * 100.0)
    peak = float(np.max(elec))
    peak_mask = (time_h % 24.0 >= PEAK_HOURS[0]) & (time_h % 24.0 < PEAK_HOURS[1])
    peak_avg_p = float(np.mean(elec[peak_mask])) if peak_mask.any() else 0.0
    t_out_mean = float(np.mean(t_out))
    avg_t = float(np.mean(t_room))
    print(f"{period}({rows[0][2]}): 电费={cost:.2f}元 违温={viol:.1f}% 峰值={peak:.3f}kW "
          f"峰时均功率={peak_avg_p:.3f}kW 室外均温={t_out_mean:.2f}°C 平均室温={avg_t:.2f}°C",
          flush=True)
    return {"period": period, "controller": rows[0][2], "cost": cost, "violation": viol,
            "peak": peak, "peak_avg_power": peak_avg_p, "t_out_mean": t_out_mean,
            "avg_temp": avg_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period-steps", type=int, default=288, help="每期步数（288=24h）")
    parser.add_argument("--output", default="/opt/gccm-pilot/output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    total_steps = args.period_steps * 4
    print(f"== ABAB 演练：每期 {args.period_steps} 步（{args.period_steps*STEP_SEC/3600:.0f}h），"
          f"共 {total_steps*STEP_SEC/3600:.0f}h ==", flush=True)

    # 1) 初始化 + 辨识（48h 探测，随后重置到同一起点）
    tc = TestCase(fmupath=FMU)
    tc.step = STEP_SEC
    tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
    tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                     "temperature_uncertainty": None, "solar_uncertainty": None, "seed": None})
    identified = identify_params(tc)

    # 2) 重置到同一起点，预取全程电价
    tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
    tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                     "temperature_uncertainty": None, "solar_uncertainty": None, "seed": None})
    prices = get_price_trajectory(tc, total_steps + 1)
    y = tc.advance(cooling_inputs_adapted(SETPOINT, cooling=False))
    yp = y[2] if isinstance(y, tuple) else y
    start_temp = float(yp.get("zon_reaTRooAir_y", SETPOINT + 273.15)) - 273.15
    print(f"  起始室温 {start_temp:.2f}°C  电价轨迹 {'OK' if prices else 'FAIL'}", flush=True)

    # 3) ABAB 四期连续交替（同一 FMU 实例，天气连续）
    periods = [("A1", "rule"), ("B1", "gccm"), ("A2", "rule"), ("B2", "gccm")]
    all_rows = []
    summaries = []
    start_idx = 0
    rate = RateLimiter(1.0)
    prev_u_holder = {}
    t_room = start_temp
    for period, controller in periods:
        print(f"\n== 期间 {period}（{controller}）==", flush=True)
        # GCCM 期重置限速器与保持状态；基线期的恒温器状态跨期保持（原控制连续运行）
        if controller == "gccm":
            rate = RateLimiter(1.0)
        rows = run_period(tc, period, controller, args.period_steps, t_room,
                          identified, prices, start_idx,
                          rate if controller == "gccm" else None, prev_u_holder)
        if not rows:
            print(f"期间 {period} 无数据，中止", flush=True)
            break
        t_room = rows[-1][3]
        start_idx += len(rows)
        all_rows += rows
        summaries.append(summarize_period(rows, period))

    # 4) CSV（逐步数据凭证）
    csv_path = os.path.join(args.output, "boptest_abab_drill.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "period", "controller", "t_room", "u_kw", "setpoint_c",
                    "elec_kw", "price", "t_out"])
        w.writerows(all_rows)

    # 5) 配对对比（§5.2 配对日法）
    by_period = {s["period"]: s for s in summaries}
    print("\n===== ABAB 演练汇总 =====", flush=True)
    savings = []
    for a, b in (("A1", "B1"), ("A2", "B2")):
        if a in by_period and b in by_period:
            sa, sb = by_period[a], by_period[b]
            saving = (sa["cost"] - sb["cost"]) / sa["cost"] * 100.0 if sa["cost"] > 0 else float("nan")
            savings.append(saving)
            print(f"{a}→{b}: 省电费 {saving:.1f}% | 违温 {sa['violation']:.1f}%→{sb['violation']:.1f}% | "
                  f"室外均温 {sa['t_out_mean']:.2f}→{sb['t_out_mean']:.2f}°C", flush=True)
    if savings:
        print(f"配对平均节省: {np.nanmean(savings):.1f}%", flush=True)
    print(f"CSV: {csv_path}", flush=True)

    # JSON 摘要（供后续分析/图表）
    import json
    with open(os.path.join(args.output, "boptest_abab_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"periods": summaries, "pair_savings_pct": savings}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
