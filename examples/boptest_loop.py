"""GCCM vs 规则基线 在 BOPTEST（EnergyPlus 标准测试台）上的公平对比。

策略（避开慢/卡死的场景预热）：
    1. 手动初始化：7 月中旬起点 + warmup=0（秒级初始化）
    2. 预热丢弃段：固定 26°C 设定跑 4h（48 步），让建筑在 7 月天气下升温到位
    3. 每个控制器（GCCM / 规则）独立重建 testid + 相同预热段 → 同起点公平对比
    4. 输出电费/违温/峰值对比 + CSV

用法：PYTHONPATH=. python3 examples/boptest_loop.py --steps 96
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import urllib.request

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import SystemState

API = os.environ.get("BOPTEST_API", "http://127.0.0.1:8000")
STEP_SEC = 300
SUMMER_START = 200 * 86400  # 7 月中旬
WARMUP_STEPS = 48  # 4h 预热丢弃段
COMFORT_MIN, COMFORT_MAX = 22.0, 27.0
SETPOINT = 24.0
Q_MAX = 8.0
SP_MIN, SP_MAX = 18.0, 26.0


def api(method, path, data=None, retries=10):
    last = None
    for attempt in range(retries):
        try:
            body = json.dumps(data).encode() if data is not None else None
            req = urllib.request.Request(API + path, data=body, method=method,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
            time.sleep(4 * (attempt + 1))
    raise last


def u_to_setpoint(u: float) -> float:
    frac = min(1.0, abs(u) / Q_MAX)
    return SP_MAX - frac * (SP_MAX - SP_MIN)


def make_engine():
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    return GCCMEngine(
        simulator=Simulator(RCBuildingModel(), HVACModel(q_min=-Q_MAX, q_max=0.0)),
        manifold=manifold,
        horizon=24,
        dt=STEP_SEC / 3600.0,
        setpoints={"T_air": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        comfort_margin=0.8,
        comfort_weight=5.0,
        energy_weight=1.5,
        smooth_weight=1e-5,
        enforce_comfort_constraints=True,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )


def warmup_phase(tid, steps=WARMUP_STEPS):
    """固定 26°C 设定跑预热段，让建筑升温到 7 月状态。返回末端室温。"""
    t_room = SETPOINT
    inputs = {
        "con_oveTSetCoo_u": SETPOINT + 273.15,
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
        "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
        "fcu_oveFan_u": 0.6, "fcu_oveFan_activate": 1,
    }
    for _ in range(steps):
        adv = api("POST", f"/advance/{tid}", inputs)
        p = adv.get("payload", {})
        t_room = float(p.get("zon_reaTRooAir_y", t_room + 273.15)) - 273.15
    print(f"  预热完成, 末端室温 {t_room:.2f}°C")
    return t_room


def run_controller(tid, steps, controller: str, start_temp: float) -> list:
    engine = make_engine() if controller == "gccm" else None
    state = SystemState([start_temp, start_temp], ["T_air", "T_wall"])
    prev = None
    rows = []
    t_room = start_temp
    price_now = 1.0

    for i in range(steps):
        if controller == "gccm":
            state = SystemState([t_room, t_room], ["T_air", "T_wall"])
            dec = engine.optimize(state, i * STEP_SEC / 3600.0, prev_control=prev, forced_mode="comfort")
            u = float(dec.control.u[0])
            sp = u_to_setpoint(u)
            prev = dec.control
        else:
            err = t_room - SETPOINT
            u = -float(np.clip(50.0 * err, 0.0, Q_MAX)) if err > 0.3 else 0.0
            sp = u_to_setpoint(u)

        inputs = {
            "con_oveTSetCoo_u": sp + 273.15, "con_oveTSetCoo_activate": 1,
            "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
            "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
            "fcu_oveFan_u": 0.6, "fcu_oveFan_activate": 1,
        }
        adv = api("POST", f"/advance/{tid}", inputs)
        p = adv.get("payload", {})
        t_room = float(p.get("zon_reaTRooAir_y", t_room + 273.15)) - 273.15
        elec = float(p.get("fcu_reaPCoo_y", 0.0)) + float(p.get("fcu_reaPFan_y", 0.0))
        price = float(p.get("Price", price_now))
        rows.append([float(i * STEP_SEC / 3600.0), float(t_room), float(u), float(sp),
                     float(elec), float(price), controller])
        price_now = price
        if (i + 1) % 24 == 0:
            print(f"  [{controller}] step {i+1}/{steps} t={t_room:.2f}°C elec={elec:.2f}kW")
    return rows


def summarize(rows, label):
    d = np.array(rows)
    cost = float(np.sum(d[:, 4].astype(float) * d[:, 5].astype(float) * (STEP_SEC / 3600.0)))
    viol = float(np.mean((d[:, 1].astype(float) > COMFORT_MAX) | (d[:, 1].astype(float) < COMFORT_MIN)) * 100.0)
    peak = float(np.max(d[:, 4].astype(float)))
    avg_t = float(np.mean(d[:, 1].astype(float)))
    print(f"{label}: 电费={cost:.2f}元 违温={viol:.1f}% 峰值={peak:.2f}kW 平均室温={avg_t:.2f}°C")
    return {"label": label, "cost": cost, "violation": viol, "peak": peak, "avg_temp": avg_t}


def fresh_test(testcase):
    """async select（立即返回）+ 轮询状态到 Running。"""
    sel = api("POST", f"/testcases/{testcase}/select-async", {}, retries=5)
    tid = sel["testid"]
    # 轮询直到 Running（worker 串行处理，每个任务几分钟）
    for _ in range(120):
        st = api("GET", f"/status/{tid}", retries=3)
        if isinstance(st, dict):
            payload = st.get("payload", "")
        else:
            payload = st
        if payload == "Running":
            break
        time.sleep(3)
    api("PUT", f"/initialize/{tid}", {"start_time": SUMMER_START, "warmup_period": 0})
    api("PUT", f"/step/{tid}", {"step": STEP_SEC})
    return tid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testcase", default="bestest_air")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--output", default="output/boptest")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    results = []

    for controller in ["gccm", "rule"]:
        print(f"\n== {controller.upper()} ==")
        tid = fresh_test(args.testcase)
        print(f"  testid={tid}")
        start_temp = warmup_phase(tid)
        rows = run_controller(tid, args.steps, controller, start_temp)
        results.append(summarize(rows, "GCCM" if controller == "gccm" else "规则"))
        all_rows += rows

    csv_path = os.path.join(args.output, "boptest_compare.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "t_room", "u_kw", "setpoint_c", "elec_kw", "price", "controller"])
        w.writerows(all_rows)

    s_gccm, s_rule = results[0], results[1]
    if s_rule["cost"] > 0:
        saving = 100 * (s_rule["cost"] - s_gccm["cost"]) / s_rule["cost"]
    else:
        saving = float("nan")
    print(f"\n===== 对比（{args.testcase}, 7 月中, {args.steps} 步）=====")
    print(f"GCCM vs 规则: 省电费 {saving:.1f}% | 违温 {s_rule['violation']:.1f}%→{s_gccm['violation']:.1f}% "
          f"| 峰值 {s_rule['peak']:.1f}→{s_gccm['peak']:.1f} kW")
    print(f"CSV: {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    main()
