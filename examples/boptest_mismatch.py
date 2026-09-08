"""BOPTEST 模型失配压力测试：GCCM 安全降级链 vs 裸 MPC。

场景：故意用失配的默认 RC 模型（对 bestest_air 严重不准）做控制，
对比：
    - GCCM：self_monitor 检测预测误差 → preemptive 降级到反馈安全控制
    - 裸 MPC：无主动安全网（preemptive_feedback=False, safe_control_mode="zero"）
指标：违温率、最高室温（安全）、电费、降级统计、预测误差。

用法（worker 镜像容器内）：
    PYTHONPATH=/opt/boptest python3 /opt/boptest/boptest_mismatch.py --steps 576
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/boptest")
from testcase import TestCase  # noqa: E402

from gccm_be import GCCMEngine  # noqa: E402
from gccm_be.geometry.manifold import StateManifold  # noqa: E402
from gccm_be.physics.external import ExternalInputProvider  # noqa: E402
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator  # noqa: E402
from gccm_be.types import ExternalInput, SystemState  # noqa: E402

FMU = "testcases/bestest_air/models/wrapped.fmu"
STEP_SEC = 300.0
PEAK_DAY = 289 * 86400
WARMUP = 86400
COMFORT_MIN, COMFORT_MAX = 22.0, 27.0
SETPOINT = 24.0
Q_MAX = 1.5
SP_MIN, SP_MAX = 18.0, 26.0
LABELS = ["T_out", "solar", "occ", "price"]
FORECAST_POINTS = ["TDryBul", "HGloHor", "PriceElectricPowerHighlyDynamic",
                   "Occupancy[1]", "InternalGainsCon[1]"]

# 失配模型：默认参数（已知对 bestest_air 严重不准）
WRONG_MODEL = dict(c_air=0.6, c_wall=4.0, r_air=0.8, r_wall=2.0, solar_gain=0.05)


def u_to_setpoint(u: float) -> float:
    frac = min(1.0, abs(u) / Q_MAX)
    return SP_MAX - frac * (SP_MAX - SP_MIN)


def cooling_inputs(sp_c: float, cooling: bool, fan: float = 0.6):
    return {
        "con_oveTSetCoo_u": sp_c + 273.15, "con_oveTSetCoo_activate": 1,
        "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
        "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
        "fcu_oveFan_u": fan if cooling else 0.0, "fcu_oveFan_activate": 1,
    }


def read_state(yp, t_room):
    t_room = float(yp.get("zon_reaTRooAir_y", t_room + 273.15)) - 273.15
    elec = (float(yp.get("fcu_reaPCoo_y", 0.0)) + float(yp.get("fcu_reaPFan_y", 0.0))) / 1000.0
    return t_room, elec


def fetch_forecast(tc, horizon_steps):
    st, msg, payload = tc.get_forecast(FORECAST_POINTS, horizon_steps * STEP_SEC, STEP_SEC)
    if not isinstance(payload, dict) or "TDryBul" not in payload:
        return None
    return payload


def occ_from_payload(payload, k):
    occ = float(payload["Occupancy[1]"][k]) * 0.12
    gains = float(payload["InternalGainsCon[1]"][k]) / 1000.0
    return occ + gains


class BOPTESTProvider(ExternalInputProvider):
    def __init__(self, tc, step_sec):
        self.tc = tc
        self.step_sec = step_sec
        self.cache = {}

    def get(self, time_h: float, horizon: int = 1) -> list:
        key = (int(time_h * 3600.0 / self.step_sec), horizon)
        if key in self.cache:
            return self.cache[key]
        payload = fetch_forecast(self.tc, horizon)
        if payload is None:
            return None
        result = []
        for k in range(horizon):
            tout = float(payload["TDryBul"][k]) - 273.15
            solar = float(payload["HGloHor"][k])
            price = float(payload["PriceElectricPowerHighlyDynamic"][k])
            occ = occ_from_payload(payload, k)
            result.append(ExternalInput(np.array([tout, solar, occ, price]), list(LABELS)))
        self.cache[key] = result
        return result


def make_engine(model_params: dict, preemptive: bool, safe_mode: str, provider):
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    model = RCBuildingModel(**model_params)
    return GCCMEngine(
        simulator=Simulator(model, HVACModel(q_min=-Q_MAX, q_max=0.0)),
        manifold=manifold,
        horizon=48,
        dt=STEP_SEC / 3600.0,
        setpoints={"T_air": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        comfort_margin=0.8,
        comfort_weight=15.0,
        energy_weight=1.5,
        smooth_weight=1e-5,
        enforce_comfort_constraints=True,
        external_provider=provider,
        preemptive_feedback=preemptive,
        safe_control_mode=safe_mode,
        identification_enabled=False,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )


def get_price_trajectory(tc, steps):
    st, msg, payload = tc.get_forecast(["PriceElectricPowerHighlyDynamic"],
                                       steps * STEP_SEC, STEP_SEC)
    if isinstance(payload, dict) and "PriceElectricPowerHighlyDynamic" in payload:
        return [float(v) for v in payload["PriceElectricPowerHighlyDynamic"]]
    return None


def run_arm(tc, steps, arm: str, prices) -> list:
    provider = BOPTESTProvider(tc, STEP_SEC)
    engine = make_engine(WRONG_MODEL,
                         preemptive=(arm == "gccm_safe"),
                         safe_mode=("feedback" if arm == "gccm_safe" else "zero"),
                         provider=provider)
    state = SystemState([SETPOINT, SETPOINT], ["T_air", "T_wall"])
    prev = None
    rows = []
    t_room = SETPOINT
    degrade_count = 0

    for i in range(steps):
        dec = engine.optimize(state, i * STEP_SEC / 3600.0, prev_control=prev, forced_mode="comfort")
        u = float(dec.control.u[0])
        sp = u_to_setpoint(u)

        y = tc.advance(cooling_inputs(sp, cooling=(u < -0.01)))
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            break
        t_next, elec = read_state(yp, t_room)
        # 预测误差喂给自监控（失配模型 → 误差大 → 触发降级）
        pred = getattr(dec, "predicted_next_state", None)
        if pred is not None:
            err = float(abs(pred.x[0] - t_next))
        else:
            err = 5.0  # 求解失败视为大误差
        engine.self_monitor.update(err)
        degraded = bool(getattr(engine, "_degraded", False))
        if degraded:
            degrade_count += 1

        price = prices[i] if prices else 0.6
        rows.append([float(i * STEP_SEC / 3600.0), float(t_room), float(u), float(sp),
                     float(elec), float(price), arm, int(degraded), float(err)])
        t_room = t_next
        state = SystemState([t_room, t_room], ["T_air", "T_wall"])
        prev = dec.control
        if (i + 1) % 96 == 0:
            print(f"  [{arm}] step {i+1}/{steps} t={t_room:.2f}°C u={u:.2f} "
                  f"err={err:.2f} degraded={degraded}", flush=True)

    print(f"  [{arm}] 降级步数: {degrade_count}/{len(rows)}", flush=True)
    return rows


def summarize(rows, label):
    d = np.array(rows)
    cost = float(np.sum(d[:, 4].astype(float) * d[:, 5].astype(float) * (STEP_SEC / 3600.0)))
    viol = float(np.mean((d[:, 1].astype(float) > COMFORT_MAX) | (d[:, 1].astype(float) < COMFORT_MIN)) * 100.0)
    peak_t = float(np.max(d[:, 1].astype(float)))
    avg_t = float(np.mean(d[:, 1].astype(float)))
    max_err = float(np.max(d[:, 8].astype(float)))
    degraded = int(np.mean(d[:, 7].astype(float)) * 100.0)
    print(f"{label}: 电费={cost:.2f}元 违温={viol:.1f}% 最高室温={peak_t:.1f}°C "
          f"平均室温={avg_t:.2f}°C 最大预测误差={max_err:.2f}°C 降级占比={degraded}%", flush=True)
    return {"label": label, "cost": cost, "violation": viol, "peak_t": peak_t,
            "avg_t": avg_t, "max_err": max_err, "degraded": degraded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=576)
    parser.add_argument("--output", default="/opt/boptest/output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    results = []

    for arm in ["gccm_safe", "naive"]:
        print(f"\n== {arm} ==", flush=True)
        tc = TestCase(fmupath=FMU)
        tc.step = STEP_SEC
        tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
        tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                         "temperature_uncertainty": None, "solar_uncertainty": None, "seed": None})
        prices = get_price_trajectory(tc, args.steps)
        rows = run_arm(tc, args.steps, arm, prices)
        results.append(summarize(rows, "GCCM安全链" if arm == "gccm_safe" else "裸MPC"))
        all_rows += rows

    csv_path = os.path.join(args.output, "boptest_mismatch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "t_room", "u_kw", "setpoint_c", "elec_kw", "price", "arm", "degraded", "pred_err"])
        w.writerows(all_rows)

    s_safe, s_naive = results[0], results[1]
    print(f"\n===== 模型失配压力测试（失配默认模型, {args.steps} 步）=====", flush=True)
    print(f"违温: 裸MPC {s_naive['violation']:.1f}% → GCCM安全链 {s_safe['violation']:.1f}%", flush=True)
    print(f"最高室温: 裸MPC {s_naive['peak_t']:.1f}°C → GCCM安全链 {s_safe['peak_t']:.1f}°C", flush=True)
    print(f"电费: 裸MPC {s_naive['cost']:.2f}元 → GCCM安全链 {s_safe['cost']:.2f}元", flush=True)
    print(f"GCCM 降级占比 {s_safe['degraded']}% | 最大预测误差 {s_safe['max_err']:.2f}°C", flush=True)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
