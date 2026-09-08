"""GCCM（在线辨识 + 真实预测 MPC）vs 规则基线 在 BOPTEST bestest_air 上的公平对比。

最终版 v2：
    1. 初始化到 peak_cool_day（day 289）+ 1 天预热 + highly_dynamic 电价
    2. 探测 8h 辨识 RC 参数（occ 用真实 Occupancy/InternalGains 数据）
    3. GCCM：BOPTEST 真实预测（TDryBul/HGloHor/电价/内热）驱动 MPC
    4. 规则基线：P 控制（同起点）  5. 48h 对比
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/boptest")
from testcase import TestCase  # noqa: E402

from gccm_be import GCCMEngine  # noqa: E402
from gccm_be.geometry.manifold import StateManifold  # noqa: E402
from gccm_be.physics.external import ExternalInputProvider  # noqa: E402
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator, ThreeRCBuildingModel  # noqa: E402
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


def read_state(yp, t_room, u, sp):
    t_room = float(yp.get("zon_reaTRooAir_y", t_room + 273.15)) - 273.15
    tout = float(yp.get("zon_weaSta_reaWeaTDryBul_y", 0.0)) - 273.15
    solar = float(yp.get("zon_weaSta_reaWeaHGloHor_y", 0.0))
    elec = (float(yp.get("fcu_reaPCoo_y", 0.0)) + float(yp.get("fcu_reaPFan_y", 0.0))) / 1000.0
    return t_room, tout, solar, elec


def fetch_forecast(tc, horizon_steps):
    """获取 horizon_steps 步的预测，返回 dict of arrays。"""
    st, msg, payload = tc.get_forecast(FORECAST_POINTS, horizon_steps * STEP_SEC, STEP_SEC)
    if not isinstance(payload, dict) or "TDryBul" not in payload:
        return None
    return payload


def occ_from_payload(payload, k):
    """从预测取第 k 步的内热（kW 当量）。"""
    occ = float(payload["Occupancy[1]"][k]) * 0.12
    gains = float(payload["InternalGainsCon[1]"][k]) / 1000.0
    return occ + gains


class BOPTESTProvider(ExternalInputProvider):
    """BOPTEST 真实预测提供器（occ 加辨识偏置）。"""

    def __init__(self, tc, step_sec, occ_bias=0.0):
        self.tc = tc
        self.step_sec = step_sec
        self.occ_bias = occ_bias
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
            occ = occ_from_payload(payload, k) + self.occ_bias
            result.append(ExternalInput(np.array([tout, solar, occ, price]), list(LABELS)))
        self.cache[key] = result
        return result


def simulate_model(theta, states0, controls, externals, dt_h):
    """按参数 theta=(c_air,c_wall,c_furn,r_air,r_wall,r_furn,solar_gain) 正向仿真 3 状态 RC，返回 T_air 序列。"""
    c_air, c_wall, c_furn, r_air, r_wall, r_furn, solar_gain = theta
    t_air, t_wall, t_furn = states0
    temps = []
    for (u, occ), (tout, solar) in zip(controls, externals):
        d_air = ((t_wall - t_air) / r_air + (t_furn - t_air) / r_furn
                 + (tout - t_air) / r_wall + solar_gain * solar + occ + u) / c_air
        d_wall = ((t_air - t_wall) / r_air + (tout - t_wall) / r_wall) / c_wall
        d_furn = ((t_air - t_furn) / r_furn) / c_furn
        t_air += d_air * dt_h
        t_wall += d_wall * dt_h
        t_furn += d_furn * dt_h
        temps.append(t_air)
    return np.array(temps)


def identify_params(tc, steps=576) -> dict:
    """灰盒辨识：正向仿真 3 状态 RC（含家具热容），least_squares 拟合 T_air（48h PRBS 探测）。"""
    from scipy.optimize import least_squares

    print("== 辨识阶段（48h 探测, 灰盒 3 状态拟合）==", flush=True)
    rng = np.random.default_rng(42)
    data_t, data_u, data_occ, data_tout, data_solar = [], [], [], [], []
    t_room = None
    for i in range(steps):
        hour = (i * STEP_SEC / 3600.0) % 24.0
        if 22.0 <= hour or hour <= 6.0:
            u = 0.0
        else:
            u = -Q_MAX * (1.0 if rng.random() < 0.5 else 0.3)
        sp = u_to_setpoint(u)
        y = tc.advance(cooling_inputs(sp, cooling=(u < -0.01)))
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            break
        t_now, tout, solar, _ = read_state(yp, t_room if t_room is not None else SETPOINT, u, sp)
        fc = fetch_forecast(tc, 1)
        occ = occ_from_payload(fc, 0) if fc else 0.3
        data_t.append(t_now)
        data_u.append(u)
        data_occ.append(occ)
        data_tout.append(tout)
        data_solar.append(solar)
        t_room = t_now
        if (i + 1) % 96 == 0:
            print(f"  probe {i+1}/{steps} t={t_now:.2f}°C", flush=True)

    dt_h = STEP_SEC / 3600.0
    t_meas = np.array(data_t)
    controls = list(zip(data_u, data_occ))
    externals = list(zip(data_tout, data_solar))

    def resid(theta):
        sim = simulate_model(theta, (t_meas[0], t_meas[0], t_meas[0]), controls, externals, dt_h)
        return sim - t_meas

    bounds = ([0.05, 0.5, 0.5, 0.1, 0.1, 0.1, 0.0],
              [5.0, 50.0, 50.0, 20.0, 30.0, 20.0, 1.0])
    sol = least_squares(resid, x0=[0.6, 4.0, 2.0, 0.8, 2.0, 1.0, 0.05],
                        bounds=bounds, max_nfev=300)
    c_air, c_wall, c_furn, r_air, r_wall, r_furn, solar_gain = sol.x
    rmse = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    print(f"  灰盒拟合 RMSE={rmse:.3f}°C", flush=True)
    print(f"  辨识: c_air={c_air:.3f} c_wall={c_wall:.2f} c_furn={c_furn:.2f} "
          f"r_air={r_air:.2f} r_wall={r_wall:.2f} r_furn={r_furn:.2f} "
          f"solar_gain={solar_gain:.5f}", flush=True)
    return {"c_air": c_air, "c_wall": c_wall, "c_furn": c_furn, "r_air": r_air,
            "r_wall": r_wall, "r_furn": r_furn, "solar_gain": solar_gain, "occ_bias": 0.0}


def make_engine(identified: dict, provider=None, use_casadi: bool = False):
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
    return GCCMEngine(
        simulator=Simulator(model, HVACModel(q_min=-Q_MAX, q_max=0.0)),
        manifold=manifold,
        horizon=48,  # 4h 预测
        dt=STEP_SEC / 3600.0,
        setpoints={"T_air": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        comfort_margin=1.0,
        comfort_weight=15.0,
        energy_weight=1.5,
        smooth_weight=1e-5,
        enforce_comfort_constraints=True,
        external_provider=provider,
        use_casadi=use_casadi,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )


def get_price_trajectory(tc, steps):
    """预取整段电价轨迹（highly_dynamic）。"""
    st, msg, payload = tc.get_forecast(["PriceElectricPowerHighlyDynamic"],
                                       steps * STEP_SEC, STEP_SEC)
    if isinstance(payload, dict) and "PriceElectricPowerHighlyDynamic" in payload:
        return [float(v) for v in payload["PriceElectricPowerHighlyDynamic"]]
    return None


def run_controller(tc, steps, controller: str, start_temp: float, identified: dict, prices,
                   use_casadi: bool = False) -> list:
    provider = BOPTESTProvider(tc, STEP_SEC, occ_bias=identified.get("occ_bias", 0.0))
    engine = make_engine(identified, provider, use_casadi) if controller == "gccm" else None
    labels3 = ["T_air", "T_wall", "T_furn"]
    state = SystemState([start_temp, start_temp, start_temp], labels3)
    prev = None
    prev_u = 0.0
    rows = []
    t_room = start_temp
    solve_times = []

    for i in range(steps):
        if controller == "gccm":
            state = SystemState([t_room, t_room, t_room], labels3)
            t0 = time.perf_counter()
            dec = engine.optimize(state, i * STEP_SEC / 3600.0, prev_control=prev, forced_mode="comfort")
            solve_times.append(time.perf_counter() - t0)
            u = float(dec.control.u[0])
            sp = u_to_setpoint(u)
            prev = dec.control
        else:
            # 真实恒温器基线：23°C 设定，2°C 滞回（T>24 全冷，T<22 停机，中间保持）
            if t_room > 24.0:
                u = -Q_MAX
            elif t_room < 22.0:
                u = 0.0
            else:
                u = prev_u
            sp = u_to_setpoint(u)
        prev_u = u

        y = tc.advance(cooling_inputs(sp, cooling=(u < -0.01)))
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            print(f"  [{controller}] scenario end at step {i}", flush=True)
            break
        t_room, tout, solar, elec = read_state(yp, t_room, u, sp)
        price = prices[i] if prices else 0.6
        rows.append([float(i * STEP_SEC / 3600.0), float(t_room), float(u), float(sp),
                     float(elec), float(price), controller])
        if (i + 1) % 96 == 0:
            print(f"  [{controller}] step {i+1}/{steps} t={t_room:.2f}°C outdoor={tout:.1f}°C "
                  f"price={price:.3f} u={u:.2f} elec={elec:.3f}kW", flush=True)
    if solve_times:
        avg_t = float(np.mean(solve_times))
        print(f"  [{controller}] 平均求解耗时: {avg_t*1000:.0f} ms/步 "
              f"(max {max(solve_times)*1000:.0f} ms)", flush=True)
    return rows


def summarize(rows, label):
    d = np.array(rows)
    cost = float(np.sum(d[:, 4].astype(float) * d[:, 5].astype(float) * (STEP_SEC / 3600.0)))
    viol = float(np.mean((d[:, 1].astype(float) > COMFORT_MAX) | (d[:, 1].astype(float) < COMFORT_MIN)) * 100.0)
    peak = float(np.max(d[:, 4].astype(float)))
    avg_t = float(np.mean(d[:, 1].astype(float)))
    print(f"{label}: 电费={cost:.2f}元 违温={viol:.1f}% 峰值={peak:.3f}kW 平均室温={avg_t:.2f}°C", flush=True)
    return {"label": label, "cost": cost, "violation": viol, "peak": peak, "avg_temp": avg_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=576)
    parser.add_argument("--output", default="/opt/boptest/output")
    parser.add_argument("--solver", choices=["scipy", "casadi"], default="scipy")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    results = []
    ident = None

    for controller in ["gccm", "rule"]:
        print(f"\n== {controller.upper()} ==", flush=True)
        tc = TestCase(fmupath=FMU)
        tc.step = STEP_SEC
        tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
        tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                         "temperature_uncertainty": None, "solar_uncertainty": None, "seed": None})
        ident = identify_params(tc)
        # 重新初始化到同一起点 + 预取电价轨迹
        tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
        tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                         "temperature_uncertainty": None, "solar_uncertainty": None, "seed": None})
        prices = get_price_trajectory(tc, args.steps)
        y = tc.advance(cooling_inputs(SETPOINT, cooling=False))
        yp = y[2] if isinstance(y, tuple) else y
        start_temp = float(yp.get("zon_reaTRooAir_y", SETPOINT + 273.15)) - 273.15
        print(f"  起始室温 {start_temp:.2f}°C  电价轨迹 {'OK' if prices else 'FAIL'}", flush=True)
        rows = run_controller(tc, args.steps, controller, start_temp, ident, prices,
                              use_casadi=(args.solver == "casadi"))
        results.append(summarize(rows, "GCCM" if controller == "gccm" else "规则"))
        all_rows += rows

    csv_path = os.path.join(args.output, "boptest_direct_compare.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "t_room", "u_kw", "setpoint_c", "elec_kw", "price", "controller"])
        w.writerows(all_rows)

    s_gccm, s_rule = results[0], results[1]
    if s_rule["cost"] > 0:
        saving = 100 * (s_rule["cost"] - s_gccm["cost"]) / s_rule["cost"]
    else:
        saving = float("nan")
    print(f"\n===== 对比（bestest_air peak_cool_day, {args.steps} 步）=====", flush=True)
    print(f"GCCM(辨识+预测) vs 规则: 省电费 {saving:.1f}% | 违温 {s_rule['violation']:.1f}%→{s_gccm['violation']:.1f}% "
          f"| 峰值 {s_rule['peak']:.3f}→{s_gccm['peak']:.3f} kW", flush=True)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
