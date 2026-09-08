"""双区引擎标准试验台验证：BOPTEST twozone_apartment_hydronic。

验证目标（PILOT_PLAN 路线"两区域配置化支持"的标准试验台背书）：
- TwoZoneRCBuildingModel + 每区独立舒适带（zone_comfort_bounds）+ 每区设定点
- 两区 PRBS 探测 → 用实测地暖功率作控制输入拟合 9 参数灰盒
- 配对对比：恒温器基线（固定设定点） vs GCCM（每区舒适带硬约束 + 限速映射）

区映射：A=Day（起居，20~24°C）、B=Nig（卧室，17~23°C）——地板供暖/热泵场景，
引擎 q>=0 制热方向，u 映射为每区温度设定点（1°C/步限速）。

用法（设备上）：
    LD_LIBRARY_PATH=/opt/miniconda3/lib PYTHONPATH=/opt/gccm-pilot \
    /opt/miniconda3/bin/python /opt/gccm-pilot/examples/boptest_twozone_drill.py \
    --steps 96 --start-day 20 --output /opt/gccm-pilot/output
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import requests
from scipy.optimize import least_squares

from gccm_be.engine import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ExternalInput, SystemState

BASE = "http://localhost:8000"
TC = "twozone_apartment_hydronic"
STEP_SEC = 900.0
CET = 273.15
SET_U = {"A": "thermostatDayZon_oveTsetZon_u", "B": "thermostatNigZon_oveTsetZon_u"}
TZON = {"A": "dayZon_reaTRooAir_y", "B": "nigZon_reaTRooAir_y"}
QHEA = {"A": "dayZon_reaPowFlooHea_y", "B": "nigZon_reaPowFlooHea_y"}
QINT = {"A": "dayZon_reaPowQint_y", "B": "nigZon_reaPowQint_y"}
# 每区独立舒适带（zone_comfort_bounds）：起居更暖、卧室允许更低
BANDS = {"A": (20.0, 24.0), "B": (17.0, 23.0)}
SETPOINTS = {"A": 22.0, "B": 20.0}
LABELS = ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"]
STATES5 = ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]
_WITH_SLAB = False
STATES = STATES5


class TwoZoneProvider(ExternalInputProvider):
    """REST 预报 → 两区外部输入（solar 两区同源，occ 用最近实测内热）。"""

    def __init__(self, tid, step_sec=STEP_SEC):
        self.tid = tid
        self.dt_h = step_sec / 3600.0
        self.labels = list(LABELS)
        self.last_occ = {"A": 0.2, "B": 0.1}

    def get(self, time_h: float, horizon: int = 1) -> list:
        try:
            r = requests.get(
                f"{BASE}/forecast/{self.tid}",
                params={"parameters": "TDryBul,HGloHor,PriceElectricPowerHighlyDynamic",
                        "horizon": int(horizon * self.dt_h * 3600) + 3600},
                timeout=60).json()
            payload = r.get("payload", r)
            t_out = [float(v) - CET for v in payload["TDryBul"]["values"]] if isinstance(payload.get("TDryBul"), dict) else payload["TDryBul"]
            solar = payload["HGloHor"]["values"] if isinstance(payload.get("HGloHor"), dict) else payload["HGloHor"]
            price = payload.get("PriceElectricPowerHighlyDynamic")
            price = (price["values"] if isinstance(price, dict) else price) or None
        except Exception:
            t_out, solar, price = None, None, None
        result = []
        for k in range(horizon):
            tout = (t_out[min(k, len(t_out) - 1)] if t_out else 5.0)
            sol = (solar[min(k, len(solar) - 1)] if solar else 0.0) / 800.0
            pr = (price[min(k, len(price) - 1)] if price else 1.0)
            result.append(ExternalInput(np.array(
                [tout, sol, sol, self.last_occ["A"], self.last_occ["B"], float(pr)]),
                list(self.labels)))
        return result


def init_tid(tid, start_day: int):
    """同 testid 重新初始化（重置到同一起点，避免泄漏新实例）。"""
    requests.put(f"{BASE}/initialize/{tid}",
                 json={"start_time": start_day * 86400, "warmup_period": 43200},
                 timeout=3600)
    requests.put(f"{BASE}/step/{tid}", json={"step": STEP_SEC}, timeout=120)


def select_init(start_day: int) -> str:
    tid = requests.post(f"{BASE}/testcases/{TC}/select", timeout=1800).json()["testid"]
    requests.put(f"{BASE}/initialize/{tid}",
                 json={"start_time": start_day * 86400, "warmup_period": 43200}, timeout=3600)
    requests.put(f"{BASE}/step/{tid}", json={"step": STEP_SEC}, timeout=60)
    return tid


def advance(tid, sp_by_zone: dict, direct: dict | None = None) -> dict:
    """direct 模式（直接执行机构）：每区质量流量 kg/s + 共享供水温度。

    设定点模式经房间恒温器 → 混水 → 地板辐射，延迟以小时计；
    direct 模式直接驱动地板环路，控制权限与模型假设（Q 注入）对齐。
    """
    u = {}
    if direct is None:
        for z, name in SET_U.items():
            u[name] = sp_by_zone[z] + CET
            u[name.replace("_u", "_activate")] = 1
    else:
        for z, name in SET_U.items():
            u[name] = 24.0 + CET                     # 恒温器让路（带内）
            u[name.replace("_u", "_activate")] = 1
        for z in ("A", "B"):
            u[f"hydronicSystem_oveM{'Day' if z == 'A' else 'Nig'}Z_u"] = direct["m"][z]
            u[f"hydronicSystem_oveM{'Day' if z == 'A' else 'Nig'}Z_activate"] = 1
        u["hydronicSystem_oveTHea_u"] = direct["t_sup"] + CET
        u["hydronicSystem_oveTHea_activate"] = 1
        u["hydronicSystem_oveMpumCon_u"] = direct.get("pump", 0.0)
        u["hydronicSystem_oveMpumCon_activate"] = 1
    y = requests.post(f"{BASE}/advance/{tid}", json=u, timeout=120).json()
    return y.get("payload", y)


def read_zone_state(yp, occ=None):
    temps = {z: float(yp.get(TZON[z], 21.0 + CET)) - CET for z in ("A", "B")}
    q_kw = {z: float(yp.get(QHEA[z], 0.0)) / 1000.0 for z in ("A", "B")}
    if occ is not None:
        for z in ("A", "B"):
            occ.last_occ[z] = float(yp.get(QINT[z], 100.0)) / 1000.0
    elec = (float(yp.get("hydronicSystem_reaPeleHeaPum_y", 0.0))
            + float(yp.get("hydronicSystem_reaPPum_y", 0.0))) / 1000.0
    return temps, q_kw, elec


# --- 两区灰盒辨识：PRBS 探测，实测地暖功率为控制输入 ---

def rollout_twozone(theta, x0, q_series, ext_series, dt_h, with_slab=False):
    c_air, c_wall, c_partition, r_air, r_wall_a, r_wall_b, r_partition, sg_a, sg_b = theta[:9]
    ta, twa, tb, twb, tp = x0
    if with_slab:
        slab_a, slab_b = theta[9], theta[10]
    out_a, out_b = [], []
    for (qa, qb), (tout, sol, oa, ob) in zip(q_series, ext_series):
        if with_slab:
            heat_a = (slab_a - ta) / 0.5   # r_slab 与引擎默认一致
            heat_b = (slab_b - tb) / 0.5
        else:
            heat_a, heat_b = qa, qb
        dta = ((twa - ta) / r_air + (tout - ta) / r_wall_a + (tp - ta) / r_partition
               + sg_a * sol + oa + heat_a) / c_air
        dtwa = ((ta - twa) / r_air + (tout - twa) / r_wall_a) / c_wall
        dtb = ((twb - tb) / r_air + (tout - tb) / r_wall_b + (tp - tb) / r_partition
               + sg_b * sol + ob + heat_b) / c_air
        dtwb = ((tb - twb) / r_air + (tout - twb) / r_wall_b) / c_wall
        dtp = ((ta - tp) / r_partition + (tb - tp) / r_partition) / c_partition
        ta += dta * dt_h; twa += dtwa * dt_h; tb += dtb * dt_h
        twb += dtwb * dt_h; tp += dtp * dt_h
        if with_slab:
            slab_a += (qa - heat_a) / 15.0 * dt_h   # c_slab 与引擎默认一致
            slab_b += (qb - heat_b) / 15.0 * dt_h
        out_a.append(ta); out_b.append(tb)
    return np.array(out_a), np.array(out_b)


def identify(tid, steps=192):
    print(f"== 两区辨识（{steps} 步 PRBS 探测，实测地暖功率为控制输入）==", flush=True)
    rng = np.random.default_rng(7)
    temps = {"A": 21.0, "B": 21.0}
    data_t, data_q, data_ext = [], [], []
    for i in range(steps):
        sp = {"A": float(20 + 6 * rng.random()), "B": float(18 + 6 * rng.random())}
        yp = advance(tid, sp)
        temps, q_kw, _ = read_zone_state(yp)
        # 当前室外/太阳：预报第 0 点
        prov = TwoZoneProvider(tid)
        w = prov.get(i * STEP_SEC / 3600.0, 1)[0]
        data_t.append((temps["A"], temps["B"]))
        data_q.append((q_kw["A"], q_kw["B"]))
        data_ext.append((w.w[0], w.w[1], occ_from_w(w, "A"), occ_from_w(w, "B")))
        if (i + 1) % 48 == 0:
            print(f"  probe {i+1}/{steps} tA={temps['A']:.2f} tB={temps['B']:.2f}", flush=True)

    dt_h = STEP_SEC / 3600.0
    ta_meas = np.array([t[0] for t in data_t])
    tb_meas = np.array([t[1] for t in data_t])

    with_slab = _WITH_SLAB
    n_struct = 11 if with_slab else 9
    def resid(theta):
        x0 = (ta_meas[0], ta_meas[0], tb_meas[0], tb_meas[0],
              (ta_meas[0] + tb_meas[0]) / 2)
        oa, ob = rollout_twozone(theta, x0, data_q, data_ext, dt_h, with_slab=with_slab)
        return np.concatenate([oa - ta_meas, ob - tb_meas])

    lo = [0.05, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
    hi = [5.0, 50.0, 50.0, 20.0, 30.0, 30.0, 30.0, 1.0, 1.0]
    x0p = [0.6, 4.0, 3.0, 0.8, 2.0, 2.0, 1.0, 0.05, 0.05]
    if with_slab:
        lo += [15.0, 15.0]     # 蓄热层初温 ≥ 室温下限
        hi += [30.0, 30.0]
        x0p += [21.0, 21.0]
    sol = least_squares(resid, x0=x0p, bounds=(lo, hi), max_nfev=400)
    rmse = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    names = ("c_air", "c_wall", "c_partition", "r_air", "r_wall_a", "r_wall_b",
             "r_partition", "solar_gain_a", "solar_gain_b")
    if with_slab:
        names += ("slab_a0", "slab_b0")
    p = dict(zip(names, sol.x))
    print(f"  两区灰盒 RMSE={rmse:.3f}°C", flush=True)
    print(f"  辨识: {', '.join(f'{k}={v:.3f}' for k, v in p.items())}", flush=True)
    return p, rmse


def occ_from_w(w, zone):
    return w.occupancy(zone=zone)


def make_engine(identified, provider):
    manifold = StateManifold(
        labels=list(STATES),
        units={lab: "°C" for lab in STATES},
        bounds={lab: (5.0, 45.0) for lab in STATES},
        scale={lab: 5.0 for lab in STATES},
    )
    building = TwoZoneRCBuildingModel(
        c_air=identified["c_air"], c_wall=identified["c_wall"],
        c_partition=identified["c_partition"], r_air=identified["r_air"],
        r_wall_a=identified["r_wall_a"], r_wall_b=identified["r_wall_b"],
        r_partition=identified["r_partition"],
        solar_gain_a=identified["solar_gain_a"], solar_gain_b=identified["solar_gain_b"],
        with_slab=_WITH_SLAB)
    hvac = HVACModel(q_min=0.0, q_max=8.0, n_units=2,
                     control_labels=["Q_hvac_A", "Q_hvac_B"])
    return GCCMEngine(
        simulator=Simulator(building, hvac),
        manifold=manifold,
        horizon=32,  # 8h 预测（热惯性大）
        dt=STEP_SEC / 3600.0,
        setpoints={"T_air_A": SETPOINTS["A"], "T_air_B": SETPOINTS["B"]},
        zone_comfort_bounds={"T_air_A": BANDS["A"], "T_air_B": BANDS["B"]},
        comfort_min=min(BANDS["A"][0], BANDS["B"][0]),
        comfort_max=max(BANDS["A"][1], BANDS["B"][1]),
        comfort_margin=0.5, comfort_weight=15.0, energy_weight=1.5, smooth_weight=1e-4,
        enforce_comfort_constraints=True, external_provider=provider,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )


def run_arm(tid, arm: str, steps: int, identified):
    """arm: baseline(固定设定点) / gccm(每区舒适带 + 限速映射)。"""
    provider = TwoZoneProvider(tid)
    engine = make_engine(identified, provider) if arm == "gccm" else None
    prev, prev_sp = None, {"A": 21.0, "B": 21.0}
    slab = {"A": 21.0, "B": 21.0}   # 蓄热层状态不可测，由引擎预测传递
    rows = []
    solve_times = []
    direct = None
    if arm == "gccm-direct":
        # 直接执行机构（BOPTEST 通道规格）：
        #   阀门 oveM{Day,Nig}Z_u ∈ [0,1]（区开度）；泵 oveMpumCon_u ∈ [0,5] kg/s；
        #   供水温度 oveTHea_u ∈ [0,45]°C。u∈[0,8]kW 归一化驱动阀门，
        #   泵流量按总需求 0~2.5 kg/s，供水温度按需求 30~45°C。
        direct = {"m": {"A": 0.0, "B": 0.0}, "t_sup": 35.0, "pump": 0.5}
    for i in range(steps):
        y = advance(tid, prev_sp, direct=direct)
        temps, q_kw, elec = read_zone_state(y, occ=provider)
        if arm == "gccm":
            state = SystemState(
                [temps["A"], temps["A"], temps["B"], temps["B"],
                 (temps["A"] + temps["B"]) / 2, slab["A"], slab["B"]],
                list(engine.manifold.labels))
            t0 = time.perf_counter()
            dec = engine.optimize(state, i * STEP_SEC / 3600.0,
                                  prev_control=prev, forced_mode="comfort")
            solve_times.append(time.perf_counter() - t0)
            prev = dec.control
            if _WITH_SLAB and dec.predicted_next_state is not None and                     dec.predicted_next_state.dim >= 7:
                slab["A"] = float(dec.predicted_next_state.x[5])
                slab["B"] = float(dec.predicted_next_state.x[6])
            u = {"A": float(dec.control.u[0]), "B": float(dec.control.u[1])}
            if direct is not None:
                for z in ("A", "B"):
                    direct["m"][z] = float(np.clip(u[z] / 8.0, 0.0, 1.0))  # 阀门 0~1
                demand = max(0.0, u["A"] + u["B"]) / 16.0                  # 0~1
                direct["pump"] = 0.5 + 2.0 * demand                        # 0.5~2.5 kg/s
                direct["t_sup"] = 35.0 + 10.0 * demand                     # 35~45°C
            # 需求 → 设定点映射：设定点钳制在本区舒适带内（防止模型失配导致过热
            # 正反馈——v4 实测无钳制时设定点漂到 26.6°C，能耗 +82%），1°C/步限速
            for z in ("A", "B"):
                target = temps[z] + 6.0 * max(0.0, u[z])
                target = float(np.clip(target, BANDS[z][0], BANDS[z][1]))
                prev_sp[z] = float(np.clip(target, prev_sp[z] - 1.0, prev_sp[z] + 1.0))
        else:
            prev_sp.update({"A": 21.0, "B": 21.0})
            u = {"A": float("nan"), "B": float("nan")}
            if direct is not None:
                direct["m"].update({"A": 0.0, "B": 0.0})
                direct["pump"] = 0.5
                direct["t_sup"] = 35.0
        rows.append([i * STEP_SEC / 3600.0, arm, temps["A"], temps["B"],
                     elec, prev_sp["A"], prev_sp["B"]])
        if (i + 1) % 48 == 0:
            print(f"  [{arm}] step {i+1}/{steps} tA={temps['A']:.2f} tB={temps['B']:.2f} "
                  f"elec={elec:.3f}kW", flush=True)
    if solve_times:
        print(f"  [{arm}] 平均求解 {np.mean(solve_times)*1000:.0f} ms/步", flush=True)
    return rows


def summarize(rows, label):
    d = np.array([[r[0], r[2], r[3], r[4]] for r in rows], dtype=float)
    dt_h = STEP_SEC / 3600.0
    energy = float(np.sum(d[:, 3]) * dt_h)
    viol_a = float(np.mean((d[:, 1] < BANDS["A"][0]) | (d[:, 1] > BANDS["A"][1])) * 100.0)
    viol_b = float(np.mean((d[:, 2] < BANDS["B"][0]) | (d[:, 2] > BANDS["B"][1])) * 100.0)
    mean_a, mean_b = float(np.mean(d[:, 1])), float(np.mean(d[:, 2]))
    print(f"{label}: 热泵电耗={energy:.2f}kWh 违温 Day={viol_a:.1f}% Nig={viol_b:.1f}% "
          f"均温 Day={mean_a:.2f}°C Nig={mean_b:.2f}°C", flush=True)
    return {"label": label, "energy_kwh": energy, "viol_day": viol_a, "viol_nig": viol_b,
            "mean_day": mean_a, "mean_nig": mean_b}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=96, help="每臂步数（96=24h）")
    parser.add_argument("--start-day", type=int, default=20, help="供暖季起始日")
    parser.add_argument("--output", default="/opt/gccm-pilot/output")
    parser.add_argument("--identify", action="store_true",
                        help="启用 PRBS 灰盒辨识（默认用模型默认参数）")
    parser.add_argument("--with-slab", action="store_true",
                        help="引擎与辨识启用地板蓄热层（7 状态）")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    global _WITH_SLAB
    _WITH_SLAB = bool(args.with_slab)

    tid = select_init(args.start_day)
    if args.identify:
        identified, rmse = identify(tid)
    else:
        d = TwoZoneRCBuildingModel()
        identified = {"c_air": d.c_air, "c_wall": d.c_wall, "c_partition": d.c_partition,
                      "r_air": d.r_air, "r_wall_a": d.r_wall_a, "r_wall_b": d.r_wall_b,
                      "r_partition": d.r_partition, "solar_gain_a": d.solar_gain_a,
                      "solar_gain_b": d.solar_gain_b}
        rmse = float("nan")
        print("== 跳过辨识：使用 TwoZoneRCBuildingModel 默认参数（硬约束+安全链兜底）==",
              flush=True)

    results = []
    all_rows = []
    for k, arm in enumerate(("baseline", "gccm", "gccm-direct")):
        print(f"\n== {arm.upper()} ==", flush=True)
        if k > 0:
            init_tid(tid, args.start_day)  # 同 testid 重置到同一起点（避免泄漏新实例）
        rows = run_arm(tid, arm, args.steps, identified)
        all_rows += rows
        results.append(summarize(rows, arm))

    base, gccm = results
    saving = (base["energy_kwh"] - gccm["energy_kwh"]) / base["energy_kwh"] * 100.0 \
        if base["energy_kwh"] > 0 else float("nan")
    print(f"\n===== 两区演练汇总 =====", flush=True)
    print(f"省电 {saving:.1f}% | 违温 Day {base['viol_day']:.1f}%→{gccm['viol_day']:.1f}% "
          f"Nig {base['viol_nig']:.1f}%→{gccm['viol_nig']:.1f}%", flush=True)

    csv_path = os.path.join(args.output, "boptest_twozone_drill.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "arm", "t_day", "t_nig", "elec_kw", "sp_day", "sp_nig"])
        w.writerows(all_rows)
    with open(os.path.join(args.output, "boptest_twozone_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"identify_rmse": rmse, "arms": results, "saving_pct": saving},
                  f, ensure_ascii=False, indent=2)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
