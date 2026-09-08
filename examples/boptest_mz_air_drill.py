"""双区引擎空气系统验证：BOPTEST multizone_office_simple_air 的 Sou/Nor 两区。

为什么选这个测试台：与 twozone_apartment_hydronic 的教训互补——风系统 FCU 响应
快、每区冷却设定点是直接执行机构（boptest_multizone_peak.py 已验证该 REST 模式），
模型结构假设（空气注入 Q）与执行器匹配。

配置：A=Sou（南向，得热大，带 22~26°C 舒适带）、B=Nor（北向，24~26 带内更松）；
共享 AHU 风量上限 0.4（两区竞争冷量）。基线 = 固定设定点 24°C；
GCCM = TwoZone 引擎（默认参数 + 每区舒适带硬约束 + 带内钳制限速映射）。

用法（设备上）：
    LD_LIBRARY_PATH=/opt/miniconda3/lib PYTHONPATH=/opt/gccm-pilot \
    /opt/miniconda3/bin/python /opt/gccm-pilot/examples/boptest_mz_air_drill.py \
    --steps 96 --start-day 210 --output /opt/gccm-pilot/output
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import requests

from gccm_be.engine import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ExternalInput, SystemState

BASE = "http://localhost:8000"
TC = "multizone_office_simple_air"
STEP_SEC = 900.0
CET = 273.15
ZONES = {"A": "Sou", "B": "Nor"}
TZON = {z: f"hvac_reaZon{r}_TZon_y" for z, r in ZONES.items()}
COOL_SET = {z: f"hvac_oveZonSup{r}_TZonCooSet_u" for z, r in ZONES.items()}
HEAT_SET_VAL = 15.0
FAN_CAP = 0.4
BANDS = {"A": (22.0, 26.0), "B": (22.0, 26.0)}
SP0 = 24.0
Q_MAX = 6.0   # 每区制冷额定（映射增益基准）
STATES5 = ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]
LABELS = ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"]


class MZProvider(ExternalInputProvider):
    """REST 预报 → 两区外部输入（solar 两区同源；occ 用办公时刻表近似）。"""

    def __init__(self, tid):
        self.tid = tid
        self.labels = list(LABELS)
        self.cache = {}

    def get(self, time_h: float, horizon: int = 1) -> list:
        key = int(time_h * 3600.0 / STEP_SEC)
        if key in self.cache:
            return self.cache[key]
        try:
            r = requests.get(f"{BASE}/forecast/{self.tid}",
                             params={"parameters": "TDryBul,HGloHor,PriceElectricPowerHighlyDynamic",
                                     "horizon": int(horizon * STEP_SEC) + 3600},
                             timeout=60).json()
            payload = r.get("payload", r)
            def series(name):
                v = payload.get(name)
                return v["values"] if isinstance(v, dict) else v
            t_out = series("TDryBul") or [25.0]
            solar = series("HGloHor") or [0.0]
            price = series("PriceElectricPowerHighlyDynamic") or [1.0]
        except Exception:
            t_out, solar, price = [25.0], [0.0], [1.0]
        result = []
        for k in range(horizon):
            hour = (time_h + k * STEP_SEC / 3600.0) % 24.0
            occ = 0.6 if 7.0 <= hour < 19.0 else 0.1
            tout = t_out[min(k, len(t_out) - 1)] - CET
            sol = max(0.0, solar[min(k, len(solar) - 1)]) / 800.0
            pr = price[min(k, len(price) - 1)]
            result.append(ExternalInput(np.array([tout, sol, sol, occ, occ, float(pr)]),
                                        list(self.labels)))
        self.cache[key] = result
        return result


def select_init(start_day: int) -> str:
    tid = requests.post(f"{BASE}/testcases/{TC}/select", timeout=1800).json()["testid"]
    requests.put(f"{BASE}/initialize/{tid}",
                 json={"start_time": start_day * 86400, "warmup_period": 43200},
                 timeout=3600)
    requests.put(f"{BASE}/step/{tid}", json={"step": STEP_SEC}, timeout=120)
    return tid


def advance(tid, sp_by_zone: dict) -> dict:
    u = {"hvac_oveAhu_yFan_u": FAN_CAP, "hvac_oveAhu_yFan_activate": 1}
    for z, r in ZONES.items():
        u[COOL_SET[z]] = sp_by_zone[z] + CET
        u[COOL_SET[z].replace("_u", "_activate")] = 1
        u[f"hvac_oveZonSup{r}_TZonHeaSet_u"] = HEAT_SET_VAL + CET
        u[f"hvac_oveZonSup{r}_TZonHeaSet_activate"] = 1
    y = requests.post(f"{BASE}/advance/{tid}", json=u, timeout=180).json()
    return y.get("payload", y)


def make_engine(provider):
    d = TwoZoneRCBuildingModel()  # 默认参数（空气注入结构与 FCU 匹配）
    manifold = StateManifold(labels=list(STATES5),
                             units={lab: "°C" for lab in STATES5},
                             bounds={lab: (5.0, 45.0) for lab in STATES5},
                             scale={lab: 5.0 for lab in STATES5})
    building = TwoZoneRCBuildingModel(
        c_air=d.c_air, c_wall=d.c_wall, c_partition=d.c_partition, r_air=d.r_air,
        r_wall_a=d.r_wall_a, r_wall_b=d.r_wall_b, r_partition=d.r_partition,
        solar_gain_a=0.08, solar_gain_b=0.02)
    return GCCMEngine(
        simulator=Simulator(building, HVACModel(q_min=-Q_MAX, q_max=0.0, n_units=2,
                                                control_labels=["Q_hvac_A", "Q_hvac_B"])),
        manifold=manifold, horizon=32, dt=STEP_SEC / 3600.0,
        setpoints={"T_air_A": SP0, "T_air_B": SP0},
        zone_comfort_bounds={"T_air_A": BANDS["A"], "T_air_B": BANDS["B"]},
        comfort_min=min(BANDS["A"][0], BANDS["B"][0]),
        comfort_max=max(BANDS["A"][1], BANDS["B"][1]),
        comfort_margin=0.5, comfort_weight=15.0, energy_weight=1.5, smooth_weight=1e-4,
        enforce_comfort_constraints=True, external_provider=provider,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )


def run_arm(tid, arm: str, steps: int):
    provider = MZProvider(tid)
    engine = make_engine(provider) if arm == "gccm" else None
    prev, prev_sp = None, {z: SP0 for z in ("A", "B")}
    rows, solve_times = [], []
    for i in range(steps):
        y = advance(tid, prev_sp)
        temps = {z: float(y.get(TZON[z], SP0 + CET)) - CET for z in ("A", "B")}
        if arm == "gccm":
            state = SystemState([temps["A"], temps["A"], temps["B"], temps["B"],
                                 (temps["A"] + temps["B"]) / 2], list(STATES5))
            t0 = time.perf_counter()
            dec = engine.optimize(state, i * STEP_SEC / 3600.0,
                                  prev_control=prev, forced_mode="energy")
            solve_times.append(time.perf_counter() - t0)
            prev = dec.control
            # 制冷需求（u≤0）→ 设定点：需求越强设定点越低；钳制舒适带 + 1°C/步限速
            for z in ("A", "B"):
                u_cool = min(0.0, float(dec.control.u[0 if z == "A" else 1]))
                target = temps[z] + (Q_MAX * 0.6) * (u_cool / Q_MAX)   # 0 ~ -3.6°C
                target = float(np.clip(target, BANDS[z][0], BANDS[z][1]))
                prev_sp[z] = float(np.clip(target, prev_sp[z] - 1.0, prev_sp[z] + 1.0))
        else:
            prev_sp.update({z: SP0 for z in ("A", "B")})
        kpi_rows = y
        for z in ("A", "B"):
            pass
        rows.append([i * STEP_SEC / 3600.0, arm, temps["A"], temps["B"],
                     prev_sp["A"], prev_sp["B"]])
        if (i + 1) % 48 == 0:
            print(f"  [{arm}] step {i+1}/{steps} tA={temps['A']:.2f} tB={temps['B']:.2f} "
                  f"spA={prev_sp['A']:.2f} spB={prev_sp['B']:.2f}", flush=True)
    if solve_times:
        print(f"  [{arm}] 平均求解 {np.mean(solve_times)*1000:.0f} ms/步", flush=True)
    return rows


def summarize(rows, label):
    d = np.array([[r[0], r[2], r[3]] for r in rows], dtype=float)
    viol = {z: float(np.mean((d[:, i] < BANDS[z][0]) | (d[:, i] > BANDS[z][1])) * 100.0)
            for i, z in enumerate(("A", "B"))}
    peak_zone = float(np.max(d[:, 1:]))
    print(f"{label}: 违温 Day/A(Sou)={viol['A']:.1f}% B(Nor)={viol['B']:.1f}% "
          f"| 峰值区温={peak_zone:.2f}°C | 均温 A={np.mean(d[:,1]):.2f} B={np.mean(d[:,2]):.2f}",
          flush=True)
    return {"label": label, "viol_a": viol["A"], "viol_b": viol["B"],
            "peak_zone": peak_zone,
            "mean_a": float(np.mean(d[:, 1])), "mean_b": float(np.mean(d[:, 2]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--start-day", type=int, default=210)
    parser.add_argument("--output", default="/opt/gccm-pilot/output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    tid = select_init(args.start_day)
    results, all_rows = [], []
    for k, arm in enumerate(("baseline", "gccm")):
        print(f"\n== {arm.upper()} ==", flush=True)
        if k > 0:
            requests.put(f"{BASE}/initialize/{tid}",
                         json={"start_time": args.start_day * 86400, "warmup_period": 43200},
                         timeout=3600)
            requests.put(f"{BASE}/step/{tid}", json={"step": STEP_SEC}, timeout=120)
        rows = run_arm(tid, arm, args.steps)
        all_rows += rows
        results.append(summarize(rows, arm))
    # KPI（BOPTEST 官方口径：tdis/ener）
    try:
        kpi = requests.get(f"{BASE}/kpi/{tid}", timeout=60).json().get("payload", {})
        print(f"BOPTEST KPI: tdis={kpi.get('tdis_tot')} ener={kpi.get('ener_tot')}", flush=True)
    except Exception:
        pass
    requests.put(f"{BASE}/stop/{tid}", timeout=60)

    csv_path = os.path.join(args.output, "boptest_mz_air_drill.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "arm", "t_sou", "t_nor", "sp_sou", "sp_nor"])
        w.writerows(all_rows)
    with open(os.path.join(args.output, "boptest_mz_air_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
