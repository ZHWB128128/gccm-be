"""两区域 GCCM vs 严格舒适 PID 对比实验。

状态: [T_air_A, T_wall_A, T_air_B, T_wall_B, T_partition]
控制: [Q_hvac_A, Q_hvac_B]
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ControlInput, ExternalInput, SystemState

STEP_H = 0.25
STEPS = 96
Q_MAX = 8.0
SETPOINT = 26.0
COMFORT_MIN = 25.0
COMFORT_MAX = 27.0

# 与单区域公平基线一致的 PID 参数
PID_KP = 2.0
PID_KI = 2.0
PID_KD = 0.1
PID_INTEGRAL_LIMIT = 5.0


class TwoZoneProvider(ExternalInputProvider):
    def __init__(self, dt_h: float = STEP_H) -> None:
        self.dt_h = dt_h

    def get(self, time_h: float, horizon: int = 1) -> List[ExternalInput]:
        result = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            if 6.0 <= hour <= 18.0:
                solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0))
            else:
                solar = 0.0
            # A 区南向得热多，B 区北向得热少
            solar_a = solar * 1.2
            solar_b = solar * 0.3
            occ_a = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            occ_b = 0.6 if 8.0 <= hour <= 18.0 else 0.2
            if hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5
            result.append(ExternalInput(
                np.array([t_out, solar_a, solar_b, occ_a, occ_b, price]),
                ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"],
            ))
        return result


def make_two_zone_simulator() -> Simulator:
    building = TwoZoneRCBuildingModel()
    hvac = HVACModel(
        q_min=-Q_MAX,
        q_max=Q_MAX,
        n_units=2,
        control_labels=["Q_hvac_A", "Q_hvac_B"],
    )
    return Simulator(building, hvac)


def run_strict_pid(sim: Simulator, provider: TwoZoneProvider):
    state = SystemState(np.full(5, 28.0), [
        "T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition",
    ])
    t = 0.0
    integral_a = 0.0
    integral_b = 0.0
    prev_err_a = 0.0
    prev_err_b = 0.0
    times, t_a, t_b, p_elec, price = [], [], [], [], []
    q_a, q_b = [], []

    for _ in range(STEPS):
        w = provider.get(t, 1)[0]
        err_a = state.x[0] - SETPOINT
        err_b = state.x[2] - SETPOINT
        integral_a = float(np.clip(integral_a + err_a * STEP_H, -PID_INTEGRAL_LIMIT, PID_INTEGRAL_LIMIT))
        integral_b = float(np.clip(integral_b + err_b * STEP_H, -PID_INTEGRAL_LIMIT, PID_INTEGRAL_LIMIT))
        der_a = (err_a - prev_err_a) / STEP_H
        der_b = (err_b - prev_err_b) / STEP_H
        cool_a = float(np.clip(PID_KP * err_a + PID_KI * integral_a + PID_KD * der_a, 0.0, Q_MAX))
        cool_b = float(np.clip(PID_KP * err_b + PID_KI * integral_b + PID_KD * der_b, 0.0, Q_MAX))
        control = ControlInput(np.array([-cool_a, -cool_b]), ["Q_hvac_A", "Q_hvac_B"])
        state = sim.step(state, control, w, STEP_H)

        times.append(t)
        t_a.append(state.x[0])
        t_b.append(state.x[2])
        q_a.append(cool_a)
        q_b.append(cool_b)
        p_elec.append(sim.hvac.electrical_power(control))
        price.append(w.w[5])
        prev_err_a = err_a
        prev_err_b = err_b
        t += STEP_H

    return {
        "times": np.array(times),
        "t_a": np.array(t_a),
        "t_b": np.array(t_b),
        "p_elec": np.array(p_elec),
        "price": np.array(price),
        "q_a": np.array(q_a),
        "q_b": np.array(q_b),
    }


def run_gccm(provider: TwoZoneProvider, horizon: int = 32):
    sim = make_two_zone_simulator()
    manifold = StateManifold(
        labels=["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"],
        units={l: "°C" for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        bounds={l: (15.0, 40.0) for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        scale={l: 5.0 for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=horizon,
        dt=STEP_H,
        setpoints={"T_air_A": SETPOINT, "T_air_B": SETPOINT},
        comfort_band=0.0,
        comfort_min=COMFORT_MIN,
        comfort_max=26.8,   # 安全裕度，保证实际不超过 27
        below_comfort_penalty=1.0,
        peak_energy_penalty=1.0,
        comfort_weight=200.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        enforce_comfort_constraints=False,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )

    state = SystemState(np.full(5, 28.0), manifold.labels)
    t = 0.0
    prev_control = None
    times, t_a, t_b, p_elec, price = [], [], [], [], []
    q_a, q_b = [], []
    solve_times = []

    for _ in range(STEPS):
        start = time.perf_counter()
        decision = engine.optimize(state, t, prev_control=prev_control, forced_mode="comfort")
        solve_times.append(time.perf_counter() - start)
        w = provider.get(t, 1)[0]
        state = sim.step(state, decision.control, w, STEP_H)

        times.append(t)
        t_a.append(state.x[0])
        t_b.append(state.x[2])
        q_a.append(decision.control.u[0])
        q_b.append(decision.control.u[1])
        p_elec.append(sim.hvac.electrical_power(decision.control))
        price.append(w.w[5])
        prev_control = decision.control
        t += STEP_H
        if (len(solve_times)) % 16 == 0:
            print(f"  GCCM 进度 {len(solve_times)}/{STEPS}，"
                  f"近16次平均求解 {np.mean(solve_times[-16:]):.2f}s", flush=True)

    return {
        "times": np.array(times),
        "t_a": np.array(t_a),
        "t_b": np.array(t_b),
        "p_elec": np.array(p_elec),
        "price": np.array(price),
        "q_a": np.array(q_a),
        "q_b": np.array(q_b),
        "solve_times": solve_times,
    }


def metrics(name, r):
    cost = float(np.sum(r["p_elec"] * r["price"] * STEP_H))
    viol_a = float(np.mean((r["t_a"] > COMFORT_MAX) | (r["t_a"] < COMFORT_MIN)) * 100.0)
    viol_b = float(np.mean((r["t_b"] > COMFORT_MAX) | (r["t_b"] < COMFORT_MIN)) * 100.0)
    peak = float(np.max(r["p_elec"]))
    avg_time = float(np.mean(r.get("solve_times", [0.0])))
    print(f"{name:<18}{cost:>10.2f}{viol_a:>10.1f}{viol_b:>10.1f}{peak:>12.2f}{avg_time:>12.3f}")
    return {"name": name, "cost": cost, "viol_a": viol_a, "viol_b": viol_b, "peak": peak, "avg_time": avg_time}


def main() -> None:
    provider = TwoZoneProvider()
    sim = make_two_zone_simulator()

    print("运行严格舒适 PID（两区域独立控制）...")
    pid = run_strict_pid(sim, provider)
    print("运行 GCCM（两区域联合优化）...")
    gccm = run_gccm(provider, horizon=32)

    print("\n" + "=" * 76)
    print(f"{'方法':<18}{'总电费(元)':>10}{'A区违规%':>10}{'B区违规%':>10}{'峰值功率(kW)':>12}{'平均求解(s)':>12}")
    print("-" * 76)
    m_pid = metrics("严格舒适 PID", pid)
    m_gccm = metrics("GCCM", gccm)
    print("=" * 76)

    saving = (m_pid["cost"] - m_gccm["cost"]) / m_pid["cost"] * 100.0
    print(f"\nGCCM 相比严格舒适 PID 节省电费: {saving:.1f}%")

    # 协同行为简单判断
    peak_mask = (gccm["times"] >= 11.0) & (gccm["times"] < 18.0)
    night_mask = gccm["times"] < 8.0
    print(f"\nGCCM 峰时总平均功率: {np.mean(gccm['p_elec'][peak_mask]):.3f} kW")
    print(f"GCCM 谷时总平均功率: {np.mean(gccm['p_elec'][night_mask]):.3f} kW")
    print(f"严格PID 峰时总平均功率: {np.mean(pid['p_elec'][peak_mask]):.3f} kW")
    print(f"严格PID 谷时总平均功率: {np.mean(pid['p_elec'][night_mask]):.3f} kW")


if __name__ == "__main__":
    main()
