"""两区域公平基线对比：GCCM vs 集中式舒适优先 MPC vs 分散式独立 MPC。

基线 A：集中式舒适优先 MPC（energy_weight=0，只优化舒适）
基线 B：分散式独立 MPC（每个房间独立优化，忽略隔墙耦合）
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ControlInput, ExternalInput, SystemState

from two_zone_compare import (
    COMFORT_MAX,
    COMFORT_MIN,
    Q_MAX,
    STEP_H,
    STEPS,
    TwoZoneProvider,
    make_two_zone_simulator,
    run_gccm,
)

# 与 GCCM 当前配置一致
HORIZON = 32
COMFORT_WEIGHT = 200.0
BASELINE_A_COMFORT_WEIGHT = 1000.0
ENERGY_WEIGHT_GCCM = 0.1
ENERGY_WEIGHT_BASELINE_A = 0.0
SMOOTH_WEIGHT = 0.1
INTERNAL_COMFORT_MAX = 26.8
BASELINE_A_COMFORT_MAX = 26.8


def run_baseline_a(provider: TwoZoneProvider):
    """集中式舒适优先 MPC：目标中电费权重为 0。"""
    sim = make_two_zone_simulator()
    manifold = StateManifold(
        labels=["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"],
        units={l: "°C" for l in manifold_labels()},
        bounds={l: (15.0, 40.0) for l in manifold_labels()},
        scale={l: 5.0 for l in manifold_labels()},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=HORIZON,
        dt=STEP_H,
        setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
        comfort_band=0.0,
        comfort_min=COMFORT_MIN,
        comfort_max=BASELINE_A_COMFORT_MAX,
        below_comfort_penalty=1.0,
        peak_energy_penalty=1.0,
        comfort_weight=BASELINE_A_COMFORT_WEIGHT,
        energy_weight=ENERGY_WEIGHT_BASELINE_A,
        smooth_weight=SMOOTH_WEIGHT,
        enforce_comfort_constraints=False,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )

    state = SystemState(np.full(5, 28.0), manifold.labels)
    return _run_engine_loop(engine, sim, provider, state, "基线A")


def run_baseline_b(provider: TwoZoneProvider):
    """分散式独立 MPC：A/B 各用一个忽略耦合的单区域 MPC。"""
    full_sim = make_two_zone_simulator()

    # 两个单区域控制器
    engine_a, engine_b = _make_single_zone_engines(provider)

    state = SystemState(np.full(5, 28.0), full_sim.building.state_labels)
    t = 0.0
    prev_a = None
    prev_b = None
    times, t_a, t_b, p_elec, price = [], [], [], [], []
    solve_times = []

    for _ in range(STEPS):
        w = provider.get(t, 1)[0]

        local_a = SystemState([state.x[0], state.x[1]], ["T_air", "T_wall"])
        local_b = SystemState([state.x[2], state.x[3]], ["T_air", "T_wall"])
        ext_a = ExternalInput(np.array([w.w[0], w.w[1], w.w[3], w.w[5]]),
                              ["T_out", "solar", "occ", "price"])
        ext_b = ExternalInput(np.array([w.w[0], w.w[2], w.w[4], w.w[5]]),
                              ["T_out", "solar", "occ", "price"])

        start = time.perf_counter()
        dec_a = engine_a.optimize(local_a, t, prev_control=prev_a, forced_mode="comfort")
        dec_b = engine_b.optimize(local_b, t, prev_control=prev_b, forced_mode="comfort")
        solve_times.append(time.perf_counter() - start)

        control = ControlInput(np.array([dec_a.control.u[0], dec_b.control.u[0]]),
                               ["Q_hvac_A", "Q_hvac_B"])
        state = full_sim.step(state, control, w, STEP_H)

        times.append(t)
        t_a.append(state.x[0])
        t_b.append(state.x[2])
        p_elec.append(full_sim.hvac.electrical_power(control))
        price.append(w.w[5])
        prev_a = dec_a.control
        prev_b = dec_b.control
        t += STEP_H

    return {
        "times": np.array(times),
        "t_a": np.array(t_a),
        "t_b": np.array(t_b),
        "p_elec": np.array(p_elec),
        "price": np.array(price),
        "solve_times": solve_times,
    }


def _make_single_zone_engines(provider: TwoZoneProvider):
    def make_engine(solar_idx: int, occ_idx: int):
        sim = Simulator(RCBuildingModel(), HVACModel(q_min=-Q_MAX, q_max=Q_MAX))
        manifold = StateManifold(
            labels=["T_air", "T_wall"],
            units={"T_air": "°C", "T_wall": "°C"},
            bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0)},
            scale={"T_air": 5.0, "T_wall": 5.0},
        )
        # 单区域控制器只使用自己的太阳辐射和人员热源
        class LocalProvider:
            def get(self, time_h: float, horizon: int = 1):
                out = []
                for k in range(horizon):
                    full = provider.get(time_h + k * STEP_H, 1)[0]
                    out.append(ExternalInput(
                        np.array([full.w[0], full.w[solar_idx], full.w[occ_idx], full.w[5]]),
                        ["T_out", "solar", "occ", "price"],
                    ))
                return out

        return GCCMEngine(
            simulator=sim,
            external_provider=LocalProvider(),  # type: ignore[arg-type]
            manifold=manifold,
            horizon=HORIZON,
            dt=STEP_H,
            setpoints={"T_air": 26.0},
            comfort_band=0.0,
            comfort_min=COMFORT_MIN,
            comfort_max=INTERNAL_COMFORT_MAX,
            below_comfort_penalty=1.0,
            peak_energy_penalty=1.0,
            comfort_weight=COMFORT_WEIGHT,
            energy_weight=ENERGY_WEIGHT_GCCM,
            smooth_weight=SMOOTH_WEIGHT,
            enforce_comfort_constraints=False,
            solver_options={"maxiter": 60, "ftol": 1e-5, "maxls": 20},
        )

    return make_engine(1, 3), make_engine(2, 4)


def _run_engine_loop(engine: GCCMEngine, sim: Simulator, provider: TwoZoneProvider,
                     state: SystemState, label: str):
    t = 0.0
    prev_control = None
    times, t_a, t_b, p_elec, price = [], [], [], [], []
    solve_times = []

    for i in range(STEPS):
        start = time.perf_counter()
        decision = engine.optimize(state, t, prev_control=prev_control, forced_mode="comfort")
        solve_times.append(time.perf_counter() - start)
        w = provider.get(t, 1)[0]
        state = sim.step(state, decision.control, w, STEP_H)

        times.append(t)
        t_a.append(state.x[0])
        t_b.append(state.x[2])
        p_elec.append(sim.hvac.electrical_power(decision.control))
        price.append(w.w[5])
        prev_control = decision.control
        t += STEP_H
        if (i + 1) % 16 == 0:
            print(f"  {label} 进度 {i + 1}/{STEPS}，近16次平均求解 {np.mean(solve_times[-16:]):.2f}s", flush=True)

    return {
        "times": np.array(times),
        "t_a": np.array(t_a),
        "t_b": np.array(t_b),
        "p_elec": np.array(p_elec),
        "price": np.array(price),
        "solve_times": solve_times,
    }


def manifold_labels():
    return ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]


def metrics(name, r):
    cost = float(np.sum(r["p_elec"] * r["price"] * STEP_H))
    viol_a = float(np.mean((r["t_a"] > COMFORT_MAX) | (r["t_a"] < COMFORT_MIN)) * 100.0)
    viol_b = float(np.mean((r["t_b"] > COMFORT_MAX) | (r["t_b"] < COMFORT_MIN)) * 100.0)
    peak = float(np.max(r["p_elec"]))
    avg_time = float(np.mean(r.get("solve_times", [0.0])))
    print(f"{name:<24}{cost:>10.2f}{viol_a:>10.1f}{viol_b:>10.1f}{peak:>12.2f}{avg_time:>12.3f}")
    return {"name": name, "cost": cost, "viol_a": viol_a, "viol_b": viol_b, "peak": peak, "avg_time": avg_time}


def main() -> None:
    provider = TwoZoneProvider()

    print("运行 GCCM（两区域联合经济优化）...")
    gccm = run_gccm(provider, horizon=HORIZON)
    print("运行基线 A：集中式舒适优先 MPC ...")
    base_a = run_baseline_a(provider)
    print("运行基线 B：分散式独立 MPC ...")
    base_b = run_baseline_b(provider)

    print("\n" + "=" * 86)
    print(f"{'方法':<24}{'总电费(元)':>10}{'A区违规%':>10}{'B区违规%':>10}{'峰值功率(kW)':>12}{'平均求解(s)':>12}")
    print("-" * 86)
    m_gccm = metrics("GCCM", gccm)
    m_a = metrics("基线A（舒适优先MPC）", base_a)
    m_b = metrics("基线B（分散式MPC）", base_b)
    print("=" * 86)

    print(f"\nGCCM vs 基线A 节省: {(m_a['cost'] - m_gccm['cost']) / m_a['cost'] * 100:.1f}%")
    print(f"GCCM vs 基线B 节省: {(m_b['cost'] - m_gccm['cost']) / m_b['cost'] * 100:.1f}%")


if __name__ == "__main__":
    main()
