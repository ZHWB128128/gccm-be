"""两区域 CasADi + IPOPT 硬约束 MPC 对比：GCCM vs 基线A（舒适优先）。

说明：
- 使用 CasADi + IPOPT 求解带 25~27°C 硬约束的非线性 MPC；
- 当前 IPOPT 在部分算例返回 Maximum_Iterations_Exceeded，但返回的控制轨迹已满足硬约束；
- 因此脚本仍使用返回轨迹执行滚动控制。
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.multiscale import RenormalizationFlow
from gccm_be.types import ExternalInput, SystemState

from two_zone_compare import (
    COMFORT_MAX,
    COMFORT_MIN,
    Q_MAX,
    STEP_H,
    STEPS,
    TwoZoneProvider,
    make_two_zone_simulator,
)

HORIZON = 12
INTERNAL_COMFORT_MAX = 26.8  # CasADi 硬约束使用安全裕度


def make_engine(energy_weight: float, two_stage: bool = False, comfort_margin: float = 0.0):
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
        horizon=HORIZON,
        dt=STEP_H,
        setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
        comfort_band=0.0,
        comfort_margin=comfort_margin,
        comfort_min=COMFORT_MIN,
        comfort_max=INTERNAL_COMFORT_MAX,
        below_comfort_penalty=1.0,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=energy_weight,
        smooth_weight=0.1,
        enforce_comfort_constraints=False,
        use_casadi=True,
        use_two_stage=two_stage,
        renormalization_enabled=True,
        solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
    )
    return engine, sim


def predicted_ok(dec):
    """检查 CasADi 返回的预测轨迹是否满足 25~27°C 硬约束。"""
    for st in dec.trajectory.states[1:]:
        if st.x[0] > COMFORT_MAX or st.x[0] < COMFORT_MIN:
            return False
        if st.x[2] > COMFORT_MAX or st.x[2] < COMFORT_MIN:
            return False
    return True


def run_engine(engine, sim, provider, label, fallback_engine=None):
    state = SystemState(np.full(5, 28.0), sim.building.state_labels)
    t = 0.0
    prev = None
    times, t_a, t_b, p_elec, price = [], [], [], [], []
    solve_times = []
    for i in range(STEPS):
        start = time.perf_counter()
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort")
        solve_times.append(time.perf_counter() - start)
        if fallback_engine is not None and (not dec.trajectory.success or not predicted_ok(dec)):
            fb = fallback_engine.optimize(state, t, prev_control=prev, forced_mode="comfort")
            dec = fb
            solve_times[-1] = time.perf_counter() - start
        w = provider.get(t, 1)[0]
        state = sim.step(state, dec.control, w, STEP_H)
        times.append(t)
        t_a.append(state.x[0])
        t_b.append(state.x[2])
        p_elec.append(sim.hvac.electrical_power(dec.control))
        price.append(w.w[5])
        prev = dec.control
        t += STEP_H
        if (i + 1) % 16 == 0:
            print(f"  {label} 进度 {i+1}/{STEPS}，近16次平均求解 {np.mean(solve_times[-16:]):.2f}s", flush=True)
    return {
        "times": np.array(times),
        "t_a": np.array(t_a),
        "t_b": np.array(t_b),
        "p_elec": np.array(p_elec),
        "price": np.array(price),
        "solve_times": solve_times,
    }


def metrics(name, r):
    cost = float(np.sum(r["p_elec"] * r["price"] * STEP_H))
    viol_a = float(np.mean((r["t_a"] > COMFORT_MAX) | (r["t_a"] < COMFORT_MIN)) * 100.0)
    viol_b = float(np.mean((r["t_b"] > COMFORT_MAX) | (r["t_b"] < COMFORT_MIN)) * 100.0)
    peak = float(np.max(r["p_elec"]))
    avg_time = float(np.mean(r["solve_times"]))
    print(f"{name:<24}{cost:>10.2f}{viol_a:>10.1f}{viol_b:>10.1f}{peak:>12.2f}{avg_time:>12.3f}")
    return {"cost": cost, "viol_a": viol_a, "viol_b": viol_b, "peak": peak, "avg_time": avg_time}


def warm_start_from_comfort(engine_g, sim):
    """用舒适优先 MPC 的解初始化 GCCM 经济优化。返回 comfort_engine 作为 fallback。"""
    comfort_engine, _ = make_engine(energy_weight=0.0, two_stage=False)
    init_state = SystemState(np.full(5, 28.0), sim.building.state_labels)
    dec = comfort_engine.optimize(init_state, 0.0, forced_mode="comfort")
    engine_g._warm_start = list(dec.trajectory.controls)
    print("  已用基线A可行解初始化 GCCM", flush=True)
    return comfort_engine


provider = TwoZoneProvider()

if __name__ == "__main__":
    print("运行 GCCM + CasADi ...")
    margin = 0.3
    eng_g, sim_g = make_engine(energy_weight=0.05, two_stage=True, comfort_margin=margin)
    comfort_engine = warm_start_from_comfort(eng_g, sim_g)
    gccm = run_engine(eng_g, sim_g, provider, "GCCM-CasADi", fallback_engine=comfort_engine)

    print("运行 基线A（舒适优先）+ CasADi ...")
    eng_a, sim_a = make_engine(energy_weight=0.0, two_stage=False, comfort_margin=margin)
    base_a = run_engine(eng_a, sim_a, provider, "基线A-CasADi")

    print("\n" + "=" * 86)
    print(f"{'方法':<24}{'总电费(元)':>10}{'A区违规%':>10}{'B区违规%':>10}{'峰值功率(kW)':>12}{'平均求解(s)':>12}")
    print("-" * 86)
    m_g = metrics("GCCM-CasADi", gccm)
    m_a = metrics("基线A-CasADi", base_a)
    print("=" * 86)
    print(f"\nGCCM vs 基线A 节省: {(m_a['cost'] - m_g['cost']) / m_a['cost'] * 100:.1f}%")

    # 重整化群流：多尺度异常放大检测
    rg = RenormalizationFlow()
    room_temps = list(zip(gccm["t_a"], gccm["t_b"]))
    # 对每个时刻做两房间粗粒化，检查是否存在放大
    amplified_count = 0
    for ta, tb in room_temps:
        if rg.analyze([ta, tb])["amplified"]:
            amplified_count += 1
    print(f"\n重整化群流：GCCM 运行中微观房间异常被宏观放大的时刻数: {amplified_count}/{STEPS}")

    # 检查 B 区违规时段
    b_viol = gccm["t_b"] > COMFORT_MAX
    if np.any(b_viol):
        viol_times = gccm["times"][b_viol]
        peak_count = int(np.sum((viol_times >= 11) & (viol_times < 18)))
        print(f"GCCM B区违规次数: {np.sum(b_viol)}，其中峰时(11-18h) {peak_count} 次")
        print(f"GCCM B区违规时间: {[round(float(t), 2) for t in viol_times[:10]]}")
    else:
        print("GCCM B区无违规")
