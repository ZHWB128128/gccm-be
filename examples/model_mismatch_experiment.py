"""模型失配实验：控制器内部模型 vs 真实建筑参数不一致。

对比：
- strict_pid：无模型，直接反馈控制
- classic_mpc：内部模型默认参数，无自指/哥德尔降级
- gccm：内部模型默认参数，带 SelfMonitor / Godel / 反事实降级

真实建筑使用“更差保温、更大得热”的参数，控制器仍使用默认模型。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState

STEP_H = 0.25
STEPS = 96
SETPOINT = 26.0
COMFORT_MIN = 25.0
COMFORT_MAX = 27.0
Q_MAX = 8.0

# 控制器内部模型：默认参数
CONTROLLER_PARAMS = dict(
    c_air=0.6,
    c_wall=4.0,
    r_air=0.8,
    r_wall=2.0,
    solar_gain=0.05,
)

# 真实建筑：保温更差、得热更大
PLANT_PARAMS = dict(
    c_air=0.5,
    c_wall=3.0,
    r_air=0.6,
    r_wall=1.2,
    solar_gain=0.08,
)


class SummerProvider(ExternalInputProvider):
    def __init__(self, dt_h: float = STEP_H) -> None:
        self.dt_h = dt_h

    def get(self, time_h: float, horizon: int = 1):
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            price = 0.3 if hour < 8.0 or hour >= 22.0 else (0.8 if hour < 11.0 or hour >= 18.0 else 1.5)
            out.append(ExternalInput(np.array([t_out, solar, occ, price]), ["T_out", "solar", "occ", "price"]))
        return out


@dataclass
class ExpResult:
    name: str
    cost: float = 0.0
    violation: float = 0.0
    max_temp: float = 0.0
    min_temp: float = 0.0
    undecidable_count: int = 0
    switch_count: int = 0
    counterfactual_count: int = 0
    avg_solve: float = 0.0
    temps: np.ndarray = field(default_factory=lambda: np.array([]))


def run_strict_pid(provider, plant_sim):
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    integral = 0.0
    prev_err = 0.0
    ts, ps, prices = [], [], []
    for _ in range(STEPS):
        w = provider.get(t, 1)[0]
        err = state.x[0] - SETPOINT
        integral = float(np.clip(integral + err * STEP_H, -5.0, 5.0))
        der = (err - prev_err) / STEP_H
        cool = float(np.clip(2.0 * err + 2.0 * integral + 0.1 * der, 0.0, Q_MAX))
        control = ControlInput([-cool], ["Q_hvac"])
        state = plant_sim.step(state, control, w, STEP_H)
        ts.append(state.x[0])
        ps.append(plant_sim.hvac.electrical_power(control))
        prices.append(w.w[3])
        prev_err = err
        t += STEP_H
    return ExpResult(
        name="strict_pid",
        cost=float(np.sum(np.array(ps) * np.array(prices) * STEP_H)),
        violation=float(np.mean((np.array(ts) > COMFORT_MAX) | (np.array(ts) < COMFORT_MIN)) * 100.0),
        max_temp=float(np.max(ts)),
        min_temp=float(np.min(ts)),
        temps=np.array(ts),
    )


def _make_engine(use_kinetic: bool, enable_guard: bool, comfort_margin: float = 0.9):
    controller_sim = Simulator(RCBuildingModel(**CONTROLLER_PARAMS), HVACModel(q_min=-Q_MAX, q_max=Q_MAX))
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15, 40), "T_wall": (15, 40)},
        scale={"T_air": 5, "T_wall": 5},
    )
    engine = GCCMEngine(
        simulator=controller_sim,
        external_provider=SummerProvider(),
        manifold=manifold,
        horizon=24,
        dt=STEP_H,
        setpoints={"T_air": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        comfort_margin=comfort_margin,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        use_kinetic=use_kinetic,
        counterfactual_enabled=enable_guard,
        safe_control_mode="worst_case",
        solver_options={"maxiter": 50, "ftol": 1e-4, "maxls": 20},
    )
    if enable_guard:
        # 降级时优先回到舒适模式，而不是 balanced
        engine.diagnoser.safe_mode = "comfort"
    return engine


def run_mpc(name: str, use_kinetic: bool, enable_guard: bool, provider, plant_sim, comfort_margin: float = 0.9, identify_steps: int = 0):
    engine = _make_engine(use_kinetic, enable_guard, comfort_margin)
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    prev = None
    prediction_error = 0.0
    ts, ps, prices = [], [], []
    solve_times = []
    undecidable_count = 0
    switch_count = 0
    cf_count = 0

    for step_idx in range(STEPS):
        start = time.perf_counter()
        w = provider.get(t, 1)[0]
        if step_idx < identify_steps:
            # 启动辨识阶段：使用无模型反馈控制，先采集数据辨识真实参数
            ctrl = engine._feedback_safe_control(state, STEP_H, w)
            dec = type("Dummy", (), {"control": ctrl, "predicted_next_state": None,
                                     "diagnosis": type("D", (), {"undecidable": False,
                                     "should_switch_mode": False, "details": {}, "suggested_mode": None})})()
        else:
            dec = engine.optimize(
                state, t,
                prev_control=prev,
                forced_mode="comfort",
                prediction_error=prediction_error,
            )
        solve_times.append(time.perf_counter() - start)
        predicted = dec.predicted_next_state.x if dec.predicted_next_state is not None else state.x
        state_before = state.copy()
        state = plant_sim.step(state, dec.control, w, STEP_H)
        prediction_error = float(np.max(np.abs(predicted - state.x)))
        engine.self_monitor.update(prediction_error)
        engine.observe_step(state_before, dec.control, w, state, STEP_H)
        engine.apply_rc_identification(min_samples=30)

        ts.append(state.x[0])
        ps.append(plant_sim.hvac.electrical_power(dec.control))
        prices.append(w.w[3])
        if dec.diagnosis.undecidable:
            undecidable_count += 1
        if dec.diagnosis.should_switch_mode:
            switch_count += 1
        if enable_guard and (dec.diagnosis.should_switch_mode or dec.diagnosis.undecidable):
            if dec.diagnosis.undecidable and not dec.diagnosis.should_switch_mode:
                cf = engine._run_counterfactual_undecidable(state, t)
            elif dec.diagnosis.suggested_mode is not None:
                old_mode = "comfort" if dec.mode != "comfort" else "comfort"
                cf = engine._run_counterfactual(state, t, old_mode, dec.diagnosis.suggested_mode)
            else:
                cf = {}
            if cf:
                dec.diagnosis.details["counterfactual"] = cf
        if "counterfactual" in dec.diagnosis.details:
            cf_count += 1
        prev = dec.control
        t += STEP_H

    return ExpResult(
        name=name,
        cost=float(np.sum(np.array(ps) * np.array(prices) * STEP_H)),
        violation=float(np.mean((np.array(ts) > COMFORT_MAX) | (np.array(ts) < COMFORT_MIN)) * 100.0),
        max_temp=float(np.max(ts)),
        min_temp=float(np.min(ts)),
        undecidable_count=undecidable_count,
        switch_count=switch_count,
        counterfactual_count=cf_count,
        avg_solve=float(np.mean(solve_times)),
        temps=np.array(ts),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--comfort-margin", type=float, default=0.9)
    parser.add_argument("--identify-steps", type=int, default=0)
    args = parser.parse_args()

    provider = SummerProvider()
    plant_sim = Simulator(RCBuildingModel(**PLANT_PARAMS), HVACModel(q_min=-Q_MAX, q_max=Q_MAX))

    print("真实建筑参数（模型失配）:")
    print(f"  c_air={PLANT_PARAMS['c_air']}, c_wall={PLANT_PARAMS['c_wall']}, "
          f"r_air={PLANT_PARAMS['r_air']}, r_wall={PLANT_PARAMS['r_wall']}, "
          f"solar_gain={PLANT_PARAMS['solar_gain']}")
    print("控制器内部模型参数:")
    print(f"  c_air={CONTROLLER_PARAMS['c_air']}, c_wall={CONTROLLER_PARAMS['c_wall']}, "
          f"r_air={CONTROLLER_PARAMS['r_air']}, r_wall={CONTROLLER_PARAMS['r_wall']}, "
          f"solar_gain={CONTROLLER_PARAMS['solar_gain']}")
    print()

    print("运行 strict_pid ...")
    pid = run_strict_pid(provider, plant_sim)
    print("运行 classic_mpc ...")
    classic = run_mpc("classic_mpc", use_kinetic=False, enable_guard=False, provider=provider, plant_sim=plant_sim, comfort_margin=args.comfort_margin, identify_steps=args.identify_steps)
    print("运行 gccm ...")
    gccm = run_mpc("gccm", use_kinetic=True, enable_guard=True, provider=provider, plant_sim=plant_sim, comfort_margin=args.comfort_margin, identify_steps=args.identify_steps)

    print("\n" + "=" * 100)
    print(f"{'方法':<14}{'电费(元)':>10}{'违规%':>8}{'最高温':>8}{'最低温':>8}"
          f"{'不可判定':>10}{'切换次数':>10}{'反事实':>8}{'平均求解s':>10}")
    print("-" * 100)
    for r in [pid, classic, gccm]:
        print(f"{r.name:<14}{r.cost:>10.2f}{r.violation:>8.1f}{r.max_temp:>8.2f}{r.min_temp:>8.2f}"
              f"{r.undecidable_count:>10}{r.switch_count:>10}{r.counterfactual_count:>8}{r.avg_solve:>10.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
