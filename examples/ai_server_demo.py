"""AI server room cooling demo: GCCM vs thermostat rule control.

AI-specific features:
    - GPU workload profile: training bursts (night, at valley price), steady inference (day), idle dips
    - ASHRAE A1 envelope: cold-aisle inlet allowed 18-27 degC (wider than human comfort 22-27)
    - Thermal storage tank: valley charge / peak discharge (arbitrage + peak shaving)
    - GCCM storage value term prevents myopic tank draining

Usage: PYTHONPATH=. python3 examples/ai_server_demo.py [--output output] [--steps 576]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.datacenter import (
    DataCenterCoolingModel,
    DataCenterProvider,
    DataCenterSimulator,
)
from gccm_be.types import ControlInput, ExternalInput, SystemState

STEP_H = 1.0 / 12.0
STEPS = 576  # 48h
SETPOINT = 24.0
# ASHRAE A1: cold-aisle inlet 18-27 degC (IT-safety band, not human comfort)
BAND_MIN, BAND_MAX = 18.0, 27.0
CHG_PRICE_MAX = 0.5
DISC_PRICE_MIN = 1.6
T_TANK_TARGET = 10.0


class AIProvider(DataCenterProvider):
    """AI server room external inputs: GPU workload + AI data-center TOU price.

    w = [T_out, solar, it_load, price]
    Workload: ~150 kW inference base + training bursts (night ~23:00-07:00 at valley
    price, plus a day-time burst 10:00-14:00) + idle dips. GPU clusters often run
    batch training overnight, so cooling load follows the AI schedule.
    """

    labels: list = ["T_out", "solar", "it_load", "price"]

    def get(self, time_h: float, horizon: int = 1) -> list:
        result = []
        for k in range(horizon):
            t = time_h + k * STEP_H
            hour = t % 24.0
            t_out = 26.0 + 6.0 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            # GPU workload
            it_load = 150.0  # inference base
            if (23.0 <= hour) or (hour < 7.0):
                it_load += 180.0  # nightly training burst (valley price)
            elif 10.0 <= hour <= 14.0:
                it_load += 120.0  # daytime training burst
            elif 19.0 <= hour <= 22.0:
                it_load += 40.0  # inference peak
            if 13.0 <= hour <= 15.0 or 2.0 <= hour <= 4.0:
                it_load -= 30.0  # idle dips
            it_load += self.rng.normal(0.0, 8.0)
            # AI DC price
            if hour < 7.0 or hour >= 23.0:
                price = 0.35  # valley
            elif 10.0 <= hour <= 12.0 or 14.0 <= hour <= 17.0:
                price = 1.8  # peak
            else:
                price = 0.9  # flat
            result.append(ExternalInput(np.array([t_out, 0.0, it_load, price]), list(self.labels)))
        return result


def _ctrl(u0: float, u1: float):
    return ControlInput(np.array([u0, u1]), ["Q_chiller", "Q_tank"])


def run_rule(provider, sim: DataCenterSimulator, steps: int = STEPS) -> dict:
    """Rule: cold-aisle thermostat (24 C) + tank charge/discharge by price thresholds."""
    state = SystemState([24.0, 20.0], ["T_aisle", "T_storage"])
    t = 0.0
    times, t_a, t_s, q0, q1, p, price = [], [], [], [], [], [], []
    for _ in range(steps):
        w = provider.get(t, 1)[0]
        err = state.x[0] - SETPOINT
        u0 = -float(np.clip(60.0 * err, 0.0, -sim.hvac.q_min)) if err > 0.2 else 0.0
        if w.price <= CHG_PRICE_MAX and state.x[1] > T_TANK_TARGET:
            u1 = sim.hvac.chg_max
        elif w.price >= DISC_PRICE_MIN and state.x[1] < state.x[0] - 1.0 and err > 0.2:
            u1 = -sim.hvac.disc_max
        else:
            u1 = 0.0
        ctrl = _ctrl(u0, u1)
        state = sim.step(state, ctrl, w, STEP_H)
        times.append(t); t_a.append(state.x[0]); t_s.append(state.x[1])
        q0.append(ctrl.u[0]); q1.append(ctrl.u[1])
        p.append(sim.hvac.electrical_power(ctrl)); price.append(w.price)
        t += STEP_H
    return {"times": np.array(times), "t_aisle": np.array(t_a), "t_tank": np.array(t_s),
            "q0": np.array(q0), "q1": np.array(q1), "p": np.array(p), "price": np.array(price)}


def run_gccm(provider, sim: DataCenterSimulator, steps: int = STEPS) -> dict:
    manifold = StateManifold(
        labels=["T_aisle", "T_storage"],
        units={"T_aisle": "°C", "T_storage": "°C"},
        bounds={"T_aisle": (15.0, 35.0), "T_storage": (5.0, 30.0)},
        scale={"T_aisle": 5.0, "T_storage": 5.0},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=48,  # 4h: covers valley-charge -> peak-discharge window
        dt=STEP_H,
        setpoints={"T_aisle": SETPOINT},
        comfort_min=BAND_MIN,
        comfort_max=BAND_MAX,
        comfort_margin=0.8,
        below_comfort_penalty=0.2,
        peak_energy_penalty=1.5,
        comfort_weight=5.0,
        energy_weight=1.5,
        smooth_weight=1e-5,
        storage_targets={"T_storage": 10.0},
        storage_weight=10.0,
        enforce_comfort_constraints=True,
        use_kinetic=True,
        safe_control_mode="feedback",
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )
    state = SystemState([24.0, 20.0], ["T_aisle", "T_storage"])
    t = 0.0
    prev = None
    prediction_error = 0.0
    times, t_a, t_s, q0, q1, p, price = [], [], [], [], [], [], []
    for _ in range(steps):
        w = provider.get(t, 1)[0]
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort",
                              prediction_error=prediction_error)
        predicted = dec.predicted_next_state
        state = sim.step(state, dec.control, w, STEP_H)
        if predicted is not None:
            prediction_error = float(np.max(np.abs(predicted.x - state.x)))
        engine.self_monitor.update(prediction_error)
        times.append(t); t_a.append(state.x[0]); t_s.append(state.x[1])
        q0.append(dec.control.u[0]); q1.append(dec.control.u[1])
        p.append(sim.hvac.electrical_power(dec.control)); price.append(w.price)
        prev = dec.control
        t += STEP_H
    return {"times": np.array(times), "t_aisle": np.array(t_a), "t_tank": np.array(t_s),
            "q0": np.array(q0), "q1": np.array(q1), "p": np.array(p), "price": np.array(price)}


def metrics(r: dict) -> tuple:
    cost = float(np.sum(r["p"] * r["price"] * STEP_H))
    viol = float(np.mean((r["t_aisle"] > BAND_MAX) | (r["t_aisle"] < BAND_MIN)) * 100.0)
    peak = float(np.max(r["p"]))
    return cost, viol, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    provider = AIProvider(peak_price=1.8)
    sim = DataCenterSimulator(DataCenterCoolingModel(
        c_aisle=30.0, c_tank=250.0, r_out=0.15,
        q_disc_max=180.0, q_chg_max=220.0,
    ))
    sim.hvac.q_min = -500.0
    sim.hvac.chg_max = 220.0
    sim.hvac.disc_max = 180.0

    print("Rule control (thermostat + price-threshold tank)...")
    rule = run_rule(provider, sim, steps=args.steps)
    print("GCCM control (MPC + storage value term)...")
    gccm = run_gccm(provider, sim, steps=args.steps)

    c_r, v_r, pk_r = metrics(rule)
    c_g, v_g, pk_g = metrics(gccm)
    saving = 100.0 * (c_r - c_g) / c_r if c_r > 0 else float("nan")

    print(f"\n===== AI Server Room Cooling: GCCM vs Rule ({args.steps} steps) =====")
    print(f"Electricity cost: rule={c_r:.1f} yuan  GCCM={c_g:.1f} yuan  savings={saving:.1f}%")
    print(f"Violation (band {BAND_MIN}-{BAND_MAX}C): rule={v_r:.1f}%  GCCM={v_g:.1f}%")
    print(f"Peak power: rule={pk_r:.1f} kW  GCCM={pk_g:.1f} kW")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(rule["times"], rule["t_aisle"], label="Rule: aisle temp", color="tab:gray", lw=1.2)
    axes[0].plot(gccm["times"], gccm["t_aisle"], label="GCCM: aisle temp", color="tab:blue", lw=1.5)
    axes[0].axhline(BAND_MAX, color="r", ls="--", lw=0.8, label=f"ASHRAE A1 max {BAND_MAX}C")
    axes[0].axhline(BAND_MIN, color="g", ls="--", lw=0.8, label=f"min {BAND_MIN}C")
    axes[0].set_ylabel("Cold-aisle temp (C)")
    axes[0].legend(fontsize=8)
    axes[1].plot(rule["times"], rule["t_tank"], label="Rule: storage tank", color="tab:gray", lw=1.2)
    axes[1].plot(gccm["times"], gccm["t_tank"], label="GCCM: storage tank", color="tab:orange", lw=1.5)
    axes[1].set_ylabel("Tank temp (C)")
    axes[1].legend(fontsize=8)
    axes[2].plot(rule["times"], rule["p"], label="Rule: electric power", color="tab:gray", lw=1.0)
    axes[2].plot(gccm["times"], gccm["p"], label="GCCM: electric power", color="tab:green", lw=1.2)
    axes[2].plot(gccm["times"], gccm["price"], label="Electricity price", color="tab:red", lw=1.0, ls=":")
    axes[2].set_ylabel("Power (kW) / price")
    axes[2].set_xlabel("Time (h)")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"AI Server Room Cooling: GCCM vs Rule (savings {saving:.1f}%)", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(args.output, "ai_server_cooling.png")
    fig.savefig(out_png, dpi=130)
    print(f"chart saved: {out_png}")


if __name__ == "__main__":
    main()
