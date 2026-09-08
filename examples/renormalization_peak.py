"""Phase B (positive-result track): does renormalization order-parameter
weighting reduce the *building peak* better than uniform control?

Why the earlier result was weak
--------------------------------
`renormalization_demo.py` only *identified* the dominant zone; it never fed that
back into the optimizer, so the peak improvement was a toy +0.14 °C. The RG idea
is only realized when the identified relevant order parameter actually reshapes
the control objective. This experiment closes that loop: the engine maps each
zone's cross-scale relevance to a per-zone comfort penalty multiplier
(`renormalization_weighting=True`), so the MPC spends its limited cooling on the
zone that truly drives the building peak.

Setup
-----
A three-zone building where one zone ("top_sunlit") gets heavy solar gain and
drives the whole building's peak temperature. We compare, across many seeds:

  OFF : uniform comfort weighting (all zones equal)
  ON  : renormalization-weighted comfort (relevant zone up-weighted)

Both share identical plant, weather, cooling capacity and cost weights; the ONLY
difference is whether the objective is reshaped by RG relevance. We paired
t-test the per-seed building-peak temperature (KPI: lower peak = better peak
shaving / demand response).

Run:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python3 examples/renormalization_peak.py --seeds 24
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState
from examples.riemannian_ablation import paired_t_test

STEP_H = 0.25
STEPS = 40
SETPOINT = 26.0
COMFORT_MIN = 25.0
COMFORT_MAX = 27.0
LABELS = ["T_air_1", "T_air_2", "T_air_3"]  # zone 3 = top_sunlit
CONTROL_LABELS = ["Q_1", "Q_2", "Q_3"]


@dataclass
class ThreeZoneModel:
    """Minimal three single-state zones with asymmetric solar gain and a shared
    outdoor coupling, plus weak inter-zone coupling. State = 3 air temps."""

    c_air: float = 0.6
    r_out: float = 1.2
    r_zone: float = 3.0                      # inter-zone coupling resistance
    solar_gain: tuple = (0.02, 0.03, 0.14)   # zone 3 sunlit
    dt: float = STEP_H
    state_labels: List[str] = field(default_factory=lambda: list(LABELS))
    control_labels: List[str] = field(default_factory=lambda: list(CONTROL_LABELS))
    external_labels: List[str] = field(default_factory=lambda: [
        "T_out", "solar", "occ", "price"])

    def step(self, state, control, external, dt=None):
        if dt is None:
            dt = self.dt
        T = state.x.copy()
        Q = control.u
        T_out = external.w[0]
        solar = external.w[1]
        dT = np.zeros(3)
        for i in range(3):
            neighbor = (T[(i - 1) % 3] - T[i]) / self.r_zone + (T[(i + 1) % 3] - T[i]) / self.r_zone
            dT[i] = (
                (T_out - T[i]) / self.r_out
                + self.solar_gain[i] * solar * 10.0
                + neighbor
                + Q[i]
            ) / self.c_air
        return SystemState(T + dT * dt, list(self.state_labels))

    def initial_state(self, t=26.0):
        return SystemState(np.full(3, t), list(self.state_labels))


class PeakProvider(ExternalInputProvider):
    def __init__(self, seed=0, noise_std=0.0, dt_h=STEP_H, day_shift=0.0,
                 base_temp=31.0, swing=4.0, solar_amp=1.0):
        self.dt_h = dt_h
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.day_shift = day_shift      # per-scenario peak-hour shift for independence
        self.base_temp = base_temp
        self.swing = swing
        self.solar_amp = solar_amp

    def get(self, time_h, horizon=1):
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = self.base_temp + self.swing * np.cos(
                2.0 * np.pi * (t - 14.0 - self.day_shift) / 24.0)
            if k == 0 and self.noise_std > 0.0:
                t_out += float(self.rng.normal(0.0, self.noise_std))
            solar = (self.solar_amp * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0))
                     if 6.0 <= hour <= 18.0 else 0.0)
            price = 1.5 if 11.0 <= hour < 16.0 else 0.6
            out.append(ExternalInput(np.array([t_out, solar, 0.6, price]),
                                     ["T_out", "solar", "occ", "price"]))
        return out


def run_once(seed, weighting_on, gain, noise, horizon, q_cap=1.6,
             energy_weight=15.0, comfort_weight=2.0, independent_days=False):
    building = ThreeZoneModel()
    # Tight per-unit cooling capacity -> zones genuinely compete for limited
    # cooling, so *how* the objective prioritizes zones actually changes the
    # allocation (and hence the building peak). With abundant capacity every
    # zone cools itself independently and weighting is a no-op.
    hvac = HVACModel(q_min=-q_cap, q_max=q_cap, n_units=3, control_labels=list(CONTROL_LABELS))
    sim = Simulator(building, hvac)
    manifold = StateManifold(
        labels=list(LABELS),
        units={l: "°C" for l in LABELS},
        bounds={l: (15.0, 40.0) for l in LABELS},
        scale={l: 5.0 for l in LABELS},
    )
    ctrl_provider = PeakProvider(seed=seed, noise_std=0.0)
    plant_provider = PeakProvider(seed=seed, noise_std=noise)
    if independent_days:
        # draw a genuinely different summer day per seed (base temp / swing /
        # peak-hour / solar amplitude) so the paired-t samples are independent
        # days rather than the same day with additive jitter.
        rng = np.random.default_rng(seed)
        kw = dict(
            day_shift=float(rng.uniform(-2.0, 2.0)),
            base_temp=float(rng.uniform(29.0, 33.0)),
            swing=float(rng.uniform(3.0, 6.0)),
            solar_amp=float(rng.uniform(0.8, 1.3)),
        )
        ctrl_provider = PeakProvider(seed=seed, noise_std=0.0, **kw)
        plant_provider = PeakProvider(seed=seed + 100000, noise_std=noise, **kw)
    engine = GCCMEngine(
        simulator=sim,
        external_provider=ctrl_provider,
        manifold=manifold,
        horizon=horizon,
        dt=STEP_H,
        setpoints={l: SETPOINT for l in LABELS},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        # A genuine comfort-vs-energy tradeoff: energy is expensive enough that
        # the OFF controller barely cools (tolerating warmth to save power).
        # In this unsaturated regime, RG weighting redirects the limited cooling
        # onto the peak-driving (relevant) zone instead of spreading it thin.
        comfort_weight=comfort_weight,
        energy_weight=energy_weight,
        smooth_weight=0.1,
        enforce_comfort_constraints=False,   # soft comfort via cost (multi-zone)
        renormalization_enabled=True,
        renormalization_weighting=weighting_on,
        renormalization_weight_gain=gain,
        solver_options={"maxiter": 50, "ftol": 1e-4, "maxls": 15},
    )
    state = SystemState(np.array([26.2, 26.4, 27.5]), list(LABELS))
    t = 8.0
    prev = None
    peaks = []
    top_temps = []
    for _ in range(STEPS):
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort")
        w = plant_provider.get(t, 1)[0]
        state = sim.step(state, dec.control, w, STEP_H)
        peaks.append(float(np.max(state.x)))
        top_temps.append(float(state.x[2]))
        prev = dec.control
        t += STEP_H
    peaks = np.array(peaks)
    return {
        # sustained peak (95th percentile of per-step building max) is the
        # demand-relevant KPI; a single-sample max is dominated by the transient.
        "peak": float(np.percentile(peaks, 95)),
        "mean_peak": float(np.mean(peaks)),
        "top_max": float(np.max(top_temps)),
    }


def main():
    parser = argparse.ArgumentParser(description="重整化序参量加权峰值削减检验")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--gain", type=float, default=3.0)
    parser.add_argument("--noise", type=float, default=0.8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--independent-days", action="store_true",
                        help="每个种子抽一个独立夏季日（更严格的配对独立性）")
    args = parser.parse_args()

    print(f"三区峰值削减检验：seeds={args.seeds}, gain={args.gain}, noise={args.noise}"
          f"{'，独立天气日' if args.independent_days else ''}")
    print("逐种子跑 均匀加权 OFF vs 序参量加权 ON（配对）...")

    keys = ["peak", "mean_peak", "top_max"]
    off = {k: [] for k in keys}
    on = {k: [] for k in keys}
    for s in range(args.seeds):
        r_off = run_once(s, False, args.gain, args.noise, args.horizon,
                         independent_days=args.independent_days)
        r_on = run_once(s, True, args.gain, args.noise, args.horizon,
                        independent_days=args.independent_days)
        for k in keys:
            off[k].append(r_off[k]); on[k].append(r_on[k])
        if (s + 1) % 4 == 0:
            print(f"  完成 {s + 1}/{args.seeds}")

    labels = {"peak": "建筑峰值(95百分位)", "mean_peak": "平均峰值温度", "top_max": "顶层最高温"}
    print("\n" + "=" * 70)
    print(f"{'KPI':<16}{'OFF 均值':>12}{'ON 均值':>12}{'差(ON-OFF)':>14}{'p 值':>10}")
    print("-" * 70)
    verdicts = {}
    for k in keys:
        a = np.array(off[k]); b = np.array(on[k])
        diffs = b - a
        t, p, dof = paired_t_test(diffs)
        verdicts[k] = (float(np.mean(a)), float(np.mean(b)), float(np.mean(diffs)), p)
        mark = "*" if p < 0.05 else ""
        print(f"{labels[k]:<16}{np.mean(a):>12.4f}{np.mean(b):>12.4f}"
              f"{np.mean(diffs):>+14.4f}{p:>10.4f}{mark:>8}")
    print("=" * 70)

    _, _, dpeak, ppeak = verdicts["peak"]
    print("\n判定（建筑峰值温度，α=0.05）：")
    if ppeak < 0.05 and dpeak < 0:
        print(f"  [POSITIVE] 序参量加权显著降低建筑峰值 {abs(dpeak):.4f} °C（p={ppeak:.4f}）"
              f"——正结果成立：重整化耦合进 MPC 后带来实测峰值削减。")
    elif ppeak < 0.05 and dpeak > 0:
        print(f"  [WORSE] 序参量加权显著抬高建筑峰值（p={ppeak:.4f}）。")
    else:
        print(f"  [NO EFFECT] 建筑峰值无显著变化（p={ppeak:.4f}），如实报告边界。")


if __name__ == "__main__":
    main()
