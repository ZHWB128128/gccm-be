"""Phase 1 experiment: Shapley cost attribution.

Question this answers
---------------------
When GCCM's bill is lower than a naive baseline on a given day, *why*? Three
things differ between the two worlds:

  1. policy  — GCCM's rolling-horizon MPC vs a fixed strict-comfort PID
  2. weather — the day GCCM ran on vs a hotter reference day
  3. price   — GCCM's tariff vs a flat reference tariff

A single "we saved 13%" number cannot separate these. This script runs the
*real* physics simulator for every factor coalition and reports the exact
Shapley decomposition, proving how much of the saving is genuinely attributable
to the control strategy rather than a lucky cooler day or a cheaper tariff.

Run:
    PYTHONPATH=. python3 examples/attribution_demo.py
"""
from __future__ import annotations

import numpy as np

from compare_baselines import (
    COMFORT_MAX,
    COMFORT_MIN,
    SETPOINT,
    STEP_H,
    STEPS,
    ScenarioExternalInputProvider,
    make_simulator,
)
from gccm_be.causal.attribution import CostAttributor
from gccm_be.types import ControlInput, SystemState


def build_gccm_policy(sim):
    """A closed-loop GCCM policy captured as a stateful (state,t)->control callable."""
    from gccm_be import GCCMEngine
    from gccm_be.geometry.manifold import StateManifold

    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 35.0), "T_wall": (15.0, 35.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    provider = ScenarioExternalInputProvider()
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=24,
        dt=STEP_H,
        setpoints={"T_air": SETPOINT},
        comfort_band=0.0,
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.5,
        comfort_weight=5.0,
        energy_weight=0.5,
        smooth_weight=0.1,
        enforce_comfort_constraints=True,
        solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
        constraint_options={"maxiter": 50, "ftol": 1e-6},
    )

    def policy(state, t):
        dec = engine.optimize(state, t, forced_mode="comfort")
        return dec.control

    return policy


def strict_pid_policy(sim):
    """Fixed strict-comfort PID baseline as a (state,t)->control callable."""
    kp, ki, kd, ilim = 2.0, 2.0, 0.1, 5.0
    st = {"integral": 0.0, "prev_error": 0.0, "last_t": None}

    def policy(state, t):
        if st["last_t"] is None:
            dt = STEP_H
        else:
            dt = max(t - st["last_t"], 1e-6)
        error = state.x[0] - SETPOINT
        st["integral"] = float(np.clip(st["integral"] + error * dt, -ilim, ilim))
        deriv = (error - st["prev_error"]) / dt
        out = kp * error + ki * st["integral"] + kd * deriv
        cooling = float(np.clip(out, 0.0, sim.hvac.q_max))
        st["prev_error"] = error
        st["last_t"] = t
        return ControlInput([-cooling], ["Q_hvac"])

    return policy


def weather_series(hot: bool):
    """Return `STEPS` external inputs for a reference (hot) or milder day."""
    provider = ScenarioExternalInputProvider()
    seq = provider.get(0.0, STEPS)
    if hot:
        # reference baseline world: +4°C hotter outdoor, +25% solar
        out = []
        for w in seq:
            ww = w.copy()
            ww.w[0] += 4.0
            ww.w[1] *= 1.25
            out.append(ww)
        return out
    return seq


def price_series(flat: bool):
    provider = ScenarioExternalInputProvider()
    seq = provider.get(0.0, STEPS)
    if flat:
        return [1.0] * STEPS  # baseline world: flat expensive tariff
    return [float(w.price) for w in seq]  # treatment: real peak/valley tariff


def main() -> None:
    sim = make_simulator()
    init = SystemState([28.0, 28.0], ["T_air", "T_wall"])

    # baseline world  = PID policy + hot day + flat tariff
    # treatment world = GCCM policy + mild day + real peak/valley tariff
    attributor = CostAttributor(
        simulator=sim,
        initial_state=init,
        dt=STEP_H,
        baseline_policy=strict_pid_policy(sim),
        treatment_policy=build_gccm_policy(sim),
        baseline_weather=weather_series(hot=True),
        treatment_weather=weather_series(hot=False),
        baseline_price=price_series(flat=True),
        treatment_price=price_series(flat=False),
    )

    print("运行 Shapley 归因分解（8 次闭环仿真）...")
    result = attributor.attribute()

    print("\n" + "=" * 60)
    print("成本差异归因（Shapley 分解）")
    print("=" * 60)
    print(result.summary())
    print("=" * 60)
    print("\n解读：")
    fracs = result.contribution_fractions()
    for name, cn in (("policy", "控制策略"), ("weather", "天气"), ("price", "电价")):
        sign = "降低" if result.contributions[name] < 0 else "增加"
        print(f"  {cn}: {sign}成本 {abs(result.contributions[name]):.3f} "
              f"（占总差异 {fracs[name]*100:+.1f}%）")
    print(f"\n可加性残差 {result.residual:.2e}（应≈0，证明分解精确可加）")


if __name__ == "__main__":
    main()
