"""Phase 2 experiment: does the Riemannian correction produce a *statistically
significant* KPI change on a geometrically non-trivial system?

Rationale
---------
Round-0 probing showed the Riemannian correction has no measurable effect on a
single-zone near-linear RC model — and the theory doc itself proves that on a
flat/constant diagonal metric the Riemannian control problem *reduces exactly*
to weighted MPC. So an honest existence test must run on a system where the
geometry is genuinely non-trivial:

  - strong inter-zone coupling (small partition resistance -> large off-diagonal
    thermal coupling), and
  - a strongly state-dependent metric (`metric_state_dependence` large), so the
    Christoffel connection Γ is far from zero.

We run a paired A/B (Riemannian OFF vs ON) across many seeds on this coupled
two-zone plant with process noise, then apply a paired t-test to the per-seed
cost and comfort-violation differences. The verdict is reported honestly:
  - if p < 0.05 and the effect favors ON  -> existence proof of a real benefit;
  - if p < 0.05 and it favors OFF          -> Riemannian correction hurts here;
  - otherwise                              -> no significant effect (the theory
    degenerates to weighted MPC on this system) — a valid negative result.

Run:
    PYTHONPATH=. python3 examples/riemannian_ablation.py --seeds 24
    PYTHONPATH=. python3 examples/riemannian_ablation.py --seeds 24 --strength 2.0 --state-dep 0.8 --coupling 0.6
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ControlInput, ExternalInput, SystemState

STEP_H = 0.25
STEPS = 48
Q_MAX = 8.0
SETPOINT = 26.0
COMFORT_MIN = 25.0
COMFORT_MAX = 27.0
LABELS = ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]


class NoisyTwoZoneProvider(ExternalInputProvider):
    """Coupled two-zone weather/price with per-seed process noise."""

    def __init__(self, seed: int = 0, noise_std: float = 0.0, dt_h: float = STEP_H) -> None:
        self.dt_h = dt_h
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std

    def get(self, time_h: float, horizon: int = 1) -> List[ExternalInput]:
        result = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            if k == 0 and self.noise_std > 0.0:
                t_out += float(self.rng.normal(0.0, self.noise_std))
            solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
            occ_a = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            occ_b = 0.6 if 8.0 <= hour <= 18.0 else 0.2
            if hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5
            result.append(ExternalInput(
                np.array([t_out, solar * 1.2, solar * 0.3, occ_a, occ_b, price]),
                ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"],
            ))
        return result


def make_coupled_simulator(r_partition: float) -> Simulator:
    """Two-zone plant. Small r_partition = strong inter-zone thermal coupling."""
    building = TwoZoneRCBuildingModel(r_partition=r_partition, solar_gain_a=0.10, solar_gain_b=0.02)
    hvac = HVACModel(q_min=-Q_MAX, q_max=Q_MAX, n_units=2, control_labels=["Q_hvac_A", "Q_hvac_B"])
    return Simulator(building, hvac)


@dataclass
class RunResult:
    cost: float
    violation: float
    peak: float


def run_once(seed: int, use_riemannian: bool, strength: float, state_dep: float,
             coupling: float, r_partition: float, noise: float, horizon: int) -> RunResult:
    sim = make_coupled_simulator(r_partition)
    manifold = StateManifold(
        labels=LABELS,
        units={l: "°C" for l in LABELS},
        bounds={l: (15.0, 40.0) for l in LABELS},
        scale={l: 5.0 for l in LABELS},
    )
    ctrl_provider = NoisyTwoZoneProvider(seed=seed, noise_std=0.0)      # forecast (clean)
    plant_provider = NoisyTwoZoneProvider(seed=seed, noise_std=noise)  # actual plant (noisy)
    engine = GCCMEngine(
        simulator=sim,
        external_provider=ctrl_provider,
        manifold=manifold,
        horizon=horizon,
        dt=STEP_H,
        setpoints={"T_air_A": SETPOINT, "T_air_B": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=20.0,
        energy_weight=0.3,
        smooth_weight=0.1,
        use_kinetic=True,
        use_riemannian=use_riemannian,
        riemannian_strength=strength,
        metric_state_dependence=state_dep if use_riemannian else 0.0,
        metric_coupling=coupling if use_riemannian else 0.0,
        solver_options={"maxiter": 40, "ftol": 1e-4, "maxls": 15},
    )
    state = SystemState(np.full(5, 28.0), LABELS)
    t = 0.0
    prev_control: Optional[ControlInput] = None
    temps_a, temps_b, powers, prices = [], [], [], []
    for _ in range(STEPS):
        dec = engine.optimize(state, t, prev_control=prev_control, forced_mode="comfort")
        w = plant_provider.get(t, 1)[0]
        state = sim.step(state, dec.control, w, STEP_H)
        temps_a.append(state.x[0]); temps_b.append(state.x[2])
        powers.append(sim.hvac.electrical_power(dec.control))
        prices.append(float(w.w[5]))
        prev_control = dec.control
        t += STEP_H
    temps_a = np.array(temps_a); temps_b = np.array(temps_b)
    powers = np.array(powers); prices = np.array(prices)
    cost = float(np.sum(powers * prices * STEP_H))
    viol = float(np.mean(
        (temps_a > COMFORT_MAX) | (temps_a < COMFORT_MIN)
        | (temps_b > COMFORT_MAX) | (temps_b < COMFORT_MIN)
    ) * 100.0)
    peak = float(np.max(powers))
    return RunResult(cost, viol, peak)


def paired_t_test(diffs: np.ndarray):
    """Paired t-test on the per-seed differences (ON - OFF). Returns (t, p, dof)."""
    n = diffs.size
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd < 1e-15:
        return 0.0, 1.0, n - 1
    se = sd / np.sqrt(n)
    t = mean / se
    dof = n - 1
    # two-sided p-value via a Student-t survival function (no scipy dependency)
    p = _student_t_sf_two_sided(abs(t), dof)
    return t, p, dof


def _student_t_sf_two_sided(t: float, dof: int) -> float:
    """Two-sided p-value for Student's t via the regularized incomplete beta."""
    x = dof / (dof + t * t)
    # P(|T|>t) = I_x(dof/2, 1/2)
    return _betai(dof / 2.0, 0.5, x)


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    import math
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    # Lentz continued fraction for the incomplete beta
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= d * c
        if abs(1.0 - d * c) < 1e-10:
            break
    result = front * (f - 1.0)
    # I_x for x on the correct side of the symmetry point
    if x < (a + 1.0) / (a + b + 2.0):
        return result
    return 1.0 - (math.exp(b * math.log(1.0 - x) + a * math.log(x) - lbeta) / b) * (f - 1.0) if False else result


def main() -> None:
    parser = argparse.ArgumentParser(description="黎曼修正统计显著性 ablation（强耦合双区）")
    parser.add_argument("--seeds", type=int, default=24, help="配对样本数（随机种子数）")
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("--state-dep", type=float, default=0.6)
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--r-partition", type=float, default=0.25, help="隔墙热阻，越小耦合越强")
    parser.add_argument("--noise", type=float, default=1.0, help="室外温度过程噪声标准差")
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()

    print(f"强耦合双区 ablation：seeds={args.seeds}, r_partition={args.r_partition} "
          f"(耦合强度), strength={args.strength}, state_dep={args.state_dep}, "
          f"coupling={args.coupling}, noise={args.noise}")
    print("逐种子跑 Riemannian OFF vs ON（配对）...")

    off_cost, on_cost, off_viol, on_viol = [], [], [], []
    for s in range(args.seeds):
        r_off = run_once(s, False, args.strength, args.state_dep, args.coupling,
                         args.r_partition, args.noise, args.horizon)
        r_on = run_once(s, True, args.strength, args.state_dep, args.coupling,
                        args.r_partition, args.noise, args.horizon)
        off_cost.append(r_off.cost); on_cost.append(r_on.cost)
        off_viol.append(r_off.violation); on_viol.append(r_on.violation)
        if (s + 1) % 4 == 0:
            print(f"  完成 {s + 1}/{args.seeds}")

    off_cost = np.array(off_cost); on_cost = np.array(on_cost)
    off_viol = np.array(off_viol); on_viol = np.array(on_viol)

    print("\n" + "=" * 68)
    print(f"{'指标':<14}{'OFF 均值':>12}{'ON 均值':>12}{'差(ON-OFF)':>14}{'p 值':>10}")
    print("-" * 68)
    for name, off, on in (("电费", off_cost, on_cost), ("违规%", off_viol, on_viol)):
        diffs = on - off
        t, p, dof = paired_t_test(diffs)
        print(f"{name:<14}{np.mean(off):>12.4f}{np.mean(on):>12.4f}"
              f"{np.mean(diffs):>+14.4f}{p:>10.4f}")
    print("=" * 68)

    # verdict on cost
    diffs = on_cost - off_cost
    t, p, dof = paired_t_test(diffs)
    print("\n判定（电费，α=0.05）：")
    if p < 0.05 and np.mean(diffs) < 0:
        print(f"  [PROVEN] 黎曼修正显著降低电费（p={p:.4f} < 0.05）——存在性证明成立。")
    elif p < 0.05 and np.mean(diffs) > 0:
        print(f"  [HARMFUL] 黎曼修正显著增加电费（p={p:.4f} < 0.05）——在此系统有害。")
    else:
        print(f"  [NO EFFECT] 无显著差异（p={p:.4f} >= 0.05）——黎曼修正在此系统退化为加权 MPC。"
              f"\n     这是有效的负结果：几何在近似线性/弱曲率工况下不产生额外增益。")


if __name__ == "__main__":
    main()
