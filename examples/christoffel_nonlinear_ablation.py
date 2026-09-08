"""边界一突破实验：非线性 COP 测试台上的 Christoffel 修正消融。

推翻条件（CONTROL_DESIGN §1.2）：若在非线性场景（度量弯曲，动能项已显著）上
Christoffel 修正得到 p<0.05 且方向有利的结果，"近线性无增益"的负结论即被推翻。

设计：
- 测试台：NonlinearCOPHVAC（COP 随室外温度衰减 → 电费项随外部状态变化 → 景观弯曲）；
- 控制器与 plant 共享同一预报（_SinProvider），保证公平；
- 唯一切换项：use_riemannian（Christoffel 动力学步进修正）；
- metric_state_dependence=0.8（度量显著状态相关 → 联络非零）；
- KPI：电费 / 违温 / 温度爬升 RMS，N seeds 配对 t 检验。

用法：PYTHONPATH=. python3 examples/christoffel_nonlinear_ablation.py --seeds 24
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState

COMFORT_MIN, COMFORT_MAX = 25.0, 27.0
Q_MAX = 15.0
DT = 1.0 / 12.0


class NonlinearCOPHVAC(HVACModel):
    """COP 随室外温度衰减（与 riemannian_smoothness.py 一致）。"""

    def __init__(self, beta: float = 0.035, t_ref: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.t_ref = t_ref
        self._t_out = t_ref

    def set_context(self, t_out: float) -> None:
        self._t_out = float(t_out)

    def electrical_power(self, control, external=None) -> float:
        cop = self.cop_cooling * (1.0 - self.beta * max(0.0, self._t_out - self.t_ref))
        total = 0.0
        for j, u in enumerate(control.u):
            q = float(u)
            abs_q = q if q >= 0 else -q
            load = min(abs_q / max(abs(self.q_min), abs(self.q_max), 1e-9), 1.0)
            cop_eff = cop * (1.0 - self.part_load_penalty * (1.0 - load) ** 2)
            total += abs_q / max(cop_eff, 1e-6)
        return total


class SinProvider(ExternalInputProvider):
    """与 plant 一致的预报：日内正弦 t_out（周期 8h），固定 solar/occ/price。"""

    def __init__(self, dt_h: float = DT):
        self.dt_h = dt_h
        self.labels = ["T_out", "solar", "occ", "price"]

    def get(self, time_h: float, horizon: int = 1):
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            t_out = 30.5 + 1.5 * math.sin(2 * math.pi * t / 8.0)
            out.append(ExternalInput(np.array([t_out, 0.5, 0.4, 1.0]),
                                     list(self.labels)))
        return out


def make_engine(strength: float, state_dep: float) -> GCCMEngine:
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    hvac = NonlinearCOPHVAC(beta=0.035, t_ref=30.0, q_min=-Q_MAX, q_max=Q_MAX)
    hvac.set_context(32.0)
    return GCCMEngine(
        simulator=Simulator(RCBuildingModel(c_air=0.6, c_wall=4.0, r_air=0.8,
                                            r_wall=2.0, solar_gain=0.05), hvac),
        external_provider=SinProvider(),
        manifold=manifold, horizon=12, dt=DT,
        setpoints={"T_air": 26.0},
        comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX, comfort_band=0.0,
        comfort_margin=0.5,  # 预测性收紧:解决"下步才出带,现在不动作"的短视
        below_comfort_penalty=1.0,
        comfort_weight=15.0, energy_weight=1.5, smooth_weight=1e-4,
        enforce_comfort_constraints=False,
        use_kinetic=False,                    # 唯一切换项 = Christoffel
        use_riemannian=strength > 0.0,
        riemannian_strength=strength,
        metric_state_dependence=state_dep,    # 度量弯曲 → 联络非零
        solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
    )


def run_arm(strength: float, state_dep: float, seed: int, steps: int = 96):
    engine = make_engine(strength, state_dep)
    state = SystemState(np.array([26.0, 26.0]), ["T_air", "T_wall"])
    temps, costs = [], []
    prev = None
    for k in range(steps):
        t_h = k * DT
        w = engine.external_provider.get(t_h, 1)[0]
        t_out = float(w.w[0])
        engine.simulator.hvac.set_context(t_out)
        dec = engine.optimize(state, t_h, prev_control=prev, forced_mode="comfort")
        prev = dec.control
        cost = engine.simulator.hvac.electrical_power(dec.control, w) \
            * (1.0 if t_out < 32 else 1.8)
        state = engine.simulator.step(state, dec.control, w, DT)
        temps.append(float(state.x[0]))
        costs.append(cost)
    temps = np.array(temps)
    return {
        "cost": float(np.sum(costs)),
        "viol": float(np.mean((temps > COMFORT_MAX) | (temps < COMFORT_MIN)) * 100.0),
        "rms": float(np.sqrt(np.mean(np.diff(temps) ** 2))),
    }


def paired_t(diffs):
    d = np.asarray(diffs, float)
    n = d.size
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1))
    if s < 1e-15:
        return (float("inf") if m != 0 else 0.0), 0.0
    t = m / (s / np.sqrt(n))
    from math import lgamma, log
    dof = n - 1
    x = dof / (dof + t * t)

    def betacf(a, b, xx):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, dd = 1.0, 1.0 - qab * xx / qap
        dd = 1.0 / dd if abs(dd) >= FPMIN else FPMIN
        h = dd
        for mm in range(1, MAXIT + 1):
            m2 = 2 * mm
            aa = mm * (b - mm) * xx / ((qam + m2) * (a + m2))
            dd = 1.0 + aa * dd
            dd = 1.0 / dd if abs(dd) >= FPMIN else FPMIN
            c = 1.0 + aa / c
            dd *= c
            h *= dd
            aa = -(a + mm) * (qab + mm) * xx / ((a + m2) * (a + m2))
            dd = 1.0 + aa * dd
            dd = 1.0 / dd if abs(dd) >= FPMIN else FPMIN
            c = 1.0 + aa / c
            dd *= c
            h *= dd
            if abs(dd - 1.0) < EPS:
                break
        return h

    def betai(a, b, xx):
        if xx <= 0.0:
            return 0.0
        if xx >= 1.0:
            return 1.0
        lb = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = np.exp(a * log(xx) + b * log(1.0 - xx) - lb) / a
        return front * betacf(a, b, xx) if xx < (a + 1.0) / (a + b + 2.0) \
            else 1.0 - front * betacf(b, a, 1.0 - xx)

    return t, float(betai(dof / 2.0, 0.5, x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--strengths", default="1.0,2.0")
    parser.add_argument("--state-dep", type=float, default=0.8)
    args = parser.parse_args()
    strengths = [float(x) for x in args.strengths.split(",")]

    print("== 边界一突破：非线性 COP 测试台上的 Christoffel 消融 ==")
    print(f"   {args.seeds} seeds × {len(strengths)} 强度档 | state_dep={args.state_dep}"
          f"（度量弯曲 → 联络非零）\n")

    for strength in strengths:
        off, on = [], []
        for seed in range(args.seeds):
            off.append(run_arm(0.0, 0.0, seed))
            on.append(run_arm(strength, args.state_dep, seed))
        print(f"--- strength={strength} ---")
        for kpi in ("cost", "viol", "rms"):
            o = np.array([r[kpi] for r in off])
            n = np.array([r[kpi] for r in on])
            t, p = paired_t(n - o)
            verdict = "*" if p < 0.05 else " "
            print(f"  {kpi:>5}: OFF={np.mean(o):9.4f}  ON={np.mean(n):9.4f}  "
                  f"Δ={np.mean(n - o):+8.4f}  p={p:.4f} {verdict}")
    print("\n判定（对照推翻条件）：任一 KPI 在任一强度档 p<0.05 且方向有利 → 负结论被推翻；")
    print("全部 p≥0.05 → '弯曲度量下 Christoffel 仍无增益'（边界更坚实）。")


if __name__ == "__main__":
    main()
