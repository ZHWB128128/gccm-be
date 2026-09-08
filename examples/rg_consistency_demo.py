"""RG cross-scale dynamical consistency check (P3, review-recommended).

把"重整化"从哲学概念固化为可检验的数学对象:

    C = || R f_micro(z) - f_macro(R z) || / || f_macro(R z) ||

R 是微→宏粗粒化算子(区均值),f_micro/f_macro 是各自尺度的单步动力学。
C 小(相对一致)= 粗粒化后动力学规律近似保持,尺度映射成立;C 随微区
失衡度(受晒差异)上升 = "序参量加权在失衡大时最有价值"的定量依据。

用法: PYTHONPATH=. python3 examples/rg_consistency_demo.py
"""
from __future__ import annotations

import numpy as np

from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState


def run_consistency_experiment(dt: float = 1.0 / 12.0, steps: int = 24):
    """微区(3 个独立 RC,不同太阳得热=失衡)→ 楼栋(单区代理,平均得热)。

    C = Σ|R·f_micro(z) − f_macro(R·z)| / Σ|f_macro(R·z)|,演化 steps 步累计。
    """
    results = []
    hvac = HVACModel(q_min=-8.0, q_max=8.0, n_units=3,
                     control_labels=["Q_A", "Q_B", "Q_C"])
    ext = ExternalInput(np.array([33.0, 0.6, 0.5, 1.0]), ["T_out", "solar", "occ", "price"])

    for imbalance in (0.0, 0.5, 1.0, 2.0, 4.0):
        zones = [RCBuildingModel(solar_gain=0.05 + imbalance * g, dt=dt)
                 for g in (0.0, 0.5, 1.0)]
        sims = [Simulator(z, hvac) for z in zones]
        states = [SystemState(np.array([27.0, 27.0]), ["T_air", "T_wall"])
                  for _ in zones]

        def macro_step(mean_t: float) -> float:
            # 宏观代理:平均温度在平均得热下演化(单区 RC)
            proxy = RCBuildingModel(solar_gain=0.05 + imbalance * 0.5, dt=dt)
            s = SystemState(np.array([mean_t, mean_t]), ["T_air", "T_wall"])
            return float(proxy.step(s, ControlInput([-4.0]), ext, dt).x[0])

        num = den = 0.0
        for _ in range(steps):
            mean_t = float(np.mean([s.x[0] for s in states]))
            macro_direct = macro_step(mean_t)
            new_states = [sim.step(s, ControlInput([-4.0]), ext, dt)
                          for sim, s in zip(sims, states)]
            mean_via_R = float(np.mean([s.x[0] for s in new_states]))
            num += abs(mean_via_R - macro_direct)
            den += abs(macro_direct)
            states = new_states
        c = num / max(den, 1e-12)
        results.append((imbalance, c))
        print(f"失衡度 {imbalance:.1f}: 一致度 C = {c:.4f}")
    return results


def main():
    print("== RG 跨尺度动力学一致性:微区均值 → 楼栋代理 ==")
    print("问题: 粗粒化后动力学是否近似保持? C 小 = 保持\n")
    res = run_consistency_experiment()
    baseline = res[0][1]
    worst = max(c for _, c in res)
    print("\n判定:")
    print(f"  均衡工况 C={baseline:.4f} — {'成立' if baseline < 0.10 else '不成立'}(<0.10 阈值)")
    print(f"  失衡最大 C={worst:.4f} — 微宏观动力学{'保持' if worst < 0.20 else '出现独立宏观动力学'}")
    trend = res[-1][1] - baseline
    print(f"  C 随失衡度变化 {trend:+.4f} → "
          f"{'失衡放大不一致(序参量加权的定量依据)' if trend > 0 else '粗粒化跨失衡稳健'}")


if __name__ == "__main__":
    main()
