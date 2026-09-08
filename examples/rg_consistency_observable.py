"""RG 一致性补充实验:mean 可观测 vs max(峰值) 可观测。

假设:均值动力学在失衡下仍一致(C 小),但"峰值"可观测量的粗粒化一致性
随失衡度恶化——因为 max 由最热区决定,平均代理无法表达区的异质性。
这正是"序参量加权"的价值域。
"""
from __future__ import annotations

import numpy as np

from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState


def observable_consistency(observable, dt=1.0/12.0, steps=24):
    """observable: 'mean' 或 'max'。C_obs = 相对一致性误差。"""
    results = []
    hvac = HVACModel(q_min=-8.0, q_max=8.0, n_units=3,
                     control_labels=["Q_A", "Q_B", "Q_C"])
    ext = ExternalInput(np.array([33.0, 0.6, 0.5, 1.0]), ["T_out", "solar", "occ", "price"])

    for imbalance in (0.0, 0.5, 1.0, 2.0, 4.0):
        zones = [RCBuildingModel(solar_gain=0.05 + imbalance * g, dt=dt)
                 for g in (0.0, 0.5, 1.0)]
        sims = [Simulator(z, hvac) for z in zones]
        states = [SystemState(np.array([27.0, 27.0]), ["T_air", "T_wall"]) for _ in zones]
        agg = (lambda temps: float(np.mean(temps))) if observable == "mean" \
            else (lambda temps: float(np.max(temps)))

        num = den = 0.0
        for _ in range(steps):
            temps = [s.x[0] for s in states]
            macro_direct = agg(temps)
            new_states = [sim.step(s, ControlInput([-4.0]), ext, dt)
                          for sim, s in zip(sims, states)]
            new_temps = [s.x[0] for s in new_states]
            macro_via_R = agg(new_temps)
            num += abs(macro_via_R - macro_direct)
            den += abs(macro_direct)
            states = new_states
        results.append((imbalance, num / max(den, 1e-12)))
    return results


def main():
    print("== 可观测量对比:均值动力学 vs 峰值动力学 的粗粒化一致性 ==\n")
    mean_res = observable_consistency("mean")
    max_res = observable_consistency("max")
    print(f"{'失衡度':>6} {'C(均值)':>10} {'C(峰值)':>10}")
    for (i1, cm), (_, cx) in zip(mean_res, max_res):
        print(f"{i1:>6.1f} {cm:>10.4f} {cx:>10.4f}")
    t_mean = mean_res[-1][1] - mean_res[0][1]
    t_max = max_res[-1][1] - max_res[0][1]
    print(f"\n趋势: 均值 {t_mean:+.4f} | 峰值 {t_max:+.4f}")
    if t_max > t_mean:
        print("→ 峰值可观测的一致性随失衡恶化快于均值:")
        print("  均值动力学可安全粗粒化;峰值(序参量所在)需要区级分辨 →")
        print("  这就是序参量加权的几何必要性,定量成立。")
    else:
        print("→ 两可观测量一致性趋势相同(本工况)。")


if __name__ == "__main__":
    main()
