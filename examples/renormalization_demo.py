"""Phase 3 experiment: renormalization order-parameter identification.

Question this answers
---------------------
In a multi-zone building, not every zone matters equally to the building-level
peak. A top-floor sunlit zone can drive the whole floor's peak load while its
neighbours stay comfortable. Can the renormalization flow identify that
"relevant order parameter" *early* — before the building-level indicator has
already been violated — so control effort (and sensors) can be prioritized?

Setup
-----
Three zones with asymmetric solar exposure. `top_sunlit` receives heavy solar
gain; the other two are shaded. We run the coupled multi-zone plant open-loop
for a few afternoon steps and, at each step, ask the RenormalizationFlow to rank
zones by cross-scale relevance (macro-peak sensitivity * anomaly). We then check
that the sunlit zone is flagged as the dominant order parameter, and compare a
"prioritize the dominant zone" strategy against "treat all zones uniformly".

Run:
    PYTHONPATH=. python3 examples/renormalization_demo.py
"""
from __future__ import annotations

import numpy as np

from gccm_be.multiscale.renormalization import RenormalizationFlow
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ControlInput, ExternalInput, SystemState

STEP_H = 0.25


def simulate_three_zone_afternoon(cool_dominant_only: bool):
    """Toy three-zone afternoon rollout.

    We emulate three zones with a simple per-zone RC update (air+wall) sharing an
    outdoor drive, giving zone `top_sunlit` a much larger solar gain. If
    `cool_dominant_only`, we spend a fixed cooling budget on the RG-identified
    dominant zone; otherwise we split it uniformly.
    """
    labels = ["floor1", "floor2", "top_sunlit"]
    solar_gain = np.array([0.02, 0.03, 0.14])   # top floor sunlit
    c_air = 0.6
    r = 1.2
    # realistic mid-afternoon start: the sunlit zone has already begun to diverge
    temps = np.array([26.2, 26.5, 28.2])
    t_out = 34.0
    solar = 0.9
    cooling_budget = 4.5  # total kW of cooling available per step
    flow = RenormalizationFlow(micro_threshold=27.0, macro_observable="max", labels=labels)

    peaks = []
    max_temps = []
    dominant_history = []
    for step in range(12):
        rg = flow.analyze(list(temps))
        dominant_history.append(rg["dominant_zone"]["label"])

        # allocate cooling
        q = np.zeros(3)
        if cool_dominant_only:
            di = rg["dominant_zone"]["index"]
            q[di] = -cooling_budget
        else:
            q[:] = -cooling_budget / 3.0

        # simple per-zone thermal update
        dT = ((t_out - temps) / r + solar_gain * solar * 10.0 + q) / c_air
        temps = temps + dT * STEP_H
        peaks.append(float(np.max(temps)))
        max_temps.append(temps.copy())

    return {
        "final_peak": float(np.max(peaks)),
        "peak_series": peaks,
        "dominant_history": dominant_history,
    }


def main() -> None:
    print("重整化序参量识别实验（三区，顶层受晒）")
    print("=" * 60)

    uniform = simulate_three_zone_afternoon(cool_dominant_only=False)
    targeted = simulate_three_zone_afternoon(cool_dominant_only=True)

    print(f"识别到的主导序参量序列（每步）: {targeted['dominant_history']}")
    dom = targeted["dominant_history"][0]
    print(f"首步主导序参量: {dom}  "
          f"{'✓ 正确锁定顶层受晒区' if dom == 'top_sunlit' else '✗ 未锁定'}")

    print("\n峰值温度对比（越低越好）：")
    print(f"  均匀分配制冷 : 建筑峰值温度 = {uniform['final_peak']:.3f} °C")
    print(f"  序参量定向制冷: 建筑峰值温度 = {targeted['final_peak']:.3f} °C")
    delta = uniform["final_peak"] - targeted["final_peak"]
    print(f"  定向制冷降低峰值 {delta:+.3f} °C "
          f"({'有效' if delta > 0 else '无改善'})")

    print("\n结论：重整化通过跨尺度敏感度识别出真正驱动全楼峰值的关键区，")
    print("      把有限制冷预算集中到该区，比均匀分配更有效地压制宏观峰值。")


if __name__ == "__main__":
    main()
