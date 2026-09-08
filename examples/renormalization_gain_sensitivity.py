"""Phase-B follow-up: sensitivity of the renormalization peak-shaving gain to the
per-zone weighting strength `gain`, and a theoretical account of why an optimum
exists.

Motivation
----------
The per-zone comfort multiplier is
    mult_i = 1 + gain · max(0, (rel_i − mean_rel) / mean_rel)
This is a *heuristic* map from cross-scale relevance to objective weight. A
reviewer will ask: is `gain` a free knob fished for significance, or does the
peak reduction have a stable, physically-bounded optimum? This script answers by
sweeping `gain` over a wide range at fixed physics and reporting the paired-t
peak reduction at each value.

Theory (what we expect, stated BEFORE the run)
----------------------------------------------
- gain = 0  → uniform weighting, Δpeak = 0 by construction.
- small gain → cooling is redirected toward the relevant (peak-driving) zone;
  peak falls. Marginal benefit is largest here.
- large gain → the multiplier saturates the objective on the single hottest zone
  and *starves* the others, so the non-relevant zones drift up and the building
  peak (a max over zones) stops improving or reverses. So we expect a concave
  curve with an interior optimum, not monotone improvement — the hallmark of a
  genuine tradeoff rather than a p-hacked knob.

Run:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python3 examples/renormalization_gain_sensitivity.py --seeds 12
"""
from __future__ import annotations

import argparse

import numpy as np

from examples.renormalization_peak import run_once
from examples.riemannian_ablation import paired_t_test


def sweep(seeds, gains, noise, horizon):
    rows = []
    # baseline OFF result is gain-independent; compute once per seed and reuse
    off_peak = np.array([run_once(s, False, 0.0, noise, horizon)["peak"] for s in range(seeds)])
    for g in gains:
        on_peak = np.array([run_once(s, True, g, noise, horizon)["peak"] for s in range(seeds)])
        diffs = on_peak - off_peak
        t, p, dof = paired_t_test(diffs)
        rows.append((g, float(np.mean(off_peak)), float(np.mean(on_peak)),
                     float(np.mean(diffs)), p))
    return rows


def main():
    ap = argparse.ArgumentParser(description="重整化加权 gain 敏感性分析")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--gains", type=float, nargs="*",
                    default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    args = ap.parse_args()

    print(f"gain 敏感性：seeds={args.seeds}, noise={args.noise}, horizon={args.horizon}")
    rows = sweep(args.seeds, args.gains, args.noise, args.horizon)

    print("\n" + "=" * 62)
    print(f"{'gain':>6}{'OFF峰值':>12}{'ON峰值':>12}{'Δ(ON-OFF)':>14}{'p 值':>10}")
    print("-" * 62)
    best = None
    for g, a, b, d, p in rows:
        mark = "*" if (p < 0.05 and d < 0) else ""
        print(f"{g:>6.1f}{a:>12.4f}{b:>12.4f}{d:>+14.4f}{p:>10.4f}{mark:>3}")
        if p < 0.05 and d < 0 and (best is None or d < best[1]):
            best = (g, d, p)
    print("=" * 62)

    if best is not None:
        print(f"\n最优 gain≈{best[0]:.1f}：峰值削减 {abs(best[1]):.4f} °C（p={best[2]:.4f}）。")
        # check concavity / saturation: is the largest gain worse than the optimum?
        last = rows[-1]
        if last[3] > best[1] + 1e-6:
            print(f"  gain 过大（{last[0]:.1f}）时削减回落至 {abs(last[3]):.4f} °C，"
                  "证实存在内部最优、非单调——符合'过度加权饿死其他区'的理论预期，"
                  "说明 gain 不是任意拟合旋钮。")
        else:
            print("  在测试范围内未观察到明显回落；如实报告：最优在区间端点或更大处。")
    else:
        print("\n未找到显著削减的 gain：如实报告该工况下方法无效。")


if __name__ == "__main__":
    main()
