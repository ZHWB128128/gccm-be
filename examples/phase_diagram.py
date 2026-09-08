"""Phase diagram / condition sweep for the two positive results.

Answers the reviewer question left open in docs: over how wide a parameter range
does each gain survive, and where is its boundary? We sweep the *governing*
parameter of each mechanism and report the paired-t effect and p at every grid
point, so the region of validity is explicit rather than a single lucky point.

(A) Riemannian smoothness — sweep process-noise σ (the thing that creates the
    state velocity the kinetic term suppresses) × kinetic_weight. KPI: temp-ramp
    RMS reduction (ON−OFF, negative = better).

(B) Renormalization peak shaving — sweep cooling capacity q_cap (saturated vs
    constrained) × energy_weight. KPI: building-peak reduction (ON−OFF).

Run:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python3 examples/phase_diagram.py --which A --seeds 8
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python3 examples/phase_diagram.py --which B --seeds 8
"""
from __future__ import annotations

import argparse

import numpy as np

from examples.riemannian_ablation import paired_t_test
from examples.riemannian_smoothness import run_once as riemann_run
from examples.renormalization_peak import run_once as peak_run


def sweep_riemannian(seeds, noises, weights, horizon, state_dep):
    print("(A) 黎曼平滑度相图：温度爬升 RMS 削减 Δ(ON-OFF)，* = p<0.05 且 Δ<0")
    header = "  σ\\w " + "".join(f"{w:>10.1f}" for w in weights)
    print(header)
    for sigma in noises:
        cells = []
        for w in weights:
            off, on = [], []
            for s in range(seeds):
                off.append(riemann_run(s, False, w, state_dep, sigma, horizon).ramp_rms)
                on.append(riemann_run(s, True, w, state_dep, sigma, horizon).ramp_rms)
            diffs = np.array(on) - np.array(off)
            _, p, _ = paired_t_test(diffs)
            d = float(np.mean(diffs))
            tag = "*" if (p < 0.05 and d < 0) else " "
            cells.append(f"{d:>+9.4f}{tag}")
        print(f"{sigma:>5.1f} " + "".join(cells))


def sweep_renormalization(seeds, caps, ews, horizon, gain, noise):
    print("(B) 重整化削峰相图：建筑峰值削减 Δ(ON-OFF)，* = p<0.05 且 Δ<0")
    header = "cap\\ew " + "".join(f"{e:>10.1f}" for e in ews)
    print(header)
    for cap in caps:
        cells = []
        for ew in ews:
            off, on = [], []
            for s in range(seeds):
                off.append(peak_run(s, False, gain, noise, horizon, q_cap=cap,
                                    energy_weight=ew)["peak"])
                on.append(peak_run(s, True, gain, noise, horizon, q_cap=cap,
                                   energy_weight=ew)["peak"])
            diffs = np.array(on) - np.array(off)
            _, p, _ = paired_t_test(diffs)
            d = float(np.mean(diffs))
            tag = "*" if (p < 0.05 and d < 0) else " "
            cells.append(f"{d:>+9.4f}{tag}")
        print(f"{cap:>5.2f} " + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description="两个正结果的工况相图")
    ap.add_argument("--which", choices=["A", "B"], required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    # A params
    ap.add_argument("--noises", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--weights", type=float, nargs="*", default=[0.5, 1.0, 3.0, 6.0])
    ap.add_argument("--state-dep", type=float, default=0.0)
    # B params
    ap.add_argument("--caps", type=float, nargs="*", default=[1.2, 1.6, 2.5, 4.0])
    ap.add_argument("--ews", type=float, nargs="*", default=[2.5, 8.0, 15.0])
    ap.add_argument("--gain", type=float, default=5.0)
    ap.add_argument("--rg-noise", type=float, default=0.8)
    args = ap.parse_args()

    if args.which == "A":
        sweep_riemannian(args.seeds, args.noises, args.weights, args.horizon, args.state_dep)
    else:
        sweep_renormalization(args.seeds, args.caps, args.ews, args.horizon,
                              args.gain, args.rg_noise)


if __name__ == "__main__":
    main()
