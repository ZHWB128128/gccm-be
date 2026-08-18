"""Riemannian 修正开关对比实验。"""
from __future__ import annotations

from compare_baselines import (
    COMFORT_MAX,
    COMFORT_MIN,
    make_simulator,
    run_gccm,
)
from hard_benchmark import BenchmarkProvider

Q_MAX = 15.0
HORIZON = 12


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=0.0)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--state-dep", type=float, default=0.0)
    parser.add_argument("--ric", action="store_true")
    parser.add_argument("--ric-weight", type=float, default=1.0)
    args = parser.parse_args()
    provider = BenchmarkProvider(peak_temp=35.0, price_mode="standard", seed=args.seed, noise_std=args.noise)
    sim = make_simulator(q_max=Q_MAX)

    print("运行 use_riemannian=False ...")
    r_off = run_gccm(
        provider, horizon=HORIZON, mode="comfort", comfort_band=0.0,
        comfort_margin=0.3, comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1, peak_energy_penalty=1.0,
        comfort_weight=5.0, energy_weight=0.5, smooth_weight=0.1,
        enforce_comfort=True, use_kinetic=True, use_riemannian=False,
        constraint_options={"maxiter": 50, "ftol": 1e-6},
        solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
        verbose=False,
    )
    print("运行 use_riemannian=True ...")
    r_on = run_gccm(
        provider, horizon=HORIZON, mode="comfort", comfort_band=0.0,
        comfort_margin=0.3, comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1, peak_energy_penalty=1.0,
        comfort_weight=5.0, energy_weight=0.5, smooth_weight=0.1,
        enforce_comfort=True, use_kinetic=True, use_riemannian=True, riemannian_strength=args.strength,
        metric_coupling=args.coupling, metric_state_dependence=args.state_dep,
        geodesic_penalty_weight=args.penalty,
        adaptive_riemannian=args.adaptive,
        use_riemannian_control=args.ric,
        riemannian_control_weight=args.ric_weight,
        constraint_options={"maxiter": 50, "ftol": 1e-6},
        solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
        verbose=False,
    )

    print("\n" + "=" * 60)
    print(f"{'配置':<24}{'电费':>8}{'违规%':>8}{'峰值':>8}")
    print("-" * 60)
    print(f"{'riemannian=False':<24}{r_off.total_cost:>8.2f}{r_off.comfort_violation*100:>8.1f}{r_off.peak_power:>8.2f}")
    print(f"{'riemannian=True':<24}{r_on.total_cost:>8.2f}{r_on.comfort_violation*100:>8.1f}{r_on.peak_power:>8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
