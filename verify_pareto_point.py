"""临时核对脚本：复现 pareto_sweep 中 energy=0.5, margin=0.3 的单个点。"""
import sys
sys.path.insert(0, "examples")
from compare_baselines import COMFORT_MAX, COMFORT_MIN, make_simulator, run_gccm
from hard_benchmark import BenchmarkProvider
from fair_compare import run_strict_pid

Q_MAX = 15.0
HORIZON = 24

provider = BenchmarkProvider(peak_temp=35.0, price_mode="standard")
sim = make_simulator(q_max=Q_MAX)
pid = run_strict_pid(sim, provider)
print(f"strict_pid: cost={pid.total_cost:.2f} viol={pid.comfort_violation*100:.1f} peak={pid.peak_power:.2f}")

r = run_gccm(
    provider, horizon=HORIZON, mode="comfort", comfort_band=0.0,
    comfort_margin=0.3, comfort_min=COMFORT_MIN, comfort_max=COMFORT_MAX,
    below_comfort_penalty=0.1, peak_energy_penalty=1.0,
    comfort_weight=5.0, energy_weight=0.5, smooth_weight=0.1,
    enforce_comfort=True, use_kinetic=True,
    constraint_options={"maxiter": 50, "ftol": 1e-6},
    solver_options={"maxiter": 60, "ftol": 1e-4, "maxls": 20},
    verbose=False,
)
print(f"gccm(ew=0.5,m=0.3): cost={r.total_cost:.2f} viol={r.comfort_violation*100:.1f} peak={r.peak_power:.2f}")
print(f"saving vs pid: {(pid.total_cost - r.total_cost) / pid.total_cost * 100:.1f}%")
