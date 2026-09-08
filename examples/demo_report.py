"""GCCM 演示图表生成器：产出对比曲线/指标/诊断时间线图。

用法：
    PYTHONPATH=. python3 examples/demo_report.py --output output

产出（output/ 目录）：
    compare_temperature.png  24h 室温对比（GCCM vs 严格舒适 PID + 舒适带）
    compare_power_price.png  24h 制冷功率与电价（展示"谷时预冷蓄冷"机制）
    compare_metrics.png      电费/违规/峰值条形对比
    diagnosis_timeline.png   模式切换 + 置信度 + 不可判定时间线
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import SystemState

from examples.compare_baselines import make_simulator, SimResult
from examples.fair_compare import run_strict_pid
from examples.hard_benchmark import BenchmarkProvider

# 中文字体（存在则用，否则退回默认）
for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

COMFORT_MIN, COMFORT_MAX = 25.0, 27.0
SETPOINT = 26.0
STEP_H = 0.25  # 15 分钟步长，与 compare_baselines/fair_compare 一致
STEPS = 96     # 24 小时


def run_gccm_loop(sim, provider, horizon: int = 24) -> tuple[SimResult, list]:
    """GCCM 闭环：返回 SimResult + 每步决策（用于诊断时间线）。"""
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 35.0), "T_wall": (15.0, 35.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=horizon,
        dt=STEP_H,
        setpoints={"T_air": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=0.5,
        smooth_weight=0.1,
        enforce_comfort_constraints=True,
        use_kinetic=True,
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    prev = None
    prediction_error = 0.0
    times, t_air, q, p, price, modes, confs, undec = [], [], [], [], [], [], [], []
    for _ in range(STEPS):
        w = provider.get(t, 1)[0]
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort",
                              prediction_error=prediction_error)
        predicted = dec.predicted_next_state
        state_before = state.copy()
        state = sim.step(state, dec.control, w, STEP_H)
        if predicted is not None:
            prediction_error = float(np.max(np.abs(predicted.x - state.x)))
        engine.self_monitor.update(prediction_error)
        times.append(t)
        t_air.append(state.x[0])
        q.append(dec.control.u[0])
        p.append(sim.hvac.electrical_power(dec.control))
        price.append(w.price)
        modes.append(dec.mode)
        confs.append(dec.confidence)
        undec.append(dec.diagnosis.undecidable)
        prev = dec.control
        t += STEP_H
    result = SimResult(
        name="GCCM", times_h=np.array(times), t_air=np.array(t_air), q_hvac=np.array(q),
        p_elec=np.array(p), price=np.array(price), solve_times=[],
    )
    return result, {"modes": modes, "confidence": confs, "undecidable": undec}


def plot_compare_temperature(gccm: SimResult, pid: SimResult, out: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(COMFORT_MIN, COMFORT_MAX, color="green", alpha=0.12, label="Comfort band 25-27°C")
    ax.plot(pid.times_h, pid.t_air, color="gray", ls="--", lw=1.5, label="Strict comfort PID")
    ax.plot(gccm.times_h, gccm.t_air, color="tab:red", lw=2, label="GCCM")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Room temp (°C)")
    ax.set_title("24h temperature comparison (GCCM vs strict comfort PID)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_compare_power_price(gccm: SimResult, pid: SimResult, out: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.bar(pid.times_h, pid.q_hvac, width=STEP_H, color="gray", alpha=0.6, label="PID cooling power")
    ax1.bar(gccm.times_h, gccm.q_hvac, width=STEP_H, color="tab:red", alpha=0.8, label="GCCM cooling power")
    ax1.set_ylabel("Cooling power (kW)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.step(gccm.times_h, gccm.price, where="post", color="tab:blue", lw=1.5, label="Price")
    ax2.fill_between(gccm.times_h, 0, gccm.price, where=gccm.price >= np.percentile(gccm.price, 66),
                     color="orange", alpha=0.25, label="Peak hours")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Price (¥/kWh)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle("24h cooling power & price (GCCM pre-cools in valley hours, reduces peak use)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_compare_metrics(gccm: SimResult, pid: SimResult, out: str) -> None:
    def metrics(r: SimResult):
        cost = float(np.sum(r.p_elec * r.price * STEP_H))
        viol = float(np.mean((r.t_air > COMFORT_MAX) | (r.t_air < COMFORT_MIN)) * 100.0)
        peak = float(np.max(r.p_elec))
        return cost, viol, peak

    gc = metrics(gccm)
    pp = metrics(pid)
    labels = ["Cost (¥)", "Violation (%)", "Peak power (kW)"]
    x = np.arange(3)
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width / 2, pp, width, color="gray", alpha=0.75, label="Strict comfort PID")
    b2 = ax.bar(x + width / 2, gc, width, color="tab:red", alpha=0.85, label="GCCM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Metrics comparison (GCCM saves {100 * (pp[0] - gc[0]) / pp[0]:.1f}%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_diagnosis_timeline(gccm_result: SimResult, diag: dict, out: str) -> None:
    modes = diag["modes"]
    confs = diag["confidence"]
    undec = diag["undecidable"]
    t = gccm_result.times_h
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    mode_idx = {m: i for i, m in enumerate(sorted(set(modes)))}
    ax1.step(t, [mode_idx[m] for m in modes], where="post", color="tab:purple", lw=1.5)
    ax1.set_yticks(list(mode_idx.values()))
    ax1.set_yticklabels(list(mode_idx.keys()))
    ax1.set_ylabel("Mode")
    ax1.grid(alpha=0.3)
    ax2.plot(t, confs, color="tab:green", lw=1.5, label="Confidence")
    if any(undec):
        ux = [t[i] for i, u in enumerate(undec) if u]
        ax2.scatter(ux, [0.05] * len(ux), marker="x", color="red", s=40, label="Undecidable")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    fig.suptitle("GCCM decision timeline (mode / confidence / undecidable)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--q-max", type=float, default=8.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    provider = BenchmarkProvider(peak_temp=35.0, price_mode="spike", seed=0)
    sim = make_simulator(q_max=args.q_max)

    print("运行严格舒适 PID 基线 ...")
    pid = run_strict_pid(sim, provider)
    print("运行 GCCM 闭环（记录决策诊断）...")
    gccm, diag = run_gccm_loop(sim, provider, horizon=args.horizon)

    plot_compare_temperature(gccm, pid, os.path.join(args.output, "compare_temperature.png"))
    plot_compare_power_price(gccm, pid, os.path.join(args.output, "compare_power_price.png"))
    plot_compare_metrics(gccm, pid, os.path.join(args.output, "compare_metrics.png"))
    plot_diagnosis_timeline(gccm, diag, os.path.join(args.output, "diagnosis_timeline.png"))

    cost_g = float(np.sum(gccm.p_elec * gccm.price * STEP_H))
    cost_p = float(np.sum(pid.p_elec * pid.price * STEP_H))
    viol_g = float(np.mean((gccm.t_air > COMFORT_MAX) | (gccm.t_air < COMFORT_MIN)) * 100.0)
    viol_p = float(np.mean((pid.t_air > COMFORT_MAX) | (pid.t_air < COMFORT_MIN)) * 100.0)
    print()
    print(f"{'指标':<10}{'严格舒适 PID':>14}{'GCCM':>12}")
    print(f"{'电费(元)':<10}{cost_p:>14.2f}{cost_g:>12.2f}")
    print(f"{'违温(%)':<10}{viol_p:>14.1f}{viol_g:>12.1f}")
    print(f"{'峰值(kW)':<10}{float(np.max(pid.p_elec)):>14.2f}{float(np.max(gccm.p_elec)):>12.2f}")
    print(f"\n省电: {100 * (cost_p - cost_g) / cost_p:.1f}%")
    print(f"图表已保存至: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
