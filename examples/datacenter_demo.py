"""数据中心机房冷却 MPC 演示：GCCM vs 恒温器式规则控制。

展示：谷时蓄冷、峰时少开机组（电价套利）+ 冷通道温度守带 + 安全降级。
用法：PYTHONPATH=. python3 examples/datacenter_demo.py [--output output]
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
from gccm_be.physics.datacenter import (
    DataCenterProvider,
    DataCenterSimulator,
    DataCenterCoolingModel,
)
from gccm_be.types import SystemState

for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

STEP_H = 1.0 / 12.0
STEPS = 288  # 24h @ 5min
SETPOINT = 24.0
COMFORT_MIN, COMFORT_MAX = 22.0, 27.0
KP_RULE = 100.0     # 规则控制比例增益
T_TANK_TARGET = 12.0  # 规则控制蓄冷目标温度
CHG_PRICE_MAX = 0.9   # 低于此价充冷
DISC_PRICE_MIN = 2.0  # 高于此价放冷


def _ctrl(u0: float, u1: float):
    from gccm_be.types import ControlInput
    return ControlInput(np.array([u0, u1]), ["Q_chiller", "Q_tank"])


def run_rule(provider, sim: DataCenterSimulator, steps: int = STEPS) -> dict:
    """规则控制：冷通道恒温器 + 蓄冷罐按电价充/放冷（无前瞻）。"""
    state = SystemState([24.0, 22.0], ["T_aisle", "T_storage"])
    t = 0.0
    times, t_a, t_s, q, p, price = [], [], [], [], [], []
    for _ in range(steps):
        w = provider.get(t, 1)[0]
        err = state.x[0] - SETPOINT
        u0 = -float(np.clip(KP_RULE * err, 0.0, -sim.hvac.q_min)) if err > 0.2 else 0.0
        if w.price <= CHG_PRICE_MAX and state.x[1] > T_TANK_TARGET:
            u1 = sim.hvac.chg_max  # 低价充冷
        elif w.price >= DISC_PRICE_MIN and state.x[1] < state.x[0] - 1.0 and err > 0.2:
            u1 = -sim.hvac.disc_max  # 峰时放冷
        else:
            u1 = 0.0
        ctrl = _ctrl(u0, u1)
        state = sim.step(state, ctrl, w, STEP_H)
        times.append(t); t_a.append(state.x[0]); t_s.append(state.x[1])
        q.append(ctrl.u[0]); p.append(sim.hvac.electrical_power(ctrl)); price.append(w.price)
        t += STEP_H
    return {"times": np.array(times), "t_aisle": np.array(t_a), "t_tank": np.array(t_s),
            "q": np.array(q), "p": np.array(p), "price": np.array(price)}


def run_gccm(provider, sim: DataCenterSimulator, steps: int = STEPS) -> dict:
    manifold = StateManifold(
        labels=["T_aisle", "T_storage"],
        units={"T_aisle": "°C", "T_storage": "°C"},
        bounds={"T_aisle": (15.0, 40.0), "T_storage": (5.0, 40.0)},
        scale={"T_aisle": 5.0, "T_storage": 5.0},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=48,  # 4h 预测时域：覆盖"谷时蓄冷→峰时放冷"的决策窗口
        dt=STEP_H,
        setpoints={"T_aisle": SETPOINT},
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        comfort_margin=0.8,  # 预防性裕度：预测约束收紧，吸收 IT 负载噪声，保证实际守带
        below_comfort_penalty=0.2,
        peak_energy_penalty=1.5,
        comfort_weight=5.0,
        energy_weight=1.5,
        smooth_weight=1e-5,  # u1 量级 ±150，平滑项需极低以免杀死充/放冷切换
        storage_targets={"T_storage": 10.0},  # 蓄冷罐储存价值：罐偏暖即惩罚，避免短视排空
        storage_weight=10.0,
        enforce_comfort_constraints=True,
        use_kinetic=True,
        safe_control_mode="feedback",
        solver_options={"maxiter": 80, "ftol": 1e-5, "maxls": 20},
    )
    state = SystemState([24.0, 22.0], ["T_aisle", "T_storage"])
    t = 0.0
    prev = None
    prediction_error = 0.0
    times, t_a, t_s, q, p, price = [], [], [], [], [], []
    for _ in range(steps):
        w = provider.get(t, 1)[0]
        dec = engine.optimize(state, t, prev_control=prev, forced_mode="comfort",
                              prediction_error=prediction_error)
        predicted = dec.predicted_next_state
        state_before = state.copy()
        state = sim.step(state, dec.control, w, STEP_H)
        if predicted is not None:
            prediction_error = float(np.max(np.abs(predicted.x - state.x)))
        engine.self_monitor.update(prediction_error)
        times.append(t); t_a.append(state.x[0]); t_s.append(state.x[1])
        q.append(dec.control.u[0]); p.append(sim.hvac.electrical_power(dec.control)); price.append(w.price)
        prev = dec.control
        t += STEP_H
    return {"times": np.array(times), "t_aisle": np.array(t_a), "t_tank": np.array(t_s),
            "q": np.array(q), "p": np.array(p), "price": np.array(price)}


def _ctrl(u0: float, u1: float):
    from gccm_be.types import ControlInput
    return ControlInput(np.array([u0, u1]), ["Q_chiller", "Q_tank"])


def _metrics(r: dict) -> tuple:
    cost = float(np.sum(r["p"] * r["price"] * STEP_H))
    viol = float(np.mean((r["t_aisle"] > COMFORT_MAX) | (r["t_aisle"] < COMFORT_MIN)) * 100.0)
    peak = float(np.max(r["p"]))
    return cost, viol, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--peak-price", type=float, default=3.0, help="峰时电价（元/kWh），越大套利空间越大")
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    provider = DataCenterProvider(peak_price=args.peak_price)
    sim = DataCenterSimulator(DataCenterCoolingModel())

    print("运行规则控制（恒温器式）...")
    rule = run_rule(provider, sim, steps=args.steps)
    print("运行 GCCM MPC（谷时蓄冷）...")
    gccm = run_gccm(provider, sim, steps=args.steps)

    rc, rv, rp = _metrics(rule)
    gc, gv, gp = _metrics(gccm)
    print()
    print(f"{'指标':<14}{'规则控制':>12}{'GCCM':>12}")
    print(f"{'制冷电费(元)':<14}{rc:>12.0f}{gc:>12.0f}")
    print(f"{'冷通道违温(%)':<14}{rv:>12.1f}{gv:>12.1f}")
    print(f"{'峰值电功率(kW)':<14}{rp:>12.1f}{gp:>12.1f}")
    print(f"\n省电: {100 * (rc - gc) / rc:.1f}%   削峰: {100 * (rp - gp) / rp:.1f}%")

    # 图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    t = gccm["times"]
    ax1.plot(t, rule["t_aisle"], color="gray", ls="--", lw=1.2, label="Rule control: cold aisle")
    ax1.plot(t, gccm["t_aisle"], color="tab:red", lw=1.6, label="GCCM: cold aisle")
    ax1.plot(t, gccm["t_tank"], color="tab:blue", lw=1.2, label="GCCM: storage tank")
    ax1.axhspan(COMFORT_MIN, COMFORT_MAX, color="green", alpha=0.1, label="Comfort band 22-27°C")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.step(t, gccm["price"], where="post", color="tab:purple", lw=1.2, label="Price")
    ax2.set_ylabel("Price (¥/kWh)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.step(t, gccm["q"], where="post", color="tab:red", lw=1.0, label="GCCM cooling power")
    ax3.step(t, rule["q"], where="post", color="gray", lw=1.0, ls="--", label="Rule control cooling power")
    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel("Cooling power (kW)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle(f"Data center cooling MPC: saves {100 * (rc - gc) / rc:.1f}% cost / "
                 f"shaves {100 * (rp - gp) / rp:.1f}% peak (charge at valley, discharge at peak)",
                 fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(args.output, "datacenter_cooling.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n图表已保存: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
