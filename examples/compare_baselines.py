"""GCCM-BE 最小对比实验：规则控制 vs PID 控制 vs GCCM 控制。

场景:
    24 小时，15 分钟步长（96 步）
    夏季室外温度正弦变化，三段式峰谷电价
    初始室内/墙体 28°C，设定温度 26°C
    舒适区间 25~27°C

运行:
    PYTHONPATH=. python3 examples/compare_baselines.py
    PYTHONPATH=. python3 examples/compare_baselines.py --horizon 48 --no-plot
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState

# 夏季场景参数
STEP_H = 0.25          # 15 分钟
STEPS = 96             # 24 小时
SETPOINT = 26.0
COMFORT_MIN = 25.0
COMFORT_MAX = 27.0

# 为保证夏季峰值时段仍有足够制冷能力，将 Q_max 设为 8 kW（热功率）
Q_MAX = 8.0
RULE_COOLING_THERMAL_KW = 3.0


class ScenarioExternalInputProvider(ExternalInputProvider):
    """按实验场景生成外部输入：T_out / solar / occ / price。"""

    def __init__(self, dt_h: float = STEP_H) -> None:
        self.dt_h = dt_h

    def get(self, time_h: float, horizon: int = 1) -> List[ExternalInput]:
        result: List[ExternalInput] = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0

            # 室外温度：凌晨约 24°C，午后最高约 35°C
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)

            # 太阳辐射：6:00-18:00
            if 6.0 <= hour <= 18.0:
                solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0))
            else:
                solar = 0.0

            # 室内人员/设备热源
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3

            # 三段式电价
            if hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5

            result.append(ExternalInput(
                np.array([t_out, solar, occ, price]),
                ["T_out", "solar", "occ", "price"],
            ))
        return result


@dataclass
class SimResult:
    name: str
    times_h: np.ndarray
    t_air: np.ndarray
    q_hvac: np.ndarray          # 热功率，正为加热，负为制冷
    p_elec: np.ndarray
    price: np.ndarray
    solve_times: List[float] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.p_elec * self.price * STEP_H))

    @property
    def comfort_violation(self) -> float:
        viol = np.sum((self.t_air > COMFORT_MAX) | (self.t_air < COMFORT_MIN))
        return float(viol / self.t_air.size)

    @property
    def peak_power(self) -> float:
        return float(np.max(self.p_elec))

    @property
    def avg_solve_time(self) -> float:
        return float(np.mean(self.solve_times)) if self.solve_times else 0.0


def make_simulator(q_max: float = Q_MAX, heavy: bool = False) -> Simulator:
    if heavy:
        building = RCBuildingModel(
            c_air=0.8,
            c_wall=8.0,
            r_air=1.2,
            r_wall=3.0,
            solar_gain=0.03,
        )
    else:
        building = RCBuildingModel()
    return Simulator(building, HVACModel(q_min=-q_max, q_max=q_max))


def run_rule(sim: Simulator, provider: ScenarioExternalInputProvider) -> SimResult:
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    on = False
    times, t_air, q, p, price = [], [], [], [], []

    for _ in range(STEPS):
        w = provider.get(t, 1)[0]
        if state.x[0] > 26.5:
            on = True
        elif state.x[0] < 25.5:
            on = False

        u = -RULE_COOLING_THERMAL_KW if on else 0.0
        control = ControlInput([u], ["Q_hvac"])
        state = sim.step(state, control, w, STEP_H)

        times.append(t)
        t_air.append(state.x[0])
        q.append(u)
        p.append(sim.hvac.electrical_power(control))
        price.append(w.w[3])
        t += STEP_H

    return SimResult(
        name="规则控制",
        times_h=np.array(times),
        t_air=np.array(t_air),
        q_hvac=np.array(q),
        p_elec=np.array(p),
        price=np.array(price),
    )


def run_pid(sim: Simulator, provider: ScenarioExternalInputProvider) -> SimResult:
    # 简单实用 PID：只制冷，输出限制在 0~Q_MAX，带积分限幅
    kp, ki, kd = 1.0, 0.05, 0.2
    integral_limit = 2.0

    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    integral = 0.0
    prev_error = 0.0
    times, t_air, q, p, price = [], [], [], [], []

    for _ in range(STEPS):
        w = provider.get(t, 1)[0]
        error = state.x[0] - SETPOINT
        integral = float(np.clip(integral + error * STEP_H, -integral_limit, integral_limit))
        derivative = (error - prev_error) / STEP_H
        output = kp * error + ki * integral + kd * derivative
        cooling = float(np.clip(output, 0.0, Q_MAX))
        u = -cooling
        control = ControlInput([u], ["Q_hvac"])
        state = sim.step(state, control, w, STEP_H)

        times.append(t)
        t_air.append(state.x[0])
        q.append(u)
        p.append(sim.hvac.electrical_power(control))
        price.append(w.w[3])
        prev_error = error
        t += STEP_H

    return SimResult(
        name="PID 控制",
        times_h=np.array(times),
        t_air=np.array(t_air),
        q_hvac=np.array(q),
        p_elec=np.array(p),
        price=np.array(price),
    )


def run_gccm(
    provider: ScenarioExternalInputProvider,
    horizon: int = STEPS,
    mode: str = "comfort",
    simulator: Optional[Simulator] = None,
    plant_provider: Optional[ScenarioExternalInputProvider] = None,
    comfort_band: float = 0.0,
    comfort_margin: float = 0.0,
    noise_adaptive_margin: bool = False,
    comfort_min: Optional[float] = 25.0,
    comfort_max: Optional[float] = 27.0,
    below_comfort_penalty: float = 0.0,
    peak_energy_penalty: float = 1.5,
    comfort_weight: Optional[float] = None,
    energy_weight: Optional[float] = None,
    smooth_weight: Optional[float] = None,
    enforce_comfort: bool = False,
    use_kinetic: bool = True,
    use_riemannian: bool = False,
    use_riemannian_control: bool = False,
    riemannian_control_weight: float = 1.0,
    riemannian_strength: float = 1.0,
    metric_coupling: float = 0.0,
    metric_state_dependence: float = 0.0,
    geodesic_penalty_weight: float = 0.0,
    adaptive_riemannian: bool = False,
    solver_options: Optional[dict] = None,
    constraint_options: Optional[dict] = None,
    verbose: bool = True,
) -> SimResult:
    sim = simulator if simulator is not None else make_simulator()
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 35.0), "T_wall": (15.0, 35.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    if solver_options is None:
        solver_options = {"maxiter": 80, "ftol": 1e-5, "maxls": 20}

    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=horizon,
        dt=STEP_H,
        setpoints={"T_air": SETPOINT},
        comfort_band=comfort_band,
        comfort_margin=comfort_margin,
        noise_adaptive_margin=noise_adaptive_margin,
        comfort_min=comfort_min,
        comfort_max=comfort_max,
        below_comfort_penalty=below_comfort_penalty,
        peak_energy_penalty=peak_energy_penalty,
        comfort_weight=comfort_weight,
        energy_weight=energy_weight,
        smooth_weight=smooth_weight,
        enforce_comfort_constraints=enforce_comfort,
        use_kinetic=use_kinetic,
        use_riemannian=use_riemannian,
        use_riemannian_control=use_riemannian_control,
        riemannian_control_weight=riemannian_control_weight,
        riemannian_strength=riemannian_strength,
        metric_coupling=metric_coupling,
        metric_state_dependence=metric_state_dependence,
        geodesic_penalty_weight=geodesic_penalty_weight,
        adaptive_riemannian=adaptive_riemannian,
        solver_options=solver_options,
        constraint_options=constraint_options or {"maxiter": 50, "ftol": 1e-6},
    )

    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    prev_control: Optional[ControlInput] = None
    prediction_error = 0.0
    times, t_air, q, p, price = [], [], [], [], []
    solve_times: List[float] = []

    for i in range(STEPS):
        start = time.perf_counter()
        decision = engine.optimize(
            state, t, prev_control=prev_control, forced_mode=mode,
            prediction_error=prediction_error,
        )
        solve_times.append(time.perf_counter() - start)
        actual_provider = plant_provider if plant_provider is not None else provider
        w = actual_provider.get(t, 1)[0]

        predicted = decision.predicted_next_state
        state_before = state.copy()
        state = sim.step(state, decision.control, w, STEP_H)
        if predicted is not None:
            # 求解失败步（predicted=None）不污染误差通道
            prediction_error = float(np.max(np.abs(predicted.x - state.x)))
            signed_error = float(state.x[0] - predicted.x[0])
            engine.online_identifier.update(np.array([1.0]), signed_error)
            engine.model_bias = float(engine.online_identifier.theta[0])
        else:
            prediction_error = 0.0
        if hasattr(engine, "self_monitor") and engine.self_monitor is not None:
            engine.self_monitor.update(prediction_error)
        engine.observe_step(state_before, decision.control, w, state, STEP_H)
        engine.apply_rc_identification(min_samples=30)
        times.append(t)
        t_air.append(state.x[0])
        q.append(decision.control.u[0])
        p.append(sim.hvac.electrical_power(decision.control))
        price.append(w.price)
        prev_control = decision.control
        t += STEP_H

        if verbose and (i + 1) % 16 == 0:
            avg = float(np.mean(solve_times[-16:]))
            print(f"  GCCM 进度 {i + 1}/{STEPS}，当前室温 {state.x[0]:.2f}°C，"
                  f"近 16 次平均求解 {avg:.2f}s")

    return SimResult(
        name=f"GCCM 控制 ({mode})",
        times_h=np.array(times),
        t_air=np.array(t_air),
        q_hvac=np.array(q),
        p_elec=np.array(p),
        price=np.array(price),
        solve_times=solve_times,
    )


def print_metrics(results: List[SimResult]) -> None:
    print("\n" + "=" * 72)
    print(f"{'方法':<18}{'总电费(元)':>12}{'舒适违规(%)':>12}{'峰值功率(kW)':>14}{'平均求解(s)':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r.name:<18}{r.total_cost:>12.2f}{r.comfort_violation * 100:>12.1f}"
              f"{r.peak_power:>14.2f}{r.avg_solve_time:>12.3f}")
    print("=" * 72)


def plot_results(results: List[SimResult], output: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    for r in results:
        ax.plot(r.times_h, r.t_air, label=r.name, linewidth=1.8)
    ax.axhspan(COMFORT_MIN, COMFORT_MAX, color="green", alpha=0.15, label="舒适区间 25~27°C")
    ax.axhline(SETPOINT, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("室内温度 (°C)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in results:
        ax.plot(r.times_h, r.p_elec, label=r.name, linewidth=1.8)
    ax.set_ylabel("电功率 (kW)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for r in results:
        ax.plot(r.times_h, np.cumsum(r.p_elec * r.price * STEP_H), label=r.name, linewidth=1.8)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("累计电费 (元)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
        print(f"\n图片已保存: {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="GCCM-BE 最小对比实验")
    parser.add_argument("--horizon", type=int, default=STEPS,
                        help="GCCM 预测时域步数，默认 96（24 小时）")
    parser.add_argument("--gccm-mode", default="comfort",
                        choices=["comfort", "balanced", "energy", "demand_response"],
                        help="GCCM 固定运行模式，默认 comfort")
    parser.add_argument("--comfort-band", type=float, default=0.0,
                        help="能量景观舒适死区半宽，0=严格跟踪设定点")
    parser.add_argument("--comfort-min", type=float, default=25.0,
                        help="夏季舒适下限，默认 25.0")
    parser.add_argument("--comfort-max", type=float, default=27.0,
                        help="夏季舒适上限，默认 27.0")
    parser.add_argument("--below-penalty", type=float, default=0.0,
                        help="低于舒适下限的惩罚系数，0 表示不惩罚")
    parser.add_argument("--peak-penalty", type=float, default=1.5,
                        help="峰时电价放大系数，>1 表示峰时用电更贵")
    parser.add_argument("--comfort-weight", type=float, default=None,
                        help="覆盖 comfort 权重")
    parser.add_argument("--energy-weight", type=float, default=None,
                        help="覆盖 energy 权重")
    parser.add_argument("--smooth-weight", type=float, default=None,
                        help="覆盖 smooth 权重")
    parser.add_argument("--enforce-comfort", action="store_true",
                        help="在求解器中加入 25~27°C 硬约束")
    parser.add_argument("--constraint-maxiter", type=int, default=50,
                        help="硬约束 SLSQP 最大迭代次数")
    parser.add_argument("--no-plot", action="store_true", help="不显示绘图")
    parser.add_argument("--output", default=None, help="图片保存路径")
    args = parser.parse_args()

    provider = ScenarioExternalInputProvider()
    sim = make_simulator()

    print("运行规则控制 ...")
    rule = run_rule(sim, provider)
    print("运行 PID 控制 ...")
    pid = run_pid(sim, provider)
    print(f"运行 GCCM 控制 (mode={args.gccm_mode}, horizon={args.horizon}) ...")
    gccm = run_gccm(
        provider,
        horizon=args.horizon,
        mode=args.gccm_mode,
        comfort_band=args.comfort_band,
        comfort_min=args.comfort_min,
        comfort_max=args.comfort_max,
        below_comfort_penalty=args.below_penalty,
        peak_energy_penalty=args.peak_penalty,
        comfort_weight=args.comfort_weight,
        energy_weight=args.energy_weight,
        smooth_weight=args.smooth_weight,
        enforce_comfort=args.enforce_comfort,
        constraint_options={"maxiter": args.constraint_maxiter, "ftol": 1e-6},
        verbose=True,
    )

    results = [rule, pid, gccm]
    print_metrics(results)

    if not args.no_plot:
        plot_results(results, args.output)


if __name__ == "__main__":
    main()
