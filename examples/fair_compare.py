"""公平基线对比实验。

在同样要求室温 25~27°C 的前提下，对比：
- 原始规则控制（不限制舒适，作为参考）
- 原始 PID（不限制舒适，作为参考）
- 严格规则控制（高功率滞回，尝试维持舒适）
- 严格舒适 PID（调参后满足 25~27°C）
- GCCM（硬约束 + 预冷 + 峰时惩罚）

运行:
    PYTHONPATH=. python3 examples/fair_compare.py --horizon 48
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np

from compare_baselines import (
    COMFORT_MAX,
    COMFORT_MIN,
    STEP_H,
    STEPS,
    ScenarioExternalInputProvider,
    SimResult,
    make_simulator,
    print_metrics,
    run_gccm,
    run_pid,
    run_rule,
)

# 严格舒适 PID 参数（通过参数搜索得到，0% 违规；积分限幅需 >=5）
STRICT_PID_KP = 2.0
STRICT_PID_KI = 2.0
STRICT_PID_KD = 0.1
STRICT_PID_INTEGRAL_LIMIT = 5.0

# 严格规则：满功率制冷，滞回 26.5/25.5
STRICT_RULE_Q = 8.0


def run_strict_rule(sim, provider: ScenarioExternalInputProvider) -> SimResult:
    state = np.array([28.0, 28.0])
    from gccm_be.types import ControlInput, SystemState
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

        u = -STRICT_RULE_Q if on else 0.0
        control = ControlInput([u], ["Q_hvac"])
        state = sim.step(state, control, w, STEP_H)

        times.append(t)
        t_air.append(state.x[0])
        q.append(u)
        p.append(sim.hvac.electrical_power(control))
        price.append(w.w[3])
        t += STEP_H

    return SimResult(
        name="严格规则控制",
        times_h=np.array(times),
        t_air=np.array(t_air),
        q_hvac=np.array(q),
        p_elec=np.array(p),
        price=np.array(price),
    )


def run_strict_pid(sim, provider: ScenarioExternalInputProvider) -> SimResult:
    from gccm_be.types import ControlInput, SystemState
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    integral = 0.0
    prev_error = 0.0
    times, t_air, q, p, price = [], [], [], [], []

    for _ in range(STEPS):
        w = provider.get(t, 1)[0]
        error = state.x[0] - 26.0
        integral = float(np.clip(
            integral + error * STEP_H,
            -STRICT_PID_INTEGRAL_LIMIT,
            STRICT_PID_INTEGRAL_LIMIT,
        ))
        derivative = (error - prev_error) / STEP_H
        output = (
            STRICT_PID_KP * error
            + STRICT_PID_KI * integral
            + STRICT_PID_KD * derivative
        )
        cooling = float(np.clip(output, 0.0, sim.hvac.q_max))
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
        name="严格舒适 PID",
        times_h=np.array(times),
        t_air=np.array(t_air),
        q_hvac=np.array(q),
        p_elec=np.array(p),
        price=np.array(price),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GCCM-BE 公平基线对比")
    parser.add_argument("--horizon", type=int, default=48,
                        help="GCCM 预测时域步数，默认 48（12 小时）")
    parser.add_argument("--constraint-maxiter", type=int, default=80,
                        help="硬约束 SLSQP 最大迭代次数")
    parser.add_argument("--peak-penalty", type=float, default=1.0,
                        help="峰时电价放大系数")
    parser.add_argument("--energy-weight", type=float, default=0.5,
                        help="energy 权重")
    args = parser.parse_args()

    provider = ScenarioExternalInputProvider()
    sim = make_simulator()

    print("运行原始规则控制 ...")
    rule = run_rule(sim, provider)
    print("运行原始 PID 控制 ...")
    pid = run_pid(sim, provider)
    print("运行严格规则控制 ...")
    strict_rule = run_strict_rule(sim, provider)
    print("运行严格舒适 PID ...")
    strict_pid = run_strict_pid(sim, provider)
    print("运行 GCCM（硬约束 + 预冷 + 峰时惩罚）...")
    gccm = run_gccm(
        provider,
        horizon=args.horizon,
        mode="comfort",
        comfort_band=0.0,
        comfort_min=COMFORT_MIN,
        comfort_max=COMFORT_MAX,
        below_comfort_penalty=0.1,
        peak_energy_penalty=args.peak_penalty,
        comfort_weight=5.0,
        energy_weight=args.energy_weight,
        smooth_weight=0.1,
        enforce_comfort=True,
        constraint_options={"maxiter": args.constraint_maxiter, "ftol": 1e-6},
        verbose=True,
    )

    results: List[SimResult] = [rule, pid, strict_rule, strict_pid, gccm]
    print_metrics(results)

    # 额外输出严格舒适 PID 与 GCCM 的谷时/峰时均功率
    def avg_power(r: SimResult, lo: float, hi: float) -> float:
        mask = (r.times_h >= lo) & (r.times_h < hi)
        return float(np.mean(r.p_elec[mask])) if np.any(mask) else 0.0

    print("\n补充：谷时(0-8h) / 峰时(11-18h) 平均电功率")
    print(f"  严格舒适 PID: 谷时 {avg_power(strict_pid, 0, 8):.3f} kW，"
          f"峰时 {avg_power(strict_pid, 11, 18):.3f} kW")
    print(f"  GCCM         : 谷时 {avg_power(gccm, 0, 8):.3f} kW，"
          f"峰时 {avg_power(gccm, 11, 18):.3f} kW")


if __name__ == "__main__":
    main()
