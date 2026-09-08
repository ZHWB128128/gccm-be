"""配置化双区闭环演示：从 examples/config_two_zone.json 构建引擎并跑闭环。

与 two_zone_compare.py 的手写装配不同，这里只靠 JSON 配置就能切到双区模型：

    PYTHONPATH=. python3 examples/two_zone_config_loop.py
    PYTHONPATH=. python3 examples/two_zone_config_loop.py --steps 48 --horizon 32
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from gccm_be.app.config import engine_from_config
from gccm_be.types import SystemState

STEP_H = 1.0 / 12.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_two_zone.json"))
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=None, help="覆盖配置里的预测时域")
    args = parser.parse_args()

    engine = engine_from_config(args.config)
    if args.horizon is not None:
        engine.horizon = args.horizon

    labels = list(engine.manifold.labels)
    air_idx = {lab: i for i, lab in enumerate(labels) if lab.startswith("T_air")}
    print(f"模型: {type(engine.simulator.building).__name__}  状态: {labels}")
    print(f"设定点: {engine.setpoints}  时域: {engine.horizon} 步  dt={engine.dt:.4f}h")

    state = SystemState(np.full(len(labels), 28.0), labels)
    decisions = engine.run_closed_loop(state, start_time_h=8.0, steps=args.steps, step_h=STEP_H)
    if not decisions:
        print("无决策输出")
        return

    print(f"\n{'步':>4} {'Q_A':>8} {'Q_B':>8} {'T_air_A':>9} {'T_air_B':>9} {'模式':>10} {'置信':>6}")
    for k, d in enumerate(decisions):
        ns = d.predicted_next_state
        ta = float(ns.x[air_idx["T_air_A"]]) if ns is not None else float("nan")
        tb = float(ns.x[air_idx["T_air_B"]]) if ns is not None else float("nan")
        print(f"{k:>4} {d.control.u[0]:>8.2f} {d.control.u[1]:>8.2f} "
              f"{ta:>9.2f} {tb:>9.2f} {d.mode:>10} {d.confidence:>6.2f}")

    print("\n各区统计:")
    for lab, i in air_idx.items():
        temps = [float(d.predicted_next_state.x[i]) for d in decisions if d.predicted_next_state is not None]
        print(f"  {lab}: 范围 [{min(temps):.2f}, {max(temps):.2f}] °C  "
              f"越界步数 {sum(1 for t in temps if t > engine.comfort_max or t < engine.comfort_min)}")
    print(f"总代价: {sum(float(d.trajectory.total_cost) for d in decisions):.2f}")


if __name__ == "__main__":
    main()
