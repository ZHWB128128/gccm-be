"""SCM 因果推断演示：do(mode=...) 对温度/电费的因果效应。"""
from __future__ import annotations

from gccm_be.causal.scm import StructuralCausalModel


def main() -> None:
    # 极简建筑控制 SCM
    # 因果顺序：mode -> comfort_weight -> metric_scale -> cooling_power -> temperature -> cost
    scm = StructuralCausalModel(
        equations={
            "comfort_weight": lambda v: 5.0 if v["mode"] == 0.0 else 0.5,
            "metric_scale": lambda v: 5.0 / (v["comfort_weight"] + 1e-6),
            "cooling_power": lambda v: 6.0 - v["metric_scale"] * 0.1,
            "temperature": lambda v: 28.0 - v["cooling_power"] * 0.4,
            "cost": lambda v: v["cooling_power"] * 1.5,
        },
        noise={"mode": 0.0},
    )

    # 观察：默认模式
    obs = scm.sample()
    print("观察值:", obs)

    # 干预：强制 demand_response
    do_dr = scm.do({"mode": 1.0})
    print("do(mode=demand_response):", do_dr)

    # 干预：强制 comfort
    do_comfort = scm.do({"mode": 0.0})
    print("do(mode=comfort):", do_comfort)

    # 因果效应
    effect_temp = do_comfort["temperature"] - do_dr["temperature"]
    effect_cost = do_comfort["cost"] - do_dr["cost"]
    print(f"\n因果效应 (comfort vs demand_response):")
    print(f"  Δtemperature = {effect_temp:.2f} °C")
    print(f"  Δcost = {effect_cost:.2f} 元")


if __name__ == "__main__":
    main()
