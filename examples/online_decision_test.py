"""在线决策层集成测试：预报噪声场景下，引擎自动计算预测误差并调整策略。

运行:
    PYTHONPATH=. python3 examples/online_decision_test.py
"""
from __future__ import annotations

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ExternalInput, SystemState

STEP_H = 0.25
STEPS = 96


class NoisyForecastProvider(ExternalInputProvider):
    def __init__(self, noise_std: float = 1.0, seed: int = 1) -> None:
        self.dt_h = STEP_H
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def get(self, time_h: float, horizon: int = 1):
        result = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            t_out += self.rng.normal(0.0, self.noise_std)
            solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            if hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5
            result.append(ExternalInput(np.array([t_out, solar, occ, price]), ["T_out", "solar", "occ", "price"]))
        return result


class CleanProvider(ExternalInputProvider):
    def __init__(self) -> None:
        self.dt_h = STEP_H

    def get(self, time_h: float, horizon: int = 1):
        result = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            if hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5
            result.append(ExternalInput(np.array([t_out, solar, occ, price]), ["T_out", "solar", "occ", "price"]))
        return result


def main() -> None:
    sim = Simulator(RCBuildingModel(), HVACModel(q_min=-8, q_max=8))
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15, 35), "T_wall": (15, 35)},
        scale={"T_air": 5, "T_wall": 5},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=NoisyForecastProvider(),
        manifold=manifold,
        horizon=48,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_band=0.0,
        comfort_min=25.0,
        comfort_max=26.9,   # 安全裕度：内部上限 26.9，实际允许 27.0
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=0.5,
        smooth_weight=0.1,
        enforce_comfort_constraints=True,
        constraint_options={"maxiter": 50, "ftol": 1e-6},
        solver_options={"maxiter": 50, "ftol": 1e-4, "maxls": 20},
    )

    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 0.0
    prev_control = None
    prediction_error = 0.0
    decisions = []
    actual_ts = []
    errors = []
    plant = CleanProvider()

    for _ in range(STEPS):
        decision = engine.optimize(
            state,
            t,
            prev_control=prev_control,
            prediction_error=prediction_error,
        )
        decisions.append(decision)
        w = plant.get(t, 1)[0]
        predicted = decision.predicted_next_state.x
        state = sim.step(state, decision.control, w, STEP_H)
        actual_ts.append(state.x[0])
        error = float(np.max(np.abs(predicted - state.x)))
        errors.append(error)
        prediction_error = error
        prev_control = decision.control
        t += STEP_H

    actual_arr = np.array(actual_ts)
    violation = float(np.sum((actual_arr > 27.0) | (actual_arr < 25.0)) / STEPS * 100.0)
    modes = {}
    for d in decisions:
        modes[d.mode] = modes.get(d.mode, 0) + 1

    print("在线决策层测试完成")
    print(f"平均预测误差: {np.mean(errors):.4f} °C，最大误差: {np.max(errors):.4f} °C")
    print(f"实际室温范围: {actual_arr.min():.2f} ~ {actual_arr.max():.2f} °C")
    print(f"实际舒适违规: {violation:.1f}%")
    print(f"模式分布: {modes}")
    print(f"不可判定次数: {sum(1 for d in decisions if d.diagnosis.undecidable)}")
    print(f"触发事件次数: {sum(len(d.diagnosis.triggers) for d in decisions)}")


if __name__ == "__main__":
    main()
