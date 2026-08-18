"""公理突变测试：验证决策层在异常事件下是否生效。

覆盖三个场景：
1. 电价尖峰 -> 自动切换到 demand_response
2. 设备性能衰减/模型失配 -> 置信度下降
3. 预测误差过大 -> 输出不可判定并降级安全模式
"""
from __future__ import annotations

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.decision.confidence import ConfidenceEvaluator
from gccm_be.decision.diagnoser import DecisionDiagnoser
from gccm_be.geometry.curvature import CurvatureAnalysis
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ExternalInput, SystemState

STEP_H = 0.25


class PriceSpikeProvider(ExternalInputProvider):
    """正常电价，但在 10:45 后电价突变为 2.5 元/kWh。"""

    def __init__(self) -> None:
        self.dt_h = STEP_H

    def get(self, time_h: float, horizon: int = 1):
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            tout = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            if t >= 10.75:
                price = 2.5  # 突变尖峰
            elif hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5
            out.append(ExternalInput(np.array([tout, solar, occ, price]), ["T_out", "solar", "occ", "price"]))
        return out


def run_price_spike_test() -> None:
    print("=" * 60)
    print("场景 1：电价尖峰触发模式切换")
    print("=" * 60)

    sim = Simulator(RCBuildingModel(), HVACModel(q_min=-8, q_max=8))
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15, 35), "T_wall": (15, 35)},
        scale={"T_air": 5, "T_wall": 5},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=PriceSpikeProvider(),
        manifold=manifold,
        horizon=24,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_band=0.0,
        comfort_min=25.0,
        comfort_max=27.0,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=0.5,
        smooth_weight=0.1,
        counterfactual_enabled=True,
        enforce_comfort_constraints=True,
        constraint_options={"maxiter": 30, "ftol": 1e-5},
        solver_options={"maxiter": 40, "ftol": 1e-4, "maxls": 20},
    )
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    t = 10.0
    prev = None
    for i in range(8):
        dec = engine.optimize(state, t, prev_control=prev)
        w = PriceSpikeProvider().get(t, 1)[0]
        state = sim.step(state, dec.control, w, STEP_H)
        print(f"t={t:.2f}h  电价={w.w[3]:.2f}  模式={dec.mode:16s}  "
              f"置信度={dec.confidence:.2f}  触发={dec.diagnosis.triggers}")
        prev = dec.control
        t += STEP_H


def run_diagnoser_synthetic_test() -> None:
    print("\n" + "=" * 60)
    print("场景 2/3：模型失配与预测误差导致置信度下降/不可判定")
    print("=" * 60)

    diagnoser = DecisionDiagnoser(confidence_threshold=0.3)
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    ext = ExternalInput(np.array([30.0, 0.2, 0.5, 0.8]), ["T_out", "solar", "occ", "price"])
    curvature = CurvatureAnalysis(
        hessian=np.eye(2),
        eigenvalues=np.array([-0.02, 0.2]),
        eigenvectors=np.eye(2),
        classification="saddle",
        stability=-0.02,
    )

    # 正常情况
    normal = diagnoser.diagnose(state, ext, prediction_error=0.05, curvature=curvature)
    print(f"正常: 置信度={normal.confidence:.2f}, 触发={normal.triggers}, 切换={normal.should_switch_mode}")

    # 设备老化：模型失配 0.4
    aging = diagnoser.diagnose(state, ext, prediction_error=0.1, curvature=curvature, model_mismatch=0.4)
    print(f"设备老化: 置信度={aging.confidence:.2f}, 触发={aging.triggers}, 不可判定={aging.undecidable}")

    # 预报严重失准：预测误差 1.5
    bad_forecast = diagnoser.diagnose(state, ext, prediction_error=1.5, curvature=curvature)
    print(f"预报失准: 置信度={bad_forecast.confidence:.2f}, 触发={bad_forecast.triggers}, "
          f"不可判定={bad_forecast.undecidable}, 建议模式={bad_forecast.suggested_mode}")


if __name__ == "__main__":
    run_price_spike_test()
    run_diagnoser_synthetic_test()
