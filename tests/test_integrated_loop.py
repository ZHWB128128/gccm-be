"""端到端集成测试：MetricTensor + use_kinetic + GodelBoundary + SelfMonitor + CounterfactualAnalyzer。"""
from __future__ import annotations

import numpy as np
import pytest

from gccm_be import GCCMEngine
from gccm_be.physics.external import ExternalInputProvider
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ControlInput, ExternalInput, SystemState

STEP_H = 0.25


class PriceSpikeProvider(ExternalInputProvider):
    def __init__(self) -> None:
        self.dt_h = STEP_H

    def get(self, time_h: float, horizon: int = 1):
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
            occ = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            if t >= 10.75:
                price = 2.5
            elif hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5
            out.append(ExternalInput(np.array([t_out, solar, occ, price]), ["T_out", "solar", "occ", "price"]))
        return out


class TwoZonePriceSpikeProvider(ExternalInputProvider):
    """两区域外部输入，带电价尖峰。"""
    def __init__(self) -> None:
        self.dt_h = STEP_H

    def get(self, time_h: float, horizon: int = 1):
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = 29.5 + 5.5 * np.cos(2.0 * np.pi * (t - 14.0) / 24.0)
            solar = 0.4 * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
            occ_a = 1.0 if 8.0 <= hour <= 18.0 else 0.3
            occ_b = 0.6 if 8.0 <= hour <= 18.0 else 0.2
            if t >= 10.75:
                price = 2.5
            elif hour < 8.0 or hour >= 22.0:
                price = 0.3
            elif hour < 11.0 or hour >= 18.0:
                price = 0.8
            else:
                price = 1.5
            out.append(ExternalInput(
                np.array([t_out, solar * 1.2, solar * 0.3, occ_a, occ_b, price]),
                ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"],
            ))
        return out



def test_integrated_loop_with_counterfactual():
    provider = PriceSpikeProvider()
    sim = Simulator(RCBuildingModel(), HVACModel(q_min=-8, q_max=8))

    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        horizon=12,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        below_comfort_penalty=0.1,
        peak_energy_penalty=1.0,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        use_kinetic=True,
        counterfactual_enabled=True,
        counterfactual_horizon=4,
        solver_options={"maxiter": 50, "ftol": 1e-4, "maxls": 20},
    )

    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    decisions = engine.run_closed_loop(
        state,
        start_time_h=10.0,
        steps=12,
        step_h=STEP_H,
    )

    assert len(decisions) == 12
    assert len(engine.self_monitor.residuals) == 12
    assert any("counterfactual" in d.diagnosis.details for d in decisions)
    assert any(d.mode == "demand_response" for d in decisions)

    # GodelBoundary 已接入 details
    assert all("godel" in d.diagnosis.details for d in decisions)


def test_renormalization_in_main_loop():
    from gccm_be.geometry.manifold import StateManifold

    provider = TwoZonePriceSpikeProvider()
    sim = Simulator(TwoZoneRCBuildingModel(), HVACModel(q_min=-8, q_max=8, n_units=2))
    manifold = StateManifold(
        labels=["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"],
        units={l: "°C" for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        bounds={l: (15, 40) for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        scale={l: 5 for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=4,
        dt=STEP_H,
        setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        renormalization_enabled=True,
        solver_options={"maxiter": 30, "ftol": 1e-4, "maxls": 20},
    )
    state = SystemState([28.0, 28.0, 28.0, 28.0, 28.0], manifold.labels)
    decisions = engine.run_closed_loop(state, start_time_h=10.0, steps=4, step_h=STEP_H)
    assert len(decisions) == 4
    assert all("renormalization" in d.diagnosis.details for d in decisions)


def test_undecidable_high_error_triggers_safe_mode():
    from gccm_be.types import ControlInput

    provider = PriceSpikeProvider()
    sim = Simulator(RCBuildingModel(), HVACModel(q_min=-8, q_max=8))
    engine = GCCMEngine(
        simulator=sim,
        external_provider=provider,
        horizon=6,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
        safe_control_mode="last_valid",
        solver_options={"maxiter": 30, "ftol": 1e-4, "maxls": 20},
    )
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    prev = ControlInput([-1.0], ["Q_hvac"])
    # 持续高预测误差，使自指误差模型发散
    for _ in range(10):
        engine.self_monitor.update(2.0)
    dec = engine.optimize(state, 10.0, prev_control=prev, prediction_error=5.0)
    assert dec.diagnosis.undecidable is True
    assert dec.mode == "balanced"
    assert np.allclose(dec.control.u, prev.u)


def test_online_identifier_rls():
    from gccm_be.physics.online_id import OnlineIdentifier

    ident = OnlineIdentifier(dim=2, lam=0.98)
    true_theta = np.array([2.0, -1.0])
    rng = np.random.default_rng(0)
    for _ in range(500):
        phi = rng.normal(size=2)
        y = float(true_theta @ phi)
        ident.update(phi, y)
    assert np.allclose(ident.theta, true_theta, atol=0.1)


def test_rc_online_identifier_converges():
    from gccm_be.physics.online_id import RCOnlineIdentifier
    from gccm_be.physics.models import RCBuildingModel

    true_model = RCBuildingModel(c_air=0.5, c_wall=3.0, r_air=0.6, r_wall=1.2, solar_gain=0.08)
    ident = RCOnlineIdentifier(lam=0.99)
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    rng = np.random.default_rng(0)
    for _ in range(300):
        control = ControlInput([-rng.uniform(0, 6)], ["Q_hvac"])
        ext = ExternalInput(np.array([
            30.0 + rng.normal(0, 0.5),
            rng.uniform(0, 0.5),
            rng.uniform(0.2, 1.0),
            0.5,
        ]), ["T_out", "solar", "occ", "price"])
        next_state = true_model.step(state, control, ext, 0.25)
        ident.update(state, control, ext, next_state, 0.25)
        state = next_state
    params = ident.parameters()
    assert params["R_air_times_C_air"] is not None
    assert abs(params["R_air_times_C_air"] - 0.6 * 0.5) < 0.2
    assert abs(params["R_wall_times_C_air"] - 1.2 * 0.5) < 0.2


def test_scm_do_operator():
    from gccm_be.causal.scm import StructuralCausalModel

    scm = StructuralCausalModel(
        equations={
            "y": lambda v: v["x"] * 2.0,
            "z": lambda v: v["y"] + 1.0,
        },
        noise={"x": 1.0},
    )
    obs = scm.sample()
    assert obs["z"] == 3.0
    inter = scm.do({"x": 5.0})
    assert inter["y"] == 10.0
    assert inter["z"] == 11.0
    assert scm.effect("z", {"x": 5.0}, {"x": 1.0}) == 8.0


def test_counterfactual_analyzer_scm():
    from gccm_be.causal.counterfactual import CounterfactualAnalyzer
    from gccm_be.causal.scm import StructuralCausalModel
    from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator

    scm = StructuralCausalModel(
        equations={
            "y": lambda v: v["x"] * 2.0,
            "z": lambda v: v["y"] + 1.0,
        },
        noise={"x": 1.0},
    )
    sim = Simulator(RCBuildingModel(), HVACModel())
    analyzer = CounterfactualAnalyzer(
        simulator=sim,
        initial_state=SystemState([26.0, 26.0], ["T_air", "T_wall"]),
        externals=[],
        scm=scm,
    )
    assert analyzer.causal_effect("z", {"x": 5.0}, {"x": 1.0}) == 8.0


def test_engine_causal_scm_in_counterfactual():
    from gccm_be.causal.scm import StructuralCausalModel
    from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
    from gccm_be.geometry.manifold import StateManifold

    scm = StructuralCausalModel(
        equations={
            "comfort_weight": lambda v: 5.0 if v["mode"] == 0.0 else 0.5,
            "cooling_power": lambda v: 6.0 - v["comfort_weight"] * 0.1,
            "temperature": lambda v: 28.0 - v["cooling_power"] * 0.4,
            "cost": lambda v: v["cooling_power"] * 1.5,
        },
        noise={"mode": 0.0},
    )
    sim = Simulator(RCBuildingModel(), HVACModel())
    manifold = StateManifold(labels=["T_air", "T_wall"], scale={"T_air": 5, "T_wall": 5})
    engine = GCCMEngine(
        simulator=sim,
        manifold=manifold,
        horizon=4,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        causal_scm=scm,
    )
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    cf = engine._run_counterfactual_undecidable(state, 10.0)
    assert "causal_effect" in cf
    assert "temperature" in cf["causal_effect"]


def test_engine_casadi_robust_backend():
    pytest.importorskip("casadi")  # casadi 未安装时跳过，避免新环境整套测试必红
    from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
    from gccm_be.geometry.manifold import StateManifold

    nominal_sim = Simulator(RCBuildingModel(), HVACModel(q_min=-8, q_max=8))
    pessimistic_sim = Simulator(
        RCBuildingModel(c_air=0.5, c_wall=3.0, r_air=0.6, r_wall=1.2, solar_gain=0.08),
        HVACModel(q_min=-8, q_max=8),
    )
    manifold = StateManifold(labels=["T_air", "T_wall"], scale={"T_air": 5, "T_wall": 5})
    engine = GCCMEngine(
        simulator=nominal_sim,
        manifold=manifold,
        horizon=4,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        use_casadi_robust=True,
        robust_scenarios=[pessimistic_sim],
        solver_options={"maxiter": 30, "ftol": 1e-4, "maxls": 20},
    )
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    dec = engine.optimize(state, 0.0, forced_mode="comfort")
    assert dec.control.u.shape == (1,)


def test_engine_riemannian_correction():
    from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
    from gccm_be.geometry.manifold import StateManifold

    sim = Simulator(RCBuildingModel(), HVACModel())
    manifold = StateManifold(labels=["T_air", "T_wall"], scale={"T_air": 5, "T_wall": 5})
    engine = GCCMEngine(
        simulator=sim,
        manifold=manifold,
        horizon=4,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        use_riemannian=True,
        solver_options={"maxiter": 30, "ftol": 1e-4, "maxls": 20},
    )
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    dec = engine.optimize(state, 0.0, forced_mode="comfort")
    assert dec.control.u.shape == (1,)


def test_build_rc_scm():
    from gccm_be.causal.scm import build_rc_scm
    from gccm_be.physics.models import RCBuildingModel, HVACModel

    scm = build_rc_scm(RCBuildingModel(), HVACModel())
    obs = scm.sample()
    do_comfort = scm.do({"mode": 0.0})
    do_dr = scm.do({"mode": 1.0})
    assert do_comfort["temperature"] < do_dr["temperature"]
    assert do_comfort["cost"] > do_dr["cost"]


def test_engine_default_scm_from_physics():
    from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
    from gccm_be.geometry.manifold import StateManifold

    sim = Simulator(RCBuildingModel(), HVACModel())
    manifold = StateManifold(labels=["T_air", "T_wall"], scale={"T_air": 5, "T_wall": 5})
    engine = GCCMEngine(
        simulator=sim,
        manifold=manifold,
        horizon=4,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
    )
    assert engine.causal_scm is not None
    # 默认 SCM 应能计算 do-效应
    effect = engine.causal_scm.effect("cost", {"mode": 0.0}, {"mode": 1.0})
    assert effect != 0.0


def test_data_driven_scm():
    from gccm_be.causal.data_driven import DataDrivenSCM

    rng = np.random.default_rng(0)
    data = []
    true_effect = 0.0
    for _ in range(2000):
        mode = float(rng.integers(0, 2))
        comfort_weight = 5.0 if mode == 0 else 0.5
        cooling = 1.0 + comfort_weight * 0.8 + rng.normal(0, 0.05)
        temp = 28.0 - cooling * 0.4 + rng.normal(0, 0.05)
        cost = cooling * 1.5 + rng.normal(0, 0.05)
        data.append({"mode": mode, "comfort_weight": comfort_weight,
                     "cooling_power": cooling, "temperature": temp, "cost": cost})
        if mode == 0:
            true_effect += temp - (28.0 - (1.0 + 0.5 * 0.8) * 0.4)
    true_effect /= 1000.0

    fitted = DataDrivenSCM().fit(data)
    scm = fitted.to_scm()
    effect = scm.effect("temperature", {"mode": 0.0}, {"mode": 1.0})
    # 只验证方向正确
    assert effect < 0
    assert abs(effect) > 0.1


def test_engine_set_causal_scm_from_data():
    from gccm_be.physics.models import RCBuildingModel, HVACModel, Simulator
    from gccm_be.geometry.manifold import StateManifold

    rng = np.random.default_rng(1)
    data = []
    for _ in range(500):
        mode = float(rng.integers(0, 2))
        cw = 5.0 if mode == 0 else 0.5
        cool = 1.0 + cw * 0.8
        temp = 28.0 - cool * 0.4
        cost = cool * 1.5
        data.append({"mode": mode, "comfort_weight": cw,
                     "cooling_power": cool, "temperature": temp, "cost": cost})

    sim = Simulator(RCBuildingModel(), HVACModel())
    manifold = StateManifold(labels=["T_air", "T_wall"], scale={"T_air": 5, "T_wall": 5})
    engine = GCCMEngine(simulator=sim, manifold=manifold, horizon=4, dt=STEP_H,
                        setpoints={"T_air": 26.0}, comfort_min=25.0, comfort_max=27.0)
    engine.set_causal_scm_from_data(data)
    effect = engine.causal_scm.effect("temperature", {"mode": 0.0}, {"mode": 1.0})
    assert effect < 0


def test_external_simulator_adapter():
    from gccm_be.physics.building_simulator_adapter import EnergyPlusAdapterStub
    from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
    from gccm_be.types import ExternalInput

    adapter = EnergyPlusAdapterStub()
    state = SystemState(np.full(5, 28.0),
                        ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"])
    adapter.reset(state)
    ctrl = ControlInput([-2.0, -2.0], ["Q_hvac_A", "Q_hvac_B"])
    ext = ExternalInput(np.array([30.0, 0.2, 0.1, 1.0, 0.5, 1.0]),
                        ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"])
    next_state = adapter.step(ctrl, ext)
    assert next_state.dim == 5
    assert "T_air_A" in adapter.get_measurements()
    assert adapter.get_state().dim == 5
