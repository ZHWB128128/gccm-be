"""数据中心冷却模型测试：模型物理 / 引擎闭环 / 谷时充冷峰前蓄冷 / 峰时削峰。"""
import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.datacenter import (
    DataCenterCoolingModel,
    DataCenterProvider,
    DataCenterSimulator,
)
from gccm_be.types import ControlInput, ExternalInput, SystemState


def _ext(it_load=150.0, price=1.0, t_out=30.0) -> ExternalInput:
    return ExternalInput(np.array([t_out, 0.0, it_load, price]),
                         ["T_out", "solar", "it_load", "price"])


def test_model_step_physics():
    """双控制物理：直供冷降冷通道；充冷降罐温；放冷降冷通道+罐升温。"""
    model = DataCenterCoolingModel()
    state = SystemState([26.0, 22.0], ["T_aisle", "T_storage"])
    w = _ext(it_load=150.0)
    # 机组直供冷（u0=-300，罐不动）
    direct = model.step(state, ControlInput(np.array([-300.0, 0.0]), ["Q_chiller", "Q_tank"]), w, 1.0 / 12.0)
    assert direct.x[0] < state.x[0], "直供冷应降低冷通道温度"
    # 充冷（u1>0：罐降温）
    chg = model.step(state, ControlInput(np.array([0.0, 150.0]), ["Q_chiller", "Q_tank"]), w, 1.0 / 12.0)
    assert chg.x[1] < state.x[1], "充冷应降低罐温"
    # 放冷（u1<0：冷通道获得冷、罐升温；IT 负载需低于放冷功率）
    w_low = _ext(it_load=80.0)
    disc = model.step(state, ControlInput(np.array([0.0, -120.0]), ["Q_chiller", "Q_tank"]), w_low, 1.0 / 12.0)
    assert disc.x[0] < state.x[0], "放冷应降低冷通道温度"
    assert disc.x[1] > state.x[1], "放冷应使罐温回升"


def _make_engine(sim, provider, energy_weight=1.0, horizon=24):
    manifold = StateManifold(
        labels=["T_aisle", "T_storage"],
        units={"T_aisle": "°C", "T_storage": "°C"},
        bounds={"T_aisle": (15.0, 40.0), "T_storage": (5.0, 40.0)},
        scale={"T_aisle": 5.0, "T_storage": 5.0},
    )
    return GCCMEngine(
        simulator=sim,
        external_provider=provider,
        manifold=manifold,
        horizon=horizon,
        dt=1.0 / 12.0,
        setpoints={"T_aisle": 24.0},
        comfort_min=22.0,
        comfort_max=27.0,
        energy_weight=energy_weight,
        smooth_weight=1e-5,  # u1 量级大，平滑项需极低以免杀死充/放冷切换
        storage_targets={"T_storage": 10.0},  # 储存价值项：避免 MPC 短视排空蓄冷罐
        storage_weight=30.0,
        enforce_comfort_constraints=True,
        solver_options={"maxiter": 30, "ftol": 1e-4, "maxls": 15},
    )


def test_engine_closed_loop_on_datacenter():
    """数据中心双控制模型可直接接入 GCCMEngine 闭环。"""
    sim = DataCenterSimulator(DataCenterCoolingModel())
    provider = DataCenterProvider()
    engine = _make_engine(sim, provider)
    state = SystemState([24.0, 22.0], ["T_aisle", "T_storage"])
    decisions = []
    for i in range(12):
        t = 8.0 + i / 12.0
        w = provider.get(t, 1)[0]
        dec = engine.optimize(state, t)
        state = sim.step(state, dec.control, w, 1.0 / 12.0)
        decisions.append(dec)
    assert len(decisions) == 12
    assert decisions[0].control.u.shape == (2,)
    assert all(np.isfinite(d.control.u).all() for d in decisions)
    assert bool(np.isfinite(state.x).all())


def test_tank_charges_before_peak():
    """峰价前 MPC 应主动充冷（罐温下降），峰时放冷（罐温回升）。"""
    sim = DataCenterSimulator(DataCenterCoolingModel())
    provider = DataCenterProvider(peak_price=5.0)
    engine = _make_engine(sim, provider)
    state = SystemState([24.0, 24.0], ["T_aisle", "T_storage"])
    tank_before_peak = None
    tank_after_peak = None
    for i in range(60):  # 8:00 起 5h（覆盖 11:00 进入峰时）
        t = 8.0 + i / 12.0
        w = provider.get(t, 1)[0]
        dec = engine.optimize(state, t)
        state = sim.step(state, dec.control, w, 1.0 / 12.0)
        if 10.9 <= t <= 11.0 and tank_before_peak is None:
            tank_before_peak = state.x[1]
        if 12.9 <= t <= 13.0:
            tank_after_peak = state.x[1]
    assert tank_before_peak is not None
    assert tank_before_peak < 23.0, f"峰前应已充冷，罐温={tank_before_peak:.2f}"
    assert tank_after_peak is not None
    assert tank_after_peak > tank_before_peak, "峰时放冷后罐温应回升"


def test_peak_hour_chiller_power_drops_with_tank():
    """峰时（11-18h）机组直供冷功率应显著低于清晨充冷功率（削峰由放冷承担）。"""
    sim = DataCenterSimulator(DataCenterCoolingModel())
    provider = DataCenterProvider(peak_price=5.0)
    engine = _make_engine(sim, provider, energy_weight=1.5)
    state = SystemState([24.0, 22.0], ["T_aisle", "T_storage"])
    q_peak, q_morning, u1_peak = [], [], []
    for i in range(96):  # 5:30 起 8h：覆盖清晨充冷 + 峰时放冷
        t = 5.5 + i / 12.0
        w = provider.get(t, 1)[0]
        dec = engine.optimize(state, t)
        state = sim.step(state, dec.control, w, 1.0 / 12.0)
        if 11.0 <= t < 13.5:
            q_peak.append(abs(dec.control.u[0]))
            u1_peak.append(dec.control.u[1])
        if 5.5 <= t < 8.0:
            q_morning.append(abs(dec.control.u[0]))
    assert len(q_peak) > 0 and len(q_morning) > 0
    # 峰时直供冷功率显著低于清晨充冷功率（清晨在给罐充冷，峰时靠罐放冷）
    assert np.mean(q_peak) < 0.8 * np.mean(q_morning), \
        f"峰时直供冷 {np.mean(q_peak):.0f}kW 应显著低于清晨 {np.mean(q_morning):.0f}kW"
    # 峰时确实在放冷
    assert np.mean(u1_peak) < 0, "峰时应处于放冷状态（u1<0）"
