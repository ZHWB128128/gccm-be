"""P1 模型完善行为级测试：SafeController 逐区边界、双区在线辨识、scipy 鲁棒路径。"""
from __future__ import annotations

import numpy as np

from gccm_be.app.config import engine_from_dict
from gccm_be.control.safe_control import SafeController
from gccm_be.physics.external import MockExternalInputProvider
from gccm_be.physics.models import (
    HVACModel, RCBuildingModel, Simulator, TwoZoneRCBuildingModel,
)
from gccm_be.physics.online_id import RCOnlineIdentifier
from gccm_be.types import ControlInput, ExternalInput, SystemState

LABS = ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]


def _two_zone_hvac():
    return HVACModel(q_min=-8, q_max=8, n_units=2, control_labels=["Q_hvac_A", "Q_hvac_B"])


# --- P1-1: SafeController 逐区边界 ---

def test_safe_controller_feedback_uses_per_zone_lower_bound():
    sim = Simulator(TwoZoneRCBuildingModel(), _two_zone_hvac())
    sc = SafeController(
        simulator=sim,
        setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
        comfort_min=25.0, comfort_max=27.0,
        zone_comfort_bounds={"T_air_B": (25.5, 26.5)},
    )
    # B 区 25.4 低于其更紧下界 25.5（但在标量下界 25.0 之上）→ 只对 B 加热
    state = SystemState(np.array([26.0, 26.0, 25.4, 26.0, 25.7]), LABS)
    ctrl = sc.feedback(state, dt=1.0 / 12.0)
    assert ctrl.u[0] == 0.0
    assert ctrl.u[1] > 0.0


def test_safe_controller_zone_bounds_fallback_scalar():
    sim = Simulator(TwoZoneRCBuildingModel(), _two_zone_hvac())
    sc = SafeController(simulator=sim, setpoints={}, comfort_min=25.0)
    assert sc.zone_bounds("T_air_A") == (25.0, None)
    assert sc.zone_bounds("T_air_B") == (25.0, None)


# --- P1-2: 双区在线辨识 ---

def test_zone_identifier_converges_on_two_zone_plant():
    sim = Simulator(TwoZoneRCBuildingModel(), _two_zone_hvac())
    provider = MockExternalInputProvider()
    ident_a = RCOnlineIdentifier(zone="A", unit_index=0)
    ident_b = RCOnlineIdentifier(zone="B", unit_index=1)
    state = SystemState(np.full(5, 28.0), LABS)
    t = 8.0
    for k in range(120):
        w = provider.get(t, 1)[0]
        # 激励：正弦调制制冷量，保证可观性
        q = -4.0 - 3.0 * np.sin(2 * np.pi * k / 24.0)
        ctrl = ControlInput(np.array([q, -q]))
        nxt = sim.step(state, ctrl, w, 1.0 / 12.0)
        ident_a.update(state, ctrl, w, nxt, 1.0 / 12.0)
        ident_b.update(state, ctrl, w, nxt, 1.0 / 12.0)
        state = nxt
        t += 1.0 / 12.0
    for ident in (ident_a, ident_b):
        assert len(ident.history) == 120
        # 收敛：近期残差显著小于前期
        early = float(np.mean(np.abs(ident.history[:20])))
        late = float(np.mean(np.abs(ident.history[-20:])))
        assert late < early, f"残差未收敛: early={early:.3f} late={late:.3f}"


def _plausible_theta() -> np.ndarray:
    # c_air=0.6, r_air=0.8, r_wall=2.0, solar_gain=0.08 对应的 a1..a6
    c_air, r_air, r_wall, sg = 0.6, 0.8, 2.0, 0.08
    return np.array([
        1.0 / (r_air * c_air),
        1.0 / (r_wall * c_air),
        sg / c_air,
        1.0 / c_air,
        1.0 / c_air,
        0.0,
    ])


def test_engine_two_zone_creates_zone_identifiers_and_trust():
    engine = engine_from_dict({"model": {"type": "two_zone"},
                               "zones": {"A": {"comfort_max": 26.5}}})
    assert set(engine.rc_identifiers_by_zone) == {"A", "B"}
    assert engine.rc_identifiers_by_zone["A"] is engine.rc_identifier
    assert engine.rc_identifiers_by_zone["A"].zone == "A"
    assert engine.rc_identifiers_by_zone["B"].unit_index == 1

    # 两个区都给出可信参数 → 逐区判定整体可信
    for ident in engine.rc_identifiers_by_zone.values():
        ident.theta = _plausible_theta()
        ident.history = [0.01] * 40
    assert engine.identification_trusted(30) is True
    # 任一区近期误差大 → 整体不可信（保守方向）
    engine.rc_identifiers_by_zone["B"].history = [2.0] * 40
    assert engine.identification_trusted(30) is False


def test_two_zone_identification_applies_candidate_with_shadow_validation():
    engine = engine_from_dict({"model": {"type": "two_zone"},
                               "controller": {"horizon": 6}})
    # 真实"楼"：r_wall_a=1.5（保温比当前模型差）；当前模型 r_wall_a=2.0
    plant = TwoZoneRCBuildingModel(r_wall_a=1.5)
    plant_sim = Simulator(plant, engine.simulator.hvac)
    state = SystemState(np.array([27.0, 26.0, 27.5, 26.0, 26.5]), LABS)
    ctrl = ControlInput(np.array([-6.0, -6.0]))
    w = MockExternalInputProvider().get(8.0, 1)[0]
    nxt = plant_sim.step(state, ctrl, w, 1.0 / 12.0)
    engine._recent_step = {"state": state, "control": ctrl, "external": w,
                           "next_state": nxt, "dt": 1.0 / 12.0}
    # 两个区辨识出的都是"真楼"参数（theta 由工厂参数反推），近期残差小
    # 真楼 A 区：c_air=0.6, r_air=0.8, r_wall_a=1.5, solar_gain_a=0.08
    # 真楼 B 区：c_air=0.6, r_air=0.8, r_wall_b=1.5(默认), solar_gain_b=0.02
    theta_a = np.array([1.0 / (0.8 * 0.6), 1.0 / (1.5 * 0.6), 0.08 / 0.6, 1.0 / 0.6, 1.0 / 0.6, 0.0])
    theta_b = np.array([1.0 / (0.8 * 0.6), 1.0 / (1.5 * 0.6), 0.02 / 0.6, 1.0 / 0.6, 1.0 / 0.6, 0.0])
    engine.rc_identifiers_by_zone["A"].theta = theta_a
    engine.rc_identifiers_by_zone["B"].theta = theta_b
    for ident in engine.rc_identifiers_by_zone.values():
        ident.history = [0.01] * 40
    # 自监控近期误差给个正值，让 0.9 门放行
    engine.self_monitor.residuals = [0.5] * 6
    # 候选模型（=真楼参数）对真楼下一步预测优于当前模型（r_wall_a=2.0 失配）
    assert engine.apply_rc_identification(30) is True
    assert abs(engine.simulator.building.r_wall_a - 1.5) < 1e-9


def test_two_zone_identification_rejected_when_shadow_worse():
    engine = engine_from_dict({"model": {"type": "two_zone"},
                               "controller": {"horizon": 6}})
    # 真实"楼"就是当前模型本身 → 候选（不同参数）影子验证必然不更优
    plant_sim = engine.simulator
    state = SystemState(np.array([27.0, 26.0, 27.5, 26.0, 26.5]), LABS)
    ctrl = ControlInput(np.array([-6.0, -6.0]))
    w = MockExternalInputProvider().get(8.0, 1)[0]
    nxt = plant_sim.step(state, ctrl, w, 1.0 / 12.0)
    engine._recent_step = {"state": state, "control": ctrl, "external": w,
                           "next_state": nxt, "dt": 1.0 / 12.0}
    for ident in engine.rc_identifiers_by_zone.values():
        ident.theta = _plausible_theta()
        ident.history = [0.01] * 40
    engine.self_monitor.residuals = [0.5] * 6
    old_building = engine.simulator.building
    assert engine.apply_rc_identification(30) is False
    assert engine.simulator.building is old_building


# --- P1-3: scipy 鲁棒路径 + 配置化 ---

def test_config_default_enables_robust_scenarios():
    engine = engine_from_dict({"controller": {"horizon": 6}})
    assert len(engine.robust_scenarios) == 2
    # 扰动方向对称：一热一冷（r_wall 增大 = 保温更差 = 更热）
    r_walls = sorted(s.building.r_wall for s in engine.robust_scenarios)
    assert r_walls[0] < engine.simulator.building.r_wall < r_walls[1]


def test_config_can_disable_robust():
    engine = engine_from_dict({"controller": {"horizon": 6},
                               "robust": {"enabled": False}})
    assert engine.robust_scenarios == []


def test_engine_uses_scipy_robust_path_in_closed_loop():
    engine = engine_from_dict({"model": {"type": "two_zone"},
                               "zones": {"A": {"setpoint": 26.0}, "B": {"setpoint": 26.0}},
                               "controller": {"horizon": 6,
                                              "enforce_comfort_constraints": False}})
    assert engine.robust_scenarios  # 默认开启
    state = SystemState(np.array([26.5, 26.0, 26.4, 26.0, 26.4]), LABS)
    decisions = engine.run_closed_loop(state, start_time_h=8.0, steps=2, step_h=engine.dt)
    assert len(decisions) == 2
    for d in decisions:
        assert d.control.dim == 2
        assert np.all(np.isfinite(d.control.u))
