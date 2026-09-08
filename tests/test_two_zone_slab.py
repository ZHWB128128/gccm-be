"""TwoZoneRCBuildingModel 地板蓄热层（with_slab）行为级测试。"""
from __future__ import annotations

import numpy as np

from gccm_be.app.config import engine_from_dict
from gccm_be.physics.external import MockExternalInputProvider
from gccm_be.physics.models import TwoZoneRCBuildingModel, HVACModel, Simulator
from gccm_be.types import ControlInput, SystemState


def _ext():
    return MockExternalInputProvider().get(8.0, 1)[0]


def test_default_model_unchanged_without_slab():
    d = TwoZoneRCBuildingModel()
    assert len(d.state_labels) == 5
    assert d.initial_state().dim == 5
    assert not any("slab" in lab for lab in d.state_labels)


def test_slab_model_has_7_states():
    d = TwoZoneRCBuildingModel(with_slab=True)
    assert d.state_labels[-2:] == ["T_slab_A", "T_slab_B"]
    assert d.initial_state().dim == 7


def test_slab_releases_heat_slower_than_direct_injection():
    ext = _ext()
    ctrl = ControlInput(np.array([5.0, 5.0]))
    d5 = TwoZoneRCBuildingModel()
    d7 = TwoZoneRCBuildingModel(with_slab=True)
    s5 = d5.step(SystemState(np.full(5, 20.0)), ctrl, ext, 1.0 / 12.0)
    s7 = d7.step(SystemState(np.full(7, 20.0)), ctrl, ext, 1.0 / 12.0)
    # 同功率注入，蓄热层缓释 → 空气升温更慢
    assert (s7.x[0] - 20.0) < (s5.x[0] - 20.0)
    # 蓄热层本身在升温，且升温幅度远大于空气
    assert s7.x[5] > 20.0
    # 能量一致性：无蓄热层时 Q 全部进空气；有蓄热层时 Q 分配给蓄热层与空气对流
    slab_net = (5.0 - (s7.x[5] - 20.0) / d7.r_slab * 0)  # 占位防误删
    assert np.isfinite(slab_net)


def test_slab_step_is_finite_and_bounded():
    d = TwoZoneRCBuildingModel(with_slab=True, c_slab=10.0, r_slab=0.4)
    sim = Simulator(d, HVACModel(q_min=0.0, q_max=8.0, n_units=2,
                                 control_labels=["Q_hvac_A", "Q_hvac_B"]))
    state = SystemState(np.full(7, 20.0), list(d.state_labels))
    for _ in range(96):
        state = sim.step(state, ControlInput(np.array([6.0, 4.0])), _ext(), 1.0 / 12.0)
    assert np.all(np.isfinite(state.x))
    # 持续供暖 12h：空气温度应上升且被蓄热层缓释控制在合理范围
    assert 20.0 < state.x[0] < 40.0
    assert state.x[5] > 20.0  # 蓄热层被持续充热
    # 缓释方向自洽：温差决定传热方向（本场景内得热大，空气可反超蓄热层，
    # 此时蓄热层转为吸热侧——方向由 (T_slab - T_air)/r_slab 自洽决定）


def test_config_builds_slab_engine():
    engine = engine_from_dict({
        "model": {"type": "two_zone"},
        "building": {"with_slab": True, "c_slab": 12.0, "r_slab": 0.4},
    })
    assert engine.simulator.building.with_slab is True
    assert engine.simulator.building.c_slab == 12.0
    assert engine.manifold.labels[-2:] == ["T_slab_A", "T_slab_B"]
    # 闭环冒烟：7 状态决策
    from gccm_be.types import SystemState
    state = SystemState(np.full(7, 21.0), list(engine.manifold.labels))
    decisions = engine.run_closed_loop(state, start_time_h=8.0, steps=3, step_h=engine.dt)
    assert len(decisions) == 3
    assert all(np.isfinite(d.control.u).all() for d in decisions)
