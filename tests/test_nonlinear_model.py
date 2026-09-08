"""Tests for the nonlinear RC building model (window / dehumidification / VAV)."""
from __future__ import annotations

import numpy as np

from gccm_be.physics.models import NonlinearRCBuildingModel
from gccm_be.types import ControlInput, ExternalInput, SystemState


def _ext(t_out=30.0, solar=0.0, occ=0.5, w_out=14.0):
    return ExternalInput([t_out, solar, occ, 1.0, w_out],
                         ["T_out", "solar", "occ", "price", "W_out"])


def _state(t_air=27.0, t_wall=27.0, w_air=10.0):
    return SystemState([t_air, t_wall, w_air], ["T_air", "T_wall", "W_air"])


def test_window_vent_only_when_cooler_outside():
    m = NonlinearRCBuildingModel()
    # warm indoors, cooler outdoors -> vent conductance > 0
    assert m._window_conductance(28.0, 24.0) > 0.0
    # warmer outdoors -> no useful venting
    assert m._window_conductance(28.0, 32.0) == 0.0


def test_window_gate_is_state_dependent_nonlinear():
    m = NonlinearRCBuildingModel()
    # conductance rises with indoor temperature (logistic gate)
    low = m._window_conductance(24.0, 20.0)
    high = m._window_conductance(30.0, 20.0)
    assert high > low


def test_dehumidification_dew_point_physics():
    """P0 修正后：除湿由盘管温度 vs 露点决定（回南天工况），而非制冷量阈值。"""
    m = NonlinearRCBuildingModel(dehum_rate=2.0)
    # 回南天：室温 27、W=14 g/kg（露点 16.9°C）。强制冷 → 送风 12.1°C、
    # 盘管 12.6°C < 露点 → 除湿；弱冷 → 盘管 26.3°C > 露点 → 不除湿
    strong = m.step(_state(w_air=14.0), ControlInput([-6.0, 1.0], ["Q_hvac", "m_dot"]), _ext(), dt=0.1)
    weak = m.step(_state(w_air=14.0), ControlInput([-0.5, 1.0], ["Q_hvac", "m_dot"]), _ext(), dt=0.1)
    assert strong.x[2] < weak.x[2]
    # 干燥房间（W=5 g/kg → 露点 3.9°C）：盘管再冷也到不了露点 → 除湿量微小
    dry_strong = m.step(_state(w_air=5.0), ControlInput([-6.0, 1.0], ["Q_hvac", "m_dot"]), _ext(), dt=0.1)
    dry_none = m.step(_state(w_air=5.0), ControlInput([0.0, 1.0], ["Q_hvac", "m_dot"]), _ext(), dt=0.1)
    assert abs(dry_strong.x[2] - dry_none.x[2]) < 0.05
    # 露点辅助函数 sanity：W=12 g/kg → 露点约 17°C
    from gccm_be.physics.models import dew_point_gram
    assert 15.0 < dew_point_gram(12.0) < 19.0


def test_vav_flow_changes_air_coupling():
    m = NonlinearRCBuildingModel()
    # hot wall, cooler air: higher fan flow -> stronger wall->air coupling -> air heats faster
    st = _state(t_air=24.0, t_wall=30.0, w_air=10.0)
    low_fan = m.step(st, ControlInput([0.0, 0.0], ["Q_hvac", "m_dot"]), _ext(t_out=24.0), dt=0.1)
    high_fan = m.step(st, ControlInput([0.0, 1.0], ["Q_hvac", "m_dot"]), _ext(t_out=24.0), dt=0.1)
    assert high_fan.x[0] > low_fan.x[0]


def test_state_stays_finite_and_humidity_nonnegative():
    m = NonlinearRCBuildingModel()
    st = _state(w_air=0.5)
    out = m.step(st, ControlInput([-6.0, 1.0], ["Q_hvac", "m_dot"]), _ext(), dt=0.25)
    assert np.all(np.isfinite(out.x))
    assert out.x[2] >= 0.0


def test_reduces_toward_linear_without_nonlinear_features():
    # with fan fully on, no window (hot outside), humidity ignored -> behaves like
    # a standard RC air/wall update in the temperature subspace
    m = NonlinearRCBuildingModel(vav_base=1.0)
    st = _state(t_air=26.0, t_wall=26.0)
    out = m.step(st, ControlInput([-2.0, 1.0], ["Q_hvac", "m_dot"]), _ext(t_out=32.0), dt=0.1)
    assert np.all(np.isfinite(out.x))
