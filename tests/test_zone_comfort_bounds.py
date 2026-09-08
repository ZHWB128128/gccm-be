"""每区独立舒适带（zone_comfort_bounds）行为级测试。

覆盖：landscape 软代价、scipy 测地线硬约束多区化、engine 闭环、
config 解析、API 越界统计。标量 comfort_min/max 行为保持不变
（未给每区覆盖时逐区回退标量边界）。
"""
from __future__ import annotations

import numpy as np

from gccm_be.app.api import SimpleAPI
from gccm_be.app.config import engine_from_dict
from gccm_be.geometry.geodesic import GeodesicSolver
from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import ExternalInput, SystemState

from gccm_be.physics.external import MockExternalInputProvider


# --- landscape: 每区边界覆盖标量 ---

def _make_landscape(**kw):
    manifold = StateManifold(
        labels=["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"],
        units={}, bounds={}, scale={lab: 5.0 for lab in
        ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
    )
    return EnergyLandscape(
        setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
        weights={"comfort": 1.0, "energy": 0.0, "smooth": 0.0},
        manifold=manifold,
        hvac=HVACModel(n_units=2, control_labels=["Q_hvac_A", "Q_hvac_B"]),
        comfort_min=25.0, comfort_max=27.0,
        **kw,
    )


def test_bounds_for_prefers_zone_override():
    ls = _make_landscape(zone_comfort_bounds={"T_air_B": (25.5, 26.5)})
    assert ls.bounds_for("T_air_B") == (25.5, 26.5)
    assert ls.bounds_for("T_air_A") == (25.0, 27.0)   # 标量回退
    assert ls.bounds_for("T_wall_A") == (25.0, 27.0)


def test_running_cost_uses_zone_specific_bounds():
    ls = _make_landscape(zone_comfort_bounds={"T_air_B": (25.5, 26.5)})
    # B 区 27.0 超出其更紧的上界 26.5 → 产生违温代价；A 区 27.0 恰好在界内
    state = SystemState(np.array([27.0, 26.0, 27.0, 26.0, 26.5]),
                        ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"])
    control = __import__("gccm_be.types", fromlist=["ControlInput"]).ControlInput(np.zeros(2))
    ext = ExternalInput(np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.6]),
                        ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"])
    cost = ls.running_cost(state, control, ext)
    # 只有 B 区贡献舒适惩罚：(27.0-26.5)/5 的平方
    assert cost > 0.0
    assert abs(cost - ((0.5 / 5.0) ** 2)) < 1e-12


# --- scipy GeodesicSolver: 硬约束逐区生效 ---

def test_geodesic_hard_constraints_respect_tighter_zone_bound():
    sim = Simulator(TwoZoneRCBuildingModel(), HVACModel(q_min=-8, q_max=8, n_units=2,
                                                        control_labels=["Q_hvac_A", "Q_hvac_B"]))
    ls = _make_landscape(zone_comfort_bounds={"T_air_B": (25.0, 26.3)})
    ls.weights["energy"] = 0.0
    solver = GeodesicSolver(
        simulator=sim, landscape=ls, horizon=6, enforce_comfort=True,
        comfort_min=25.0, comfort_max=27.0,
        zone_comfort_bounds={"T_air_B": (25.0, 26.3)},
    )
    # 初始状态在收紧后的 B 区边界内（26.2 < 26.3）：硬约束应使轨迹全程保持在界内。
    # 注：若初值本身超界（如 26.9），单步最大制冷无法瞬间拉回，SLSQP 只能最小化
    # 违约——这是物理不可行，不是约束编码错误，因此这里用可行初值。
    state = SystemState(np.array([26.5, 26.0, 26.2, 26.0, 26.5]),
                        ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"])
    provider = MockExternalInputProvider()
    traj = solver.solve(state, provider.get(8.0, 6))
    assert traj.controls
    b_max = max(s.x[2] for s in traj.states[1:])
    assert b_max <= 26.3 + 1e-4, f"T_air_B max={b_max} violated tighter zone bound"
    # 对照：同样的可行初值，若无每区收紧，B 区允许冲到标量上界 27.0 附近
    ls_scalar = _make_landscape()
    ls_scalar.weights["energy"] = 0.0
    solver_scalar = GeodesicSolver(
        simulator=sim, landscape=ls_scalar, horizon=6, enforce_comfort=True,
        comfort_min=25.0, comfort_max=27.0,
    )
    traj_scalar = solver_scalar.solve(state, provider.get(8.0, 6))
    assert max(s.x[2] for s in traj_scalar.states[1:]) > b_max + 0.3


def test_geodesic_single_zone_hard_constraints_unchanged():
    sim = Simulator(RCBuildingModel(), HVACModel())
    manifold = StateManifold(labels=["T_air", "T_wall"], units={}, bounds={},
                             scale={"T_air": 5.0, "T_wall": 5.0})
    ls = EnergyLandscape(setpoints={"T_air": 26.0},
                         weights={"comfort": 1.0, "energy": 0.0, "smooth": 0.0},
                         manifold=manifold, hvac=sim.hvac,
                         comfort_min=25.0, comfort_max=26.5)
    solver = GeodesicSolver(simulator=sim, landscape=ls, horizon=6,
                            enforce_comfort=True, comfort_min=25.0, comfort_max=26.5)
    # 可行初值（26.4 在 26.5 界内）：全程保持在界内
    state = SystemState(np.array([26.4, 26.0]), ["T_air", "T_wall"])
    provider = MockExternalInputProvider()
    traj = solver.solve(state, provider.get(8.0, 6))
    assert traj.controls
    assert max(s.x[0] for s in traj.states[1:]) <= 26.5 + 1e-4


# --- config + engine 闭环 ---

def test_config_per_zone_comfort_bounds():
    engine = engine_from_dict({
        "model": {"type": "two_zone"},
        "zones": {
            "A": {"setpoint": 26.0},
            "B": {"setpoint": 25.5, "comfort_max": 26.8},
        },
        "controller": {"horizon": 6, "comfort_min": 25.0, "comfort_max": 27.0},
    })
    assert engine.zone_comfort_bounds == {"T_air_B": (25.0, 26.8)}
    # 未覆盖的 A 区回退标量
    assert engine.zone_comfort_bounds.get("T_air_A") is None


def test_config_rejects_inverted_zone_bounds():
    import pytest
    with pytest.raises(ValueError, match="comfort_min"):
        engine_from_dict({
            "model": {"type": "two_zone"},
            "zones": {"A": {"comfort_min": 27.0, "comfort_max": 26.0}},
        })


def test_two_zone_closed_loop_respects_per_zone_bound():
    engine = engine_from_dict({
        "model": {"type": "two_zone"},
        "zones": {"A": {"setpoint": 26.0}, "B": {"setpoint": 25.5, "comfort_max": 26.5}},
        "controller": {"horizon": 6, "comfort_min": 25.0, "comfort_max": 27.0,
                       "enforce_comfort_constraints": True},
    })
    labels = list(engine.manifold.labels)
    # B 区初值在其收紧边界 26.5 内
    state = SystemState(np.array([26.5, 26.0, 26.2, 26.0, 26.5]), labels)
    decisions = engine.run_closed_loop(state, start_time_h=8.0, steps=3, step_h=engine.dt)
    assert len(decisions) == 3
    ib = labels.index("T_air_B")
    for d in decisions:
        if d.predicted_next_state is not None:
            assert d.predicted_next_state.x[ib] <= 26.5 + 1e-4


# --- API: 越界统计按每区边界 ---

def test_simulate_violations_use_zone_specific_bounds():
    engine = engine_from_dict({
        "model": {"type": "two_zone"},
        "zones": {"A": {"setpoint": 26.0}, "B": {"setpoint": 25.5, "comfort_max": 26.3}},
        "controller": {"horizon": 6, "comfort_min": 25.0, "comfort_max": 27.0,
                       "enforce_comfort_constraints": False},
    })
    res = SimpleAPI(engine).simulate({"t0": 29.0, "steps": 5})
    # B 区返回其有效边界，供前端逐区画带
    zb = {z["label"]: (z["comfort_min"], z["comfort_max"]) for z in res["zones"]}
    assert zb["T_air_B"] == (25.0, 26.3)
    assert zb["T_air_A"] == (25.0, 27.0)
    # violations 与逐区边界一致
    expected = sum(
        1 for z in res["zones"] for t in z["temps"] if t > z["comfort_max"] or t < z["comfort_min"]
    )
    assert res["violations"] == expected
