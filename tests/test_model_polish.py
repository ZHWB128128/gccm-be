"""代码/模型完善行为级测试：每区设备容量、鲁棒扰动覆盖面、试点 CSV 日志。"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from gccm_be.app.config import _perturbed_building, engine_from_dict
from gccm_be.app.pilot_log import FIELDNAMES, append_row
from gccm_be.geometry.geodesic import GeodesicSolver
from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.datacenter import DataCenterCoolingModel
from gccm_be.physics.models import (
    HVACModel, NonlinearRCBuildingModel, RCBuildingModel, Simulator,
    TwoZoneRCBuildingModel,
)
from gccm_be.physics.external import MockExternalInputProvider
from gccm_be.types import ControlInput, SystemState


# --- 每区独立设备容量 ---

def test_hvac_unit_bounds_override_uniform():
    hvac = HVACModel(q_min=-8, q_max=8, n_units=2,
                     control_labels=["Q_hvac_A", "Q_hvac_B"],
                     unit_bounds=[(-8, 8), (-6, 6)])
    assert hvac.bounds() == [(-8, 8), (-6, 6)]
    # B 区制冷额定 6kW：两台均满载 → cop_eff 相同 → 电功率 = (8+6)/3.8
    ctrl = ControlInput(np.array([-8.0, -6.0]))
    assert abs(hvac.electrical_power(ctrl) - 14.0 / 3.8) < 1e-9
    # B 区半载（3/6）有部分负荷惩罚，效率低于满载
    half = hvac.electrical_power(ControlInput(np.array([-8.0, -3.0])))
    assert half > (8.0 + 3.0) / 3.8


def test_hvac_unit_bounds_length_mismatch_raises():
    with pytest.raises(ValueError, match="unit_bounds"):
        HVACModel(n_units=2, unit_bounds=[(-8, 8)])


def test_geodesic_respects_per_unit_capacity():
    sim = Simulator(TwoZoneRCBuildingModel(),
                    HVACModel(q_min=-8, q_max=8, n_units=2,
                              control_labels=["Q_hvac_A", "Q_hvac_B"],
                              unit_bounds=[(-8, 8), (-4, 4)]))
    labs = ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]
    manifold = StateManifold(labels=labs, units={}, bounds={},
                             scale={l: 5.0 for l in labs})
    ls = EnergyLandscape(setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
                         weights={"comfort": 1.0, "energy": 0.0, "smooth": 0.0},
                         manifold=manifold, hvac=sim.hvac,
                         comfort_min=25.0, comfort_max=26.8)
    solver = GeodesicSolver(simulator=sim, landscape=ls, horizon=6)
    state = SystemState(np.array([27.0, 26.0, 27.0, 26.0, 26.5]), labs)
    traj = solver.solve(state, MockExternalInputProvider().get(8.0, 6))
    assert traj.controls
    for c in traj.controls:
        assert -8.0 - 1e-9 <= c.u[0] <= 8.0 + 1e-9
        assert -4.0 - 1e-9 <= c.u[1] <= 4.0 + 1e-9


def test_config_per_zone_capacity():
    engine = engine_from_dict({
        "model": {"type": "two_zone"},
        "zones": {"A": {"q_max": 8.0}, "B": {"q_max": 6.0}},
    })
    assert engine.simulator.hvac.bounds() == [(-8.0, 8.0), (-8.0, 6.0)]


def test_config_rejects_unknown_zone_key():
    with pytest.raises(ValueError, match="未知键"):
        engine_from_dict({"model": {"type": "two_zone"},
                          "zones": {"A": {"capacity": 5.0}}})


# --- 鲁棒扰动覆盖面（各模型类型安全） ---

def test_perturbation_covers_all_model_types():
    delta, sign = 0.15, 1.0
    p1 = _perturbed_building(RCBuildingModel(), sign, delta)
    assert p1.r_wall != RCBuildingModel().r_wall
    p2 = _perturbed_building(TwoZoneRCBuildingModel(), sign, delta)
    assert p2.r_wall_a != TwoZoneRCBuildingModel().r_wall_a
    assert p2.r_wall_b != TwoZoneRCBuildingModel().r_wall_b
    p3 = _perturbed_building(NonlinearRCBuildingModel(), sign, delta)
    assert p3.r_wall != NonlinearRCBuildingModel().r_wall
    p4 = _perturbed_building(DataCenterCoolingModel(), sign, delta)
    assert p4.r_out != DataCenterCoolingModel().r_out
    assert p4.c_tank != DataCenterCoolingModel().c_tank
    # 设备能力上限不被扰动
    assert p4.q_disc_max == DataCenterCoolingModel().q_disc_max
    # dt 永不扰动
    assert p1.dt == RCBuildingModel().dt


def test_perturbed_datacenter_model_steps():
    p = _perturbed_building(DataCenterCoolingModel(), 1.0, 0.15)
    sim = Simulator(p, HVACModel(q_min=-150, q_max=150, n_units=2,
                                 control_labels=["Q_chiller", "Q_tank"]))
    state = SystemState(np.array([24.0, 22.0]), ["T_aisle", "T_storage"])
    w = MockExternalInputProvider().get(8.0, 1)[0]
    nxt = sim.step(state, ControlInput(np.array([-80.0, 60.0])), w, 1.0 / 12.0)
    assert np.all(np.isfinite(nxt.x))


# --- 试点 CSV 日志 ---

def test_append_row_writes_header_once(tmp_path):
    path = str(tmp_path / "mv" / "pilot.csv")
    append_row(path, {"time_iso": "2026-08-30T10:00", "T_air": 26.1,
                      "u_kw": -7.5, "setpoint": 20.4, "extra_key": "ignored"})
    append_row(path, {"time_iso": "2026-08-30T10:15", "T_air": 25.8})
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(FIELDNAMES)
    assert len(rows) == 3  # 表头 + 2 行
    assert "extra_key" not in rows[1]  # 多余键忽略
    assert rows[2][FIELDNAMES.index("T_air")] == "25.8"
    assert rows[2][FIELDNAMES.index("power_w")] == ""  # 缺失键留空


def test_append_row_appends_to_existing_file_with_data(tmp_path):
    path = str(tmp_path / "pilot.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("pre-existing header\n")
    append_row(path, {"time_iso": "x"})
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    # 已有内容的文件不再写表头（避免破坏既有文件）
    assert len(lines) == 2
    assert lines[0] == "pre-existing header"
    assert lines[1].startswith("x,")


# --- 有人时段违温 KPI ---

def test_violations_occupied_excludes_night_drift():
    from gccm_be.app.pilot_log import violations_occupied
    rows = []
    # 白天 25°C（达标），夜间 19°C（漂移，但无人）
    for h in range(24):
        t = 25.0 if 7.0 <= h < 19.0 else 19.0
        rows.append([float(h), t, t])
    bands = {"A": (22.0, 26.0), "B": (22.0, 26.0)}
    full = violations_occupied(rows, bands, occupied_hours=(0.0, 24.0))
    occ = violations_occupied(rows, bands, occupied_hours=(7.0, 19.0))
    assert full["A"] > 40.0        # 旧口径：夜间漂移全算违温
    assert occ["A"] == 0.0         # 新口径：有人时段全部达标


# --- _perturbed_building 保留非扰动字段(with_slab 传递) ---

def test_perturbed_building_preserves_with_slab_and_custom_params():
    from gccm_be.app.config import _perturbed_building
    from gccm_be.physics.models import TwoZoneRCBuildingModel
    b = TwoZoneRCBuildingModel(with_slab=True, c_slab=12.0, r_slab=0.4)
    p = _perturbed_building(b, 1.0, 0.15)
    # 结构字段保留（旧实现 type(building)(**fields) 会丢 with_slab）
    assert p.with_slab is True
    # 用户自定义的未扰动参数保留（旧实现静默落回默认 15.0/0.5）
    assert p.c_slab == 12.0 and p.r_slab == 0.4
    # 热参数被扰动
    assert p.r_wall_a != b.r_wall_a
    # 标签与 dt 保留
    assert p.state_labels == b.state_labels and p.dt == b.dt
