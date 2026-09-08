"""配置化构建引擎测试。"""
import json
import os

import pytest

from gccm_be.app.config import DEFAULTS, engine_from_config, engine_from_dict, load_config
from gccm_be.physics.models import TwoZoneRCBuildingModel


def test_load_config_merges_defaults(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"controller": {"horizon": 12}}), encoding="utf-8")
    cfg = load_config(str(cfg_file))
    # 覆盖值生效
    assert cfg["controller"]["horizon"] == 12
    # 默认值保留
    assert cfg["controller"]["comfort_min"] == DEFAULTS["controller"]["comfort_min"]
    assert cfg["building"]["c_air"] == DEFAULTS["building"]["c_air"]


def test_engine_from_config(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({
        "controller": {"horizon": 6, "comfort_max": 26.5, "safe_control_mode": "worst_case"},
    }), encoding="utf-8")
    engine = engine_from_config(str(cfg_file))
    assert engine.horizon == 6
    assert engine.comfort_max == 26.5
    assert engine.safe_control_mode == "worst_case"
    assert engine.simulator.building is not None
    assert engine.dt == engine.simulator.building.dt  # dt 统一解析


def test_example_config_is_valid():
    example = os.path.join(os.path.dirname(__file__), "..", "examples", "config.json")
    if os.path.exists(example):
        engine = engine_from_config(example)
        assert engine.horizon > 0


# --- 双区配置化（model.type = two_zone）---

def test_engine_from_dict_two_zone_builds_5_state_model():
    engine = engine_from_dict({
        "model": {"type": "two_zone"},
        "controller": {"horizon": 6},
    })
    assert isinstance(engine.simulator.building, TwoZoneRCBuildingModel)
    assert engine.simulator.hvac.n_units == 2
    assert engine.simulator.hvac.control_labels == ["Q_hvac_A", "Q_hvac_B"]
    assert engine.manifold.labels == [
        "T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition",
    ]
    # 未给 zones 时两区都用 controller.setpoint
    assert engine.setpoints == {"T_air_A": 26.0, "T_air_B": 26.0}
    assert engine.dt == engine.simulator.building.dt


def test_two_zone_per_zone_setpoints_and_building_params():
    engine = engine_from_dict({
        "model": {"type": "two_zone"},
        "building": {"r_wall_b": 1.2, "solar_gain_a": 0.1},
        "zones": {"A": {"setpoint": 26.0}, "B": {"setpoint": 25.5}},
    })
    assert engine.setpoints == {"T_air_A": 26.0, "T_air_B": 25.5}
    assert engine.simulator.building.r_wall_b == 1.2
    assert engine.simulator.building.solar_gain_a == 0.1


def test_two_zone_rejects_unknown_building_param():
    with pytest.raises(ValueError, match="unknown|未知|未知参数"):
        engine_from_dict({"model": {"type": "two_zone"}, "building": {"nope": 1.0}})


def test_two_zone_rejects_unknown_zone_id():
    with pytest.raises(ValueError):
        engine_from_dict({"model": {"type": "two_zone"}, "zones": {"C": {"setpoint": 25.0}}})


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="model.type"):
        engine_from_dict({"model": {"type": "three_zone"}})


def test_two_zone_example_config_is_valid():
    example = os.path.join(os.path.dirname(__file__), "..", "examples", "config_two_zone.json")
    if os.path.exists(example):
        engine = engine_from_config(example)
        assert isinstance(engine.simulator.building, TwoZoneRCBuildingModel)
        assert engine.setpoints == {"T_air_A": 26.0, "T_air_B": 25.5}
