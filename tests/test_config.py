"""配置化构建引擎测试。"""
import json
import os

from gccm_be.app.config import DEFAULTS, engine_from_config, load_config


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
