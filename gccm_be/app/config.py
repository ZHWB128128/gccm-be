"""Build GCCMEngine from a JSON config file (productized: change building without changing code).

See examples/config.json.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..engine import GCCMEngine
from ..geometry.manifold import StateManifold
from ..physics.models import HVACModel, RCBuildingModel, Simulator


DEFAULTS: Dict[str, Any] = {
    "building": {
        "c_air": 0.6,
        "c_wall": 4.0,
        "r_air": 0.8,
        "r_wall": 2.0,
        "solar_gain": 0.05,
        "dt": 1.0 / 12.0,
    },
    "hvac": {
        "q_min": -8.0,
        "q_max": 8.0,
        "cop_heating": 3.2,
        "cop_cooling": 3.8,
    },
    "controller": {
        "horizon": 24,
        "comfort_min": 25.0,
        "comfort_max": 27.0,
        "setpoint": 26.0,
        "comfort_weight": 5.0,
        "energy_weight": 0.5,
        "smooth_weight": 0.1,
        "comfort_margin": 0.0,
        "peak_price_threshold": 1.0,
        "peak_energy_penalty": 1.5,
        "enforce_comfort_constraints": True,
        "use_kinetic": True,
        "safe_control_mode": "feedback",
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8080,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    return _deep_merge(DEFAULTS, user)


def engine_from_config(path: str) -> GCCMEngine:
    """从配置文件构建引擎（当前支持单区域 RC 模型）。"""
    cfg = load_config(path)
    b = cfg["building"]
    h = cfg["hvac"]
    c = cfg["controller"]

    building = RCBuildingModel(
        c_air=b["c_air"], c_wall=b["c_wall"], r_air=b["r_air"],
        r_wall=b["r_wall"], solar_gain=b["solar_gain"], dt=b["dt"],
    )
    hvac = HVACModel(
        q_min=h["q_min"], q_max=h["q_max"],
        cop_heating=h["cop_heating"], cop_cooling=h["cop_cooling"],
    )
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    return GCCMEngine(
        simulator=Simulator(building, hvac),
        manifold=manifold,
        horizon=c["horizon"],
        setpoints={"T_air": c["setpoint"]},
        comfort_min=c["comfort_min"],
        comfort_max=c["comfort_max"],
        comfort_margin=c["comfort_margin"],
        comfort_weight=c["comfort_weight"],
        energy_weight=c["energy_weight"],
        smooth_weight=c["smooth_weight"],
        peak_price_threshold=c["peak_price_threshold"],
        peak_energy_penalty=c["peak_energy_penalty"],
        enforce_comfort_constraints=c["enforce_comfort_constraints"],
        use_kinetic=c["use_kinetic"],
        safe_control_mode=c["safe_control_mode"],
    )
