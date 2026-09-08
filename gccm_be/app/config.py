"""Build GCCMEngine from a JSON config file (productized: change building without changing code).

Model selection: `"model": {"type": "single_zone"}` (default) or
`"model": {"type": "two_zone"}`. See examples/config.json and
examples/config_two_zone.json.
"""
from __future__ import annotations

import json
from typing import Any

from ..engine import GCCMEngine
from ..geometry.manifold import StateManifold
from ..physics.models import (
    HVACModel,
    RCBuildingModel,
    Simulator,
    TwoZoneRCBuildingModel,
)

DEFAULTS: dict[str, Any] = {
    "model": {
        "type": "single_zone",
    },
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
    # 每区设定点/舒适带覆盖：{"A": {"setpoint": 26.0, "comfort_max": 26.5}}。
    # 未给出的字段回退到 controller 对应标量值。
    "zones": {},
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
        "confidence_calibrated": True,
        "confidence_quantile": 0.90,
    },
    # 预报误差鲁棒：自动生成 ±delta RC 参数扰动的场景仿真器，启用多场景共享
    # 控制序列的鲁棒 MPC（纯 numpy 路径；CasADi 鲁棒后端仍由 use_casadi_robust 控制）。
    # delta 校准依据：tools_local/calibrate_robust.py 用真实 NWP 误差分布测算
    # （温度 P95 4.6K / 辐射 P90 66% → delta 显著大于早期拍脑袋的 0.15）；
    # 设 "delta": null 时从 calibration 文件读取 recommended_delta。
    "robust": {
        "enabled": True,
        "delta": 0.30,
        "calibration_file": None,
    },
    # 真实天气预报（Open-Meteo，无需 key）：provider=nwp 时生效；拉取失败自动
    # 回退确定性 mock 并打印警告。zones 为双区模型的外部输入区分标签。
    "weather": {
        "provider": "mock",
        "latitude": 30.0,
        "longitude": 120.0,
        "solar_scale": 800.0,
        "price_peak": 1.5,
        "price_flat": 0.8,
        "price_valley": 0.3,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8080,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_user_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> dict[str, Any]:
    return _deep_merge(DEFAULTS, _read_user_config(path))


def _controller_kwargs(c: dict[str, Any]) -> dict[str, Any]:
    """Controller section shared by all model types."""
    return dict(
        horizon=c["horizon"],
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
        confidence_calibrated=c["confidence_calibrated"],
        confidence_quantile=c["confidence_quantile"],
    )


def _hvac_kwargs(h: dict[str, Any]) -> dict[str, Any]:
    return dict(
        q_min=h["q_min"],
        q_max=h["q_max"],
        cop_heating=h["cop_heating"],
        cop_cooling=h["cop_cooling"],
    )


def _single_zone_engine(cfg: dict[str, Any], _user_building: dict[str, Any]) -> GCCMEngine:
    b = cfg["building"]
    c = cfg["controller"]

    building = RCBuildingModel(
        c_air=b["c_air"], c_wall=b["c_wall"], r_air=b["r_air"],
        r_wall=b["r_wall"], solar_gain=b["solar_gain"], dt=b["dt"],
    )
    hvac = HVACModel(**_hvac_kwargs(cfg["hvac"]))
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        units={"T_air": "°C", "T_wall": "°C"},
        bounds={"T_air": (15.0, 40.0), "T_wall": (15.0, 40.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    return GCCMEngine(
        simulator=Simulator(building, hvac),
        manifold=manifold,
        setpoints={"T_air": c["setpoint"]},
        **_controller_kwargs(c),
    )


# TwoZoneRCBuildingModel 的结构字段（标签列表）不由配置提供，只开放物理参数。
_TWO_ZONE_BUILDING_FIELDS = (
    "c_air", "c_wall", "c_partition", "r_air", "r_wall_a", "r_wall_b",
    "r_partition", "solar_gain_a", "solar_gain_b", "dt",
    "with_slab", "c_slab", "r_slab",
)


def _two_zone_engine(cfg: dict[str, Any], user_building: dict[str, Any]) -> GCCMEngine:
    # 双区 building 参数默认值由 TwoZoneRCBuildingModel 提供；
    # 只接受用户显式给出的键，避免混入单区 DEFAULTS 里的同名不同义参数
    # （如 r_wall/solar_gain 在双区对应 r_wall_a/r_wall_b/solar_gain_a/solar_gain_b）。
    b = user_building
    c = cfg["controller"]

    unknown = sorted(set(b) - set(_TWO_ZONE_BUILDING_FIELDS))
    if unknown:
        raise ValueError(
            f"two_zone building 配置含未知参数 {unknown}，"
            f"可用参数：{list(_TWO_ZONE_BUILDING_FIELDS)}"
        )
    building = TwoZoneRCBuildingModel(**b)

    # 每区设定点：zones 以模型固定区号 A/B 为键，缺省回退 controller.setpoint
    zones_cfg = cfg.get("zones") or {}
    if not isinstance(zones_cfg, dict):
        raise ValueError("zones 必须是 {\"A\": {...}, \"B\": {...}} 形式的对象")
    bad = sorted(set(zones_cfg) - {"A", "B"})
    if bad:
        raise ValueError(f"zones 只支持区号 A/B（双区模型固定），收到 {bad}")
    setpoints: dict[str, float] = {}
    zone_bounds: dict[str, tuple] = {}
    unit_bounds: list[tuple] = []
    zone_keys_allowed = {"setpoint", "comfort_min", "comfort_max", "q_min", "q_max"}
    h = cfg["hvac"]
    for zid in ("A", "B"):
        entry = zones_cfg.get(zid) or {}
        if not isinstance(entry, dict):
            raise ValueError(f"zones.{zid} 必须是对象")
        unknown_zone_keys = sorted(set(entry) - zone_keys_allowed)
        if unknown_zone_keys:
            raise ValueError(
                f"zones.{zid} 含未知键 {unknown_zone_keys}，可用：{sorted(zone_keys_allowed)}"
            )
        setpoints[f"T_air_{zid}"] = float(entry.get("setpoint", c["setpoint"]))
        # 每区独立舒适带
        lo = entry.get("comfort_min")
        hi = entry.get("comfort_max")
        if lo is not None or hi is not None:
            lo_f = float(lo) if lo is not None else float(c["comfort_min"])
            hi_f = float(hi) if hi is not None else float(c["comfort_max"])
            if lo_f >= hi_f:
                raise ValueError(f"zones.{zid} comfort_min >= comfort_max: {lo_f} >= {hi_f}")
            zone_bounds[f"T_air_{zid}"] = (lo_f, hi_f)
        # 每区独立设备容量
        unit_bounds.append((
            float(entry["q_min"]) if "q_min" in entry else float(h["q_min"]),
            float(entry["q_max"]) if "q_max" in entry else float(h["q_max"]),
        ))

    hvac_kwargs = _hvac_kwargs(h)
    if any((lo, hi) != (h["q_min"], h["q_max"]) for lo, hi in unit_bounds):
        hvac_kwargs["unit_bounds"] = unit_bounds
    hvac = HVACModel(n_units=2, control_labels=["Q_hvac_A", "Q_hvac_B"], **hvac_kwargs)
    manifold = StateManifold(
        labels=list(building.state_labels),
        units={lab: "°C" for lab in building.state_labels},
        bounds={lab: (15.0, 40.0) for lab in building.state_labels},
        scale={lab: 5.0 for lab in building.state_labels},
    )

    return GCCMEngine(
        simulator=Simulator(building, hvac),
        manifold=manifold,
        setpoints=setpoints,
        zone_comfort_bounds=zone_bounds,
        **_controller_kwargs(c),
    )


def _perturbed_building(building, sign: float, delta: float):
    """RC 参数整体扰动 ±delta，生成鲁棒场景建筑（dt 不扰动）。

    覆盖建筑/数据中心两类模型的热参数；设备能力上限（q_disc_max 等）是
    物理铭牌值，不属于预报不确定性，不扰动。
    """
    fields = {}
    for f in ("c_air", "c_wall", "c_partition", "c_furn", "c_aisle", "c_tank",
              "r_air", "r_wall", "r_wall_a", "r_wall_b", "r_partition", "r_out",
              "solar_gain", "solar_gain_a", "solar_gain_b"):
        if hasattr(building, f):
            fields[f] = getattr(building, f) * (1.0 + sign * delta)
    return type(building)(**fields)


def _apply_robust_scenarios(engine: GCCMEngine, cfg: dict[str, Any]) -> GCCMEngine:
    r = cfg.get("robust") or {}
    if not r.get("enabled"):
        return engine
    delta = r.get("delta")
    if delta is None:
        cal_file = r.get("calibration_file")
        if not cal_file:
            raise ValueError('robust.delta 为 null 时必须提供 robust.calibration_file')
        delta = json.load(open(cal_file, encoding="utf-8"))["calibration"]["recommended_delta"]
    delta = float(delta)
    building = engine.simulator.building
    hvac = engine.simulator.hvac
    engine.robust_scenarios = [
        Simulator(_perturbed_building(building, sign, delta), hvac)
        for sign in (1.0, -1.0)
    ]
    return engine


def _build_external_provider(cfg: dict[str, Any]):
    """按 weather 配置构建外部输入提供者；NWP 拉取失败回退 mock（带警告）。"""
    w = cfg.get("weather") or {}
    provider = str(w.get("provider", "mock")).lower()
    if provider != "nwp":
        return None  # 引擎默认 mock
    from ..physics.weather import NWPWeatherProvider
    zones = ("A", "B") if str(cfg.get("model", {}).get("type", "single_zone")).lower() == "two_zone" else ()
    nwp = NWPWeatherProvider(
        latitude=float(w["latitude"]),
        longitude=float(w["longitude"]),
        zones=zones,
        solar_scale=float(w.get("solar_scale", 800.0)),
        price_peak=float(w.get("price_peak", 1.5)),
        price_flat=float(w.get("price_flat", 0.8)),
        price_valley=float(w.get("price_valley", 0.3)),
    )
    try:
        nwp.refresh()
    except Exception as exc:
        print(f"[gccm-be] 警告: 天气预报拉取失败（{exc}），回退确定性 mock 外部输入。")
        return None
    return nwp


_MODEL_BUILDERS = {
    "single_zone": _single_zone_engine,
    "two_zone": _two_zone_engine,
}


def engine_from_dict(cfg: dict[str, Any]) -> GCCMEngine:
    """从配置字典构建引擎（不落盘，供 API/测试复用）。

    model.type 经 physics.registry 分发：single_zone/two_zone 走带严格校验的
    专属 builder；其余注册类型（three_rc/nonlinear_rc/datacenter）走通用
    registry 构建路径（结构自描述，新 HVAC 类型无需逐类 config 代码）。
    """
    user_building = dict(cfg.get("building") or {})
    cfg = _deep_merge(DEFAULTS, cfg)
    mtype = str(cfg.get("model", {}).get("type", "single_zone")).lower()
    builder = _MODEL_BUILDERS.get(mtype)
    if builder is not None:
        engine = builder(cfg, user_building)
    else:
        from ..physics.registry import build_building, registered_buildings
        if mtype not in registered_buildings():
            raise ValueError(
                f"未知 model.type: {mtype!r}，可选："
                f"{sorted(set(_MODEL_BUILDERS) | set(registered_buildings()))}"
            )
        building = build_building(mtype, **user_building)
        hvac = HVACModel(**_hvac_kwargs(cfg["hvac"]))
        labels = list(building.state_labels)
        c = cfg["controller"]
        manifold = StateManifold(
            labels=labels,
            units={lab: "°C" for lab in labels},
            bounds={lab: (5.0, 45.0) for lab in labels},
            scale={lab: 5.0 for lab in labels},
        )
        engine = GCCMEngine(
            simulator=Simulator(building, hvac),
            manifold=manifold,
            setpoints={lab: c["setpoint"] for lab in labels if lab.startswith("T_air")}
            or {"T_air": c["setpoint"]},
            **_controller_kwargs(c),
        )
    provider = _build_external_provider(cfg)
    if provider is not None:
        engine.external_provider = provider
    return _apply_robust_scenarios(engine, cfg)


def engine_from_config(path: str) -> GCCMEngine:
    # 传原始用户配置，由 engine_from_dict 统一做 DEFAULTS 合并
    return engine_from_dict(_read_user_config(path))
