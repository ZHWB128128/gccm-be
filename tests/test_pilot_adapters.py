"""试点数据通道行为级测试：NWP 天气预报适配器 + Home Assistant 适配器。"""
from __future__ import annotations

import pytest

from gccm_be.app.config import engine_from_dict
from gccm_be.physics.external import MockExternalInputProvider
from gccm_be.physics.homeassistant import HABuildingAdapter
from gccm_be.physics.weather import NWPWeatherProvider
from gccm_be.physics import weather as weather_mod
from gccm_be.types import SystemState


def _open_meteo_payload(n_hours: int = 48) -> dict:
    """构造 Open-Meteo /forecast 响应桩：正弦室外温度 + 白天辐射。"""
    times, temps, rads = [], [], []
    for h in range(n_hours):
        day, hod = divmod(h, 24)
        times.append(f"2026-08-{30 + day:02d}T{hod:02d}:00")
        temps.append(29.0 + 4.0 * ((hod - 5.0) / 12.0))
        rads.append(700.0 if 8 <= hod <= 16 else 0.0)
    return {"hourly": {"time": times, "temperature_2m": temps, "shortwave_radiation": rads}}


# --- NWPWeatherProvider ---

def test_nwp_single_zone_labels_and_values():
    p = NWPWeatherProvider(latitude=30.0, longitude=120.0,
                           fetch=lambda url: _open_meteo_payload())
    p.refresh()
    assert p.labels == ["T_out", "solar", "occ", "price"]
    seq = p.get(8.0, 3)
    assert len(seq) == 3
    w = seq[0]
    # 8 点室外温度 = 桩数据 8 点值
    assert abs(w.get("T_out") - (29.0 + 4.0 * (8.0 - 5.0) / 12.0)) < 1e-9
    # 8 点辐射 700 W/m² → 700/800 归一化
    assert abs(w.solar() - 700.0 / 800.0) < 1e-9
    # 8 点在工作时段
    assert abs(w.occupancy() - 0.75) < 1e-9
    # 8 点平段电价
    assert abs(w.price - 0.8) < 1e-9


def test_nwp_two_zone_labels_and_schedules():
    p = NWPWeatherProvider(latitude=30.0, longitude=120.0, zones=("A", "B"),
                           zone_occ_scale={"A": 1.0, "B": 0.5},
                           fetch=lambda url: _open_meteo_payload())
    p.refresh()
    assert p.labels == ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"]
    w = p.get(14.0, 1)[0]
    assert abs(w.solar(zone="A") - 700.0 / 800.0) < 1e-9
    assert abs(w.occupancy(zone="B") - 0.375) < 1e-9   # 0.75 * 0.5 峰时电价
    assert abs(w.price - 1.5) < 1e-9
    # 谷时（23 点，跨午夜谷段）
    w_night = p.get(23.0, 1)[0]
    assert abs(w_night.price - 0.3) < 1e-9
    assert abs(w_night.occupancy(zone="A") - 0.2) < 1e-9


def test_nwp_missing_hourly_raises():
    p = NWPWeatherProvider(fetch=lambda url: {"error": True})
    with pytest.raises(RuntimeError):
        p.refresh()


def test_config_nwp_falls_back_to_mock_on_fetch_failure(monkeypatch):
    def boom(url):
        raise OSError("network down")
    monkeypatch.setattr(weather_mod, "_http_get_json", boom)
    engine = engine_from_dict({"controller": {"horizon": 6},
                               "weather": {"provider": "nwp",
                                           "latitude": 30.0, "longitude": 120.0}})
    assert isinstance(engine.external_provider, MockExternalInputProvider)


def test_config_nwp_success_wires_provider(monkeypatch):
    monkeypatch.setattr(weather_mod, "_http_get_json", lambda url: _open_meteo_payload())
    engine = engine_from_dict({"model": {"type": "two_zone"},
                               "controller": {"horizon": 6},
                               "weather": {"provider": "nwp",
                                           "latitude": 30.0, "longitude": 120.0}})
    assert isinstance(engine.external_provider, NWPWeatherProvider)
    # 双区模型的外部输入标签自动对齐
    assert engine.external_provider.labels == [
        "T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price",
    ]


# --- HABuildingAdapter ---

def _fake_ha(calls):
    def request(method, path, body=None):
        calls.append((method, path, body))
        if path == "/api/states/sensor.room_temp":
            return {"state": "24.5"}
        if path == "/api/states/sensor.ac_power":
            return {"state": "1300.0"}
        if path == "/api/services/climate/set_temperature":
            return {"ok": True}
        raise AssertionError(f"unexpected path {path}")
    return request


def test_ha_adapter_reads_measurements():
    calls = []
    ad = HABuildingAdapter(base_url="http://ha:8123", token="x",
                           temp_entity="sensor.room_temp",
                           power_entity="sensor.ac_power",
                           request=_fake_ha(calls))
    m = ad.get_measurements()
    assert m == {"temperature": 24.5, "power": 1300.0}
    assert calls[0][0] == "GET"


def test_ha_adapter_reset_sync_and_state():
    calls = []
    ad = HABuildingAdapter(temp_entity="sensor.room_temp",
                           request=_fake_ha(calls))
    ad.reset(SystemState([26.0, 25.0], ["T_air", "T_wall"]))
    state = ad.step(None, None)  # 真实建筑：不做仿真推进，只同步测量
    assert abs(state.x[0] - 24.5) < 1e-9   # 测量刷新进 T_air 通道
    assert abs(state.x[1] - 25.0) < 1e-9


def test_ha_adapter_sets_target_temperature():
    calls = []
    ad = HABuildingAdapter(temp_entity="sensor.room_temp",
                           climate_entity="climate.ac",
                           request=_fake_ha(calls))
    ad.set_target_temperature(26.0)
    method, path, body = calls[0]
    assert (method, path) == ("POST", "/api/services/climate/set_temperature")
    assert body == {"entity_id": "climate.ac", "temperature": 26.0}


def test_ha_adapter_power_level_control_not_supported():
    ad = HABuildingAdapter(temp_entity="sensor.room_temp",
                           request=_fake_ha([]))
    import numpy as np
    from gccm_be.types import ControlInput
    with pytest.raises(NotImplementedError):
        ad.set_actuators(ControlInput([-6.0]))
