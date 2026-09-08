"""Home Assistant adapter: pilot data channel (measurements in, setpoint out).

实楼试点的第一版数据通道（见 docs/PILOT_PLAN.md）：米家插座/温湿度计经 HA 网关
接入，GCCM 读取室温/功率、下发空调设定温度。BACnet/Modbus 是楼宇级的事，
试点阶段用 HA REST API 即可。

- **可注入传输**：`request` 注入 HTTP 函数 `(method, path, body) -> dict`，
  生产用 stdlib urllib + Long-Lived Access Token，测试用桩函数；
- 被控对象是真实建筑，`step()` 不做仿真推进（真实世界自己走），只暴露
  读测量 / 写设定温度两个动作；策略层（控制功率 → 设定温度的映射）由调用方持有。
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from ..types import ControlInput, ExternalInput, SystemState
from .building_simulator_adapter import BuildingSimulatorAdapter


def _http_request(base_url: str, token: str, timeout: float):
    """生产传输：返回 (method, path, body) -> dict。"""
    def request(method: str, path: str, body: dict | None = None) -> dict:
        url = base_url.rstrip("/") + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    return request


@dataclass
class HABuildingAdapter(BuildingSimulatorAdapter):
    """Home Assistant REST 适配器：读传感器状态，写 climate 设定温度。"""

    base_url: str = "http://homeassistant.local:8123"
    token: str = ""
    temp_entity: str = ""                    # 室温传感器，如 sensor.room_temp
    power_entity: str | None = None       # 空调功率/电量传感器（可选）
    climate_entity: str | None = None     # 空调 climate 实体（写设定温度）
    timeout: float = 5.0
    request: Callable[[str, str, dict | None], dict] | None = None

    state: SystemState | None = None

    def __post_init__(self) -> None:
        if self.request is None:
            self.request = _http_request(self.base_url, self.token, self.timeout)

    # ---- BuildingSimulatorAdapter 契约 ----
    def reset(self, initial_state: SystemState) -> None:
        self.state = initial_state.copy()

    def step(self, control: ControlInput, external: ExternalInput) -> SystemState:
        """真实建筑实时演化，不做仿真推进；返回最近一次读到的测量状态。"""
        if self.state is None:
            raise RuntimeError("HABuildingAdapter not reset")
        self.sync_measurements()
        return self.get_state()

    def get_measurements(self) -> dict[str, float]:
        out: dict[str, float] = {"temperature": self._read_state(self.temp_entity)}
        if self.power_entity:
            out["power"] = self._read_state(self.power_entity)
        return out

    def set_actuators(self, control: ControlInput) -> None:
        """真实执行器由 set_target_temperature 驱动；功率级控制不下发。"""
        raise NotImplementedError(
            "真实空调下发请用 set_target_temperature()；功率→设定温度的映射策略由试点层持有"
        )

    # ---- 试点动作 ----
    def sync_measurements(self) -> SystemState:
        """读 HA 传感器，刷新内部状态（首个状态即 T_air 通道）。"""
        m = self.get_measurements()
        if self.state is None:
            raise RuntimeError("HABuildingAdapter not reset")
        if self.state.labels:
            self.state.x[0] = m["temperature"]
        return self.get_state()

    def get_state(self) -> SystemState:
        if self.state is None:
            raise RuntimeError("HABuildingAdapter not reset")
        return self.state.copy()

    def set_target_temperature(self, value: float) -> dict:
        """经 HA 下发空调设定温度（climate.set_temperature）。"""
        if not self.climate_entity:
            raise ValueError("未配置 climate_entity，无法下发设定温度")
        return self.request("POST", "/api/services/climate/set_temperature", {
            "entity_id": self.climate_entity,
            "temperature": float(value),
        })

    # ---- 内部 ----
    def _read_state(self, entity: str) -> float:
        if not entity:
            raise ValueError("未配置传感器实体")
        data = self.request("GET", f"/api/states/{entity}", None)
        return float(data["state"])
