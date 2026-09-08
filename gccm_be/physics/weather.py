"""Real weather forecast provider: Open-Meteo NWP adapter (no API key required).

把真实数值天气预报接入 MPC 的外部输入通道。设计要点：

- **可注入传输**：`fetch` 参数注入取数函数（生产用 stdlib urllib，测试/离线用
  桩函数），核心逻辑不碰网络即可测试；
- **按时钟对齐**：引擎的 time_h 是相对小时，`start_hour` 声明 t=0 对应的当日
  时刻（默认 0 点），预报按"当日小时"取最近整点值——避免日期对齐问题；
- **安全降级**：拉取失败抛 RuntimeError，由 config 层捕获并回退确定性 mock；
  `refresh()` 支持定期更新预报。

多区：`zones=("A","B")` 时输出 `solar_A/solar_B/occ_A/occ_B` 标签（各区得热 =
同一气象源 × 区系数，人员按区系数缩放），单区输出通用 `solar/occ`。
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..types import ExternalInput
from .external import ExternalInputProvider


def _http_get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class NWPWeatherProvider(ExternalInputProvider):
    """Open-Meteo 逐时预报 → GCCM 外部输入（T_out / solar / occ / price）。"""

    latitude: float = 30.0
    longitude: float = 120.0
    zones: tuple[str, ...] = ()          # 双区 ("A","B")；单区 ()
    dt_h: float = 1.0 / 12.0
    start_hour: float = 0.0              # 引擎 t=0 对应的当日时刻（小时）
    solar_scale: float = 800.0           # W/m² → 归一化得热（800 W/m² ≈ 1.0）
    occ_day: float = 0.75
    occ_night: float = 0.2
    occ_start: float = 8.0
    occ_end: float = 18.0
    zone_occ_scale: dict[str, float] = field(default_factory=dict)  # 区人数系数
    price_peak: float = 1.5
    price_flat: float = 0.8
    price_valley: float = 0.3
    price_peak_hours: tuple[float, float] = (11.0, 18.0)
    price_valley_hours: tuple[float, float] = (22.0, 6.0)
    fetch: Callable[[str], dict] | None = None   # 注入传输（测试/换源）
    timeout: float = 10.0
    timezone: str = "auto"               # 预报时间戳时区：auto=坐标当地时区
    labels: list[str] = None  # type: ignore[assignment]

    _hours: list[float] = field(default_factory=list, init=False)     # 当日小时
    _t_out: list[float] = field(default_factory=list, init=False)
    _solar: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.labels is None:
            if self.zones:
                self.labels = (["T_out"]
                               + [f"solar_{z}" for z in self.zones]
                               + [f"occ_{z}" for z in self.zones] + ["price"])
            else:
                self.labels = ["T_out", "solar", "occ", "price"]

    # ---- 预报拉取与解析 ----
    def _url(self) -> str:
        vars_ = "temperature_2m,shortwave_radiation"
        return (f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={self.latitude}&longitude={self.longitude}"
                f"&hourly={vars_}&forecast_days=2&timeformat=iso8601"
                f"&timezone={self.timezone}")

    def refresh(self) -> None:
        """拉取/更新预报。失败抛 RuntimeError，由调用方决定降级策略。"""
        transport = self.fetch or _http_get_json
        data = transport(self._url())
        try:
            hourly = data["hourly"]
            times, temps, rads = hourly["time"], hourly["temperature_2m"], hourly["shortwave_radiation"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Open-Meteo 响应缺少 hourly 字段: {exc}") from exc
        hours, t_out, solar = [], [], []
        for ts, tv, rv in zip(times, temps, rads):
            # ISO 时间戳 "2026-08-30T14:00" → 当日小时
            hour = float(ts.split("T")[1].split(":")[0]) + float(ts.split("T")[1].split(":")[1]) / 60.0
            hours.append(hour)
            t_out.append(float(tv))
            solar.append(max(0.0, float(rv or 0.0)) / max(self.solar_scale, 1e-9))
        if not hours:
            raise RuntimeError("Open-Meteo 返回空预报")
        self._hours, self._t_out, self._solar = hours, t_out, solar

    def _forecast_at(self, hour_of_day: float) -> tuple[float, float]:
        """取该当日小时最近整点的预报值；无数据时取最近可用点。"""
        if not self._hours:
            raise RuntimeError("天气预报未拉取，先调用 refresh()")
        best_i, best_d = 0, float("inf")
        for i, h in enumerate(self._hours):
            d = abs(h - hour_of_day)
            if d < best_d:
                best_i, best_d = i, d
        return self._t_out[best_i], self._solar[best_i]

    def _price_at(self, hour: float) -> float:
        p0, p1 = self.price_peak_hours
        v0, v1 = self.price_valley_hours
        if v0 > v1:  # 跨午夜谷段
            if hour >= v0 or hour < v1:
                return self.price_valley
        if p0 <= hour < p1:
            return self.price_peak
        return self.price_flat

    def _occ_at(self, hour: float, zone_scale: float) -> float:
        base = self.occ_day if self.occ_start <= hour < self.occ_end else self.occ_night
        return base * zone_scale

    def get(self, time_h: float, horizon: int = 1) -> list[ExternalInput]:
        result = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = (t + self.start_hour) % 24.0
            t_out, solar = self._forecast_at(hour)
            w: list[float] = [t_out]
            if self.zones:
                for z in self.zones:
                    w.append(solar)
                for z in self.zones:
                    w.append(self._occ_at(hour, float(self.zone_occ_scale.get(z, 1.0))))
            else:
                w += [solar, self._occ_at(hour, 1.0)]
            w.append(self._price_at(hour))
            result.append(ExternalInput(np.array(w, dtype=float), list(self.labels)))
        return result
