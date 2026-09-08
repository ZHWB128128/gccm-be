"""Building model adapter protocol: one registry for all HVAC structure variants.

缺点 #3 修正：此前每换一类 HVAC（水暖/VAV/数据中心），结构适配散落在
模型类、config 分发器、CasADi _casadi_step 三处，规模化时每类都要改三处。

本模块定义适配协议 + 注册表，把"结构适配"收敛为单一注册动作：

1. `BuildingModelProtocol`：一类 HVAC 模型必须声明
   - `state_labels` / `control_labels` / `external_labels`（结构自描述）
   - `step(state, control, external, dt)`（物理）
   - `state_index(label)`（按名定位，求解器/引擎不再猜位置）
   - `air_state_labels()`（温度区标签，硬约束/KPI 用）
2. `register_building(name, cls)` + `build_building(name, **params)`：
   config 的 `model.type` 经注册表分发，新 HVAC 类型 = 一个注册调用 +
   CasADi 分支（仍需手写，但只写一处）。

现有模型（RCBuildingModel/TwoZoneRCBuildingModel/ThreeRCBuildingModel/
NonlinearRCBuildingModel/DataCenterCoolingModel）在 import 时自动注册，
行为不变——这是重构不是重写。
"""
from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, tuple[type, Callable]] = {}


def register_building(name: str, cls: type,
                      config_builder: Callable | None = None) -> None:
    """注册一类建筑/HVAC 模型。

    name: config `model.type` 的取值（如 "two_zone"）。
    cls: 模型类（须含 state_labels/control_labels/step）。
    config_builder: 可选 (params: dict) -> 模型实例；缺省用 cls(**params)。
    """
    _REGISTRY[name] = (cls, config_builder or cls)


def build_building(model_type: str, **params):
    """按注册表构建模型实例；未知类型报可用清单。"""
    entry = _REGISTRY.get(model_type)
    if entry is None:
        raise ValueError(
            f"未知 model.type: {model_type!r}，已注册：{sorted(_REGISTRY)}"
        )
    _, builder = entry
    return builder(**params)


def registered_buildings() -> dict[str, type]:
    return {name: cls for name, cls, _ in
            ((n, *v) for n, v in _REGISTRY.items())} if False else {
        name: cls for name, (cls, _) in _REGISTRY.items()
    }


# --- 协议检查：注册时验证模型类满足最小结构契约 ---

def _validate_protocol(cls: type) -> None:
    for attr in ("state_labels", "control_labels", "step"):
        if not hasattr(cls, attr) and not (isinstance(getattr(cls, attr, None), property)):
            # dataclass 字段在类上有默认值即可；实例一定有
            import dataclasses
            if not (dataclasses.is_dataclass(cls)
                    and any(f.name == attr for f in dataclasses.fields(cls))):
                raise TypeError(
                    f"{cls.__name__} 不满足 BuildingModelProtocol：缺少 {attr}"
                )


def register_building_checked(name: str, cls: type,
                              config_builder: Callable | None = None) -> None:
    _validate_protocol(cls)
    register_building(name, cls, config_builder)


def air_state_labels(building) -> list[str]:
    """该模型的温度区标签（T_air* 前缀；数据中心等特殊模型自行覆盖命名）。"""
    labels = getattr(building, "state_labels", [])
    air = [lab for lab in labels if str(lab).startswith("T_air")]
    if not air:  # 旧约定：无 T_air* 标签时第一个状态是主温区
        air = [labels[0]] if labels else []
    return air


def state_index(building, label: str) -> int:
    """按名定位状态索引（O(n) 查找，结构变化安全）。"""
    labels = list(getattr(building, "state_labels", []))
    return labels.index(label)


# --- 现有模型自动注册（import 时生效）---

def _register_builtin() -> None:
    from .datacenter import DataCenterCoolingModel
    from .models import (
        NonlinearRCBuildingModel,
        RCBuildingModel,
        ThreeRCBuildingModel,
        TwoZoneRCBuildingModel,
    )
    for name, cls in (
        ("single_zone", RCBuildingModel),
        ("two_zone", TwoZoneRCBuildingModel),
        ("three_rc", ThreeRCBuildingModel),
        ("nonlinear_rc", NonlinearRCBuildingModel),
        ("datacenter", DataCenterCoolingModel),
    ):
        register_building_checked(name, cls)


_register_builtin()
