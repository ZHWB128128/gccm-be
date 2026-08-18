"""外部建筑仿真适配器：连接 EnergyPlus/BOPTEST/真实建筑仿真器。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..types import ControlInput, ExternalInput, SystemState
from .models import Simulator, TwoZoneRCBuildingModel, HVACModel


class BuildingSimulatorAdapter:
    """外部建筑仿真器统一接口。

    真实实现可对接 EnergyPlus、BOPTEST、Modelica 等。
    """

    def reset(self, initial_state: SystemState) -> None:
        raise NotImplementedError

    def step(self, control: ControlInput, external: ExternalInput) -> SystemState:
        """推进外部仿真一步，返回测量状态。"""
        raise NotImplementedError

    def get_measurements(self) -> Dict[str, float]:
        raise NotImplementedError

    def set_actuators(self, control: ControlInput) -> None:
        raise NotImplementedError


@dataclass
class EnergyPlusAdapterStub(BuildingSimulatorAdapter):
    """EnergyPlus 适配器模拟实现。

    实际部署时，将内部 simulator 替换为 EnergyPlus 的 Python API 调用。
    """

    simulator: Simulator = None  # type: ignore[assignment]
    state: SystemState = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.simulator is None:
            building = TwoZoneRCBuildingModel()
            hvac = HVACModel(q_min=-8, q_max=8, n_units=2,
                             control_labels=["Q_hvac_A", "Q_hvac_B"])
            self.simulator = Simulator(building, hvac)

    def reset(self, initial_state: SystemState) -> None:
        self.state = initial_state.copy()

    def step(self, control: ControlInput, external: ExternalInput) -> SystemState:
        if self.state is None:
            raise RuntimeError("Adapter not reset")
        self.state = self.simulator.step(self.state, control, external, 0.25)
        return self.state.copy()

    def get_state(self) -> SystemState:
        if self.state is None:
            raise RuntimeError("Adapter not reset")
        return self.state.copy()

    def get_measurements(self) -> Dict[str, float]:
        labels = self.state.labels
        return {lab: float(v) for lab, v in zip(labels, self.state.x)}

    def set_actuators(self, control: ControlInput) -> None:
        # 真实 EnergyPlus 中这里会写 BCVTB/外部接口
        pass
