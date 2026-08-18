"""GCCM-BE: 建筑能源几何因果推理引擎.

分层架构:
    app        -> 应用层
    decision   -> 决策与诊断层
    normative  -> 规范层
    geometry   -> 几何推理核心层
    physics    -> 物理世界仿真层
"""

__version__ = "0.1.0"

from .types import (
    SystemState,
    ControlInput,
    ExternalInput,
    EnergyLandscapeParams,
    Trajectory,
    ControlDecision,
    DiagnosisReport,
)
from .engine import GCCMEngine

__all__ = [
    "__version__",
    "SystemState",
    "ControlInput",
    "ExternalInput",
    "EnergyLandscapeParams",
    "Trajectory",
    "ControlDecision",
    "DiagnosisReport",
    "GCCMEngine",
]
