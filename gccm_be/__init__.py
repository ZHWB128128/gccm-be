"""GCCM-BE: Geometry-based Causal Control Model for Building Energy.

Layered architecture:
    app        -> application layer
    decision   -> decision & diagnosis layer
    normative  -> normative layer
    geometry   -> geometric reasoning core
    physics    -> physical world simulation layer
"""

__version__ = "0.1.0"

from .engine import GCCMEngine
from .types import (
    ControlDecision,
    ControlInput,
    DiagnosisReport,
    EnergyLandscapeParams,
    ExternalInput,
    SystemState,
    Trajectory,
)

__all__ = [
    "ControlDecision",
    "ControlInput",
    "DiagnosisReport",
    "EnergyLandscapeParams",
    "ExternalInput",
    "GCCMEngine",
    "SystemState",
    "Trajectory",
    "__version__",
]
