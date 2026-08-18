"""GCCM-BE: Geometry-based Causal Control Model for Building Energy.

Layered architecture:
    app        -> application layer
    decision   -> decision & diagnosis layer
    normative  -> normative layer
    geometry   -> geometric reasoning core
    physics    -> physical world simulation layer
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
