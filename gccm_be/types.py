"""Core data type definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class SystemState:
    """System state vector, e.g. [indoor temp, wall temp]."""

    x: np.ndarray
    labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        if not self.labels:
            self.labels = [f"x{i}" for i in range(self.x.size)]

    @property
    def dim(self) -> int:
        return self.x.size

    def copy(self) -> "SystemState":
        return SystemState(self.x.copy(), list(self.labels))

    def __repr__(self) -> str:
        parts = ", ".join(f"{label}={value:.4g}" for label, value in zip(self.labels, self.x))
        return f"SystemState({parts})"


@dataclass
class ControlInput:
    """Control vector, e.g. [HVAC heating/cooling power]."""

    u: np.ndarray
    labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=float)
        if not self.labels:
            self.labels = [f"u{i}" for i in range(self.u.size)]

    @property
    def dim(self) -> int:
        return self.u.size

    def copy(self) -> "ControlInput":
        return ControlInput(self.u.copy(), list(self.labels))

    def __repr__(self) -> str:
        parts = ", ".join(f"{label}={value:.4g}" for label, value in zip(self.labels, self.u))
        return f"ControlInput({parts})"


@dataclass
class ExternalInput:
    """External input vector, e.g. [outdoor temp, solar, internal heat, price]."""

    w: np.ndarray
    labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=float)
        if not self.labels:
            self.labels = [f"w{i}" for i in range(self.w.size)]

    @property
    def dim(self) -> int:
        return self.w.size

    def get(self, label: str, default: Optional[float] = None) -> Optional[float]:
        """Look up a value by label (safer than positional indexing in multi-zone layouts)."""
        for i, lab in enumerate(self.labels):
            if lab == label:
                return float(self.w[i])
        return default

    @property
    def price(self) -> float:
        """Electricity price: prefer the 'price' label; fall back to convention (w[3] single-zone, w[5] two-zone)."""
        p = self.get("price")
        if p is not None:
            return float(p)
        if self.w.size > 5:
            return float(self.w[5])
        if self.w.size > 3:
            return float(self.w[3])
        return 1.0

    def copy(self) -> "ExternalInput":
        return ExternalInput(self.w.copy(), list(self.labels))

    def __repr__(self) -> str:
        parts = ", ".join(f"{label}={value:.4g}" for label, value in zip(self.labels, self.w))
        return f"ExternalInput({parts})"


@dataclass
class EnergyLandscapeParams:
    """Energy landscape parameters, produced by the normative layer."""

    weights: dict[str, float] = field(default_factory=dict)
    setpoints: dict[str, float] = field(default_factory=dict)
    mode: str = "balanced"
    metadata: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Geodesic solver result: optimal control sequence, predicted state trajectory and costs."""

    controls: List[ControlInput]
    states: List[SystemState]
    costs: List[float] = field(default_factory=list)
    total_cost: float = 0.0
    success: bool = True
    message: str = ""


@dataclass
class DiagnosisReport:
    """Decision & diagnosis layer output."""

    confidence: float = 1.0
    should_switch_mode: bool = False
    suggested_mode: Optional[str] = None
    undecidable: bool = False
    triggers: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class ControlDecision:
    """Full output of one engine rolling-horizon optimization."""

    control: ControlInput
    trajectory: Trajectory
    mode: str
    confidence: float
    diagnosis: DiagnosisReport
    weights: dict[str, float] = field(default_factory=dict)
    predicted_next_state: Optional[SystemState] = None
    solver_success: bool = True

    def as_dict(self) -> dict:
        return {
            "control": {label: float(v) for label, v in zip(self.control.labels, self.control.u)},
            "mode": self.mode,
            "confidence": self.confidence,
            "undecidable": self.diagnosis.undecidable,
            "predicted_next_state": {
                label: float(v) for label, v in zip(self.predicted_next_state.labels, self.predicted_next_state.x)
            }
            if self.predicted_next_state is not None else None,
            "total_cost": float(self.trajectory.total_cost),
            "solver_success": self.solver_success,
            "diagnosis": {
                "should_switch_mode": self.diagnosis.should_switch_mode,
                "suggested_mode": self.diagnosis.suggested_mode,
                "triggers": list(self.diagnosis.triggers),
                "details": self.diagnosis.details,
            },
        }
