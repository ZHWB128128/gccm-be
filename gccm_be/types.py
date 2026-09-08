"""Core data type definitions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SystemState:
    """System state vector, e.g. [indoor temp, wall temp]."""

    x: np.ndarray
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        if not self.labels:
            self.labels = [f"x{i}" for i in range(self.x.size)]

    @property
    def dim(self) -> int:
        return self.x.size

    def copy(self) -> SystemState:
        return SystemState(self.x.copy(), list(self.labels))

    def __repr__(self) -> str:
        parts = ", ".join(f"{label}={value:.4g}" for label, value in zip(self.labels, self.x))
        return f"SystemState({parts})"


@dataclass
class ControlInput:
    """Control vector, e.g. [HVAC heating/cooling power]."""

    u: np.ndarray
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=float)
        if not self.labels:
            self.labels = [f"u{i}" for i in range(self.u.size)]

    @property
    def dim(self) -> int:
        return self.u.size

    def copy(self) -> ControlInput:
        return ControlInput(self.u.copy(), list(self.labels))

    def __repr__(self) -> str:
        parts = ", ".join(f"{label}={value:.4g}" for label, value in zip(self.labels, self.u))
        return f"ControlInput({parts})"


@dataclass
class ExternalInput:
    """External input vector, e.g. [outdoor temp, solar, internal heat, price].

    Access convention
    ------------------
    Positional indexing (``w[3]``/``w[5]``) is fragile because single-zone and
    multi-zone layouts differ. Prefer the **named accessors** below, which look
    values up by label and never depend on array position:

        ext.price, ext.t_out, ext.solar(zone=None), ext.occupancy(zone=None)
        ext.get("solar_A"), ext.require("price")
    """

    w: np.ndarray
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=float)
        if not self.labels:
            self.labels = [f"w{i}" for i in range(self.w.size)]
        # index for O(1), position-independent label lookup
        self._index = {lab: i for i, lab in enumerate(self.labels)}

    @property
    def dim(self) -> int:
        return self.w.size

    def get(self, label: str, default: float | None = None) -> float | None:
        """Look up a value by label (safer than positional indexing)."""
        idx = self._index.get(label) if hasattr(self, "_index") else None
        if idx is None:
            # fall back to a linear scan if the cached index is unavailable
            for i, lab in enumerate(self.labels):
                if lab == label:
                    return float(self.w[i])
            return default
        return float(self.w[idx])

    def require(self, label: str) -> float:
        """Look up a value by label, raising if the label is absent (no silent guess)."""
        val = self.get(label)
        if val is None:
            raise KeyError(f"ExternalInput has no label {label!r}; available: {self.labels}")
        return val

    @property
    def t_out(self) -> float:
        """Outdoor temperature: prefer 'T_out' label, fall back to first element."""
        v = self.get("T_out")
        return float(v) if v is not None else float(self.w[0]) if self.w.size else 0.0

    def solar(self, zone: str | None = None) -> float:
        """Solar gain by zone label ('solar_A'/'solar_B') or generic 'solar'."""
        if zone is not None:
            v = self.get(f"solar_{zone}")
            if v is not None:
                return float(v)
        v = self.get("solar")
        return float(v) if v is not None else 0.0

    def occupancy(self, zone: str | None = None) -> float:
        """Occupancy/internal-gain by zone label ('occ_A'/'occ_B') or generic 'occ'."""
        if zone is not None:
            v = self.get(f"occ_{zone}")
            if v is not None:
                return float(v)
        v = self.get("occ")
        return float(v) if v is not None else 0.0

    @property
    def price(self) -> float:
        """Electricity price by label; last resort falls back to legacy position.

        The positional fallback (w[5] two-zone, w[3] single-zone) is retained
        only for legacy arrays built without a 'price' label; new code should
        always label the price channel so this branch is never taken.
        """
        p = self.get("price")
        if p is not None:
            return float(p)
        if self.w.size > 5:
            return float(self.w[5])
        if self.w.size > 3:
            return float(self.w[3])
        return 1.0

    def copy(self) -> ExternalInput:
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

    controls: list[ControlInput]
    states: list[SystemState]
    costs: list[float] = field(default_factory=list)
    total_cost: float = 0.0
    success: bool = True
    message: str = ""


@dataclass
class DiagnosisReport:
    """Decision & diagnosis layer output."""

    confidence: float = 1.0
    should_switch_mode: bool = False
    suggested_mode: str | None = None
    undecidable: bool = False
    triggers: list[str] = field(default_factory=list)
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
    predicted_next_state: SystemState | None = None
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
