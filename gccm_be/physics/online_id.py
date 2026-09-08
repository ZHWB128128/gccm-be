"""Online parameter identification: recursive least squares (RLS) estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..types import ControlInput, ExternalInput, SystemState


@dataclass
class OnlineIdentifier:
    """Simple RLS identifier for online estimation of linearized model parameters.

    A general framework: inputs feature phi and observation y, updates theta.
    Can be specialized for RC models (temperature differences, control, solar, etc.).
    """

    dim: int = 3
    lam: float = 0.99
    theta: np.ndarray = None  # type: ignore[assignment]
    P: np.ndarray = None  # type: ignore[assignment]
    history: list[float] = field(default_factory=list)
    history_maxlen: int = 500

    def __post_init__(self) -> None:
        self.theta = np.zeros(self.dim)
        self.P = np.eye(self.dim) * 100.0

    def _append_history(self, err: float) -> None:
        self.history.append(float(err))
        if len(self.history) > self.history_maxlen:
            del self.history[: len(self.history) - self.history_maxlen]

    def update(self, phi: np.ndarray, y: float) -> float:
        """输入特征向量 phi 和观测 y，返回预测残差。"""
        phi = np.asarray(phi, dtype=float).reshape(-1)
        pred = float(self.theta @ phi)
        error = y - pred
        # RLS 更新
        P_phi = self.P @ phi
        denom = float(phi @ P_phi + self.lam)
        gain = P_phi / denom
        self.theta = self.theta + gain * error
        self.P = (self.P - np.outer(gain, phi @ self.P)) / self.lam
        self._append_history(error)
        return error

    def predict(self, phi: np.ndarray) -> float:
        return float(self.theta @ np.asarray(phi, dtype=float).reshape(-1))


@dataclass
class RCOnlineIdentifier:
    """Online parameter identification for single-zone RC models.

    Estimates the discretized T_air dynamics:
        dT_air/dt =
            a1*(T_wall - T_air)
            + a2*(T_out - T_air)
            + a3*solar
            + a4*occ
            + a5*Q
            + a6
    where a1=1/(R_air*C_air), a2=1/(R_wall*C_air),
          a3=solar_gain/C_air, a4=1/C_air, a5=1/C_air.

    Multi-zone: set ``zone="A"``（可 "B"…）后，状态按 ``T_air_{zone}``/``T_wall_{zone}``
    标签定位、控制取 ``unit_index`` 台设备、solar/occ 按 ``solar_{zone}``/``occ_{zone}``
    标签查（无区标签时回退通用 ``solar``/``occ``）。zone=None 保持单区位置约定。
    注：多区模式下隔墙耦合项未显式建模，其影响被 a2/a6 吸收——辨识结果是有偏估计，
    由引擎侧的置信门控 + 影子验证兜底（诚实标注）。
    """

    lam: float = 0.98
    theta: np.ndarray = None  # type: ignore[assignment]
    P: np.ndarray = None  # type: ignore[assignment]
    n: int = 6
    zone: str | None = None
    unit_index: int = 0
    history: list[float] = field(default_factory=list)
    history_maxlen: int = 500

    def __post_init__(self) -> None:
        self.theta = np.zeros(self.n)
        self.P = np.eye(self.n) * 100.0

    def _append_history(self, err: float) -> None:
        self.history.append(float(err))
        if len(self.history) > self.history_maxlen:
            del self.history[: len(self.history) - self.history_maxlen]

    def _air_index(self, state: SystemState) -> int:
        if self.zone is None:
            return 0
        return list(state.labels).index(f"T_air_{self.zone}")

    def features(self, state: SystemState, control: ControlInput, external: ExternalInput) -> np.ndarray:
        if self.zone is None:
            T_air, T_wall = state.x[0], state.x[1]
            Q = control.u[0]
            T_out = external.w[0]
            solar = external.w[1]
            occ = external.w[2]
        else:
            labels = list(state.labels)
            i_air = labels.index(f"T_air_{self.zone}")
            i_wall = labels.index(f"T_wall_{self.zone}")
            T_air, T_wall = state.x[i_air], state.x[i_wall]
            Q = float(control.u[self.unit_index]) if control.u.size > self.unit_index else 0.0
            T_out = external.get("T_out", float(external.w[0]))
            solar = external.solar(zone=self.zone)
            occ = external.occupancy(zone=self.zone)
        return np.array([
            T_wall - T_air,
            T_out - T_air,
            solar,
            occ,
            Q,
            1.0,
        ])

    def update(self, state: SystemState, control: ControlInput, external: ExternalInput,
               next_state: SystemState, dt: float = 0.25) -> float:
        phi = self.features(state, control, external)
        i_air = self._air_index(state)
        y = (next_state.x[i_air] - state.x[i_air]) / dt
        pred = float(self.theta @ phi)
        error = y - pred
        P_phi = self.P @ phi
        denom = float(phi @ P_phi + self.lam)
        gain = P_phi / denom
        self.theta = self.theta + gain * error
        self.P = (self.P - np.outer(gain, phi @ self.P)) / self.lam
        self._append_history(error)
        return error

    def parameters(self) -> dict:
        a1, a2, a3, a4, a5, a6 = self.theta
        return {
            "R_air_times_C_air": 1.0 / a1 if abs(a1) > 1e-6 else None,
            "R_wall_times_C_air": 1.0 / a2 if abs(a2) > 1e-6 else None,
            "solar_gain_over_C_air": a3,
            "one_over_C_air": a4,
            "bias": a6,
        }
