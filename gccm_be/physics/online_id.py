"""在线参数辨识：递归最小二乘（RLS）估计模型参数。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class OnlineIdentifier:
    """简单 RLS 辨识器，用于在线估计线性化模型参数。

    当前作为通用框架：输入特征 phi 和观测 y，更新 theta。
    后续可针对 RC 模型构造具体特征（温差、控制量、太阳辐射等）。
    """

    dim: int = 3
    lam: float = 0.99
    theta: np.ndarray = None  # type: ignore[assignment]
    P: np.ndarray = None  # type: ignore[assignment]
    history: List[float] = field(default_factory=list)
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
    """面向单区域 RC 模型的在线参数辨识。

    估计离散化后的 T_air 动态：
        dT_air/dt =
            a1*(T_wall - T_air)
            + a2*(T_out - T_air)
            + a3*solar
            + a4*occ
            + a5*Q
            + a6
    其中 a1=1/(R_air*C_air), a2=1/(R_wall*C_air),
          a3=solar_gain/C_air, a4=1/C_air, a5=1/C_air。
    """

    lam: float = 0.98
    theta: np.ndarray = None  # type: ignore[assignment]
    P: np.ndarray = None  # type: ignore[assignment]
    n: int = 6
    history: List[float] = field(default_factory=list)
    history_maxlen: int = 500

    def __post_init__(self) -> None:
        self.theta = np.zeros(self.n)
        self.P = np.eye(self.n) * 100.0

    def _append_history(self, err: float) -> None:
        self.history.append(float(err))
        if len(self.history) > self.history_maxlen:
            del self.history[: len(self.history) - self.history_maxlen]

    def features(self, state: SystemState, control: ControlInput, external: ExternalInput) -> np.ndarray:
        T_air, T_wall = state.x[0], state.x[1]
        Q = control.u[0]
        T_out = external.w[0]
        solar = external.w[1]
        occ = external.w[2]
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
        y = (next_state.x[0] - state.x[0]) / dt
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
