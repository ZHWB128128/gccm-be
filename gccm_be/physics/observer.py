"""State observer: EKF for estimating unmeasurable RC states from T_air measurements.

P0 修正（评审意见）：全部历史闭环都假设全状态可测（T_wall/T_slab/T_storage 直接
从仿真 state 读取），但真实建筑只有室温传感器。缺观测器时：①RC 模型缺 T_wall
真实初值，一步预测即系统性偏移；②在线辨识的特征 (T_wall − T_air) 无从计算。

本模块提供扩展 Kalman 滤波（EKF）：

- **通用性**：任何实现了 `step(state, control, external, dt)` 与 `state_labels`
  的物理模型（registry 协议）均可估计；雅可比用中心差分数值线性化，无需逐模型
  手写解析雅可比。
- **观测模型**：温度类状态（T_air* 前缀，及数据中心 T_aisle）视为可测；其余
  （墙/蓄热层/隔墙/罐）为隐状态。观测噪声由传感器精度决定（默认 σ=0.3K）。
- **过程噪声**：按状态类型区分——温度类小（模型可信），蓄热类更小（大热容
  慢漂移），可在构造时覆盖。

用法（试点闭环）：
    obs = EKFStateObserver(building, hvac)
    obs.initialize(t_air_measured)          # 用首测室温初始化全状态
    x_hat = obs.update(t_air_measured, control, external, dt)   # 每步
    state = x_hat                            # 喂给引擎 optimize
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import ControlInput, ExternalInput, SystemState
from .models import Simulator

# 可测（室温类）状态的前缀/名称
_MEASURABLE_PREFIXES = ("T_air",)
_MEASURABLE_NAMES = ("T_aisle",)


def _is_measurable(label: str) -> bool:
    return (any(label.startswith(p) for p in _MEASURABLE_PREFIXES)
            or label in _MEASURABLE_NAMES)


@dataclass
class EKFStateObserver:
    """Extended Kalman filter over a registry-protocol building model."""

    simulator: Simulator
    # 传感器噪声 σ (K)：观测方程 R = σ²
    measurement_sigma: float = 0.3
    # 过程噪声 σ (K/√step)：温度状态；蓄热/罐类默认更小
    process_sigma_temp: float = 0.05
    process_sigma_slow: float = 0.02
    q_max_clamp: float = 8.0    # 反演控制功率时的界
    _x_hat: np.ndarray | None = None
    _P: np.ndarray | None = None
    _labels: list[str] = field(default_factory=list, init=False)
    _meas_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False)

    # ---- 构造与初始化 ----
    def __post_init__(self) -> None:
        self._labels = list(self.simulator.building.state_labels)
        self._meas_idx = np.array(
            [i for i, lab in enumerate(self._labels) if _is_measurable(lab)], dtype=int)

    def _slow_mask(self) -> np.ndarray:
        """慢状态（蓄热/罐/墙类）掩码：过程噪声取小值。"""
        return np.array([not _is_measurable(lab) for lab in self._labels])

    def initialize(self, t_air_measured: float, ambient: float = 26.0) -> SystemState:
        """用首测室温初始化：可测状态 = 实测值，隐状态 = ambient（弱先验，靠 EKF 收敛）。"""
        n = len(self._labels)
        x0 = np.full(n, float(ambient))
        for i in self._meas_idx:
            x0[i] = float(t_air_measured)
        self._x_hat = x0
        # 初始协方差：可测状态小（信任传感器），隐状态大（不知道）
        self._P = np.diag([0.1 if i in set(self._meas_idx) else 4.0 for i in range(n)])
        return self.state()

    def state(self) -> SystemState:
        if self._x_hat is None:
            raise RuntimeError("EKFStateObserver not initialized — call initialize() first")
        return SystemState(self._x_hat.copy(), list(self._labels))

    # ---- 数值雅可比 ----
    def _f_jacobian(self, x: np.ndarray, control: ControlInput,
                    external: ExternalInput, dt: float) -> np.ndarray:
        """F = ∂f/∂x（中心差分）。"""
        n = x.size
        F = np.zeros((n, n))
        eps = 1e-4
        for j in range(n):
            xp, xm = x.copy(), x.copy()
            xp[j] += eps
            xm[j] -= eps
            fp = self.simulator.step(SystemState(xp, list(self._labels)),
                                     control, external, dt).x
            fm = self.simulator.step(SystemState(xm, list(self._labels)),
                                     control, external, dt).x
            F[:, j] = (fp - fm) / (2 * eps)
        return F

    # ---- 每步更新 ----
    def update(self, measurements: dict, control: ControlInput,
               external: ExternalInput, dt: float) -> SystemState:
        """一步 EKF：predict（模型）→ update（室温观测）。

        measurements: {可测状态标签: 实测值}，如 {"T_air": 24.5} 或
        {"T_air_A": 24.5, "T_air_B": 23.8}。返回估计的全状态。
        """
        if self._x_hat is None:
            raise RuntimeError("EKFStateObserver not initialized — call initialize() first")
        x = self._x_hat

        # --- predict ---
        F = self._f_jacobian(x, control, external, dt)
        x_pred = self.simulator.step(SystemState(x, list(self._labels)),
                                     control, external, dt).x
        n = x.size
        Q = np.diag([self.process_sigma_slow if slow else self.process_sigma_temp
                     for slow in self._slow_mask()])
        P_pred = F @ self._P @ F.T + Q

        # --- update（观测 = 提供的室温测量） ---
        used = [(i, float(measurements[lab]))
                for i, lab in enumerate(self._labels)
                if _is_measurable(lab) and lab in measurements]
        if used:
            idx = np.array([i for i, _ in used], dtype=int)
            z = np.array([v for _, v in used])
            H = np.zeros((idx.size, n))
            H[np.arange(idx.size), idx] = 1.0
            R = np.full((idx.size, idx.size), self.measurement_sigma ** 2)
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y
            self._P = (np.eye(n) - K @ H) @ P_pred
        else:
            x = x_pred
            self._P = P_pred

        self._x_hat = x
        return self.state()
