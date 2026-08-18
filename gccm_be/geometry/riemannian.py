"""严格黎曼几何基础：Christoffel 联络与测地线方程数值积分。"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np

from ..types import SystemState


def christoffel_symbols(
    metric_fn: Callable[[SystemState], np.ndarray],
    state: SystemState,
    eps: float = 1e-4,
) -> np.ndarray:
    """数值计算 Christoffel 联络 Γ^k_ij = ½ g^{kl}(∂_j g_li + ∂_i g_lj − ∂_l g_ij)。

    实现：先对每个方向 a 求度量导数张量 ∂_a g_bc（2n+1 次度量求值），
    再按公式组装——原实现按 (k,i,j) 三重循环内逐 l 扰动，需 O(6n³) 次度量求值。
    """
    n = state.dim
    g = metric_fn(state)
    g_inv = np.linalg.inv(g)
    Gamma = np.zeros((n, n, n))

    def g_at(x: np.ndarray) -> np.ndarray:
        return metric_fn(SystemState(x, list(state.labels)))

    # ∂_a g_bc：每个方向一次中心差分（O(n) 次度量求值）
    dg = np.zeros((n, n, n))
    for a in range(n):
        xp = state.x.copy(); xp[a] += eps
        xm = state.x.copy(); xm[a] -= eps
        dg[a] = (g_at(xp) - g_at(xm)) / (2 * eps)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                val = 0.0
                for l in range(n):
                    val += 0.5 * g_inv[k, l] * (dg[j, l, i] + dg[i, l, j] - dg[l, i, j])
                Gamma[k, i, j] = val
    return Gamma


def geodesic_step(
    state: SystemState,
    velocity: np.ndarray,
    metric_fn: Callable[[SystemState], np.ndarray],
    dt: float = 0.01,
) -> tuple[SystemState, np.ndarray]:
    """欧拉积分测地线方程：dz = v dt, dv^k = -Γ^k_ij v^i v^j dt。"""
    Gamma = christoffel_symbols(metric_fn, state)
    n = state.dim
    acc = np.zeros(n)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                acc[k] += -Gamma[k, i, j] * velocity[i] * velocity[j]
    new_v = velocity + acc * dt
    new_x = state.x + new_v * dt
    return SystemState(new_x, list(state.labels)), new_v


def christoffel_symbols_autodiff(
    metric_fn,
    state: SystemState,
) -> np.ndarray:
    """用 CasADi 自动微分计算 Christoffel 联络（更严格）。

    注意：度量若含非光滑元素（如正定性钳制 fmin/fmax），符号逆 ca.inv 求值会失败，
    因此 g^{kl} 在固定点用数值逆，雅可比保持符号 AD。
    """
    try:
        import casadi as ca
    except Exception:
        raise RuntimeError("CasADi required for autodiff Christoffel")

    n = state.dim
    z = ca.MX.sym("z", n)
    sym_state = SimpleNamespace(x=z, labels=list(state.labels), dim=n)
    G = metric_fn(sym_state)
    # G is numpy array of MX? Ensure casadi matrix
    G = ca.blockcat([[G[i, j] for j in range(n)] for i in range(n)])
    z0 = state.x
    G_fn = ca.Function("G_eval", [z], [G])
    g_inv = np.linalg.inv(np.array(G_fn(z0)))
    Gamma = np.zeros((n, n, n))
    for k in range(n):
        for i in range(n):
            for j in range(n):
                expr = 0.0
                for l in range(n):
                    # 1/2 g^{kl} (∂_j g_{li} + ∂_i g_{lj} - ∂_l g_{ij})
                    term = 0.5 * g_inv[k, l] * (
                        ca.jacobian(G[l, i], z)[0, j]
                        + ca.jacobian(G[l, j], z)[0, i]
                        - ca.jacobian(G[i, j], z)[0, l]
                    )
                    expr += term
                f = ca.Function("Gamma", [z], [expr])
                Gamma[k, i, j] = float(f(z0))
    return Gamma


def christoffel_symbols_analytic(landscape, state: SystemState) -> np.ndarray:
    """解析计算当前 EnergyLandscape 度量的 Christoffel 联络。

    纯对角度量（coupling=0）时给出精确闭式解：
        g_ii = base_i * exp(α*(z_i - setpoint_i))（仅 setpoints 内维度状态相关）
        Γ^k_ii = ½ g^{ki} ∂_i g_ii = ½ g^{ki} α g_ii，其余为 0
    含非对角耦合时（可能被正定性钳制为状态相关），委托数值版（O(n) 度量求值）。
    """
    n = state.dim
    alpha = getattr(landscape, "metric_state_dependence", 0.0)
    coupling = getattr(landscape, "metric_coupling", 0.0)
    if alpha == 0.0:
        return np.zeros((n, n, n))  # 平坦度量：联络恒为 0
    if coupling != 0.0:
        # 非对角度量的解析导数含钳制分支，公式繁琐且易错——数值版更可靠
        return christoffel_symbols(landscape.metric, state)

    g = landscape.metric(state)
    g_inv = np.linalg.inv(g)
    Gamma = np.zeros((n, n, n))
    setpoints = getattr(landscape, "setpoints", None) or {}
    labels = list(landscape.manifold.labels)
    for k in range(n):
        for i in range(n):
            if labels[i] in setpoints:
                Gamma[k, i, i] = 0.5 * g_inv[k, i] * alpha * g[i, i]
    return Gamma
