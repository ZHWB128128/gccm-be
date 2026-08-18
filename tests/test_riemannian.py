"""Christoffel 联络三版实现一致性测试（数值/解析/自动微分）。"""
import numpy as np
import pytest

from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.geometry.riemannian import (
    christoffel_symbols,
    christoffel_symbols_analytic,
    christoffel_symbols_autodiff,
)
from gccm_be.physics.models import HVACModel
from gccm_be.types import SystemState


def make_landscape(coupling=0.5, alpha=0.2, labels=("T_air", "T_wall")):
    manifold = StateManifold(
        labels=list(labels),
        units={l: "°C" for l in labels},
        bounds={l: (15, 40) for l in labels},
        scale={l: 5.0 for l in labels},
    )
    return EnergyLandscape(
        setpoints={"T_air": 26.0},
        weights={"comfort": 1.2, "energy": 0.5, "smooth": 0.2},
        manifold=manifold,
        hvac=HVACModel(),
        metric_coupling=coupling,
        metric_state_dependence=alpha,
    )


def christoffel_reference(metric_fn, state, eps=1e-5):
    """按定义暴力实现：Γ^k_ij = ½ g^{kl}(∂_j g_li + ∂_i g_lj − ∂_l g_ij)。"""
    n = state.dim
    g = metric_fn(state)
    g_inv = np.linalg.inv(g)
    Gamma = np.zeros((n, n, n))

    def g_at(x):
        return metric_fn(SystemState(x, list(state.labels)))

    for k in range(n):
        for i in range(n):
            for j in range(n):
                val = 0.0
                for l in range(n):
                    xp = state.x.copy(); xp[j] += eps
                    xm = state.x.copy(); xm[j] -= eps
                    dj_gl = (g_at(xp)[l, i] - g_at(xm)[l, i]) / (2 * eps)
                    xp = state.x.copy(); xp[i] += eps
                    xm = state.x.copy(); xm[i] -= eps
                    di_gl = (g_at(xp)[l, j] - g_at(xm)[l, j]) / (2 * eps)
                    xp = state.x.copy(); xp[l] += eps
                    xm = state.x.copy(); xm[l] -= eps
                    dl_g = (g_at(xp)[i, j] - g_at(xm)[i, j]) / (2 * eps)
                    val += 0.5 * g_inv[k, l] * (dj_gl + di_gl - dl_g)
                Gamma[k, i, j] = val
    return Gamma


def test_numeric_christoffel_matches_reference_with_coupling():
    """数值版（修复后）：coupling≠0 + 状态相关度量时与暴力参考一致。"""
    ls = make_landscape(coupling=0.5, alpha=0.2)
    state = SystemState([27.0, 28.0], list(ls.manifold.labels))
    ref = christoffel_reference(ls.metric, state)
    num = christoffel_symbols(ls.metric, state, eps=1e-4)
    assert np.max(np.abs(num - ref)) < 1e-3, f"数值版偏差 {np.max(np.abs(num - ref))}"


def test_analytic_christoffel_matches_reference():
    """解析版（纯对角度量）：含非 setpoints 维度（T_wall）时与暴力参考一致。"""
    ls = make_landscape(coupling=0.0, alpha=0.2)
    state = SystemState([27.0, 28.0], list(ls.manifold.labels))
    ref = christoffel_reference(ls.metric, state)
    ana = christoffel_symbols_analytic(ls, state)
    assert np.max(np.abs(ana - ref)) < 1e-4, f"解析版偏差 {np.max(np.abs(ana - ref))}"
    # 非 setpoints 维度（T_wall）相关项必须为 0：旧 bug 给出 Γ[1,1,1]=0.1
    assert abs(ana[1, 1, 1]) < 1e-12, f"T_wall 维度不应有联络：{ana[1, 1, 1]}"


def test_analytic_with_coupling_delegates_to_numeric():
    """含非对角耦合时解析版委托数值版，两者一致且都匹配参考。"""
    ls = make_landscape(coupling=0.5, alpha=0.2)
    state = SystemState([27.0, 28.0], list(ls.manifold.labels))
    ref = christoffel_reference(ls.metric, state)
    ana = christoffel_symbols_analytic(ls, state)
    num = christoffel_symbols(ls.metric, state, eps=1e-4)
    assert np.max(np.abs(ana - num)) < 1e-9
    assert np.max(np.abs(num - ref)) < 1e-3, f"数值版偏差 {np.max(np.abs(num - ref))}"


def test_analytic_vs_numeric_agree():
    """解析版与数值版互相一致（非对角度量下）。"""
    ls = make_landscape(coupling=0.8, alpha=0.3)
    state = SystemState([26.5, 27.5], list(ls.manifold.labels))
    ana = christoffel_symbols_analytic(ls, state)
    num = christoffel_symbols(ls.metric, state, eps=1e-4)
    assert np.max(np.abs(ana - num)) < 1e-3


def test_flat_metric_zero_christoffel():
    """平坦度量（α=0, coupling=0）下联络恒为 0。"""
    ls = make_landscape(coupling=0.0, alpha=0.0)
    state = SystemState([27.0, 28.0], list(ls.manifold.labels))
    assert np.all(christoffel_symbols_analytic(ls, state) == 0.0)
    assert np.max(np.abs(christoffel_symbols(ls.metric, state))) < 1e-8


def test_autodiff_christoffel_matches_analytic():
    """AD 版与解析版一致（需 casadi）。"""
    pytest.importorskip("casadi")
    ls = make_landscape(coupling=0.5, alpha=0.2)
    state = SystemState([27.0, 28.0], list(ls.manifold.labels))
    ad = christoffel_symbols_autodiff(lambda s: ls.metric_casadi(s), state)
    ana = christoffel_symbols_analytic(ls, state)
    assert np.max(np.abs(ad - ana)) < 1e-6
