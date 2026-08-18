"""严格黎曼测地线基础演示：Christoffel 联络 + 测地线积分。"""
from __future__ import annotations

import numpy as np

from gccm_be.geometry.riemannian import christoffel_symbols, geodesic_step
from gccm_be.types import SystemState


def poincare_metric(state: SystemState) -> np.ndarray:
    """二维 Poincare 半平面度量：g = diag(1/y^2, 1/y^2)。"""
    y = max(state.x[1], 0.1)
    return np.diag([1.0 / y**2, 1.0 / y**2])


def main() -> None:
    state = SystemState(np.array([0.0, 1.0]), ["x", "y"])
    Gamma = christoffel_symbols(poincare_metric, state)
    print("Christoffel 联络 Γ^k_ij (at [0,1]):")
    print(np.round(Gamma, 3))

    # 沿 x 方向给初速度，积分几步测地线
    v = np.array([1.0, 0.0])
    s = state
    print("\n测地线轨迹:")
    for i in range(5):
        s, v = geodesic_step(s, v, poincare_metric, dt=0.05)
        print(f"  step {i+1}: z=({s.x[0]:.3f}, {s.x[1]:.3f}), v=({v[0]:.3f}, {v[1]:.3f})")


if __name__ == "__main__":
    main()
