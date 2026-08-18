"""严格黎曼控制理论验证：CasADi 自动微分梯度 vs 数值梯度。"""
from __future__ import annotations

import casadi as ca
import numpy as np

DT = 0.25
H = 4
ALPHA = 0.1
COUPLING = 0.5


def main() -> None:
    u_sym = ca.MX.sym("u", H)
    z = ca.MX([28.0, 28.0])
    total = 0.0
    for k in range(H):
        u = u_sym[k]
        # 度量（符号）
        g00 = 0.2 * ca.exp(ALPHA * (z[0] - 26.0))
        G = ca.MX.zeros(2, 2)
        G[0, 0] = g00
        G[0, 1] = COUPLING
        G[1, 0] = COUPLING
        G[1, 1] = 1.0
        # 线性动力学
        A = ca.DM([[ -0.1, 0.05], [0.02, -0.08]])
        B = ca.MX([DT, 0.0])
        z_next = z + DT * (A @ z) + B * u
        delta = z_next - z
        dev = delta - ca.vertcat(DT * u, 0.0)
        total += 0.5 * ca.dot(dev, G @ dev) / DT**2 + 0.5 * (z[0] - 26.0)**2 + 0.1 * u**2
        z = z_next
    total += 0.5 * (z[0] - 26.0)**2

    f = ca.Function("J", [u_sym], [total])
    grad_f = ca.Function("dJ", [u_sym], [ca.gradient(total, u_sym)])

    rng = np.random.default_rng(0)
    u = rng.normal(size=H)
    grad_ad = np.array(grad_f(u)).flatten()

    eps = 1e-6
    grad_num = np.zeros(H)
    for k in range(H):
        up = u.copy(); up[k] += eps
        um = u.copy(); um[k] -= eps
        grad_num[k] = (float(f(up)) - float(f(um))) / (2 * eps)

    err = np.max(np.abs(grad_ad - grad_num))
    print(f"自动微分梯度: {grad_ad}")
    print(f"数值梯度:     {grad_num}")
    print(f"最大误差: {err:.2e}")
    print("理论验证:", "PASS" if err < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
