"""数据驱动 SCM 量化验证：估计因果效应 vs 真实物理 SCM。"""
from __future__ import annotations

import numpy as np

from gccm_be.causal.data_driven import DataDrivenSCM
from gccm_be.causal.scm import build_rc_scm
from gccm_be.physics.models import RCBuildingModel, HVACModel


def main() -> None:
    true_scm = build_rc_scm(RCBuildingModel(), HVACModel())
    true_effect = true_scm.effect("temperature", {"mode": 0.0}, {"mode": 1.0})

    rng = np.random.default_rng(0)
    data = []
    for _ in range(5000):
        mode = float(rng.integers(0, 2))
        # 从真实 SCM 采样
        sample = true_scm.do({"mode": mode})
        data.append(sample)

    fitted = DataDrivenSCM().fit(data)
    est_scm = fitted.to_scm()
    est_effect = est_scm.effect("temperature", {"mode": 0.0}, {"mode": 1.0})

    print(f"真实因果效应: {true_effect:.4f}")
    print(f"估计因果效应: {est_effect:.4f}")
    print(f"绝对误差: {abs(true_effect - est_effect):.4f}")
    print(f"相对误差: {abs(true_effect - est_effect) / abs(true_effect) * 100:.2f}%")


if __name__ == "__main__":
    main()
