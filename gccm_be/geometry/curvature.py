"""曲率分析器：计算能量景观关于状态的二阶结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..types import ControlInput, ExternalInput, SystemState
from .landscape import EnergyLandscape


@dataclass
class CurvatureAnalysis:
    """曲率分析结果。"""

    hessian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    classification: str
    stability: float

    def as_dict(self) -> dict:
        return {
            "hessian": self.hessian.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "classification": self.classification,
            "stability": float(self.stability),
        }


class CurvatureAnalyzer:
    """基于有限差分的局部能量景观二阶结构分析。"""

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

    def analyze(
        self,
        landscape: EnergyLandscape,
        state: SystemState,
        control: ControlInput,
        external: ExternalInput,
    ) -> CurvatureAnalysis:
        n = state.dim
        hessian = np.zeros((n, n))
        eps = self.eps

        def energy(x: np.ndarray) -> float:
            return landscape.running_cost(SystemState(x, list(state.labels)), control, external)

        for i in range(n):
            for j in range(i, n):
                xp = state.x.copy()
                xm = state.x.copy()
                if i == j:
                    xp[i] += eps
                    xm[i] -= eps
                    hessian[i, i] = (energy(xp) - 2.0 * energy(state.x) + energy(xm)) / (eps * eps)
                else:
                    xpp = state.x.copy()
                    xpm = state.x.copy()
                    xmp = state.x.copy()
                    xmm = state.x.copy()
                    xpp[i] += eps; xpp[j] += eps
                    xpm[i] += eps; xpm[j] -= eps
                    xmp[i] -= eps; xmp[j] += eps
                    xmm[i] -= eps; xmm[j] -= eps
                    hessian[i, j] = hessian[j, i] = (
                        (energy(xpp) - energy(xpm) - energy(xmp) + energy(xmm)) / (4.0 * eps * eps)
                    )

        eigvals, eigvecs = np.linalg.eigh(hessian)
        if eigvals.size == 0:
            return CurvatureAnalysis(hessian, eigvals, eigvecs, "unknown", 0.0)

        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))
        if min_eig > 1e-8:
            classification = "stable"
            stability = min_eig
        elif max_eig < -1e-8:
            classification = "unstable"
            stability = max_eig
        elif min_eig < -1e-8 and max_eig > 1e-8:
            classification = "saddle"
            stability = min_eig
        else:
            classification = "flat"
            stability = 0.0

        return CurvatureAnalysis(hessian, eigvals, eigvecs, classification, stability)
