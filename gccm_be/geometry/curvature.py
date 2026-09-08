"""Curvature analyzer: second-order structure of the energy landscape w.r.t. state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import ControlInput, ExternalInput, SystemState
from .landscape import EnergyLandscape
from .riemannian import christoffel_symbols


@dataclass
class CurvatureAnalysis:
    """Curvature analysis result."""

    hessian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    classification: str
    stability: float
    covariant: bool = False

    def as_dict(self) -> dict:
        return {
            "hessian": self.hessian.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "classification": self.classification,
            "stability": float(self.stability),
            "covariant": self.covariant,
        }


class CurvatureAnalyzer:
    """Local second-order structure analysis of the energy landscape.

    Two modes:
      - Euclidean (default, `covariant=False`): the plain Hessian ∂²E of the
        running cost w.r.t. state. This is the coordinate second derivative and
        depends on the chosen coordinates.
      - Covariant (`covariant=True`): the Riemannian covariant Hessian
        (∇²E)_ij = ∂_i∂_j E − Γ^k_ij ∂_k E, using the energy landscape's metric
        g(z) to supply the Christoffel connection Γ. This is the geometrically
        invariant second-order structure of E on the manifold (g, ∇). It is the
        object that "曲率分析基于真实几何" actually requires — distinct from the
        cost Hessian, and distinct again from the metric's own Riemann curvature.
    """

    def __init__(self, eps: float = 1e-3, covariant: bool = False) -> None:
        self.eps = eps
        self.covariant = covariant

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

        if self.covariant:
            # Covariant Hessian: (∇²E)_ij = ∂_i∂_j E − Γ^k_ij ∂_k E.
            # ∂_k E via central differences; Γ from the landscape metric.
            grad = np.zeros(n)
            for k in range(n):
                xp = state.x.copy(); xp[k] += eps
                xm = state.x.copy(); xm[k] -= eps
                grad[k] = (energy(xp) - energy(xm)) / (2.0 * eps)
            try:
                Gamma = christoffel_symbols(landscape.metric, state)
                correction = np.einsum("kij,k->ij", Gamma, grad)
                hessian = hessian - correction
                # symmetrize against numerical asymmetry
                hessian = 0.5 * (hessian + hessian.T)
            except Exception:
                pass  # fall back to Euclidean Hessian if metric is degenerate

        eigvals, eigvecs = np.linalg.eigh(hessian)
        if eigvals.size == 0:
            return CurvatureAnalysis(hessian, eigvals, eigvecs, "unknown", 0.0, self.covariant)

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

        return CurvatureAnalysis(hessian, eigvals, eigvecs, classification, stability, self.covariant)
