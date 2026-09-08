"""Tests for the Riemannian switch specification (four orthogonal mechanisms)."""
from __future__ import annotations

import pytest

from gccm_be.geometry.geodesic import GeodesicSolver
from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator


def _solver(**kw):
    labels = ["T_air", "T_wall"]
    manifold = StateManifold(labels=labels, bounds={l: (15.0, 40.0) for l in labels},
                             scale={l: 5.0 for l in labels})
    landscape = EnergyLandscape(
        setpoints={"T_air": 26.0},
        weights={"comfort": 1.0, "energy": 0.5, "smooth": 0.2},
        manifold=manifold,
        hvac=HVACModel(),
    )
    sim = Simulator(RCBuildingModel(), HVACModel())
    return GeodesicSolver(simulator=sim, landscape=landscape, horizon=4, dt=0.25, **kw)


def test_geodesic_penalty_requires_riemannian():
    with pytest.raises(ValueError, match="requires use_riemannian"):
        _solver(use_riemannian=False, geodesic_penalty_weight=0.5)


def test_geodesic_penalty_ok_with_riemannian():
    s = _solver(use_riemannian=True, geodesic_penalty_weight=0.5)
    assert s.geodesic_penalty_weight == 0.5


def test_negative_strength_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        _solver(riemannian_strength=-1.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        _solver(riemannian_control_weight=-0.5)


def test_default_preset_is_valid():
    # weighted-MPC baseline: all Riemannian mechanisms off
    s = _solver()
    assert s.use_riemannian is False
    assert s.geodesic_penalty_weight == 0.0


def test_smoothness_preset_valid():
    # smoothness mode: kinetic on, others off
    s = _solver(use_kinetic=True)
    assert s.use_kinetic is True
    assert s.use_riemannian is False
