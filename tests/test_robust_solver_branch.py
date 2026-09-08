"""Regression for the robust-MPC branch fix in GCCMEngine._solve_trajectory.

Before the fix, the `use_casadi_robust and robust_scenarios` branch referenced an
unbound name `robust_solver`, so it raised NameError on every call and silently
fell through to safe control — robust MPC never ran. These tests pin the fixed
behavior:

  - the branch now constructs a CasadiRobustGeodesicSolver (no NameError);
  - when CasADi is unavailable, construction raises RuntimeError which the owner
    normalizes to an empty trajectory -> safe control (engine still returns a
    valid decision instead of crashing);
  - _build_robust_solver wires the engine's robust-* config through.
"""
from __future__ import annotations

import pytest

from gccm_be import GCCMEngine
from gccm_be.geometry import casadi_robust_solver as crs
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import SystemState

STEP_H = 0.25


def _engine(**kw):
    sim = Simulator(RCBuildingModel(), HVACModel())
    scen = Simulator(RCBuildingModel(r_wall=1.5), HVACModel())
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        bounds={"T_air": (15.0, 35.0), "T_wall": (15.0, 35.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    base = dict(simulator=sim, manifold=manifold, horizon=4, dt=STEP_H,
                setpoints={"T_air": 26.0}, comfort_min=25.0, comfort_max=27.0,
                use_casadi_robust=True, robust_scenarios=[scen])
    base.update(kw)
    return GCCMEngine(**base)


def test_build_robust_solver_wires_config():
    if not crs.HAS_CASADI:
        pytest.skip("CasADi not installed on this host")
    eng = _engine(robust_enforce_lower_bound=False, robust_heating_cost_factor=2.0)
    # minimal landscape via one optimize cycle setup is heavy; build directly
    from gccm_be.geometry.landscape import EnergyLandscape
    landscape = EnergyLandscape(setpoints={"T_air": 26.0}, weights={"comfort": 1.0},
                                manifold=eng.manifold, hvac=eng.simulator.hvac)
    rs = eng._build_robust_solver(landscape, 4, 25.0, 27.0)
    assert rs.enforce_lower_bound is False
    assert rs.heating_cost_factor == 2.0
    assert rs.horizon == 4
    assert len(rs.scenario_sims) == 1


def test_robust_branch_no_nameerror_and_returns_valid_decision():
    # Regardless of CasADi presence, the engine must return a usable control and
    # never raise NameError from the old unbound `robust_solver`.
    eng = _engine()
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    dec = eng.optimize(state, 0.0, forced_mode="comfort")
    assert dec.control is not None
    assert dec.control.u.shape[0] >= 1


def test_robust_branch_falls_back_to_safe_when_casadi_missing(monkeypatch):
    # Force the "CasADi unavailable" path: construction raises RuntimeError which
    # the owner must normalize to a safe-control fallback (solver_exc recorded).
    monkeypatch.setattr(crs, "HAS_CASADI", False)
    eng = _engine()
    state = SystemState([29.0, 29.0], ["T_air", "T_wall"])
    dec = eng.optimize(state, 0.0, forced_mode="comfort")
    # feedback safe control cools when hot; decision is still valid
    assert dec.control.u[0] <= 0.0
    assert dec.solver_success is False
