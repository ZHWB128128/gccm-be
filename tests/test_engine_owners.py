"""Tests locking the behavior of the owner methods extracted from the
GCCMEngine.optimize God method: horizon/margin planning, degradation recovery,
solver-trajectory normalization, first-control selection, and diagnosis mode
application. These pin the split so future edits stay behavior-preserving."""
from __future__ import annotations

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ControlInput, SystemState, Trajectory

STEP_H = 0.25


def _engine(**kw):
    sim = Simulator(RCBuildingModel(), HVACModel())
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        bounds={"T_air": (15.0, 35.0), "T_wall": (15.0, 35.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    base = dict(simulator=sim, manifold=manifold, horizon=6, dt=STEP_H,
                setpoints={"T_air": 26.0}, comfort_min=25.0, comfort_max=27.0,
                nominal_horizon=6, min_horizon=3)
    base.update(kw)
    return GCCMEngine(**base)


def test_plan_horizon_default_no_adaptation():
    eng = _engine()
    h, cmin, cmax, mode_override = eng._plan_horizon_and_margin(
        SystemState([26.0, 26.0], ["T_air", "T_wall"]), None)
    assert h == eng.horizon
    assert cmin == 25.0 and cmax == 27.0
    assert mode_override is None


def test_plan_horizon_singular_shortens_and_overrides():
    eng = _engine(curvature_adaptive=True, tightened_comfort_max=26.5)
    eng._last_curvature_min = eng.curvature_singular_threshold - 1.0
    h, cmin, cmax, mode_override = eng._plan_horizon_and_margin(
        SystemState([26.0, 26.0], ["T_air", "T_wall"]), None)
    assert h == eng.min_horizon
    assert cmax == 26.5
    assert mode_override == "balanced"


def test_plan_horizon_consumes_geometry_margin_boost_once():
    eng = _engine()
    eng._geometry_margin_boost = 0.3
    _, cmin, cmax, _ = eng._plan_horizon_and_margin(
        SystemState([26.0, 26.0], ["T_air", "T_wall"]), None)
    # margin 0.3 tightens both bounds inward
    assert abs(cmin - 25.3) < 1e-9 and abs(cmax - 26.7) < 1e-9
    # single-shot: boost reset to 0
    assert eng._geometry_margin_boost == 0.0


def test_degradation_recovery_exits_after_streak():
    eng = _engine(min_degradation_steps=1, recovery_steps_required=1,
                  degradation_exit_error=1.0)
    eng._degraded = True
    eng._pre_degradation_mode = "comfort"
    eng._degraded_steps = 5
    eng.mode_manager.switch("balanced")
    # self_monitor default predicts near-zero error, not diverging -> recovers
    eng._attempt_degradation_recovery(SystemState([25.5, 25.5], ["T_air", "T_wall"]))
    assert eng._degraded is False
    assert eng.mode_manager.current_mode == "comfort"


def test_solve_trajectory_normalizes_exception_to_empty():
    eng = _engine()

    class _BoomSolver:
        def solve(self, *a, **k):
            raise RuntimeError("boom")

    traj, exc = eng._solve_trajectory(
        _BoomSolver(), SystemState([28.0, 28.0], ["T_air", "T_wall"]),
        [], None, eng.horizon)
    assert traj.controls == []
    assert traj.success is False
    assert isinstance(exc, RuntimeError)


def test_select_first_control_falls_back_on_failure():
    eng = _engine()
    empty = Trajectory(controls=[], states=[], success=False, message="x")
    ctrl = eng._select_first_control(
        empty, SystemState([29.0, 29.0], ["T_air", "T_wall"]), None, solver_failed=True)
    # feedback safe control cools when hot
    assert ctrl.u[0] <= 0.0


def test_select_first_control_passthrough_on_success():
    eng = _engine()
    u0 = ControlInput(np.array([-1.5]), ["Q_hvac"])
    traj = Trajectory(controls=[u0], states=[], success=True, message="ok")
    ctrl = eng._select_first_control(
        traj, SystemState([26.0, 26.0], ["T_air", "T_wall"]), None, solver_failed=False)
    assert ctrl is u0
