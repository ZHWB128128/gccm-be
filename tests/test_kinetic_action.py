"""Tests for the Riemannian action (kinetic) term as a state-smoothness owner.

These lock in the Phase-A positive result: the term ½ ż^T g(z) ż is wired into
the objective, uses the *current* state's metric (so state-dependence actually
bites), and behaves as a smoothness regularizer.
"""
from __future__ import annotations

import numpy as np

from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel
from gccm_be.types import SystemState


def _landscape(state_dep=0.0, kinetic_weight=1.0):
    labels = ["T_air", "T_wall"]
    manifold = StateManifold(
        labels=labels,
        units={l: "°C" for l in labels},
        bounds={l: (15.0, 40.0) for l in labels},
        scale={l: 5.0 for l in labels},
    )
    return EnergyLandscape(
        setpoints={"T_air": 26.0},
        weights={"comfort": 1.0, "energy": 0.5, "smooth": 0.2},
        manifold=manifold,
        hvac=HVACModel(),
        metric_state_dependence=state_dep,
        kinetic_weight=kinetic_weight,
    )


def test_kinetic_term_zero_for_no_motion():
    ls = _landscape()
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    assert ls.kinetic_term(np.zeros(2), dt=0.25, state=state) == 0.0


def test_kinetic_term_penalizes_larger_motion():
    ls = _landscape()
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    small = ls.kinetic_term(np.array([0.1, 0.0]), dt=0.25, state=state)
    large = ls.kinetic_term(np.array([1.0, 0.0]), dt=0.25, state=state)
    assert large > small > 0.0


def test_kinetic_weight_scales_linearly():
    base = _landscape(kinetic_weight=1.0)
    scaled = _landscape(kinetic_weight=3.0)
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    dz = np.array([0.5, 0.2])
    b = base.kinetic_term(dz, dt=0.25, state=state)
    s = scaled.kinetic_term(dz, dt=0.25, state=state)
    assert abs(s - 3.0 * b) < 1e-9


def test_kinetic_term_uses_current_state_metric_when_state_dependent():
    # With state-dependence on, the metric (hence the kinetic term) must differ
    # between a state at setpoint and one far from it. This is the bug the
    # Phase-A fix addressed: previously a zero-point dummy collapsed the metric.
    ls = _landscape(state_dep=0.4)
    dz = np.array([0.5, 0.0])
    at_setpoint = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    far = SystemState([31.0, 31.0], ["T_air", "T_wall"])
    k_setpoint = ls.kinetic_term(dz, dt=0.25, state=at_setpoint)
    k_far = ls.kinetic_term(dz, dt=0.25, state=far)
    assert abs(k_far - k_setpoint) > 1e-6
    # metric grows as exp(state_dep * dev) above setpoint -> far term is larger
    assert k_far > k_setpoint


def test_default_state_falls_back_without_error():
    # Backwards compatibility: kinetic_term still callable without a state arg.
    ls = _landscape()
    val = ls.kinetic_term(np.array([0.3, 0.1]), dt=0.25)
    assert val > 0.0
