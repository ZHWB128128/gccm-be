"""Tests for the SafeController owner extracted from GCCMEngine, and the
RenormalizationFlow.zone_comfort_weights RG->MPC coupling owner."""
from __future__ import annotations

import numpy as np

from gccm_be.control import SafeController
from gccm_be.multiscale.renormalization import RenormalizationFlow
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ExternalInput, SystemState


def _controller(**kw):
    sim = Simulator(RCBuildingModel(), HVACModel(q_min=-6.0, q_max=6.0))
    return SafeController(simulator=sim, setpoints={"T_air": 26.0},
                          comfort_min=25.0, comfort_max=27.0, **kw)


def test_feedback_cools_when_hot():
    sc = _controller()
    state = SystemState([29.0, 29.0], ["T_air", "T_wall"])
    u = sc.feedback(state, 0.25, ExternalInput([34.0, 0.5, 0.6], ["T_out", "solar", "occ"]))
    assert u.u[0] < 0.0  # cooling


def test_feedback_heats_when_below_comfort_min():
    sc = _controller()
    state = SystemState([23.0, 23.0], ["T_air", "T_wall"])
    u = sc.feedback(state, 0.25)
    assert u.u[0] > 0.0  # heating


def test_feedback_idle_near_setpoint():
    sc = _controller()
    state = SystemState([26.0, 26.0], ["T_air", "T_wall"])
    u = sc.feedback(state, 0.25)
    assert abs(u.u[0]) < 1e-9


def test_worst_case_falls_back_to_feedback_on_nonrc_model():
    # non-RC building -> worst_case delegates to feedback (still returns a control)
    sc = _controller()
    state = SystemState([29.0, 29.0], ["T_air", "T_wall"])
    u = sc.worst_case(state, 0.25, ExternalInput([35.0, 0.6, 0.7], ["T_out", "solar", "occ"]))
    assert u.u[0] <= 0.0


def test_snapshot_restore_roundtrips_integral():
    sc = _controller()
    state = SystemState([29.0, 29.0], ["T_air", "T_wall"])
    sc.feedback(state, 0.25)  # build up integral
    snap = sc.snapshot()
    sc.feedback(state, 0.25)  # mutate further
    sc.restore(snap)
    assert sc._integral == snap["integral"]


def test_cool_cap_monotone_in_error():
    sc = _controller()
    assert sc.cool_cap(0.5, 10.0) <= sc.cool_cap(1.5, 10.0) <= sc.cool_cap(2.5, 10.0)


# ---- RG -> MPC coupling owner ----

def test_zone_weights_upweights_hot_relevant_zone():
    flow = RenormalizationFlow()
    labels = ["T_air_1", "T_air_2", "T_air_3"]
    # zone 3 hottest and peak-driving -> largest multiplier
    weights = flow.zone_comfort_weights([26.0, 26.2, 29.0], gain=5.0, labels=labels)
    assert weights["T_air_3"] > weights["T_air_1"]
    assert weights["T_air_3"] > 1.0


def test_zone_weights_empty_for_single_zone():
    flow = RenormalizationFlow()
    assert flow.zone_comfort_weights([26.0], gain=5.0, labels=["T_air_1"]) == {}


def test_zone_weights_gain_zero_is_unity():
    flow = RenormalizationFlow()
    labels = ["T_air_1", "T_air_2", "T_air_3"]
    weights = flow.zone_comfort_weights([26.0, 27.0, 29.0], gain=0.0, labels=labels)
    assert all(abs(v - 1.0) < 1e-12 for v in weights.values())
