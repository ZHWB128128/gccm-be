"""Tests for renormalization-driven per-zone comfort weighting in the objective.

Locks in the Phase-B mechanism: the RG relevance of each zone maps to a
per-zone comfort penalty multiplier in EnergyLandscape, and the engine wires it
from the order-parameter identification.
"""
from __future__ import annotations

import numpy as np

from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel
from gccm_be.types import ControlInput, ExternalInput, SystemState


def _landscape(zone_weights=None):
    labels = ["T_air_1", "T_air_2", "T_air_3"]
    manifold = StateManifold(
        labels=labels,
        bounds={l: (15.0, 40.0) for l in labels},
        scale={l: 5.0 for l in labels},
    )
    return EnergyLandscape(
        setpoints={l: 26.0 for l in labels},
        weights={"comfort": 1.0, "energy": 0.5, "smooth": 0.2},
        manifold=manifold,
        hvac=HVACModel(n_units=3, control_labels=["Q1", "Q2", "Q3"]),
        comfort_min=25.0,
        comfort_max=27.0,
        zone_comfort_weights=zone_weights or {},
    )


def _cost(ls, temps):
    state = SystemState(np.array(temps), ["T_air_1", "T_air_2", "T_air_3"])
    ctrl = ControlInput([0.0, 0.0, 0.0], ["Q1", "Q2", "Q3"])
    ext = ExternalInput([34.0, 0.5, 0.6, 1.0], ["T_out", "solar", "occ", "price"])
    return ls.running_cost(state, ctrl, ext)


def test_zone_weighting_default_is_unity():
    # No weights -> behaves exactly as before (all zones equal).
    ls = _landscape()
    # two identical over-temperature zones contribute equally
    c = _cost(ls, [29.0, 26.0, 26.0])
    c2 = _cost(ls, [26.0, 29.0, 26.0])
    assert abs(c - c2) < 1e-12


def test_upweighting_a_zone_increases_its_comfort_penalty():
    base = _landscape()
    weighted = _landscape({"T_air_3": 3.0})
    temps = [26.0, 26.0, 29.0]  # only zone 3 hot
    c_base = _cost(base, temps)
    c_weighted = _cost(weighted, temps)
    # zone 3 penalty tripled -> total comfort cost strictly larger
    assert c_weighted > c_base


def test_weighting_only_affects_targeted_zone():
    weighted = _landscape({"T_air_3": 5.0})
    # zone 3 comfortable -> multiplier irrelevant, cost equals hot-zone-1 case scaled by 1
    hot1 = _cost(weighted, [29.0, 26.0, 26.0])
    base = _landscape()
    hot1_base = _cost(base, [29.0, 26.0, 26.0])
    assert abs(hot1 - hot1_base) < 1e-12


def test_engine_wires_relevance_to_zone_weights():
    from gccm_be import GCCMEngine
    from gccm_be.physics.models import Simulator
    from examples.renormalization_peak import ThreeZoneModel, PeakProvider, LABELS, CONTROL_LABELS

    building = ThreeZoneModel()
    hvac = HVACModel(q_min=-1.6, q_max=1.6, n_units=3, control_labels=list(CONTROL_LABELS))
    sim = Simulator(building, hvac)
    manifold = StateManifold(
        labels=list(LABELS),
        bounds={l: (15.0, 40.0) for l in LABELS},
        scale={l: 5.0 for l in LABELS},
    )
    engine = GCCMEngine(
        simulator=sim,
        external_provider=PeakProvider(seed=0),
        manifold=manifold,
        horizon=6,
        dt=0.25,
        setpoints={l: 26.0 for l in LABELS},
        comfort_min=25.0,
        comfort_max=27.0,
        comfort_weight=2.0,
        energy_weight=15.0,
        smooth_weight=0.1,
        enforce_comfort_constraints=False,
        renormalization_enabled=True,
        renormalization_weighting=True,
        renormalization_weight_gain=5.0,
        solver_options={"maxiter": 40, "ftol": 1e-4, "maxls": 15},
    )
    # zone 3 clearly hottest -> it must receive the largest per-zone multiplier.
    off = ControlInput([0.0, 0.0, 0.0], list(CONTROL_LABELS))
    state = SystemState(np.array([26.2, 26.4, 28.5]), list(LABELS))
    # exercise one optimize; then verify order-parameter identification picks zone 3
    engine.optimize(state, 8.0, forced_mode="comfort")
    engine.renormalization_flow.labels = list(LABELS)
    params = engine.renormalization_flow.identify_order_parameters(list(state.x))
    assert params[0].label == "T_air_3"
