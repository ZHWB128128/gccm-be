"""Tests for Shapley-based cost attribution (Phase 1: causal attribution)."""
from __future__ import annotations

import numpy as np

from gccm_be.causal.attribution import CostAttributor, shapley_values
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import ControlInput, ExternalInput, SystemState

STEP_H = 0.25


def _weather(n, t_out, price):
    return [
        ExternalInput(np.array([t_out, 0.2, 0.5, price]), ["T_out", "solar", "occ", "price"])
        for _ in range(n)
    ]


def test_shapley_efficiency_and_symmetry():
    # A purely additive value function: contributions equal marginal values,
    # and they must sum exactly to v(full) - v(empty).
    base = {"a": 1.0, "b": 2.0, "c": 3.0}

    def value_fn(S):
        return sum(base[k] for k in S)

    phi, cache = shapley_values(["a", "b", "c"], value_fn)
    assert abs(phi["a"] - 1.0) < 1e-9
    assert abs(phi["b"] - 2.0) < 1e-9
    assert abs(phi["c"] - 3.0) < 1e-9
    total = cache[frozenset(["a", "b", "c"])] - cache[frozenset()]
    assert abs(sum(phi.values()) - total) < 1e-9


def test_shapley_handles_interaction_exactly():
    # A value function with a pairwise interaction term: naive one-at-a-time
    # attribution would leave a residual, Shapley must not.
    def value_fn(S):
        v = 0.0
        if "x" in S:
            v += 10.0
        if "y" in S:
            v += 5.0
        if "x" in S and "y" in S:
            v += 4.0  # interaction, split evenly by symmetry
        return v

    phi, cache = shapley_values(["x", "y"], value_fn)
    # interaction of 4 is split 2/2
    assert abs(phi["x"] - 12.0) < 1e-9
    assert abs(phi["y"] - 7.0) < 1e-9
    total = cache[frozenset(["x", "y"])] - cache[frozenset()]
    assert abs(sum(phi.values()) - total) < 1e-9


def _make_attributor():
    sim = Simulator(RCBuildingModel(), HVACModel(q_min=-8.0, q_max=8.0))
    init = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    n = 48

    # baseline policy: no cooling; treatment policy: constant cooling
    def baseline_policy(state, t):
        return ControlInput([0.0], ["Q_hvac"])

    def treatment_policy(state, t):
        return ControlInput([-4.0], ["Q_hvac"])

    return CostAttributor(
        simulator=sim,
        initial_state=init,
        dt=STEP_H,
        baseline_policy=baseline_policy,
        treatment_policy=treatment_policy,
        baseline_weather=_weather(n, 34.0, 1.0),
        treatment_weather=_weather(n, 30.0, 1.0),
        baseline_price=[1.0] * n,
        treatment_price=[0.5] * n,
    )


def test_attribution_additivity_residual_is_zero():
    attr = _make_attributor()
    result = attr.attribute()
    # Efficiency: contributions must reconstruct the total difference exactly.
    assert abs(result.residual) < 1e-9
    assert abs(sum(result.contributions.values()) - result.total_difference) < 1e-9
    # all three factors are present
    assert set(result.contributions) == {"policy", "weather", "price"}


def test_attribution_price_drop_reduces_cost():
    attr = _make_attributor()
    result = attr.attribute()
    # treatment halves the price -> the price factor must contribute a cost *reduction*
    assert result.contributions["price"] < 0.0
    # fractions sum to ~1
    fracs = result.contribution_fractions()
    assert abs(sum(fracs.values()) - 1.0) < 1e-6


def test_attribution_policy_cooling_adds_cost_under_equal_weather():
    # Isolate policy: identical weather & price on both sides, only policy differs.
    sim = Simulator(RCBuildingModel(), HVACModel(q_min=-8.0, q_max=8.0))
    init = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    n = 24
    w = _weather(n, 32.0, 1.0)
    attr = CostAttributor(
        simulator=sim,
        initial_state=init,
        dt=STEP_H,
        baseline_policy=lambda s, t: ControlInput([0.0], ["Q_hvac"]),
        treatment_policy=lambda s, t: ControlInput([-5.0], ["Q_hvac"]),
        baseline_weather=w,
        treatment_weather=w,
        baseline_price=[1.0] * n,
        treatment_price=[1.0] * n,
    )
    result = attr.attribute()
    # only policy differs -> policy carries the whole difference, others ~0
    assert abs(result.contributions["weather"]) < 1e-9
    assert abs(result.contributions["price"]) < 1e-9
    assert result.contributions["policy"] > 0.0  # cooling consumes electricity
