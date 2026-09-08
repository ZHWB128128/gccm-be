"""Tests for the independent multi-day weather/price scenario sampler."""
from __future__ import annotations

import numpy as np

from examples.weather_scenarios import ScenarioProvider, sample_scenario


def test_sampled_scenarios_differ_across_seeds():
    s0 = sample_scenario(np.random.default_rng(0))
    s1 = sample_scenario(np.random.default_rng(1))
    # different seeds -> different physical day (at least one DOF differs materially)
    assert (s0.base_temp, s0.peak_hour, s0.price_peak_val) != (
        s1.base_temp, s1.peak_hour, s1.price_peak_val)


def test_scenario_is_reproducible_for_same_seed():
    a = sample_scenario(np.random.default_rng(7))
    b = sample_scenario(np.random.default_rng(7))
    assert a == b


def test_provider_returns_labeled_external_inputs():
    scen = sample_scenario(np.random.default_rng(3))
    prov = ScenarioProvider(scen, dt_h=0.25, seed=3)
    seq = prov.get(8.0, horizon=4)
    assert len(seq) == 4
    for w in seq:
        # labels present -> price/solar resolved by name, not position
        assert "price" in w.labels and "T_out" in w.labels
        assert w.price in (scen.price_base, scen.price_peak_val)


def test_plant_cloud_disturbance_only_when_plant_true():
    scen = sample_scenario(np.random.default_rng(5))
    # force a cloud so the disturbance is active
    scen.cloud = 0.4
    ctrl = ScenarioProvider(scen, dt_h=0.25, seed=5, plant=False)
    plant = ScenarioProvider(scen, dt_h=0.25, seed=5, plant=True)
    # at a daytime hour with sun, plant solar may be reduced by cloud; forecast is not
    c = ctrl.get(12.0, 1)[0].solar()
    p = plant.get(12.0, 1)[0].solar()
    assert p <= c + 1e-9  # cloud never increases irradiance
