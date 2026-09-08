"""Tests for renormalization: order-parameter identification and tiered action."""
from __future__ import annotations

import numpy as np

from gccm_be.multiscale.renormalization import RenormalizationFlow, coarse_grain


def test_coarse_grain_basic():
    m = coarse_grain([25.0, 27.0, 29.0])
    assert abs(m["building_mean"] - 27.0) < 1e-9
    assert abs(m["building_max"] - 29.0) < 1e-9
    assert m["n_rooms"] == 3


def test_order_parameter_picks_hot_peak_driving_zone():
    # zone 2 is both the hottest and (via soft-max) the peak driver
    flow = RenormalizationFlow(micro_threshold=27.0, macro_observable="max",
                               labels=["A", "B", "C"])
    params = flow.identify_order_parameters([25.5, 26.0, 31.0])
    assert params[0].label == "C"          # dominant order parameter
    assert params[0].index == 2
    # dominant zone must carry the largest relevance
    assert params[0].relevance >= params[1].relevance >= params[2].relevance
    # its anomaly (above 27) is positive, cool zones' is zero
    assert params[0].anomaly > 0.0
    assert params[-1].anomaly == 0.0


def test_macro_sensitivity_mean_is_uniform():
    flow = RenormalizationFlow(macro_observable="mean")
    sens = flow._macro_sensitivity(np.array([26.0, 27.0, 28.0]))
    assert np.allclose(sens, 1.0 / 3.0)


def test_softmax_sensitivity_sums_to_one():
    # d(soft-max value)/d(zones) should sum to ~1 (it's a weighted average value)
    flow = RenormalizationFlow(macro_observable="max", softmax_beta=6.0)
    temps = np.array([25.0, 28.0, 30.0])
    sens = flow._macro_sensitivity(temps)
    assert abs(np.sum(sens) - 1.0) < 1e-6
    # hotter zone must have larger sensitivity than the cool one
    assert sens[2] > sens[0]


def test_tiered_action_axiom_mutation_on_macro_amplification():
    # all zones hot -> building mean far from center -> amplified -> axiom mutation
    flow = RenormalizationFlow(micro_threshold=27.0, macro_amplification_ratio=0.5,
                               comfort_center=26.0)
    out = flow.analyze([29.0, 29.5, 30.0])
    assert out["amplified"] is True
    assert out["action"] == "axiom_mutation"
    assert out["scale"] == "macro"
    assert out["dominant_zone"] is not None


def test_tiered_action_observe_when_all_comfortable():
    flow = RenormalizationFlow(micro_threshold=27.0)
    out = flow.analyze([25.5, 26.0, 26.2])
    assert out["amplified"] is False
    assert out["action"] == "observe"


def test_top_floor_sunlit_zone_flagged_before_building_violation():
    # One zone (top floor, sunlit) is hot and drives the peak, while the
    # building MEAN is still near comfort -> should NOT be full macro amplification
    # yet, but the dominant order parameter must already single out that zone.
    flow = RenormalizationFlow(micro_threshold=27.0, macro_amplification_ratio=1.0,
                               comfort_center=26.0, labels=["floor1", "floor2", "top_sunlit"])
    out = flow.analyze([25.5, 25.8, 28.5])
    dom = out["dominant_zone"]
    assert dom["label"] == "top_sunlit"
    # early cross-scale warning: dominant zone identified before macro amplifies
    assert dom["anomaly"] > 0.0
    assert out["action"] in ("mode_switch", "local")
