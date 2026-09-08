"""Tests for label-based ExternalInput accessors (retires positional w[3]/w[5])."""
from __future__ import annotations

import numpy as np
import pytest

from gccm_be.types import ExternalInput


def test_named_lookup_single_zone():
    ext = ExternalInput([32.0, 0.5, 0.6, 1.4], ["T_out", "solar", "occ", "price"])
    assert ext.t_out == 32.0
    assert ext.solar() == 0.5
    assert ext.occupancy() == 0.6
    assert ext.price == 1.4


def test_named_lookup_two_zone_by_position_independent_label():
    # two-zone layout: price is at index 5, solar split by zone
    ext = ExternalInput(
        [30.0, 0.8, 0.2, 1.0, 0.6, 0.9],
        ["T_out", "solar_A", "solar_B", "occ_A", "occ_B", "price"],
    )
    assert ext.price == 0.9          # not fooled by position
    assert ext.solar(zone="A") == 0.8
    assert ext.solar(zone="B") == 0.2
    assert ext.occupancy(zone="A") == 1.0
    assert ext.occupancy(zone="B") == 0.6


def test_require_raises_on_missing_label():
    ext = ExternalInput([30.0, 0.5], ["T_out", "solar"])
    with pytest.raises(KeyError):
        ext.require("price")


def test_price_legacy_positional_fallback_still_works():
    # arrays built without a 'price' label must still resolve via legacy position
    single = ExternalInput([30.0, 0.5, 0.6, 1.7], ["T_out", "solar", "occ", "w3"])
    assert single.price == 1.7       # w[3] fallback
    two = ExternalInput([30, 0.5, 0.2, 1, 0.6, 2.1], [f"c{i}" for i in range(6)])
    assert two.price == 2.1          # w[5] fallback


def test_index_survives_copy():
    ext = ExternalInput([30.0, 0.5, 0.6, 1.4], ["T_out", "solar", "occ", "price"])
    c = ext.copy()
    assert c.price == 1.4
    assert c.solar() == 0.5
