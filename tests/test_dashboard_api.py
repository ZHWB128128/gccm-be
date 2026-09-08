"""Tests for the Web dashboard API + the introspection registry that drives it.

The key test here is `test_registry_fields_exist_on_engine`: it enforces the
long-term contract that the introspection registry (the single source of truth
the dashboard renders from) stays in sync with the actual engine. If a model
change adds/renames a switch or weight but forgets the registry — or the registry
references a field the engine no longer has — this test fails.
"""
from __future__ import annotations

import json
from dataclasses import fields

import pytest

from gccm_be import GCCMEngine
from gccm_be.app import introspection as intro
from gccm_be.app.api import SimpleAPI
from gccm_be.app.dashboard import DASHBOARD_HTML


@pytest.fixture
def api():
    return SimpleAPI(GCCMEngine())


def test_registry_fields_exist_on_engine():
    engine_fields = {f.name for f in fields(GCCMEngine)}
    referenced = set(intro.editable_fields())
    for name in referenced:
        assert name in engine_fields, f"introspection references unknown engine field: {name}"


def test_registry_covers_known_switch_weight_fields():
    # Reverse guard: engine fields that clearly are switches/weights should be
    # either documented in the registry or explicitly listed as non-surfaced, so
    # a new model switch can't silently bypass the page.
    engine_fields = {f.name for f in fields(GCCMEngine)}
    documented = set(intro.editable_fields()) | set(intro.NON_SURFACED_FIELDS)
    suspicious = {
        n for n in engine_fields
        if n.startswith("use_") or n.endswith("_weight")
    }
    missing = suspicious - documented
    assert not missing, (
        f"engine switches/weights missing from introspection registry: {missing}. "
        "Add them to a registry section or to NON_SURFACED_FIELDS with a reason.")


def test_introspection_attaches_live_values():
    engine = GCCMEngine(use_kinetic=True, kinetic_weight=2.5, comfort_weight=7.0)
    spec = intro.describe(engine)
    kin = next(e for e in spec["riemannian_switches"] if e["field"] == "use_kinetic")
    assert kin["value"] is True
    assert kin["strength_value"] == 2.5
    cw = next(e for e in spec["weights"] if e["field"] == "comfort_weight")
    assert cw["value"] == 7.0
    assert spec["state_labels"]  # non-empty


def test_dashboard_html_is_data_driven():
    # The page must fetch the registry rather than hardcode the model surface.
    assert "/introspection" in DASHBOARD_HTML
    assert "/simulate" in DASHBOARD_HTML
    assert "/config" in DASHBOARD_HTML


def test_toggle_switch_is_clickable_label():
    # Regression: the on/off switch must wrap the checkbox in a <label> so that
    # clicking the visible slider actually toggles it. A bare <span class="switch">
    # over the checkbox is inert (clicks never reach the input).
    assert 'class="switch"><input type="checkbox"' in DASHBOARD_HTML
    assert '<label class="switch">' in DASHBOARD_HTML
    assert '<span class="switch"><input' not in DASHBOARD_HTML


def test_set_config_applies_and_rejects(api):
    res = api.set_config({"comfort_weight": 9.0, "use_kinetic": False, "not_a_field": 1})
    assert res["applied"]["comfort_weight"] == 9.0
    assert res["applied"]["use_kinetic"] is False
    assert "not_a_field" in res["rejected"]
    assert api.engine.comfort_weight == 9.0
    assert api.engine.use_kinetic is False


def test_control_endpoint_returns_decision(api):
    res = api.handle_control({"state": [28.0, 28.0], "labels": ["T_air", "T_wall"],
                              "forced_mode": "comfort", "time_h": 8.0})
    assert "control" in res
    assert "mode" in res
    assert "solver_success" in res


def test_simulate_returns_plottable_trajectories(api):
    res = api.simulate({"t0": 29.0, "steps": 12, "forced_mode": "comfort"})
    assert len(res["temps"]) == 12
    assert len(res["controls"]) == 12
    assert len(res["prices"]) == 12
    assert "comfort_min" in res and "comfort_max" in res
    assert res["last_decision"] is not None


def test_status_and_introspection_json_serializable(api):
    json.dumps(api.status())
    json.dumps(api.introspection())


def test_physics_models_registry_are_real_classes():
    # Each physics model advertised on the dashboard must be an importable class,
    # so removing/renaming a model can't leave a dangling entry on the page.
    import gccm_be.physics as physics
    for m in intro.PHYSICS_MODELS:
        assert hasattr(physics, m["id"]) or _is_datacenter_model(m["id"]), (
            f"introspection PHYSICS_MODELS lists unknown model: {m['id']}")


def _is_datacenter_model(name: str) -> bool:
    import gccm_be.physics.datacenter as dc
    return hasattr(dc, name)


def test_modes_registry_matches_allowed_modes():
    from gccm_be.app.api import ALLOWED_MODES
    registry_ids = {m["id"] for m in intro.MODES}
    assert registry_ids == set(ALLOWED_MODES), (
        f"introspection MODES {registry_ids} must match ALLOWED_MODES {set(ALLOWED_MODES)}")


def test_diagnostic_fields_present_in_decision(api):
    # Diagnostic fields advertised on the monitor must actually appear in a real
    # decision dict, so the monitor can't reference a field the engine dropped.
    dec = api.handle_control({"state": [28.0, 28.0], "labels": ["T_air", "T_wall"],
                              "forced_mode": "comfort", "time_h": 8.0})
    top_level = set(dec.keys())
    nested = set(dec.get("diagnosis", {}).keys())
    detail = set(dec.get("diagnosis", {}).get("details", {}).keys())
    available = top_level | nested | detail
    for f in intro.DIAGNOSTIC_FIELDS:
        assert f["id"] in available, (
            f"diagnostic field '{f['id']}' advertised on dashboard but not in decision dict")


def test_dashboard_has_arch_tab():
    # The "架构图" tab must exist and be data-driven off the introspection registry.
    assert 'id="tab-arch"' in DASHBOARD_HTML
    assert 'id="panel-arch"' in DASHBOARD_HTML
    assert 'id="arch-layers"' in DASHBOARD_HTML
    assert "renderArch" in DASHBOARD_HTML
    assert "switchTab" in DASHBOARD_HTML
    # Arch tab must not be hardcoded HTML: it renders SPEC.architecture at runtime.
    assert ".architecture" in DASHBOARD_HTML  # JS references SPEC.architecture


def test_introspection_architecture_blocks():
    spec = intro.describe()
    arch = spec["architecture"]
    assert set(arch.keys()) == {"layers", "pipeline", "fallback_points", "theory_layers"}
    # layers: 7 layers, each with module + help
    assert len(arch["layers"]) == 7
    for l in arch["layers"]:
        assert l["id"] and l["label"] and l["module"] and l["help"]
    # pipeline: sequential steps 1..13, each with method
    steps = [s["step"] for s in arch["pipeline"]]
    assert steps == list(range(1, 14)), f"pipeline steps must be 1..13, got {steps}"
    for s in arch["pipeline"]:
        assert s["label"] and s["method"] and s["help"]
    # fallback points: non-empty, each declares point + path
    assert len(arch["fallback_points"]) >= 3
    for f in arch["fallback_points"]:
        assert f["point"] and f["path"]
    # theory layers: the honest-positioning table (no magic claims)
    assert {t["id"] for t in arch["theory_layers"]} == {"riemannian", "scm", "rg"}
    for t in arch["theory_layers"]:
        assert t["label"] and t["math"] and t["role"] and t["when"] and t["kpi"], (
            f"theory layer {t['id']} must carry honest positioning fields")
