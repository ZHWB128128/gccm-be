"""Verifies the covariant (Riemannian) curvature is actually consumed by the
engine's decision path, not computed and discarded."""
from __future__ import annotations


from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator
from gccm_be.types import SystemState

STEP_H = 0.25


def _engine(covariant: bool, adaptive: bool = False):
    sim = Simulator(RCBuildingModel(), HVACModel())
    manifold = StateManifold(
        labels=["T_air", "T_wall"],
        bounds={"T_air": (15.0, 35.0), "T_wall": (15.0, 35.0)},
        scale={"T_air": 5.0, "T_wall": 5.0},
    )
    return GCCMEngine(
        simulator=sim,
        manifold=manifold,
        horizon=4,
        dt=STEP_H,
        setpoints={"T_air": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        covariant_curvature=covariant,
        curvature_adaptive=adaptive,
        metric_state_dependence=0.5,
        nominal_horizon=4,
        min_horizon=2,
    )


def test_curvature_geometry_reported_in_diagnosis():
    engine = _engine(covariant=True)
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    dec = engine.optimize(state, 0.0, forced_mode="comfort")
    geom = dec.diagnosis.details.get("curvature_geometry")
    assert geom is not None
    # the engine used the covariant (Riemannian) Hessian, and it is recorded
    assert geom["covariant"] is True
    assert geom["classification"] in ("stable", "unstable", "saddle", "flat")


def test_euclidean_mode_flag_is_false():
    engine = _engine(covariant=False)
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])
    dec = engine.optimize(state, 0.0, forced_mode="comfort")
    assert dec.diagnosis.details["curvature_geometry"]["covariant"] is False


def test_landscape_mutation_trigger_recorded_on_singular_geometry():
    # Force a saddle/singular classification by monkeypatching the analyzer
    engine = _engine(covariant=True, adaptive=True)
    state = SystemState([28.0, 28.0], ["T_air", "T_wall"])

    class _FakeCurv:
        covariant = True
        classification = "saddle"
        stability = -1.0

        def as_dict(self):
            return {"classification": self.classification, "stability": self.stability}

    engine.curvature_analyzer.analyze = lambda *a, **k: _FakeCurv()  # type: ignore
    dec = engine.optimize(state, 0.0, forced_mode="comfort")
    geom = dec.diagnosis.details["curvature_geometry"]
    assert geom["landscape_mutation"] is not None
    assert geom["landscape_mutation"]["reason"] == "saddle_or_singular"
    assert any(t.startswith("landscape_mutation:") for t in dec.diagnosis.triggers)
    # the geometry-driven margin boost must be armed for the next cycle
    assert getattr(engine, "_geometry_margin_boost", 0.0) >= 0.3
