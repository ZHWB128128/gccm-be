"""置信度 conformal 式校准行为级测试。"""
from __future__ import annotations

import numpy as np

from gccm_be.app.config import engine_from_dict
from gccm_be.decision.confidence import ConfidenceEvaluator


def test_fixed_scale_unchanged_by_default():
    ev = ConfidenceEvaluator()
    assert ev.calibrated is False
    # 误差 = error_scale 时置信度扣满 1.0
    assert ev.evaluate(prediction_error=2.0) == 0.0
    assert ev.evaluate(prediction_error=0.0) == 1.0


def test_quantile_calibration_adapts_scale():
    ev = ConfidenceEvaluator(calibrated=True, quantile=0.9, min_history=20)
    # 喂 100 个误差在 [0.8, 1.0] 的历史 → P90 ≈ 0.98，远小于固定尺度 2.0
    rng = np.random.default_rng(0)
    for e in rng.uniform(0.8, 1.0, size=100):
        ev.observe_error(float(e))
    scale = ev.error_scale_current()
    assert 0.8 <= scale <= 1.0
    # 同样 0.9 的误差：校准模式下扣减更多（尺度更小）
    loose = ConfidenceEvaluator().evaluate(prediction_error=0.9)
    strict = ev.evaluate(prediction_error=0.9)
    assert strict < loose


def test_calibration_cold_start_falls_back_to_fixed():
    ev = ConfidenceEvaluator(calibrated=True, min_history=20)
    for e in [0.5] * 10:        # 少于 min_history
        ev.observe_error(e)
    assert ev.error_scale_current() == ev.error_scale


def test_calibrated_engine_confidence_reflects_history():
    engine = engine_from_dict({"controller": {"horizon": 6, "confidence_calibrated": True}})
    ev = engine.diagnoser.confidence_evaluator
    assert ev.calibrated is True
    from gccm_be.physics.external import MockExternalInputProvider
    from gccm_be.types import SystemState
    labels = list(engine.manifold.labels)
    state = SystemState(np.full(len(labels), 26.0), labels)
    # 跑闭环：optimize 预测 → observe_step 喂已实现误差 → 校准历史应积累
    decisions = engine.run_closed_loop(state, start_time_h=8.0, steps=30, step_h=engine.dt)
    assert len(ev._error_history) > 0
    assert all(np.isfinite(e) for e in ev._error_history)


def test_confidence_calibrated_false_disables_history():
    engine = engine_from_dict({"controller": {"horizon": 6, "confidence_calibrated": False}})
    ev = engine.diagnoser.confidence_evaluator
    assert ev.calibrated is False
    from gccm_be.types import SystemState
    state = SystemState(np.full(len(engine.manifold.labels), 26.0), engine.manifold.labels)
    engine.run_closed_loop(state, start_time_h=8.0, steps=5, step_h=engine.dt)
    assert len(ev._error_history) == 0


# --- P0-3: 室外温度相关 COP ---

def test_outdoor_dependent_cop_degrades_with_t_out():
    from gccm_be.physics.models import OutdoorDependentCOP
    cop = OutdoorDependentCOP(cop_nominal=3.8, t_out_nominal=30.0, degrade_per_K=0.05)
    assert abs(cop.cop(30.0) - 3.8) < 1e-9
    assert cop.cop(40.0) < cop.cop(30.0)          # 越热越低效
    assert cop.cop(50.0) == max(3.8 - 0.05 * 20, cop.cop_floor)  # 20K 降额后仍未触底


def test_hvac_electrical_power_uses_outdoor_cop_when_wired():
    from gccm_be.physics.models import HVACModel, OutdoorDependentCOP
    from gccm_be.types import ControlInput, ExternalInput
    import numpy as np
    hv = HVACModel(q_min=-8, q_max=8,
                   cop_provider=OutdoorDependentCOP(cop_nominal=3.8,
                                                    t_out_nominal=30.0,
                                                    degrade_per_K=0.1))
    ctrl = ControlInput(np.array([-8.0]))
    ext_cool = ExternalInput(np.array([25.0, 0, 0, 0.5]), ["T_out", "solar", "occ", "price"])
    ext_hot = ExternalInput(np.array([45.0, 0, 0, 0.5]), ["T_out", "solar", "occ", "price"])
    p_cool = hv.electrical_power(ctrl, ext_cool)
    p_hot = hv.electrical_power(ctrl, ext_hot)
    assert p_hot > p_cool          # 同样制冷量，室外越热耗电越多
    # 不传 external 时用名义 COP（向后兼容）
    assert abs(hv.electrical_power(ctrl) - 8.0 / 3.8) < 1e-9


def test_landscape_running_cost_passes_external_to_hvac():
    """景观代价里的电费项必须感知室外温度（否则 MPC 看不到 COP 降额）。"""
    from gccm_be.geometry.landscape import EnergyLandscape
    from gccm_be.geometry.manifold import StateManifold
    from gccm_be.physics.models import HVACModel, OutdoorDependentCOP
    from gccm_be.types import ControlInput, ExternalInput, SystemState
    import numpy as np
    manifold = StateManifold(labels=["T_air", "T_wall"], units={}, bounds={},
                             scale={"T_air": 5.0, "T_wall": 5.0})
    hv = HVACModel(q_min=-8, q_max=8,
                   cop_provider=OutdoorDependentCOP(cop_nominal=3.8,
                                                    t_out_nominal=30.0, degrade_per_K=0.1))
    ls = EnergyLandscape(setpoints={"T_air": 26.0},
                         weights={"comfort": 1.0, "energy": 1.0, "smooth": 0.0},
                         manifold=manifold, hvac=hv,
                         comfort_min=25.0, comfort_max=27.0)
    state = SystemState(np.array([27.0, 26.0]), ["T_air", "T_wall"])
    ctrl = ControlInput(np.array([-8.0]))
    def ext(t_out):
        return ExternalInput(np.array([t_out, 0, 0, 0.6]), ["T_out", "solar", "occ", "price"])
    assert ls.running_cost(state, ctrl, ext(45.0)) > ls.running_cost(state, ctrl, ext(25.0))


# --- P2-3: API token 鉴权 ---

def test_api_token_auth():
    from gccm_be.app.api import SimpleAPI
    api = SimpleAPI(engine_from_dict({"controller": {"horizon": 6}}), auth_token="secret")
    import pytest
    with pytest.raises(PermissionError):
        api.status() if False else api._check_auth({})
    api._check_auth({"Authorization": "Bearer secret"})   # 正确 token 通过


def test_api_token_none_keeps_open_behavior():
    from gccm_be.app.api import SimpleAPI
    api = SimpleAPI(engine_from_dict({"controller": {"horizon": 6}}))
    api._check_auth({})   # 无 token 配置 → 不校验


# --- 边界二:GI 连续化指数 ---

def test_gi_index_continuous_and_gated():
    from gccm_be.decision.godel_boundary import GodelBoundary
    g = GodelBoundary()
    hi = g.geometric_indeterminacy_index(0.9, 0.02, 0.8)
    lo = g.geometric_indeterminacy_index(0.1, 0.08, 0.1)
    assert hi["gi"] > 0.7 and hi["level"] == "degrade"
    assert lo["gi"] < 0.4 and lo["level"] == "normal"
    # AND 门语义保持:GI 高但单通道未全坏 → undecidable False,GI 补位
    r = g.evaluate(0.9, None, 0.8)  # curvature=None → curvature_bad False → AND False
    assert r["undecidable"] is False
    assert hi["gi"] > 0.7           # GI 连续通道仍能感知


def test_engine_emits_geometric_indeterminacy():
    from gccm_be.app.config import engine_from_dict
    engine = engine_from_dict({"controller": {"horizon": 6}})
    from gccm_be.types import SystemState
    import numpy as np
    state = SystemState(np.full(len(engine.manifold.labels), 27.5), engine.manifold.labels)
    dec = engine.optimize(state, 8.0, forced_mode="comfort")
    assert "geometric_indeterminacy" in dec.diagnosis.details
    gi = dec.diagnosis.details["geometric_indeterminacy"]
    assert 0.0 <= gi["gi"] <= 1.0 and gi["level"] in ("normal", "watch", "degrade")
