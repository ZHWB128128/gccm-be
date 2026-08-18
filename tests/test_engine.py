import numpy as np
from types import SimpleNamespace
from unittest import mock

from gccm_be import GCCMEngine
from gccm_be.geometry.geodesic import GeodesicSolver
from gccm_be.types import ControlInput, DiagnosisReport, SystemState, Trajectory


def test_engine_single_optimization():
    engine = GCCMEngine(horizon=6)
    state = SystemState([26.5, 26.0], ["T_air", "T_wall"])
    decision = engine.optimize(state, time_h=8.0)
    assert decision.control.u.shape == (1,)
    assert decision.trajectory.success
    assert 0.0 <= decision.confidence <= 1.0
    assert len(decision.trajectory.controls) == 6
    assert len(decision.trajectory.states) == 7


def test_engine_closed_loop():
    engine = GCCMEngine(horizon=4)
    state = SystemState([26.5, 26.0], ["T_air", "T_wall"])
    decisions = engine.run_closed_loop(state, steps=4, step_h=1.0 / 12.0)
    assert len(decisions) == 4
    assert all(np.isfinite(d.control.u).all() for d in decisions)


# ---------- 行为级断言（修复验证） ----------

def test_hot_state_produces_cooling():
    """高温状态下 MPC 必须输出制冷（u < 0）。"""
    engine = GCCMEngine(horizon=6)
    state = SystemState([28.5, 27.0], ["T_air", "T_wall"])
    decision = engine.optimize(state, time_h=14.0)
    assert decision.control.u.shape == (1,)
    assert decision.control.u[0] < 0.0, f"高温应制冷，实际 u={decision.control.u[0]}"


def test_worst_case_more_cooling_when_hotter():
    """worst_case 安全控制：越热制冷越强（反馈项方向正确）。"""
    engine = GCCMEngine()
    q28 = engine._worst_case_safe_control(
        SystemState([28.0, 27.0], ["T_air", "T_wall"]), 0.25
    ).u[0]
    q30 = engine._worst_case_safe_control(
        SystemState([30.0, 27.0], ["T_air", "T_wall"]), 0.25
    ).u[0]
    assert q28 <= 0.0
    assert q30 <= q28, f"越热应制冷越强：q30={q30} 应 <= q28={q28}"


def test_undecidable_preserves_pre_degradation_mode_and_recovers():
    """不可判定切换安全模式时必须保存降级前真实模式，恢复时能还原。"""
    engine = GCCMEngine(horizon=4)
    engine.mode_manager.switch("energy")
    stub = SimpleNamespace(
        diagnose=lambda **kw: DiagnosisReport(undecidable=True, confidence=0.05),
        safe_mode="demand_response",
    )
    engine.diagnoser = stub
    state = SystemState([28.0, 26.0], ["T_air", "T_wall"])
    engine.optimize(state, time_h=8.0)
    # 已切到安全模式，且保存了降级前的真实模式（energy），而不是 safe_mode
    assert engine.mode_manager.current_mode == "demand_response"
    assert engine._pre_degradation_mode == "energy"
    assert engine._degraded

    # 恢复：误差回落 + 温度正常 + 连续低误差步数足够 → 应还原为 energy
    engine.self_monitor.predicted_error = lambda: 0.05
    engine.self_monitor.is_diverging = lambda: False
    engine._degraded_steps = engine.min_degradation_steps
    engine._low_error_streak = engine.recovery_steps_required - 1
    stub.diagnose = lambda **kw: DiagnosisReport(undecidable=False, confidence=0.95)
    cool_state = SystemState([26.2, 25.8], ["T_air", "T_wall"])
    engine.optimize(cool_state, time_h=9.0)
    assert engine.mode_manager.current_mode == "energy"
    assert not engine._degraded


def test_counterfactual_does_not_pollute_state():
    """反事实滚动不得污染真实控制状态（warm start / 模式 / 降级标志 / 曲率）。"""
    engine = GCCMEngine(horizon=4, counterfactual_enabled=True, counterfactual_horizon=2)
    state = SystemState([26.5, 26.0], ["T_air", "T_wall"])
    engine.optimize(state, time_h=8.0)
    before_warm = engine._warm_start
    before_mode = engine.mode_manager.current_mode
    before_degraded = engine._degraded
    before_curv = engine._last_curvature_min
    before_integral = dict(engine._safe_integral)

    engine._run_counterfactual(state, 8.0, "balanced", "energy")
    engine._run_counterfactual_undecidable(state, 8.0)

    assert engine.mode_manager.current_mode == before_mode
    assert engine._degraded == before_degraded
    assert engine._warm_start is before_warm
    assert engine._last_curvature_min == before_curv
    assert dict(engine._safe_integral) == before_integral


def test_solver_failure_falls_back_safely():
    """求解器返回空轨迹（如 IPOPT 失败）时不得崩溃，应回退反馈安全控制并记录。"""
    engine = GCCMEngine(horizon=4)
    state = SystemState([28.0, 26.0], ["T_air", "T_wall"])
    with mock.patch.object(
        GeodesicSolver, "solve",
        return_value=Trajectory(controls=[], states=[], success=False),
    ):
        decision = engine.optimize(state, time_h=8.0)
    assert decision.control.u.shape == (1,)
    assert decision.control.u[0] <= 0.0  # 高温下回退控制也应制冷
    assert decision.solver_success is False
    assert "solver_failure" in decision.diagnosis.details


def test_solver_unreliable_records_warning():
    """求解器返回 success=False 但有轨迹时，标记不可靠并记录告警。"""
    engine = GCCMEngine(horizon=4)
    state = SystemState([26.5, 26.0], ["T_air", "T_wall"])
    traj = Trajectory(
        controls=[ControlInput(np.array([-2.0]), ["Q_hvac"])],
        states=[state.copy()],
        success=False,
        message="Maximum_Iterations_Exceeded",
    )
    with mock.patch.object(GeodesicSolver, "solve", return_value=traj):
        decision = engine.optimize(state, time_h=8.0)
    assert decision.solver_success is False
    assert "solver_warning" in decision.diagnosis.details
    assert "Maximum_Iterations_Exceeded" in decision.diagnosis.details["solver_warning"]


def test_solver_exception_falls_back_safely():
    """求解器直接抛异常（非空轨迹约定）时也不得崩溃，应回退并记录。"""
    engine = GCCMEngine(horizon=4)
    state = SystemState([28.0, 26.0], ["T_air", "T_wall"])

    def boom(self, *a, **kw):
        raise RuntimeError("IPOPT blowup")

    with mock.patch.object(GeodesicSolver, "solve", boom):
        decision = engine.optimize(state, time_h=8.0)
    assert decision.control.u.shape == (1,)
    assert decision.control.u[0] <= 0.0
    assert decision.solver_success is False
    assert "solver_failure" in decision.diagnosis.details
    assert "IPOPT blowup" in decision.diagnosis.details["solver_failure"]


def test_solver_failure_does_not_pollute_prediction():
    """求解失败步的 predicted_next_state 应为 None，避免把状态变化误当模型误差。"""
    engine = GCCMEngine(horizon=4)
    state = SystemState([28.0, 26.0], ["T_air", "T_wall"])
    with mock.patch.object(
        GeodesicSolver, "solve",
        return_value=Trajectory(controls=[], states=[], success=False),
    ):
        decision = engine.optimize(state, time_h=8.0)
    assert decision.predicted_next_state is None
