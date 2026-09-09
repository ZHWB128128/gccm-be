"""GCCM-BE top-level engine: rolling-horizon optimization and event-driven decision orchestration."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .causal.data_driven import DataDrivenSCM
from .causal.scm import StructuralCausalModel, build_rc_scm
from .control import SafeController
from .decision.diagnoser import DecisionDiagnoser
from .decision.self_monitor import SelfMonitor
from .geometry.casadi_robust_solver import CasadiRobustGeodesicSolver
from .geometry.curvature import CurvatureAnalyzer
from .geometry.geodesic import GeodesicSolver
from .geometry.landscape import EnergyLandscape
from .geometry.manifold import StateManifold
from .geometry.robust_solver import RobustGeodesicSolver
from .multiscale.renormalization import RenormalizationFlow
from .normative.context import ContextLabeler
from .normative.modes import ModeManager
from .normative.weights import WeightMapper
from .physics.external import ExternalInputProvider, MockExternalInputProvider
from .physics.models import (
    HVACModel,
    RCBuildingModel,
    Simulator,
    TwoZoneRCBuildingModel,
)
from .physics.online_id import OnlineIdentifier, RCOnlineIdentifier
from .types import ControlDecision, ControlInput, ExternalInput, SystemState, Trajectory


@dataclass
class GCCMEngine:
    """Composes the five layers and provides a unified control-decision entry point."""

    simulator: Simulator = None  # type: ignore[assignment]
    external_provider: ExternalInputProvider = None  # type: ignore[assignment]
    manifold: StateManifold = None  # type: ignore[assignment]
    context_labeler: ContextLabeler = None  # type: ignore[assignment]
    mode_manager: ModeManager = None  # type: ignore[assignment]
    weight_mapper: WeightMapper = None  # type: ignore[assignment]
    curvature_analyzer: CurvatureAnalyzer = None  # type: ignore[assignment]
    diagnoser: DecisionDiagnoser = None  # type: ignore[assignment]
    self_monitor: SelfMonitor = None  # type: ignore[assignment]
    horizon: int = 12
    dt: float | None = None
    solver_options: dict = field(default_factory=lambda: {"maxiter": 200, "ftol": 1e-8})
    comfort_band: float = 1.0
    comfort_margin: float = 0.0
    noise_adaptive_margin: bool = False
    comfort_min: float | None = None
    comfort_max: float | None = None
    # 每区独立舒适带 label -> (lo, hi)：覆盖标量 comfort_min/max，作用于
    # 软代价与硬约束；标量边界仍驱动 SafeController 与自适应裕度。
    zone_comfort_bounds: dict[str, tuple] = field(default_factory=dict)
    below_comfort_penalty: float = 0.0
    peak_price_threshold: float = 1.0
    peak_energy_penalty: float = 1.0
    storage_targets: dict[str, float] = field(default_factory=dict)
    storage_weight: float = 0.0
    comfort_weight: float | None = None
    energy_weight: float | None = None
    smooth_weight: float | None = None
    enforce_comfort_constraints: bool = False
    use_casadi: bool = False
    use_casadi_robust: bool = False
    robust_scenarios: list[Simulator] = field(default_factory=list)
    robust_enforce_lower_bound: bool = True
    robust_heating_cost_factor: float = 1.0
    use_riemannian: bool = False
    use_riemannian_control: bool = False
    riemannian_control_weight: float = 1.0
    riemannian_strength: float = 1.0
    metric_coupling: float = 0.0
    metric_state_dependence: float = 0.0
    geodesic_penalty_weight: float = 0.0
    kinetic_weight: float = 1.0
    adaptive_riemannian: bool = False
    use_two_stage: bool = False
    use_kinetic: bool = True
    counterfactual_enabled: bool = False
    counterfactual_horizon: int = 4
    causal_scm: object | None = None
    safe_control_mode: str = "zero"
    preemptive_feedback: bool = False
    online_identifier: OnlineIdentifier = None  # type: ignore[assignment]
    rc_identifier: RCOnlineIdentifier = None  # type: ignore[assignment]
    identification_enabled: bool = True
    model_bias: float = 0.0
    safe_kp: float = 2.0
    # 置信度 conformal 式校准：以已实现预测误差的滚动分位数为尺度
    confidence_calibrated: bool = True
    confidence_quantile: float = 0.90
    safe_ki: float = 0.2
    safe_feedforward_gain: float = 0.5
    safe_integral_limit: float = 2.0
    safe_max_cooling: float = 2.5
    safe_weather_margin: float = 1.0
    degradation_enter_error: float = 0.6
    degradation_exit_error: float = 0.2
    recovery_steps_required: int = 3
    min_degradation_steps: int = 5
    worst_case_outdoor_temp: float = 40.0
    worst_case_solar: float = 0.8
    worst_case_occ: float = 1.2
    renormalization_enabled: bool = False
    renormalization_flow: RenormalizationFlow = None  # type: ignore[assignment]
    # 重整化序参量驱动分区舒适权重：把 identify_order_parameters 的相关度映射为
    # 每区舒适惩罚乘子，使 MPC 优先照顾驱动全楼峰值的关键区。
    renormalization_weighting: bool = False
    renormalization_weight_gain: float = 2.0
    curvature_adaptive: bool = False
    covariant_curvature: bool = False
    nominal_horizon: int | None = None
    min_horizon: int | None = None
    curvature_flat_threshold: float = 1e-6
    curvature_singular_threshold: float = -1e-3
    tightened_comfort_max: float | None = None
    constraint_options: dict = field(default_factory=lambda: {"maxiter": 30, "ftol": 1e-5})
    setpoints: dict[str, float] = field(default_factory=lambda: {"T_air": 24.0})

    def __post_init__(self) -> None:
        if self.simulator is None:
            self.simulator = Simulator(RCBuildingModel(), HVACModel())
        # 统一时间步长：未显式指定时继承建筑模型默认步长（1/12h），消除 0.25/1.0 回退不一致
        if self.dt is None:
            self.dt = getattr(self.simulator.building, "dt", 1.0 / 12.0)
        if self.external_provider is None:
            self.external_provider = MockExternalInputProvider()
        if self.manifold is None:
            self.manifold = StateManifold(
                labels=["T_air", "T_wall"],
                units={"T_air": "°C", "T_wall": "°C"},
                bounds={"T_air": (15.0, 35.0), "T_wall": (15.0, 35.0)},
                scale={"T_air": 5.0, "T_wall": 5.0},
            )
        if self.context_labeler is None:
            self.context_labeler = ContextLabeler()
        if self.mode_manager is None:
            self.mode_manager = ModeManager()
        if self.weight_mapper is None:
            self.weight_mapper = WeightMapper()
        if self.curvature_analyzer is None:
            self.curvature_analyzer = CurvatureAnalyzer(covariant=self.covariant_curvature)
        if self.diagnoser is None:
            self.diagnoser = DecisionDiagnoser()
        # 校准开关在条件外应用：自带 diagnoser 的构造同样生效
        self.diagnoser.confidence_evaluator.calibrated = self.confidence_calibrated
        self.diagnoser.confidence_evaluator.quantile = self.confidence_quantile
        if self.self_monitor is None:
            self.self_monitor = SelfMonitor()
        if self.causal_scm is None:
            self.causal_scm = self._build_default_scm()
        if self.renormalization_flow is None:
            self.renormalization_flow = RenormalizationFlow()
        if self.online_identifier is None:
            self.online_identifier = OnlineIdentifier(dim=1, lam=0.98)
        if self.rc_identifier is None:
            self.rc_identifier = RCOnlineIdentifier(lam=0.99)
        # 双区模型：按区各建一个辨识器（zone 模式按标签取状态/按区名取外部输入）。
        # rc_identifier 保留为 A 区标识器，供既有引用（warm/mismatch 检查）继续工作。
        self.rc_identifiers_by_zone: dict[str, RCOnlineIdentifier] = {}
        if isinstance(self.simulator.building, TwoZoneRCBuildingModel):
            self.rc_identifiers_by_zone = {
                "A": self.rc_identifier if self.rc_identifier.zone == "A"
                     else RCOnlineIdentifier(lam=0.99, zone="A", unit_index=0),
                "B": RCOnlineIdentifier(lam=0.99, zone="B", unit_index=1),
            }
            self.rc_identifier = self.rc_identifiers_by_zone["A"]
        self._warm_start = None
        self._last_curvature_min: float | None = None

        # Canonical owner of the fallback/safe control laws (extracted from this
        # God Object). The engine no longer implements the laws itself; it
        # delegates and shares its identification-trust collaborators.
        self.safe_controller = SafeController(
            simulator=self.simulator,
            setpoints=self.setpoints,
            comfort_min=self.comfort_min,
            comfort_max=self.comfort_max,
            zone_comfort_bounds=dict(self.zone_comfort_bounds),
            safe_kp=self.safe_kp,
            safe_ki=self.safe_ki,
            safe_feedforward_gain=self.safe_feedforward_gain,
            safe_integral_limit=self.safe_integral_limit,
            safe_weather_margin=self.safe_weather_margin,
            worst_case_outdoor_temp=self.worst_case_outdoor_temp,
            worst_case_solar=self.worst_case_solar,
            worst_case_occ=self.worst_case_occ,
            identification_trusted=self.identification_trusted,
            rc_parameters=lambda: self.rc_identifier.parameters(),
        )
        self._degraded = False
        self._pre_degradation_mode: str | None = None
        self._recent_step: dict | None = None
        self._low_error_streak = 0
        self._degraded_steps = 0
        if self.nominal_horizon is None:
            self.nominal_horizon = self.horizon
        if self.min_horizon is None:
            self.min_horizon = max(4, self.nominal_horizon // 2)

    def _plan_horizon_and_margin(self, state: SystemState, forced_mode: str | None):
        """Curvature-adaptive planning owner: resolve effective horizon, comfort
        bounds (with predictive + geometry-driven margin tightening) and any mode
        override from last cycle's local geometry. Pure decision, no side effects
        except the single-shot consumption of the geometry margin boost.

        Returns (effective_horizon, effective_comfort_min, effective_comfort_max,
        mode_override).
        """
        effective_horizon = self.horizon
        effective_comfort_min = self.comfort_min
        effective_comfort_max = self.comfort_max
        if self.curvature_adaptive and self._last_curvature_min is not None:
            if self._last_curvature_min < self.curvature_singular_threshold:
                effective_horizon = self.min_horizon
                if self.tightened_comfort_max is not None:
                    effective_comfort_max = self.tightened_comfort_max
                mode_override = "balanced"
            elif self._last_curvature_min < self.curvature_flat_threshold:
                effective_horizon = max(self.min_horizon, self.nominal_horizon - 4)
                if self.tightened_comfort_max is not None:
                    effective_comfort_max = self.tightened_comfort_max
                mode_override = None
            else:
                effective_horizon = self.nominal_horizon
                effective_comfort_max = self.comfort_max
                mode_override = None
        else:
            effective_horizon = self.horizon
            effective_comfort_min = self.comfort_min
            effective_comfort_max = self.comfort_max
            mode_override = None

        # 约束收紧：预防性保守，不改变真实舒适指标
        margin = self.comfort_margin
        if self.noise_adaptive_margin and self.self_monitor is not None:
            pred_err = self.self_monitor.predicted_error()
            margin = max(margin, min(0.8, pred_err * 0.8))
        # 上一周期曲率几何突变（saddle/奇异）驱动的额外收紧：真正消费协变 Hessian 的结论
        geometry_boost = getattr(self, "_geometry_margin_boost", 0.0)
        if geometry_boost > 0.0:
            margin = max(margin, geometry_boost)
            self._geometry_margin_boost = 0.0  # 单次消费，避免长期累积
        if margin > 0.0:
            if effective_comfort_min is not None:
                effective_comfort_min = effective_comfort_min + margin
            if effective_comfort_max is not None:
                effective_comfort_max = effective_comfort_max - margin
        return effective_horizon, effective_comfort_min, effective_comfort_max, mode_override

    def _attempt_degradation_recovery(self, state: SystemState) -> None:
        """Degradation-recovery owner: graceful exit from degraded mode with
        hysteresis, a minimum degradation dwell, and a temperature gate. Mutates
        the engine's degradation bookkeeping in place (bit-identical to inline)."""
        if self._degraded:
            self._degraded_steps += 1
        if self._degraded and self.self_monitor is not None:
            temp_ok = state.x[0] <= (self.comfort_max or 27.0) - 0.3
            recovered = (
                self._degraded_steps >= self.min_degradation_steps
                and not self.self_monitor.is_diverging()
                and self.self_monitor.predicted_error() < self.degradation_exit_error
                and temp_ok
            )
            if recovered:
                self._low_error_streak += 1
            else:
                self._low_error_streak = 0
            if self._low_error_streak >= self.recovery_steps_required:
                self._degraded = False
                self._low_error_streak = 0
                self._degraded_steps = 0
                if self._pre_degradation_mode is not None:
                    self.mode_manager.switch(self._pre_degradation_mode)

    def _renormalization_zone_weights(self, state: SystemState) -> dict[str, float]:
        """Renormalization owner (engine side): read the air-zone temps and defer
        to RenormalizationFlow.zone_comfort_weights for the RG->MPC coupling."""
        if self.renormalization_weighting and state.dim > 2:
            air_idx = [i for i, lab in enumerate(state.labels) if lab.startswith("T_air")]
            if len(air_idx) >= 2:
                return self.renormalization_flow.zone_comfort_weights(
                    [state.x[i] for i in air_idx],
                    gain=self.renormalization_weight_gain,
                    labels=[state.labels[i] for i in air_idx],
                )
        return {}

    def _consume_curvature_geometry(self, diagnosis, curvature, landscape_mutation) -> None:
        """Curvature-geometry consumption owner: record the covariant-Hessian
        classification + mutation suggestion into the diagnosis, append a trigger,
        and (when curvature_adaptive) arm the next-cycle margin boost on a
        saddle/singular geometry. Mutates diagnosis + engine state in place."""
        diagnosis.details["curvature_geometry"] = {
            "covariant": bool(getattr(curvature, "covariant", False)),
            "classification": curvature.classification,
            "stability": float(curvature.stability),
            "landscape_mutation": landscape_mutation,
        }
        if landscape_mutation is not None:
            diagnosis.triggers.append(f"landscape_mutation:{landscape_mutation['type']}")
            if self.curvature_adaptive and landscape_mutation.get("reason") == "saddle_or_singular":
                self._geometry_margin_boost = max(
                    getattr(self, "_geometry_margin_boost", 0.0), 0.3
                )

    def _build_robust_solver(self, landscape, effective_horizon,
                             effective_comfort_min, effective_comfort_max):
        """Construct the CasADi robust geodesic solver for this cycle.

        Shares the current landscape, nominal simulator, robust scenario
        simulators and the robust-* engine config. Raises RuntimeError if CasADi
        is unavailable (handled by the caller's try/except -> safe fallback).
        """
        return CasadiRobustGeodesicSolver(
            nominal_sim=self.simulator,
            landscape=landscape,
            scenario_sims=list(self.robust_scenarios),
            horizon=effective_horizon,
            dt=self.dt,
            comfort_min=effective_comfort_min if effective_comfort_min is not None else 25.0,
            comfort_max=effective_comfort_max if effective_comfort_max is not None else 27.0,
            enforce_lower_bound=self.robust_enforce_lower_bound,
            heating_cost_factor=self.robust_heating_cost_factor,
        )

    def _solve_trajectory(self, solver, state, external_seq, prev_control,
                          effective_horizon, landscape=None,
                          effective_comfort_min=None, effective_comfort_max=None):
        """Solver owner: build warm-start initial controls, run the (optionally
        robust) geodesic solve, and normalize failures to an empty Trajectory.

        Returns (trajectory, solver_exc).

        Robust branch: when ``use_casadi_robust and robust_scenarios`` are set, a
        CasadiRobustGeodesicSolver is built for this cycle (previously an unbound
        ``robust_solver`` name silently NameError'd into the safe-control
        fallback, so robust MPC never actually ran). The nominal solve seeds its
        warm start; if the robust solve returns no controls we fall back to the
        nominal solve. If CasADi is unavailable the construction raises and the
        outer try normalizes it to an empty trajectory -> safe control.
        """
        initial_controls = None
        if self._warm_start is not None and len(self._warm_start) == effective_horizon:
            initial_controls = self._warm_start[1:] + [self._warm_start[-1]]
        elif self._warm_start is None:
            initial_controls = self._heuristic_initial_controls(external_seq)

        solver_exc = None
        try:
            if self.use_casadi_robust and self.robust_scenarios:
                # 每一步都用普通 MPC 解作为鲁棒 MPC 初始猜测，提高 IPOPT 成功率
                nominal_traj = solver.solve(
                    state, external_seq, prev_control=prev_control, initial_controls=None,
                )
                initial_controls = nominal_traj.controls if nominal_traj.controls else None
                robust_solver = self._build_robust_solver(
                    landscape, effective_horizon,
                    effective_comfort_min, effective_comfort_max,
                )
                trajectory = robust_solver.solve(
                    state, external_seq, prev_control=prev_control,
                    initial_controls=initial_controls,
                )
                if not trajectory.controls:
                    trajectory = solver.solve(
                        state, external_seq, prev_control=prev_control,
                        initial_controls=initial_controls,
                    )
            elif self.robust_scenarios:
                # 纯 numpy 鲁棒路径（无 CasADi 时的预报误差鲁棒）：多场景共享控制序列
                robust_solver = RobustGeodesicSolver(
                    nominal_sim=self.simulator,
                    landscape=landscape,
                    scenario_sims=list(self.robust_scenarios),
                    horizon=effective_horizon,
                    dt=self.dt,
                    comfort_min=(effective_comfort_min
                                 if effective_comfort_min is not None else 25.0),
                    comfort_max=(effective_comfort_max
                                 if effective_comfort_max is not None else 27.0),
                )
                trajectory = robust_solver.solve(
                    state, external_seq, prev_control=prev_control,
                    initial_controls=initial_controls,
                )
            else:
                trajectory = solver.solve(
                    state, external_seq, prev_control=prev_control,
                    initial_controls=initial_controls,
                )
        except Exception as exc:
            # 求解器抛异常（如 SLSQP/IPOPT 数值故障、或 CasADi 未安装）：与空轨迹同等对待，回退安全控制
            solver_exc = exc
            trajectory = Trajectory(
                controls=[], states=[], success=False, message=f"solver exception: {exc}"
            )
        if trajectory.controls:
            self._warm_start = trajectory.controls
        return trajectory, solver_exc

    def _select_first_control(self, trajectory, state, current_external, solver_failed):
        """Degradation/mode owner: choose the executed first control, applying the
        preemptive-feedback degradation entry logic. Mutates degradation
        bookkeeping in place. Bit-identical to the former inline block.
        """
        if solver_failed:
            # 求解失败（如 IPOPT 未收敛返回空轨迹）：回退无模型反馈安全控制，避免 controls[0] 越界崩溃
            return self._feedback_safe_control(state, self.dt, current_external)
        if self.preemptive_feedback and self.self_monitor is not None:
            early_mismatch = False
            if self.rc_identifier is not None and len(self.rc_identifier.history) >= 5:
                early_mismatch = float(np.mean(np.abs(self.rc_identifier.history[-5:]))) > 0.5
            high_error = (
                self.self_monitor.is_diverging()
                or self.self_monitor.predicted_error() > self.degradation_enter_error
                or early_mismatch
            )
            if high_error:
                if not self._degraded:
                    self._degraded_steps = 0
                self._degraded = True
                self._pre_degradation_mode = self.mode_manager.current_mode
            if self._degraded:
                # 持续保守安全控制，直到误差确认恢复
                if self.safe_control_mode == "worst_case" and state.x[0] > 26.5:
                    return self._worst_case_safe_control(state, self.dt, current_external)
                return self._feedback_safe_control(state, self.dt, current_external)
            return trajectory.controls[0]
        return trajectory.controls[0]

    def _apply_diagnosis_mode(self, diagnosis, state, current_external, prev_control,
                              first_control, mode):
        """Mode-switch owner: apply undecidable safe-mode entry or a suggested mode
        switch after diagnosis. Returns (first_control, mode). Mutates the mode
        manager and degradation bookkeeping in place. Bit-identical to inline.
        """
        if diagnosis.undecidable:
            # 保存降级前管理器的真实模式（而非被 mode_override/forced_mode 覆盖后的有效模式）
            pre_switch_mode = self.mode_manager.current_mode
            self.mode_manager.switch(self.diagnoser.safe_mode)
            if self.safe_control_mode == "worst_case":
                first_control = self._worst_case_safe_control(state, self.dt, current_external)
            elif self.safe_control_mode == "feedback":
                first_control = self._feedback_safe_control(state, self.dt, current_external)
            elif self.safe_control_mode == "last_valid" and prev_control is not None:
                first_control = prev_control.copy()
            else:
                safe_u = np.zeros(self.simulator.hvac.bounds().__len__() or 1)
                first_control = ControlInput(safe_u, ["Q_hvac"])
            mode = self.mode_manager.current_mode
            if not self._degraded:
                self._degraded_steps = 0
                self._degraded = True
                self._pre_degradation_mode = pre_switch_mode
        elif diagnosis.should_switch_mode and diagnosis.suggested_mode is not None:
            self.mode_manager.switch(diagnosis.suggested_mode)
            mode = self.mode_manager.current_mode
        return first_control, mode

    def optimize(
        self,
        state: SystemState,
        time_h: float,
        prev_control: ControlInput | None = None,
        forced_mode: str | None = None,
        prediction_error: float = 0.0,
        model_mismatch: float = 0.0,
    ) -> ControlDecision:
        # 曲率驱动的自适应规划（时域/舒适裕度/模式覆盖）已抽出为 owner。
        (effective_horizon, effective_comfort_min, effective_comfort_max,
         mode_override) = self._plan_horizon_and_margin(state, forced_mode)

        external_seq = self._apply_model_bias(self.external_provider.get(time_h, effective_horizon))
        current_external = external_seq[0]
        context = self.context_labeler.generate(state, current_external, time_h, self.mode_manager.current_mode)

        # 优雅恢复：自指误差回落后退出降级（带滞回 + 最短降级期 + 温度门控）已抽出为 owner。
        self._attempt_degradation_recovery(state)

        mode = forced_mode or mode_override or self.mode_manager.current_mode
        params = self.weight_mapper.map(mode, context, self.setpoints)
        if self.comfort_weight is not None:
            params.weights["comfort"] = self.comfort_weight
        if self.energy_weight is not None:
            params.weights["energy"] = self.energy_weight
        if self.smooth_weight is not None:
            params.weights["smooth"] = self.smooth_weight
        effective_metric_coupling = self.metric_coupling
        if self.adaptive_riemannian and self.self_monitor is not None:
            if self.self_monitor.predicted_error() > 0.2 or self.self_monitor.is_diverging():
                effective_metric_coupling = 0.0
        # 重整化序参量驱动的分区舒适权重（RG->MPC 耦合）已抽出为 owner。
        zone_comfort_weights = self._renormalization_zone_weights(state)
        landscape = EnergyLandscape(
            setpoints=params.setpoints,
            weights=params.weights,
            manifold=self.manifold,
            hvac=self.simulator.hvac,
            comfort_band=self.comfort_band,
            metric_coupling=effective_metric_coupling,
            metric_state_dependence=self.metric_state_dependence,
            kinetic_weight=self.kinetic_weight,
            comfort_min=effective_comfort_min,
            comfort_max=effective_comfort_max,
            below_comfort_penalty=self.below_comfort_penalty,
            peak_price_threshold=self.peak_price_threshold,
            peak_energy_penalty=self.peak_energy_penalty,
            storage_targets=self.storage_targets,
            storage_weight=self.storage_weight,
            zone_comfort_weights=zone_comfort_weights,
            zone_comfort_bounds=dict(self.zone_comfort_bounds),
        )

        solver = GeodesicSolver(
            simulator=self.simulator,
            landscape=landscape,
            horizon=effective_horizon,
            dt=self.dt,
            options=self.solver_options,
            enforce_comfort=self.enforce_comfort_constraints,
            # 硬约束用收紧后的 effective 边界：comfort_margin 同时作用于代价与硬约束
            comfort_min=effective_comfort_min,
            comfort_max=effective_comfort_max,
            zone_comfort_bounds=dict(self.zone_comfort_bounds),
            constraint_options=self.constraint_options,
            use_casadi=self.use_casadi,
            two_stage=self.use_two_stage,
            use_kinetic=self.use_kinetic,
            use_riemannian=self.use_riemannian,
            use_riemannian_control=self.use_riemannian_control,
            riemannian_control_weight=self.riemannian_control_weight,
            riemannian_strength=self.riemannian_strength,
            geodesic_penalty_weight=self.geodesic_penalty_weight,
        )
        # 求解（含 warm-start 与鲁棒分支、异常归一）已抽出为 owner。
        trajectory, solver_exc = self._solve_trajectory(
            solver, state, external_seq, prev_control, effective_horizon,
            landscape=landscape,
            effective_comfort_min=effective_comfort_min,
            effective_comfort_max=effective_comfort_max,
        )
        solver_failed = not trajectory.controls
        solver_unreliable = bool(trajectory.controls) and not trajectory.success

        # 执行控制量的选择（含预防性反馈降级进入）已抽出为 owner。
        first_control = self._select_first_control(
            trajectory, state, current_external, solver_failed
        )
        curvature = self.curvature_analyzer.analyze(landscape, state, first_control, current_external)
        self._last_curvature_min = float(curvature.stability)
        # 让曲率几何结构真正驱动引擎决策：把曲率分类映射为能量景观突变建议，
        # 并记录是否使用了协变（黎曼）Hessian，供诊断/审计消费而非悬空。
        trigger = getattr(self.diagnoser, "trigger", None)
        landscape_mutation = (
            trigger.suggest_landscape_mutation(curvature) if trigger is not None else None
        )
        self_predicted_error = self.self_monitor.predicted_error() if self.self_monitor else 0.0
        diagnosis = self.diagnoser.diagnose(
            state=state,
            external=current_external,
            prediction_error=prediction_error,
            curvature=curvature,
            model_mismatch=model_mismatch,
            current_mode=mode,
            forced_mode=forced_mode,
            self_predicted_error=self_predicted_error,
        )

        if solver_failed:
            if solver_exc is not None:
                diagnosis.details["solver_failure"] = (
                    f"solver exception: {solver_exc}; fell back to feedback safe control"
                )
            else:
                diagnosis.details["solver_failure"] = "empty trajectory; fell back to feedback safe control"
        elif solver_unreliable:
            diagnosis.details["solver_warning"] = (
                f"solver success=False ({getattr(trajectory, 'message', '') or 'no message'}); "
                "control may be unreliable"
            )

        # 曲率几何被引擎决策真正消费（记录协变 Hessian 标记/突变建议，并在
        # curvature_adaptive 下把几何突变转为下一周期裕度信号）已抽出为 owner。
        self._consume_curvature_geometry(diagnosis, curvature, landscape_mutation)

        # 连续 Geometric Indeterminacy 指数（分级响应：normal/watch/degrade）
        gb = getattr(self.diagnoser, "godel_boundary", None)
        if gb is not None and hasattr(gb, "geometric_indeterminacy_index"):
            gi = gb.geometric_indeterminacy_index(
                prediction_error,
                float(curvature.stability) if curvature is not None else 0.1,
                self_predicted_error,
            )
            diagnosis.details["geometric_indeterminacy"] = gi
            if gi["level"] == "degrade" and not diagnosis.undecidable:
                # 连续指数先于 AND 门触发降级（分级响应增强）
                diagnosis.details["gi_preemptive"] = (
                    "GI 连续指数进入降级区（AND 门未触发）")

        # undecidable 安全模式进入 / 建议模式切换已抽出为 owner。
        first_control, mode = self._apply_diagnosis_mode(
            diagnosis, state, current_external, prev_control, first_control, mode
        )

        if solver_failed:
            # 求解失败时预测状态不可信：置 None，避免把"实际状态变化"误当模型误差喂给自监控/辨识器
            predicted_next_state = None
            self._last_predicted_state = None
        else:
            predicted_next_state = trajectory.states[1] if len(trajectory.states) > 1 else state.copy()
            # 记录预测全状态：下一步实测后算 max-norm 已实现误差（与 evaluate 同口径）
            self._last_predicted_state = predicted_next_state.x.copy()

        return ControlDecision(
            control=first_control,
            trajectory=trajectory,
            mode=mode,
            confidence=diagnosis.confidence,
            diagnosis=diagnosis,
            weights=params.weights,
            predicted_next_state=predicted_next_state,
            solver_success=not (solver_failed or solver_unreliable),
        )

    def _heuristic_initial_controls(self, external_seq: Sequence[ExternalInput]) -> list[ControlInput]:
        """首次优化的启发式初值：谷时预冷、峰时降载。"""
        bounds = self.simulator.hvac.bounds()
        n_u = len(bounds)
        labels = getattr(self.simulator.hvac, "control_labels", None)
        if not labels or len(labels) != n_u:
            labels = [f"u{i}" for i in range(n_u)]
        controls: list[ControlInput] = []
        for w in external_seq:
            price = w.price
            if price < 0.5:
                # 谷时预冷
                frac = 0.7
            elif price > 1.0:
                # 峰时降载
                frac = 0.3
            else:
                # 平时适度制冷
                frac = 0.5
            u_by_unit = [frac * bounds[j][0] for j in range(n_u)]  # 每台按自身 q_min
            controls.append(ControlInput(np.array(u_by_unit), list(labels)))
        return controls

    def _rollout_mode(self, state: SystemState, time_h: float, mode: str, steps: int) -> dict:
        """Rolls out `steps` steps under a given mode for true counterfactual comparison."""
        s = state.copy()
        t = time_h
        prev = None
        temps = []
        powers = []
        prices = []
        for _ in range(steps):
            dec = self.optimize(s, t, prev_control=prev, forced_mode=mode)
            w = self.external_provider.get(t, 1)[0]
            s = self.simulator.step(s, dec.control, w, self.dt)
            temps.append(s.x[0])
            powers.append(self.simulator.hvac.electrical_power(dec.control))
            prices.append(w.price)
            prev = dec.control
            t += self.dt
        temps = np.array(temps)
        cost = float(np.sum(np.array(powers) * np.array(prices) * (self.dt)))
        viol = float(np.mean((temps > (self.comfort_max or 27.0)) | (temps < (self.comfort_min or 25.0))) * 100.0)
        peak = float(np.max(powers))
        return {"total_cost": cost, "comfort_violation": viol, "peak_power": peak}

    def observe_step(self, state: SystemState, control: ControlInput, external: ExternalInput,
                    next_state: SystemState, dt: float | None = None) -> None:
        """Records one step of real data and updates the online identifiers."""
        self._recent_step = {
            "state": state.copy(),
            "control": control.copy(),
            "external": external.copy(),
            "next_state": next_state.copy(),
            "dt": dt,
        }
        if dt is None:
            dt = self.dt or self.simulator.building.dt
        if self.rc_identifier is not None:
            self.rc_identifier.update(state, control, external, next_state, dt)
        # 已实现误差 → 置信度校准历史（conformal 分位数）
        try:
            pred_state = self._last_predicted_state
            if pred_state is not None:
                # 口径统一：与 evaluate() 的 prediction_error 同为全状态 max-norm
                self.diagnoser.confidence_evaluator.observe_error(
                    float(np.max(np.abs(next_state.x - pred_state))))
        except Exception:
            pass
        # 双区：逐区喂样本（rc_identifier 即 A 区，避免重复更新）
        for ident in getattr(self, "rc_identifiers_by_zone", {}).values():
            if ident is not self.rc_identifier:
                ident.update(state, control, external, next_state, dt)

    def _validate_config(self) -> None:
        """运行时配置一致性校验（/config 热更新后调用）；非法抛 ValueError。

        汇总各开关/权重的语义约束（与 GeodesicSolver._validate_riemannian_switches
        同源的规则 + 本引擎层的新增约束）。set_config 失败时先恢复旧值再抛 400。
        """
        if self.geodesic_penalty_weight > 0.0 and not self.use_riemannian:
            raise ValueError("geodesic_penalty_weight>0 需要 use_riemannian=True")
        for name, val in (("riemannian_strength", self.riemannian_strength),
                          ("riemannian_control_weight", self.riemannian_control_weight),
                          ("geodesic_penalty_weight", self.geodesic_penalty_weight)):
            if val < 0.0:
                raise ValueError(f"{name} 必须 >= 0, 收到 {val}")
        if self.horizon < 1:
            raise ValueError(f"horizon 必须 >= 1, 收到 {self.horizon}")
        if self.comfort_min is not None and self.comfort_max is not None                 and self.comfort_min >= self.comfort_max:
            raise ValueError(
                f"comfort_min({self.comfort_min}) >= comfort_max({self.comfort_max})")

    @staticmethod
    def _rc_params_plausible(params: dict) -> bool:
        """物理合理性门控：c/r/solar 均有限且在经验范围内。

        换算约定：parameters() 返回 R_air_times_C_air = 1/a1 = R_air·C_air、
        R_wall_times_C_air = 1/a2 = R_wall·C_air、one_over_C_air = 1/C_air、
        solar_gain_over_C_air = solar_gain/C_air。因此 R = param / C_air。
        """
        try:
            c_air = 1.0 / params["one_over_C_air"]
            r_air = params["R_air_times_C_air"] / c_air
            r_wall = params["R_wall_times_C_air"] / c_air
            solar_gain = params["solar_gain_over_C_air"] * c_air
        except Exception:
            return False
        if not all(np.isfinite([c_air, r_air, r_wall, solar_gain])):
            return False
        return (0.1 <= c_air <= 10.0 and 0.05 <= r_air <= 5.0
                and 0.05 <= r_wall <= 5.0 and 0.0 <= solar_gain <= 1.0)

    def apply_rc_identification(self, min_samples: int = 30) -> bool:
        """用 RCOnlineIdentifier 的估计结果更新控制器内部模型（带置信门控）。"""
        if not self.identification_enabled:
            return False
        # 双区：逐区辨识（分区参数 + 两区平均影子验证）
        if getattr(self, "rc_identifiers_by_zone", None):
            return self._apply_two_zone_identification(min_samples)
        if self.rc_identifier is None or len(self.rc_identifier.history) < min_samples:
            return False
        # 候选模型误差必须低于当前模型近期误差，才允许更新
        current_err = self.self_monitor.recent_mean(6) if self.self_monitor is not None else float("inf")
        # 单位统一（量纲 bug 修复）: history 存 dT/dt 残差（K/h），乘回 dt → K
        dt_for_err = self.dt or self.simulator.building.dt
        rc_err = float(np.mean(np.abs(self.rc_identifier.history[-20:]))) * dt_for_err
        if np.isfinite(current_err) and current_err > 0 and rc_err >= current_err * 0.9:
            return False
        params = self.rc_identifier.parameters()
        if not self._rc_params_plausible(params):
            return False
        c_air = 1.0 / params["one_over_C_air"]
        r_air = params["R_air_times_C_air"] / c_air
        r_wall = params["R_wall_times_C_air"] / c_air
        solar_gain = params["solar_gain_over_C_air"] * c_air

        # 影子预测验证：候选模型必须比当前模型更准
        if self._recent_step is not None:
            step = self._recent_step
            candidate_building = RCBuildingModel(
                c_air=c_air,
                c_wall=getattr(self.simulator.building, "c_wall", 4.0),
                r_air=r_air,
                r_wall=r_wall,
                solar_gain=solar_gain,
            )
            candidate_sim = Simulator(candidate_building, self.simulator.hvac)
            try:
                pred_candidate = candidate_sim.step(
                    step["state"], step["control"], step["external"], step["dt"]
                ).x[0]
                pred_current = self.simulator.step(
                    step["state"], step["control"], step["external"], step["dt"]
                ).x[0]
                err_candidate = abs(pred_candidate - step["next_state"].x[0])
                err_current = abs(pred_current - step["next_state"].x[0])
                if err_candidate >= err_current:
                    return False
            except Exception:
                return False

        old = self.simulator.building
        self.simulator.building = RCBuildingModel(
            c_air=c_air,
            c_wall=getattr(old, "c_wall", 4.0),
            r_air=r_air,
            r_wall=r_wall,
            solar_gain=solar_gain,
        )
        return True

    def _apply_two_zone_identification(self, min_samples: int) -> bool:
        """双区逐区辨识：分区参数估计 + 物理门控 + 两区平均影子验证。

        c_air/r_air 双区共享（模型结构如此），取两区估计的均值；
        r_wall_a/solar_gain_a 取 A 区、r_wall_b/solar_gain_b 取 B 区。
        """
        idents = self.rc_identifiers_by_zone
        if not idents or any(len(i.history) < min_samples for i in idents.values()):
            return False
        dt_for_err = self.dt or self.simulator.building.dt
        current_err = self.self_monitor.recent_mean(6) if self.self_monitor is not None else float("inf")
        zone_params: dict[str, dict] = {}
        for zid, ident in idents.items():
            # 单位统一（量纲 bug 修复）: K/h × dt → K
            rc_err = float(np.mean(np.abs(ident.history[-20:]))) * dt_for_err
            if np.isfinite(current_err) and current_err > 0 and rc_err >= current_err * 0.9:
                return False
            params = ident.parameters()
            if not self._rc_params_plausible(params):
                return False
            c_air = 1.0 / params["one_over_C_air"]
            r_air = params["R_air_times_C_air"] / c_air
            zone_params[zid] = {
                "c_air": c_air,
                "r_air": r_air,
                "r_wall": params["R_wall_times_C_air"] / c_air,
                "solar_gain": params["solar_gain_over_C_air"] * c_air,
            }

        old = self.simulator.building
        candidate = TwoZoneRCBuildingModel(
            c_air=float(np.mean([p["c_air"] for p in zone_params.values()])),
            c_wall=old.c_wall,
            c_partition=old.c_partition,
            r_air=float(np.mean([p["r_air"] for p in zone_params.values()])),
            r_wall_a=zone_params["A"]["r_wall"],
            r_wall_b=zone_params["B"]["r_wall"],
            r_partition=old.r_partition,
            solar_gain_a=zone_params["A"]["solar_gain"],
            solar_gain_b=zone_params["B"]["solar_gain"],
            dt=old.dt,
        )

        # 影子预测验证：候选模型两区平均一步预测误差必须低于当前模型
        if self._recent_step is not None:
            step = self._recent_step
            labels = list(step["state"].labels)
            i_a = labels.index("T_air_A")
            i_b = labels.index("T_air_B")
            try:
                candidate_sim = Simulator(candidate, self.simulator.hvac)
                pred_candidate = candidate_sim.step(
                    step["state"], step["control"], step["external"], step["dt"]
                ).x
                pred_current = self.simulator.step(
                    step["state"], step["control"], step["external"], step["dt"]
                ).x
                err_candidate = 0.5 * (
                    abs(pred_candidate[i_a] - step["next_state"].x[i_a])
                    + abs(pred_candidate[i_b] - step["next_state"].x[i_b])
                )
                err_current = 0.5 * (
                    abs(pred_current[i_a] - step["next_state"].x[i_a])
                    + abs(pred_current[i_b] - step["next_state"].x[i_b])
                )
                if err_candidate >= err_current:
                    return False
            except Exception:
                return False

        self.simulator.building = candidate
        return True

    def _apply_model_bias(self, external_seq: Sequence[ExternalInput]) -> list[ExternalInput]:
        """在线辨识的模型偏置应用（量纲修正后的语义）。

        model_bias 来自 RLS 对 signed_error = (actual − predicted) 的一维估计，
        单位是 **K/步**（空气温度的每步偏差），物理含义是"模型系统性低估室温"。
        它不属于任何外部输入通道——把它加到 occ(kW) 是量纲错误（0.3K 会变成
        43% 的内热扰动，过度矫正）。

        正确语义：作为**状态预测的加性偏置**直接作用于预测的 T_air——模型误差
        未必来自内热，加性偏置是更中性的表达。实现方式：把 K/步偏置折算为等效
        的外部得热增量 ΔQ = c_air·bias/dt（kW），保留原"改外部输入"的机制但
        量纲正确；等效地每步预测 T_air 会多出 bias(K)。
        """
        if abs(self.model_bias) < 1e-6:
            return list(external_seq)
        dt = self.dt or self.simulator.building.dt
        c_air = getattr(self.simulator.building, "c_air", 1.0)
        delta_q = c_air * self.model_bias / dt   # K/步 → kW（等效内热增量）
        corrected = []
        for w in external_seq:
            arr = w.w.copy()
            for i, lab in enumerate(w.labels):
                if lab.startswith("occ"):
                    arr[i] += delta_q
            corrected.append(ExternalInput(arr, list(w.labels)))
        return corrected

    def identification_trusted(self, min_samples: int = 30) -> bool:
        """Identification trust check: enough samples, physically plausible parameters, and small recent error."""
        # 双区：逐区都可信才算可信（任一区参数漂移即关闭信任，保守方向）
        idents = (list(self.rc_identifiers_by_zone.values())
                  if getattr(self, "rc_identifiers_by_zone", None)
                  else [self.rc_identifier])
        if not idents or any(i is None for i in idents):
            return False
        if any(len(i.history) < min_samples for i in idents):
            return False
        recent_err = max(float(np.mean(np.abs(i.history[-10:]))) for i in idents)
        if recent_err > 1.0:
            return False
        for ident in idents:
            if not self._rc_params_plausible(ident.parameters()):
                return False
        return True

    def _worst_case_safe_control(self, state: SystemState, dt: float, external: ExternalInput | None = None) -> ControlInput:
        """Delegate to the SafeController owner (kept as a thin wrapper for callers/tests)."""
        return self.safe_controller.worst_case(state, dt, external)

    def _safe_cool_cap(self, err: float, q_max: float) -> float:
        """Delegate to the SafeController owner."""
        return self.safe_controller.cool_cap(err, q_max)

    def _feedback_safe_control(self, state: SystemState, dt: float, external: ExternalInput | None = None) -> ControlInput:
        """Delegate to the SafeController owner (kept as a thin wrapper for callers/tests)."""
        return self.safe_controller.feedback(state, dt, external)

    def _rollout_feedback(self, state: SystemState, time_h: float, steps: int) -> dict:
        """Counterfactual: short-horizon rollout of the model-free feedback safe control."""
        s = state.copy()
        t = time_h
        temps = []
        powers = []
        prices = []
        for _ in range(steps):
            w = self.external_provider.get(t, 1)[0]
            control = self._feedback_safe_control(s, self.dt, w)
            s = self.simulator.step(s, control, w, self.dt)
            temps.append(s.x[0])
            powers.append(self.simulator.hvac.electrical_power(control))
            prices.append(w.price)
            t += self.dt
        temps = np.array(temps)
        cost = float(np.sum(np.array(powers) * np.array(prices) * (self.dt)))
        viol = float(np.mean((temps > (self.comfort_max or 27.0)) | (temps < (self.comfort_min or 25.0))) * 100.0)
        peak = float(np.max(powers))
        return {"total_cost": cost, "comfort_violation": viol, "peak_power": peak}

    def _snapshot_control_state(self) -> dict:
        """Snapshots all mutable controller state so counterfactual (hypothetical) rollouts can restore it and never pollute the real control."""
        return {
            "warm_start": self._warm_start,
            "last_curvature_min": self._last_curvature_min,
            "safe_controller": self.safe_controller.snapshot(),
            "degraded": self._degraded,
            "pre_degradation_mode": self._pre_degradation_mode,
            "low_error_streak": self._low_error_streak,
            "degraded_steps": self._degraded_steps,
            "recent_step": self._recent_step,
            "mode": self.mode_manager.current_mode,
        }

    def _restore_control_state(self, snap: dict) -> None:
        """Restores the controller state snapshot, undoing all side effects of counterfactual rollouts."""
        self._warm_start = snap["warm_start"]
        self._last_curvature_min = snap["last_curvature_min"]
        if "safe_controller" in snap:
            self.safe_controller.restore(snap["safe_controller"])
        self._degraded = snap["degraded"]
        self._pre_degradation_mode = snap["pre_degradation_mode"]
        self._low_error_streak = snap["low_error_streak"]
        self._degraded_steps = snap["degraded_steps"]
        self._recent_step = snap["recent_step"]
        self.mode_manager.current_mode = snap["mode"]

    def _run_counterfactual_undecidable(self, state: SystemState, time_h: float) -> dict:
        """Counterfactual under undecidable: keep trusting the model vs model-free degradation."""
        snap = self._snapshot_control_state()
        try:
            steps = max(1, self.counterfactual_horizon)
            result_model = self._rollout_mode(state, time_h, "comfort", steps)
            result_feedback = self._rollout_feedback(state, time_h, steps)
            out = {
                "trust_model": result_model,
                "feedback_safe": result_feedback,
                "note": f"不可判定反事实（{steps} 步）：继续信任模型 vs 无模型反馈降级",
            }
            if self.causal_scm is not None:
                try:
                    out["causal_effect"] = {
                        "temperature": self.causal_scm.effect(
                            "temperature", {"mode": 0.0}, {"mode": 1.0}
                        ),
                        "cost": self.causal_scm.effect(
                            "cost", {"mode": 0.0}, {"mode": 1.0}
                        ),
                    }
                except Exception:
                    pass
            return out
        finally:
            self._restore_control_state(snap)

    def set_causal_scm_from_data(self, data: list) -> None:
        """Fits an SCM from intervention data and sets it as the current causal model."""
        self.causal_scm = DataDrivenSCM().fit(data).to_scm()

    def _build_default_scm(self) -> StructuralCausalModel:
        """Default structural causal model: auto-generated from the current physical model."""
        return build_rc_scm(self.simulator.building, self.simulator.hvac)

    def _run_counterfactual(
        self,
        state: SystemState,
        time_h: float,
        current_mode: str,
        suggested_mode: str,
    ) -> dict:
        """Runs a short-horizon counterfactual after a mode switch, using true rolling optimization."""
        snap = self._snapshot_control_state()
        try:
            steps = max(1, self.counterfactual_horizon)
            try:
                result_current = self._rollout_mode(state, time_h, current_mode, steps)
            except Exception:
                result_current = {"total_cost": float("nan"), "comfort_violation": float("nan"), "peak_power": float("nan")}
            try:
                result_suggested = self._rollout_mode(state, time_h, suggested_mode, steps)
            except Exception:
                result_suggested = {"total_cost": float("nan"), "comfort_violation": float("nan"), "peak_power": float("nan")}
            return {
                "keep_current": result_current,
                "suggested_mode": result_suggested,
                "note": f"短时域真实滚动反事实对比（{steps} 步）",
            }
        finally:
            self._restore_control_state(snap)

    def run_closed_loop(
        self,
        initial_state: SystemState,
        start_time_h: float = 8.0,
        steps: int = 24,
        step_h: float = 1.0 / 12.0,
        plant_provider: ExternalInputProvider | None = None,
    ) -> list[ControlDecision]:
        """Rolling-horizon closed-loop simulation for demos and tests.

        Computes the prediction error after each step and passes it to the next optimization.
        """
        state = initial_state.copy()
        time_h = start_time_h
        prev_control: ControlInput | None = None
        prediction_error = 0.0
        decisions: list[ControlDecision] = []

        for _ in range(steps):
            old_mode = self.mode_manager.current_mode
            decision = self.optimize(
                state,
                time_h,
                prev_control=prev_control,
                prediction_error=prediction_error,
            )
            decisions.append(decision)
            if (
                self.counterfactual_enabled
                and (
                    decision.diagnosis.should_switch_mode
                    or decision.diagnosis.undecidable
                )
            ):
                try:
                    if decision.diagnosis.undecidable and not decision.diagnosis.should_switch_mode:
                        cf = self._run_counterfactual_undecidable(state, time_h)
                    elif decision.diagnosis.suggested_mode is not None:
                        cf = self._run_counterfactual(
                            state,
                            time_h,
                            old_mode,
                            decision.diagnosis.suggested_mode,
                        )
                    else:
                        cf = {}
                except Exception as exc:
                    # 反事实是"假设性"分析，失败不应杀死真实闭环
                    cf = {"error": str(exc)}
                if cf:
                    decision.diagnosis.details["counterfactual"] = cf
            # 重整化群流：多区域宏观异常检测 + 序参量识别进入主循环
            if self.renormalization_enabled and state.dim > 2:
                air_idx = [i for i, lab in enumerate(state.labels) if lab.startswith("T_air")]
                if air_idx:
                    self.renormalization_flow.labels = [state.labels[i] for i in air_idx]
                    rg = self.renormalization_flow.analyze([state.x[i] for i in air_idx])
                    decision.diagnosis.details["renormalization"] = rg
            # 执行第一个控制量并推进真实世界（以仿真器代替）
            actual_provider = plant_provider if plant_provider is not None else self.external_provider
            external = actual_provider.get(time_h, 1)[0]
            predicted = decision.predicted_next_state
            state_before = state.copy()
            state = self.simulator.step(state, decision.control, external, step_h)
            if predicted is not None:
                prediction_error = float(np.max(np.abs(predicted.x - state.x)))
                signed_error = float(state.x[0] - predicted.x[0])
                self.online_identifier.update(np.array([1.0]), signed_error)
                self.model_bias = float(self.online_identifier.theta[0])
            self.self_monitor.update(prediction_error)
            self.observe_step(
                state_before, decision.control, external, state, step_h
            )
            self.apply_rc_identification(min_samples=30)
            prev_control = decision.control
            time_h += step_h

        return decisions
