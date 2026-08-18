"""GCCM-BE 顶层引擎：周期滚动优化与事件驱动决策编排。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .decision.diagnoser import DecisionDiagnoser
from .decision.self_monitor import SelfMonitor
from .causal.counterfactual import CounterfactualAnalyzer
from .causal.scm import StructuralCausalModel, build_rc_scm
from .causal.data_driven import DataDrivenSCM
from .multiscale.renormalization import RenormalizationFlow
from .geometry.curvature import CurvatureAnalyzer
from .geometry.geodesic import GeodesicSolver
from .geometry.casadi_robust_solver import CasadiRobustGeodesicSolver
from .geometry.landscape import EnergyLandscape
from .geometry.manifold import StateManifold
from .normative.context import ContextLabeler
from .normative.modes import ModeManager
from .normative.weights import WeightMapper
from .physics.external import ExternalInputProvider, MockExternalInputProvider
from .physics.online_id import OnlineIdentifier, RCOnlineIdentifier
from .physics.models import HVACModel, RCBuildingModel, Simulator
from .types import ControlDecision, ControlInput, ExternalInput, SystemState, Trajectory


@dataclass
class GCCMEngine:
    """组合五层模块，提供统一控制决策入口。"""

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
    dt: Optional[float] = None
    solver_options: dict = field(default_factory=lambda: {"maxiter": 200, "ftol": 1e-8})
    comfort_band: float = 1.0
    comfort_margin: float = 0.0
    noise_adaptive_margin: bool = False
    comfort_min: Optional[float] = None
    comfort_max: Optional[float] = None
    below_comfort_penalty: float = 0.0
    peak_price_threshold: float = 1.0
    peak_energy_penalty: float = 1.0
    storage_targets: Dict[str, float] = field(default_factory=dict)
    storage_weight: float = 0.0
    comfort_weight: Optional[float] = None
    energy_weight: Optional[float] = None
    smooth_weight: Optional[float] = None
    enforce_comfort_constraints: bool = False
    use_casadi: bool = False
    use_casadi_robust: bool = False
    robust_scenarios: List[Simulator] = field(default_factory=list)
    robust_enforce_lower_bound: bool = True
    robust_heating_cost_factor: float = 1.0
    use_riemannian: bool = False
    use_riemannian_control: bool = False
    riemannian_control_weight: float = 1.0
    riemannian_strength: float = 1.0
    metric_coupling: float = 0.0
    metric_state_dependence: float = 0.0
    geodesic_penalty_weight: float = 0.0
    adaptive_riemannian: bool = False
    use_two_stage: bool = False
    use_kinetic: bool = True
    counterfactual_enabled: bool = False
    counterfactual_horizon: int = 4
    causal_scm: Optional[object] = None
    safe_control_mode: str = "zero"
    preemptive_feedback: bool = False
    online_identifier: OnlineIdentifier = None  # type: ignore[assignment]
    rc_identifier: RCOnlineIdentifier = None  # type: ignore[assignment]
    identification_enabled: bool = True
    model_bias: float = 0.0
    safe_kp: float = 2.0
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
    curvature_adaptive: bool = False
    nominal_horizon: Optional[int] = None
    min_horizon: Optional[int] = None
    curvature_flat_threshold: float = 1e-6
    curvature_singular_threshold: float = -1e-3
    tightened_comfort_max: Optional[float] = None
    constraint_options: dict = field(default_factory=lambda: {"maxiter": 30, "ftol": 1e-5})
    setpoints: Dict[str, float] = field(default_factory=lambda: {"T_air": 24.0})

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
            self.curvature_analyzer = CurvatureAnalyzer()
        if self.diagnoser is None:
            self.diagnoser = DecisionDiagnoser()
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
        self._warm_start = None
        self._last_curvature_min: Optional[float] = None
        self._safe_integral: Dict[int, float] = {}
        self._safe_prev_error: Dict[int, float] = {}
        self._degraded = False
        self._pre_degradation_mode: Optional[str] = None
        self._recent_step: Optional[dict] = None
        self._low_error_streak = 0
        self._degraded_steps = 0
        if self.nominal_horizon is None:
            self.nominal_horizon = self.horizon
        if self.min_horizon is None:
            self.min_horizon = max(4, self.nominal_horizon // 2)

    def optimize(
        self,
        state: SystemState,
        time_h: float,
        prev_control: Optional[ControlInput] = None,
        forced_mode: Optional[str] = None,
        prediction_error: float = 0.0,
        model_mismatch: float = 0.0,
    ) -> ControlDecision:
        # 曲率驱动的自适应：根据上一周期局部几何稳定性调整时域与安全裕度
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
        if margin > 0.0:
            if effective_comfort_min is not None:
                effective_comfort_min = effective_comfort_min + margin
            if effective_comfort_max is not None:
                effective_comfort_max = effective_comfort_max - margin

        external_seq = self._apply_model_bias(self.external_provider.get(time_h, effective_horizon))
        current_external = external_seq[0]
        context = self.context_labeler.generate(state, current_external, time_h, self.mode_manager.current_mode)

        # 优雅恢复：自指误差回落后退出降级（带滞回 + 最短降级期 + 温度门控）
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
        landscape = EnergyLandscape(
            setpoints=params.setpoints,
            weights=params.weights,
            manifold=self.manifold,
            hvac=self.simulator.hvac,
            comfort_band=self.comfort_band,
            metric_coupling=effective_metric_coupling,
            comfort_min=effective_comfort_min,
            comfort_max=effective_comfort_max,
            below_comfort_penalty=self.below_comfort_penalty,
            peak_price_threshold=self.peak_price_threshold,
            peak_energy_penalty=self.peak_energy_penalty,
            storage_targets=self.storage_targets,
            storage_weight=self.storage_weight,
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
                    state,
                    external_seq,
                    prev_control=prev_control,
                    initial_controls=None,
                )
                initial_controls = nominal_traj.controls if nominal_traj.controls else None
                trajectory = robust_solver.solve(
                    state,
                    external_seq,
                    prev_control=prev_control,
                    initial_controls=initial_controls,
                )
                if not trajectory.controls:
                    trajectory = solver.solve(
                        state,
                        external_seq,
                        prev_control=prev_control,
                        initial_controls=initial_controls,
                    )
            else:
                trajectory = solver.solve(
                    state,
                    external_seq,
                    prev_control=prev_control,
                    initial_controls=initial_controls,
                )
        except Exception as exc:  # noqa: BLE001
            # 求解器抛异常（如 SLSQP/IPOPT 数值故障）：与空轨迹同等对待，回退安全控制
            solver_exc = exc
            trajectory = Trajectory(
                controls=[], states=[], success=False, message=f"solver exception: {exc}"
            )
        if trajectory.controls:
            self._warm_start = trajectory.controls
        solver_failed = not trajectory.controls
        solver_unreliable = bool(trajectory.controls) and not trajectory.success

        if solver_failed:
            # 求解失败（如 IPOPT 未收敛返回空轨迹）：回退无模型反馈安全控制，避免 controls[0] 越界崩溃
            first_control = self._feedback_safe_control(state, self.dt, current_external)
        elif self.preemptive_feedback and self.self_monitor is not None:
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
                    first_control = self._worst_case_safe_control(state, self.dt, current_external)
                else:
                    first_control = self._feedback_safe_control(state, self.dt, current_external)
            else:
                first_control = trajectory.controls[0]
        else:
            first_control = trajectory.controls[0]
        curvature = self.curvature_analyzer.analyze(landscape, state, first_control, current_external)
        self._last_curvature_min = float(curvature.stability)
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

        if solver_failed:
            # 求解失败时预测状态不可信：置 None，避免把"实际状态变化"误当模型误差喂给自监控/辨识器
            predicted_next_state = None
        else:
            predicted_next_state = trajectory.states[1] if len(trajectory.states) > 1 else state.copy()

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

    def _heuristic_initial_controls(self, external_seq: Sequence[ExternalInput]) -> List[ControlInput]:
        """首次优化的启发式初值：谷时预冷、峰时降载。"""
        q_min = self.simulator.hvac.q_min
        bounds = self.simulator.hvac.bounds()
        n_u = len(bounds)
        labels = getattr(self.simulator.hvac, "control_labels", None)
        if not labels or len(labels) != n_u:
            labels = [f"u{i}" for i in range(n_u)]
        controls: List[ControlInput] = []
        for w in external_seq:
            price = w.price
            if price < 0.5:
                # 谷时预冷
                u = 0.7 * q_min
            elif price > 1.0:
                # 峰时降载
                u = 0.3 * q_min
            else:
                # 平时适度制冷
                u = 0.5 * q_min
            controls.append(ControlInput(np.full(n_u, u), list(labels)))
        return controls

    def _rollout_mode(self, state: SystemState, time_h: float, mode: str, steps: int) -> dict:
        """在指定模式下滚动仿真 steps 步，用于真正的反事实对比。"""
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
                    next_state: SystemState, dt: float = 0.25) -> None:
        """记录一步真实数据，并更新在线辨识器。"""
        self._recent_step = {
            "state": state.copy(),
            "control": control.copy(),
            "external": external.copy(),
            "next_state": next_state.copy(),
            "dt": dt,
        }
        if self.rc_identifier is not None:
            self.rc_identifier.update(state, control, external, next_state, dt)

    def apply_rc_identification(self, min_samples: int = 30) -> bool:
        """用 RCOnlineIdentifier 的估计结果更新控制器内部模型（带置信门控）。"""
        if not self.identification_enabled:
            return False
        if self.rc_identifier is None or len(self.rc_identifier.history) < min_samples:
            return False
        # 候选模型误差必须低于当前模型近期误差，才允许更新
        current_err = self.self_monitor.recent_mean(6) if self.self_monitor is not None else float("inf")
        rc_err = float(np.mean(np.abs(self.rc_identifier.history[-20:])))
        if np.isfinite(current_err) and rc_err >= current_err * 0.9:
            return False
        params = self.rc_identifier.parameters()
        try:
            c_air = 1.0 / params["one_over_C_air"]
            r_air = 1.0 / (params["R_air_times_C_air"] * c_air)
            r_wall = 1.0 / (params["R_wall_times_C_air"] * c_air)
            solar_gain = params["solar_gain_over_C_air"] * c_air
            if not (np.isfinite(c_air) and np.isfinite(r_air) and np.isfinite(r_wall) and np.isfinite(solar_gain)):
                return False
            if not (0.1 <= c_air <= 10.0):
                return False
            if not (0.05 <= r_air <= 5.0):
                return False
            if not (0.05 <= r_wall <= 5.0):
                return False
            if not (0.0 <= solar_gain <= 1.0):
                return False
        except Exception:
            return False

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

    def _apply_model_bias(self, external_seq: Sequence[ExternalInput]) -> List[ExternalInput]:
        """将在线辨识得到的模型偏差叠加到预测外部输入上。"""
        if abs(self.model_bias) < 1e-6:
            return list(external_seq)
        corrected = []
        for w in external_seq:
            arr = w.w.copy()
            # 对 occupancy / internal heat 维度叠加偏差
            for i, lab in enumerate(w.labels):
                if lab.startswith("occ"):
                    arr[i] += self.model_bias
            corrected.append(ExternalInput(arr, list(w.labels)))
        return corrected

    def identification_trusted(self, min_samples: int = 30) -> bool:
        """辨识结果可信度检查：样本足够、参数物理合理、近期误差不大。"""
        if self.rc_identifier is None or len(self.rc_identifier.history) < min_samples:
            return False
        recent_err = float(np.mean(np.abs(self.rc_identifier.history[-10:])))
        if recent_err > 1.0:
            return False
        try:
            params = self.rc_identifier.parameters()
            c_air = 1.0 / params["one_over_C_air"]
            r_air = 1.0 / (params["R_air_times_C_air"] * c_air)
            r_wall = 1.0 / (params["R_wall_times_C_air"] * c_air)
            solar_gain = params["solar_gain_over_C_air"] * c_air
        except Exception:
            return False
        if not all(np.isfinite([c_air, r_air, r_wall, solar_gain])):
            return False
        # 物理合理范围
        if not (0.1 <= c_air <= 10.0):
            return False
        if not (0.05 <= r_air <= 5.0):
            return False
        if not (0.05 <= r_wall <= 5.0):
            return False
        if not (0.0 <= solar_gain <= 1.0):
            return False
        return True

    def _worst_case_safe_control(self, state: SystemState, dt: float, external: Optional[ExternalInput] = None) -> ControlInput:
        """保守安全控制：优先用实际天气+安全裕度，否则用物理上界。"""
        building = self.simulator.building
        bounds = self.simulator.hvac.bounds()
        n_u = len(bounds)
        labels = getattr(self.simulator.hvac, "control_labels", None)
        if not labels or len(labels) != n_u:
            labels = [f"u{i}" for i in range(n_u)]

        if not isinstance(building, RCBuildingModel):
            # 多区域/未知模型暂时回退到反馈 PI
            return self._feedback_safe_control(state, dt)

        T_air, T_wall = state.x[0], state.x[1]
        target = (self.comfort_max or 27.0) - 0.1
        setpoint = self.setpoints.get("T_air", 26.0)

        # 优先使用在线辨识得到的真实参数重建保守功率
        C = building.c_air
        r_air = building.r_air
        r_wall = building.r_wall
        solar_gain = building.solar_gain
        if not self.identification_trusted(min_samples=30):
            # 辨识不可信：不退回慢速反馈，而是使用保守参数（小热阻/小热容 -> 偏强制冷）
            C = min(getattr(building, "c_air", 0.6), 0.4)
            r_air = min(getattr(building, "r_air", 0.8), 0.3)
            r_wall = min(getattr(building, "r_wall", 2.0), 0.5)
            solar_gain = max(getattr(building, "solar_gain", 0.05), 0.1)
        else:
            try:
                params = self.rc_identifier.parameters()
                C = 1.0 / params["one_over_C_air"]
                r_air = 1.0 / (params["R_air_times_C_air"] * C)
                r_wall = 1.0 / (params["R_wall_times_C_air"] * C)
                solar_gain = params["solar_gain_over_C_air"] * C
            except Exception:
                C = min(getattr(building, "c_air", 0.6), 0.4)
                r_air = min(getattr(building, "r_air", 0.8), 0.3)
                r_wall = min(getattr(building, "r_wall", 2.0), 0.5)
                solar_gain = max(getattr(building, "solar_gain", 0.05), 0.1)

        # 温度不高时不要激进制冷，避免过冷振荡
        if T_air <= setpoint + 0.5:
            fb = self.safe_kp * (setpoint - T_air)
            q = min(0.0, float(np.clip(fb, bounds[0][0], 0.0)))
            return ControlInput(np.array([q]), list(labels))

        # 无制冷时保守情况下的一步后温度
        if external is not None and external.w.size >= 3:
            t_out_eff = external.w[0] + self.safe_weather_margin
            solar_eff = external.w[1] * 1.2 + 0.05
            occ_eff = external.w[2] + 0.2
        else:
            t_out_eff = self.worst_case_outdoor_temp
            solar_eff = self.worst_case_solar
            occ_eff = self.worst_case_occ
        q_none = (
            (T_wall - T_air) / r_air
            + (t_out_eff - T_air) / r_wall
            + solar_gain * solar_eff
            + occ_eff
        )
        T_nocool = T_air + dt / C * q_none
        required = C / dt * (target - T_nocool)
        # 反馈修正项（越热 fb 越负 → 制冷越强）
        fb = self.safe_kp * (setpoint - T_air)
        q = min(0.0, required + fb)
        # 防止一步过冷到舒适下限以下
        t_low = (self.comfort_min or 25.0) + 0.2
        lower_bound = C / dt * (t_low - T_nocool)
        q = max(q, lower_bound)
        # 再限制最大安全制冷功率
        cool_cap = self._safe_cool_cap(max(0.0, T_air - setpoint), bounds[0][1])
        q = float(np.clip(q, -cool_cap, 0.0))
        return ControlInput(np.array([q]), list(labels))

    def _safe_cool_cap(self, err: float, q_max: float) -> float:
        """根据温度偏差动态限制安全制冷功率，避免过冷。"""
        if err > 2.0:
            return min(q_max, 6.0)
        if err > 1.0:
            return min(q_max, 4.0)
        return min(q_max, 2.5)

    def _feedback_safe_control(self, state: SystemState, dt: float, external: Optional[ExternalInput] = None) -> ControlInput:
        """无模型反馈安全控制：PI + 天气前馈，不依赖模型预测。"""
        bounds = self.simulator.hvac.bounds()
        n_u = len(bounds)
        air_idx = [i for i, lab in enumerate(state.labels) if lab.startswith("T_air")]
        if not air_idx:
            air_idx = [0]
        safe_u = np.zeros(n_u)
        for j, idx in enumerate(air_idx[:n_u]):
            setpoint = self.setpoints.get(state.labels[idx], 26.0)
            err = state.x[idx] - setpoint
            # 低温保护：低于舒适下限时主动加热
            if self.comfort_min is not None and state.x[idx] < self.comfort_min:
                heat = self.safe_kp * (self.comfort_min - state.x[idx])
                safe_u[j] = float(np.clip(heat, 0.0, bounds[j][1]))
                self._safe_integral[idx] = 0.0
                continue
            # 天气前馈：外部高温时主动增加制冷
            ff = 0.0
            if external is not None and external.w.size > 0:
                ff = self.safe_feedforward_gain * max(0.0, external.w[0] - 26.0)
            # 防过冷：接近或低于设定点时停止制冷
            if err <= 0.2 and ff <= 0.0:
                self._safe_integral[idx] = 0.0
                safe_u[j] = 0.0
                continue
            integral = self._safe_integral.get(idx, 0.0) + err * dt
            integral = float(np.clip(integral, -self.safe_integral_limit, self.safe_integral_limit))
            self._safe_integral[idx] = integral
            cool_cap = self._safe_cool_cap(err, bounds[j][1])
            cool = float(np.clip(self.safe_kp * err + self.safe_ki * integral + ff, 0.0, cool_cap))
            safe_u[j] = -cool
        labels = getattr(self.simulator.hvac, "control_labels", None)
        if not labels or len(labels) != n_u:
            labels = [f"u{i}" for i in range(n_u)]
        return ControlInput(safe_u, list(labels))

    def _rollout_feedback(self, state: SystemState, time_h: float, steps: int) -> dict:
        """反事实：无模型反馈安全控制的短时域滚动。"""
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
        """快照控制器全部可变状态，供反事实等"假设性"滚动前后恢复，避免污染真实控制。"""
        return {
            "warm_start": self._warm_start,
            "last_curvature_min": self._last_curvature_min,
            "safe_integral": dict(self._safe_integral),
            "safe_prev_error": dict(self._safe_prev_error),
            "degraded": self._degraded,
            "pre_degradation_mode": self._pre_degradation_mode,
            "low_error_streak": self._low_error_streak,
            "degraded_steps": self._degraded_steps,
            "recent_step": self._recent_step,
            "mode": self.mode_manager.current_mode,
        }

    def _restore_control_state(self, snap: dict) -> None:
        """恢复控制器状态快照，撤销反事实滚动产生的所有副作用。"""
        self._warm_start = snap["warm_start"]
        self._last_curvature_min = snap["last_curvature_min"]
        self._safe_integral = dict(snap["safe_integral"])
        self._safe_prev_error = dict(snap["safe_prev_error"])
        self._degraded = snap["degraded"]
        self._pre_degradation_mode = snap["pre_degradation_mode"]
        self._low_error_streak = snap["low_error_streak"]
        self._degraded_steps = snap["degraded_steps"]
        self._recent_step = snap["recent_step"]
        self.mode_manager.current_mode = snap["mode"]

    def _run_counterfactual_undecidable(self, state: SystemState, time_h: float) -> dict:
        """不可判定时的反事实：继续信任模型 vs 无模型降级。"""
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
        """用干预实验数据拟合 SCM 并设为当前因果模型。"""
        self.causal_scm = DataDrivenSCM().fit(data).to_scm()

    def _build_default_scm(self) -> StructuralCausalModel:
        """默认结构因果模型：从当前物理模型自动生成。"""
        return build_rc_scm(self.simulator.building, self.simulator.hvac)

    def _run_counterfactual(
        self,
        state: SystemState,
        time_h: float,
        current_mode: str,
        suggested_mode: str,
    ) -> dict:
        """模式切换后运行短时域反事实，使用真正的滚动优化。"""
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
        plant_provider: Optional[ExternalInputProvider] = None,
    ) -> List[ControlDecision]:
        """滚动时域闭环仿真，用于演示与测试。

        每次执行后计算预测误差，并在下一次优化时传递给决策层。
        """
        state = initial_state.copy()
        time_h = start_time_h
        prev_control: Optional[ControlInput] = None
        prediction_error = 0.0
        decisions: List[ControlDecision] = []

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
                except Exception as exc:  # noqa: BLE001
                    # 反事实是"假设性"分析，失败不应杀死真实闭环
                    cf = {"error": str(exc)}
                if cf:
                    decision.diagnosis.details["counterfactual"] = cf
            # 重整化群流：多区域宏观异常检测进入主循环
            if self.renormalization_enabled and state.dim > 2:
                air_idx = [i for i, lab in enumerate(state.labels) if lab.startswith("T_air")]
                if air_idx:
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
