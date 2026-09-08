"""Model introspection registry — the SINGLE SOURCE OF TRUTH the Web dashboard
reads to render controls, monitors and documentation.

WHY THIS EXISTS
---------------
The user requirement is: *every time the model changes, the visualization page
changes with it.* To make that maintainable instead of hand-editing HTML, the
dashboard is DATA-DRIVEN: it fetches this registry at runtime and renders the
model's capabilities dynamically. So the maintenance rule collapses to a single,
enforceable action:

    When you add/rename/remove an engine switch, weight, physics model, control
    mode, or diagnostic trigger, update the corresponding entry HERE.

`tests/test_introspection.py` guards this: it cross-checks that every switch /
weight field named here actually exists on ``GCCMEngine`` (and flags engine
fields that look like switches/weights but are undocumented), so a model change
that forgets to update this registry fails CI.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any

from ..engine import GCCMEngine

# --- Riemannian / geometry switches (the four orthogonal mechanisms) ---
RIEMANNIAN_SWITCHES: list[dict[str, Any]] = [
    {
        "field": "use_kinetic",
        "type": "bool",
        "strength_field": "kinetic_weight",
        "label": "动能项（状态平滑）",
        "help": "作用量 ½żᵀg(z)ż 惩罚状态速度（温度爬升），产品面向的状态平滑正则子。",
    },
    {
        "field": "use_riemannian",
        "type": "bool",
        "strength_field": "riemannian_strength",
        "label": "黎曼动力学修正（Christoffel）",
        "help": "在动力学步进中加入 Christoffel 联络修正，使推进沿流形测地线。",
    },
    {
        "field": "use_riemannian_control",
        "type": "bool",
        "strength_field": "riemannian_control_weight",
        "label": "度量范数控制代价",
        "help": "用度量范数度量控制努力，度量耦合下惩罚控制量。",
    },
    {
        "field": "geodesic_penalty_weight",
        "type": "float",
        "strength_field": None,
        "label": "测地线软惩罚权重",
        "help": "偏离测地线的软惩罚（>0 需 use_riemannian 开启）。",
    },
]

# --- objective weights ---
WEIGHTS: list[dict[str, Any]] = [
    {"field": "comfort_weight", "label": "舒适权重", "help": "温度偏离设定点的惩罚强度。"},
    {"field": "energy_weight", "label": "能耗权重", "help": "电费（功率×电价）惩罚强度。"},
    {"field": "smooth_weight", "label": "平滑权重", "help": "相邻控制量变化的惩罚强度。"},
    {"field": "peak_energy_penalty", "label": "峰值能耗惩罚", "help": "峰时电价超阈值时的额外能耗惩罚。"},
    {"field": "below_comfort_penalty", "label": "低于舒适下限惩罚", "help": "温度低于舒适下限的额外惩罚。"},
]

# --- other controller toggles surfaced for operation ---
CONTROLLER_TOGGLES: list[dict[str, Any]] = [
    {"field": "enforce_comfort_constraints", "type": "bool", "label": "硬舒适约束",
     "help": "在求解器里以硬约束强制温度落在舒适区间。"},
    {"field": "curvature_adaptive", "type": "bool", "label": "曲率自适应时域",
     "help": "根据上周期局部几何稳定性调整预测时域与安全裕度。"},
    {"field": "covariant_curvature", "type": "bool", "label": "协变（黎曼）Hessian",
     "help": "曲率分析用协变 Hessian 而非欧氏 Hessian。"},
    {"field": "renormalization_weighting", "type": "bool", "label": "重整化分区加权（削峰）",
     "help": "把序参量相关度映射为分区舒适权重，多区受限时削峰。",
     "strength_field": "renormalization_weight_gain"},
    {"field": "preemptive_feedback", "type": "bool", "label": "预防性反馈降级",
     "help": "自指误差高时提前切换到安全反馈控制。"},
    {"field": "confidence_calibrated", "type": "bool", "label": "置信度校准",
     "help": "以已实现预测误差的滚动分位数为尺度（conformal 式），关闭则用固定尺度。"},
    {"field": "noise_adaptive_margin", "type": "bool", "label": "噪声自适应裕度",
     "help": "预测误差大时自动收紧舒适裕度。"},
]

# --- control modes ---
MODES: list[dict[str, str]] = [
    {"id": "comfort", "label": "舒适优先", "help": "优先保证温度贴近设定点。"},
    {"id": "balanced", "label": "均衡", "help": "舒适与能耗折中。"},
    {"id": "energy", "label": "节能", "help": "优先降低电费。"},
    {"id": "demand_response", "label": "需求响应", "help": "峰时削峰/降载。"},
]

# --- physics models available ---
PHYSICS_MODELS: list[dict[str, str]] = [
    {"id": "RCBuildingModel", "label": "单区 RC（2 状态）",
     "help": "T_air/T_wall 单区热网络，线性。"},
    {"id": "TwoZoneRCBuildingModel", "label": "双区 RC（5 状态）",
     "help": "两区+隔墙耦合。"},
    {"id": "ThreeRCBuildingModel", "label": "三容 RC（3 状态）",
     "help": "T_air/T_wall/T_furn，家具热容。"},
    {"id": "NonlinearRCBuildingModel", "label": "非线性 RC（3 状态）",
     "help": "含开窗自然通风/除湿/变风量非线性。"},
    {"id": "DataCenterCoolingModel", "label": "数据中心冷却",
     "help": "服务器发热 + 冷却回路。"},
]

# --- diagnostic outputs surfaced in the monitor ---
DIAGNOSTIC_FIELDS: list[dict[str, str]] = [
    {"id": "mode", "label": "当前模式"},
    {"id": "confidence", "label": "置信度"},
    {"id": "undecidable", "label": "不可判定（哥德尔边界）"},
    {"id": "solver_success", "label": "求解成功"},
    {"id": "total_cost", "label": "轨迹总代价"},
    {"id": "triggers", "label": "触发器"},
    {"id": "curvature_geometry", "label": "曲率几何（协变/分类/稳定性/景观突变）"},
]


# --- architecture diagram data (rendered by the dashboard "架构图" tab) ---
ARCHITECTURE_LAYERS: list[dict[str, str]] = [
    {"id": "app", "label": "应用层", "module": "gccm_be/app",
     "help": "Web 控制台 + REST API（introspection 驱动）+ JSON 配置 + 报告"},
    {"id": "decision", "label": "决策与诊断层", "module": "gccm_be/decision",
     "help": "置信度 / 哥德尔边界（不可判定）/ 反事实 / 自监控(AR1) / 触发器"},
    {"id": "normative", "label": "规范层", "module": "gccm_be/normative",
     "help": "模式(comfort/balanced/energy/demand_response) → 权重映射"},
    {"id": "geometry", "label": "几何推理层", "module": "gccm_be/geometry",
     "help": "能量景观 + 度量g(z)=λI + 测地线求解(SLSQP/CasADi/鲁棒) + 曲率(协变Hessian)"},
    {"id": "multiscale", "label": "多尺度层", "module": "gccm_be/multiscale",
     "help": "重整化群流 RG→MPC：序参量相关度→分区舒适权重（削峰）"},
    {"id": "physics", "label": "物理层", "module": "gccm_be/physics",
     "help": "RC 灰盒(单区/双区/三容/非线性/数据中心) + 外部输入 + 在线辨识"},
    {"id": "safety", "label": "安全控制层（横切）", "module": "gccm_be/control",
     "help": "SafeController：worst_case / feedback / cool_cap 无模型兜底"},
]

# 一次 optimize() 决策的 13 步管线（对应 engine.py 各 owner 方法）
DECISION_PIPELINE: list[dict[str, str]] = [
    {"step": 1, "label": "规划时域/裕度", "method": "_plan_horizon_and_margin",
     "help": "曲率自适应时域 + 舒适裕度收紧 + 模式覆盖"},
    {"step": 2, "label": "取外部输入序列", "method": "external_provider.get",
     "help": "未来 horizon 步天气/电价/占用（+模型偏置校正）"},
    {"step": 3, "label": "优雅恢复", "method": "_attempt_degradation_recovery",
     "help": "自指误差回落→退出降级（滞回+最短期+温度门控）"},
    {"step": 4, "label": "权重映射", "method": "weight_mapper.map",
     "help": "模式→comfort/energy/smooth 权重"},
    {"step": 5, "label": "RG→MPC 分区权重", "method": "_renormalization_zone_weights",
     "help": "重整化序参量→分区舒适权重（多区削峰）"},
    {"step": 6, "label": "构建能量景观", "method": "EnergyLandscape",
     "help": "舒适+电费+平滑+削峰+储能 景观"},
    {"step": 7, "label": "构建求解器", "method": "GeodesicSolver",
     "help": "四黎曼开关(kinetic/riemannian/riemannian_control/geodesic_penalty)"},
    {"step": 8, "label": "求解", "method": "_solve_trajectory",
     "help": "warm-start + 可选鲁棒多场景 + 异常→空轨迹"},
    {"step": 9, "label": "选首个控制量", "method": "_select_first_control",
     "help": "预防性反馈降级进入"},
    {"step": 10, "label": "曲率分析", "method": "curvature_analyzer.analyze",
     "help": "协变 Hessian 稳定性/分类/景观突变"},
    {"step": 11, "label": "诊断", "method": "diagnoser.diagnose",
     "help": "置信度 / 哥德尔边界 / 触发器"},
    {"step": 12, "label": "消费曲率几何", "method": "_consume_curvature_geometry",
     "help": "突变建议→下周期裕度信号"},
    {"step": 13, "label": "诊断模式", "method": "_apply_diagnosis_mode",
     "help": "不可判定→安全模式 / 建议模式切换"},
]

# 安全降级入口：任何一步失败都不会失控
SAFETY_FALLBACK_POINTS: list[dict[str, str]] = [
    {"point": "8 求解异常", "path": "求解器抛异常/空轨迹 → SafeController"},
    {"point": "9 误差高", "path": "预测误差超标 → 预防性反馈降级"},
    {"point": "13 不可判定", "path": "哥德尔边界 → 无条件安全模式"},
]

# 三个理论层的作用定位（诚实标注：均为正则子/自适应，不是玄学）
THEORY_LAYERS: list[dict[str, str]] = [
    {"id": "riemannian", "label": "黎曼/作用量", "math": "常对角度量 g=λI，动能项 ½żᵀg(z)ż",
     "role": "状态平滑正则子（抑制温度爬升 RMS）", "when": "冷量受限/强扰动、状态真正摆动时",
     "kpi": "温度爬升 RMS（非电费）——g=λI 退化为加权 MPC"},
    {"id": "scm", "label": "SCM 因果", "math": "结构因果模型 + 反事实 + Shapley",
     "role": "模式切换/降级的归因与对比、成本归因", "when": "需要解释“为什么这样控制”",
     "kpi": "归因报告正确性与可解释性"},
    {"id": "rg", "label": "重整化 RG", "math": "序参量相关度→分区权重 mult_i",
     "role": "多区竞争有限冷量时的削峰再分配", "when": "多区共享 AHU、冷量不足时",
     "kpi": "峰值温度（配对 t 检验）"},
]


def _field_default(name: str) -> Any:
    for f in fields(GCCMEngine):
        if f.name == name:
            if f.default is not None and repr(f.default) != "<dataclasses._MISSING_TYPE>":
                try:
                    return f.default
                except Exception:
                    return None
            return None
    return None


def describe(engine: GCCMEngine | None = None) -> dict[str, Any]:
    """Return the full model-capability description for the dashboard.

    If an engine instance is given, current live values are attached so the page
    can render actual toggle/slider positions.
    """
    def live(name: str) -> Any:
        if engine is None:
            return None
        return getattr(engine, name, None)

    def enrich(entries):
        out = []
        for e in entries:
            d = dict(e)
            d["value"] = live(e["field"])
            if e.get("strength_field"):
                d["strength_value"] = live(e["strength_field"])
            out.append(d)
        return out

    return {
        "version": _version(),
        "riemannian_switches": enrich(RIEMANNIAN_SWITCHES),
        "weights": enrich(WEIGHTS),
        "controller_toggles": enrich(CONTROLLER_TOGGLES),
        "modes": MODES,
        "physics_models": PHYSICS_MODELS,
        "diagnostic_fields": DIAGNOSTIC_FIELDS,
        "state_labels": (list(engine.manifold.labels) if engine is not None else ["T_air", "T_wall"]),
        # 当前激活的物理模型类名（对应 physics_models 注册表中的 id）
        "active_model": (type(engine.simulator.building).__name__ if engine is not None else None),
        # 架构图页签数据（分层 / 一次决策管线 / 安全降级点 / 理论层定位）
        "architecture": {
            "layers": ARCHITECTURE_LAYERS,
            "pipeline": DECISION_PIPELINE,
            "fallback_points": SAFETY_FALLBACK_POINTS,
            "theory_layers": THEORY_LAYERS,
        },
    }


def editable_fields() -> list[str]:
    """Names of engine attributes the dashboard is allowed to set at runtime."""
    names: list[str] = []
    for e in RIEMANNIAN_SWITCHES + WEIGHTS + CONTROLLER_TOGGLES:
        names.append(e["field"])
        if e.get("strength_field"):
            names.append(e["strength_field"])
    return names


# Engine switches/weights intentionally NOT surfaced on the dashboard, with the
# reason. These are solver-backend selectors or infra knobs, not model-behavior
# toggles an operator flips. Listed explicitly so the introspection sync test can
# tell "deliberately excluded" apart from "forgot to document a new switch".
NON_SURFACED_FIELDS: dict[str, str] = {
    "use_casadi": "求解器后端选择（是否用 CasADi），非模型行为开关。",
    "use_casadi_robust": "鲁棒 MPC 后端开关，需多场景配置，非日常操作项。",
    "use_two_stage": "两阶段求解策略，求解器内部选项。",
    "storage_weight": "储能目标权重，储能功能未在页面开放。",
}


def _version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "0.1.0"
