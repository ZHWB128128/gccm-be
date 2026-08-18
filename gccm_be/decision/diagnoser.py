"""决策与诊断层：输出置信度、是否切换模式、是否不可判定。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..geometry.curvature import CurvatureAnalysis
from ..types import DiagnosisReport, ExternalInput, SystemState
from .confidence import ConfidenceEvaluator
from .godel_boundary import GodelBoundary
from .triggers import AxiomMutationTrigger


@dataclass
class DecisionDiagnoser:
    """组合触发器、置信度评估与不可判定输出。"""

    trigger: AxiomMutationTrigger = None  # type: ignore[assignment]
    confidence_evaluator: ConfidenceEvaluator = None  # type: ignore[assignment]
    confidence_threshold: float = 0.3
    safe_mode: str = "balanced"
    godel_boundary: GodelBoundary = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.trigger is None:
            self.trigger = AxiomMutationTrigger()
        if self.confidence_evaluator is None:
            self.confidence_evaluator = ConfidenceEvaluator()
        if self.godel_boundary is None:
            self.godel_boundary = GodelBoundary()

    def diagnose(
        self,
        state: SystemState,
        external: ExternalInput,
        prediction_error: float = 0.0,
        curvature: Optional[CurvatureAnalysis] = None,
        model_mismatch: float = 0.0,
        current_mode: str = "balanced",
        forced_mode: Optional[str] = None,
        self_predicted_error: float = 0.0,
    ) -> DiagnosisReport:
        triggers = self.trigger.check(state, external, prediction_error, curvature)
        confidence = self.confidence_evaluator.evaluate(prediction_error, curvature, model_mismatch)

        curvature_min_eig = curvature.stability if curvature is not None else 0.0
        godel = self.godel_boundary.evaluate(
            prediction_error=prediction_error,
            curvature_min_eig=curvature_min_eig,
            self_predicted_error=self_predicted_error,
        )
        undecidable = confidence < self.confidence_threshold or godel["undecidable"]
        suggested = None
        if undecidable:
            suggested = self.safe_mode
        elif forced_mode is not None:
            suggested = forced_mode
        else:
            suggested = self.trigger.suggest_mode(triggers)

        should_switch = bool(suggested is not None and suggested != current_mode)
        return DiagnosisReport(
            confidence=confidence,
            should_switch_mode=should_switch,
            suggested_mode=suggested,
            undecidable=undecidable,
            triggers=triggers,
            details={
                "prediction_error": prediction_error,
                "model_mismatch": model_mismatch,
                "curvature": curvature.as_dict() if curvature is not None else None,
                "godel": godel,
            },
        )
