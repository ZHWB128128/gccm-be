"""Report generator: readable energy-saving / diagnosis reports."""
from __future__ import annotations

from typing import List, Sequence

from ..engine import GCCMEngine
from ..types import ControlDecision


class ReportGenerator:
    """Generate Markdown/text reports from run records."""

    def __init__(self, engine: GCCMEngine) -> None:
        self.engine = engine

    def generate_markdown(self, decisions: Sequence[ControlDecision]) -> str:
        lines = ["# GCCM-BE 运行报告", ""]
        if not decisions:
            lines.append("无运行数据。")
            return "\n".join(lines)

        avg_cost = sum(d.trajectory.total_cost for d in decisions) / len(decisions)
        avg_conf = sum(d.confidence for d in decisions) / len(decisions)
        lines.append(f"- 决策次数: {len(decisions)}")
        lines.append(f"- 平均轨迹总成本: {avg_cost:.3f}")
        lines.append(f"- 平均置信度: {avg_conf:.3f}")
        lines.append(f"- 最终模式: {self.engine.mode_manager.current_mode}")
        lines.append("")
        lines.append("| 步骤 | 控制量 | 预测下一状态 | 模式 | 置信度 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i, d in enumerate(decisions):
            u = ", ".join(f"{k}={v:.2f}" for k, v in zip(d.control.labels, d.control.u))
            x = ", ".join(f"{k}={v:.2f}" for k, v in zip(d.predicted_next_state.labels, d.predicted_next_state.x))
            lines.append(f"| {i} | {u} | {x} | {d.mode} | {d.confidence:.2f} |")
        lines.append("")
        return "\n".join(lines)
