"""GCCM-BE MVP 演示：单区域空调滚动优化。"""
from gccm_be import GCCMEngine
from gccm_be.app.report import ReportGenerator
from gccm_be.types import SystemState


def main() -> None:
    engine = GCCMEngine(horizon=12, counterfactual_enabled=True)
    state = SystemState([27.0, 26.5], ["T_air", "T_wall"])

    print("=== 单步优化 ===")
    decision = engine.optimize(state, time_h=8.0)
    print(decision.as_dict())

    print("\n=== 滚动闭环 24 步（5 分钟/步）===")
    decisions = engine.run_closed_loop(state, start_time_h=8.0, steps=24, step_h=1.0 / 12.0)
    report = ReportGenerator(engine).generate_markdown(decisions)
    print(report)


if __name__ == "__main__":
    main()
