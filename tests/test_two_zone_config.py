"""双区配置化行为级测试：config 构建的引擎跑闭环 + API 多区数据面。

与 examples/two_zone_*.py 的手写装配实验不同，这里验证的是"只改 JSON
就能切到双区模型"这条产品化路径端到端可用。
"""
from __future__ import annotations

import numpy as np

from gccm_be.app.api import SimpleAPI
from gccm_be.app.config import engine_from_dict
from gccm_be.types import SystemState


TWO_ZONE_CFG = {
    "model": {"type": "two_zone"},
    "zones": {"A": {"setpoint": 26.0}, "B": {"setpoint": 26.0}},
    "controller": {
        "horizon": 6,
        "comfort_min": 25.0,
        "comfort_max": 27.0,
        "enforce_comfort_constraints": False,
    },
}


def _make_engine():
    return engine_from_dict(TWO_ZONE_CFG)


def test_two_zone_closed_loop_short_rollout():
    engine = _make_engine()
    labels = list(engine.manifold.labels)
    state = SystemState(np.full(len(labels), 28.0), labels)
    decisions = engine.run_closed_loop(state, start_time_h=8.0, steps=6, step_h=engine.dt)

    assert len(decisions) == 6
    for d in decisions:
        # 每步输出两台设备的控制
        assert d.control.dim == 2
        assert np.all(np.isfinite(d.control.u))
        assert np.all(d.control.u >= -8.0) and np.all(d.control.u <= 8.0)
        if d.predicted_next_state is not None:
            assert np.all(np.isfinite(d.predicted_next_state.x))


def test_two_zone_simulate_returns_per_zone_series():
    api = SimpleAPI(_make_engine())
    res = api.simulate({"t0": 29.0, "steps": 6})

    # 每区一条室温曲线，共两区
    assert [z["label"] for z in res["zones"]] == ["T_air_A", "T_air_B"]
    assert all(len(z["temps"]) == 6 for z in res["zones"])
    # 每台设备一条控制序列
    assert [u["label"] for u in res["unit_controls"]] == ["Q_hvac_A", "Q_hvac_B"]
    assert all(len(u["data"]) == 6 for u in res["unit_controls"])
    # 兼容字段指向第一个区/第一台设备
    assert res["temps"] == res["zones"][0]["temps"]
    assert res["controls"] == res["unit_controls"][0]["data"]
    # 越界按全部区的室温统计（非只第一区）
    lo, hi = res["comfort_min"], res["comfort_max"]
    expected = sum(
        1 for z in res["zones"] for t in z["temps"] if t > hi or t < lo
    )
    assert res["violations"] == expected


def test_single_zone_simulate_response_shape_unchanged():
    """单区引擎走同一 API：zones/unit_controls 各只有一项，兼容字段一致。"""
    api = SimpleAPI(engine_from_dict({"controller": {"horizon": 6}}))
    res = api.simulate({"t0": 29.0, "steps": 6})

    assert [z["label"] for z in res["zones"]] == ["T_air"]
    assert len(res["unit_controls"]) == 1
    assert res["temps"] == res["zones"][0]["temps"]
    assert len(res["temps"]) == 6
