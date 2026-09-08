"""EKF 状态观测器行为级测试（P0-1 修正）。"""
from __future__ import annotations

import numpy as np

from gccm_be.physics.observer import EKFStateObserver
from gccm_be.physics.external import MockExternalInputProvider
from gccm_be.physics.models import (
    HVACModel, RCBuildingModel, Simulator, TwoZoneRCBuildingModel, HVACModel as _H,
)
from gccm_be.types import ControlInput, SystemState


def _single_zone_sim():
    return Simulator(RCBuildingModel(c_air=0.6, c_wall=4.0, r_air=0.8, r_wall=2.0,
                                     solar_gain=0.05),
                     HVACModel(q_min=-8.0, q_max=8.0))


def _two_zone_sim():
    d = TwoZoneRCBuildingModel()
    return Simulator(d, HVACModel(q_min=-8.0, q_max=8.0, n_units=2,
                                  control_labels=["Q_hvac_A", "Q_hvac_B"]))


def test_ekf_converges_wall_state_from_air_measurements():
    """真值仿真只暴露 T_air；EKF 应收敛到真值 T_wall（不可测状态）。"""
    sim = _single_zone_sim()
    truth = SystemState(np.array([24.0, 30.0]), ["T_air", "T_wall"])  # 墙温与空气差 6K
    ext = MockExternalInputProvider().get(8.0, 1)[0]
    ctrl = ControlInput(np.array([-4.0]))

    obs = EKFStateObserver(sim)
    obs.initialize(24.0)
    rng = np.random.default_rng(0)
    err_wall = []
    for k in range(120):  # 10h @ 5min
        truth = sim.step(truth, ctrl, ext, 1.0 / 12.0)
        z = float(truth.x[0]) + rng.normal(0, 0.2)   # 传感器噪声 σ=0.2K
        x_hat = obs.update({"T_air": z}, ctrl, ext, 1.0 / 12.0)
        err_wall.append(abs(x_hat.x[1] - truth.x[1]))
    # 收敛：后期墙温估计误差 < 0.5K（无观测器的开环漂移会是数 K）
    assert np.mean(err_wall[-30:]) < 0.5, f"wall est err {np.mean(err_wall[-30:]):.3f}"


def test_ekf_two_zone_estimates_partition_and_walls():
    sim = _two_zone_sim()
    truth = SystemState(np.array([25.0, 28.0, 27.0, 26.0, 25.5]),
                        list(sim.building.state_labels))
    ext = MockExternalInputProvider().get(8.0, 1)[0]
    ctrl = ControlInput(np.array([-3.0, -5.0]))
    obs = EKFStateObserver(sim)
    obs.initialize(25.0, ambient=27.0)
    rng = np.random.default_rng(1)
    errs = []
    for k in range(120):
        truth = sim.step(truth, ctrl, ext, 1.0 / 12.0)
        z_a = float(truth.x[0]) + rng.normal(0, 0.2)
        obs.update({"T_air_A": z_a, "T_air_B": float(truth.x[2]) + rng.normal(0, 0.2)}, ctrl, ext, 1.0 / 12.0)
        x_hat = obs.state()
        errs.append([abs(x_hat.x[i] - truth.x[i]) for i in (1, 4)])  # wall_A, partition
    late = np.mean(errs[-30:], axis=0)
    assert np.all(late < 0.8), f"late hidden-state err {late}"


def test_ekf_prediction_beats_open_loop_drift():
    """关键场景：EKF 一步预测误差应显著小于开环（错初值）预测。"""
    sim = _single_zone_sim()
    truth = SystemState(np.array([26.0, 31.0]), ["T_air", "T_wall"])
    ext = MockExternalInputProvider().get(8.0, 1)[0]
    ctrl = ControlInput(np.array([-4.0]))
    obs = EKFStateObserver(sim)
    obs.initialize(26.0, ambient=26.0)   # 初始以为 T_wall=26，真值 31
    # 预热让 EKF 收敛
    truth_k = truth
    for _ in range(60):
        truth_k = sim.step(truth_k, ctrl, ext, 1.0 / 12.0)
        obs.update({"T_air": float(truth_k.x[0])}, ctrl, ext, 1.0 / 12.0)
    # 比较一步预测：EKF 状态起步 vs 错误初值起步
    ekf_next = sim.step(obs.state(), ctrl, ext, 1.0 / 12.0)
    bad_next = sim.step(SystemState(np.array([truth_k.x[0], 26.0]),
                                    ["T_air", "T_wall"]), ctrl, ext, 1.0 / 12.0)
    truth_next = sim.step(truth_k, ctrl, ext, 1.0 / 12.0)
    err_ekf = abs(ekf_next.x[0] - truth_next.x[0])
    err_bad = abs(bad_next.x[0] - truth_next.x[0])
    assert err_ekf < err_bad * 0.5, f"ekf {err_ekf:.4f} vs open-loop {err_bad:.4f}"


def test_ekf_state_labels_match_model():
    sim = _two_zone_sim()
    obs = EKFStateObserver(sim)
    obs.initialize(21.0)
    assert obs.state().labels == list(sim.building.state_labels)
    assert obs.state().dim == 5
