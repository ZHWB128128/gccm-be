"""New testbed experiment: EKF observer validation on BOPTEST bestest_air (FMU direct).

内容（试点前置的关键验证，此前只在本机仿真验证过）：
1. 从 bestest_air 闭环采集「仅 T_air 可测」的数据流；
2. EKF 在线估计 T_wall（不可测隐状态）；
3. 评估指标：①墙温估计收敛误差（对比仿真真值）；②EKF 预测 vs 开环预测的
   一步温度误差；③辨识特征 (T_wall - T_air) 的可用性。

用法（设备上）：
    cd /opt/boptest && LD_LIBRARY_PATH=/opt/miniforge3/lib PYTHONPATH=/opt/gccm-pilot \
    /opt/miniforge3/bin/python /opt/gccm-pilot/examples/boptest_ekf_validation.py \
    --steps 288 --output /opt/gccm-pilot/output
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/boptest")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testcase import TestCase  # noqa: E402
from boptest_direct import (  # noqa: E402
    FMU, PEAK_DAY, Q_MAX, SETPOINT, STEP_SEC, WARMUP,
    get_price_trajectory, read_state,
)
from gccm_be.physics.observer import EKFStateObserver  # noqa: E402
from gccm_be.physics.models import HVACModel, RCBuildingModel, Simulator  # noqa: E402
from gccm_be.types import ControlInput, SystemState  # noqa: E402


def init_fmu(tc):
    tc.initialize(start_time=PEAK_DAY, warmup_period=WARMUP)
    tc.set_scenario({"electricity_price": "highly_dynamic", "time_period": None,
                     "temperature_uncertainty": None, "solar_uncertainty": None,
                     "seed": None})


def advance_one(tc, sp_c: float, cooling: bool, fan: float = 0.6):
    y = tc.advance({
        "con_oveTSetCoo_u": sp_c + 273.15, "con_oveTSetCoo_activate": 1,
        "con_oveTSetHea_u": 285.15, "con_oveTSetHea_activate": 1,
        "fcu_oveTSup_u": 288.15, "fcu_oveTSup_activate": 1,
        "fcu_oveFan_u": fan if cooling else 0.0, "fcu_oveFan_activate": 1,
    })
    return y[2] if isinstance(y, tuple) else y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--sensor-sigma", type=float, default=0.2,
                        help="模拟传感器噪声 σ (K)")
    parser.add_argument("--output", default="/opt/gccm-pilot/output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # EKF 用「项目默认参数」的单区 RC（模拟试点：无先验辨识）
    ekf_sim = Simulator(RCBuildingModel(), HVACModel(q_min=-Q_MAX, q_max=Q_MAX))
    observer = EKFStateObserver(ekf_sim)

    tc = TestCase(fmupath=FMU)
    tc.step = STEP_SEC
    init_fmu(tc)

    y0 = advance_one(tc, SETPOINT, cooling=False)
    yp0 = y0 if y0 is not None else {}
    t0 = float(yp0.get("zon_reaTRooAir_y", SETPOINT + 273.15)) - 273.15
    observer.initialize(t0, ambient=t0)
    print(f"EKF 初始化: T_air={t0:.2f}°C（实测），T_wall 先验={t0:.2f}°C（未知）")

    rng = np.random.default_rng(0)
    t_room = t0
    rows = []
    ekf_errs, ol_errs = [], []
    prev_ctrl = ControlInput(np.array([0.0]), ["Q_hvac"])
    # 开环对照:一个"错墙温初值"的裸模型,用同样控制 rollout
    open_loop = SystemState(np.array([t0, t0]), ["T_air", "T_wall"])

    for k in range(args.steps):
        # 伪控制:白天制冷(与 ABAB 恒温器类似),产生动态激励
        hour = (k * STEP_SEC / 3600.0) % 24.0
        cooling = 22.0 <= hour < 24.0 or hour < 6.0
        u = -Q_MAX * 0.5 if t_room > 24.0 else (-Q_MAX * 0.3 if cooling else 0.0)
        sp = 24.0 if u < -0.01 else 28.0

        y = advance_one(tc, sp, cooling=(u < -0.01))
        yp = y[2] if isinstance(y, tuple) else y
        if yp is None:
            print(f"scenario end at step {k}", flush=True)
            break
        t_true, tout, solar, elec = read_state(yp, t_room, u, sp)
        z = t_true + rng.normal(0, args.sensor_sigma)   # 传感器噪声

        ctrl = ControlInput(np.array([u]), ["Q_hvac"])
        from gccm_be.types import ExternalInput as _EI
        ext = _EI(np.array([tout, solar, 0.5, 0.6]),
                  ["T_out", "solar", "occ", "price"])
        x_hat = observer.update({"T_air": z}, ctrl, ext, STEP_SEC / 3600.0)

        # 一步预测对比:EKF 状态起步 vs 开环(错墙温)起步
        ekf_pred = ekf_sim.step(x_hat, ctrl, ext, STEP_SEC / 3600.0).x[0]
        ol_pred = ekf_sim.step(open_loop, ctrl, ext, STEP_SEC / 3600.0).x[0]

        # 真值墙温:单区 2 状态模型的 T_wall 在 bestest_air 里没有直接对应,
        # 用 EKF 自身前后一致性 + 一步预测误差作为主要指标;墙温真值用
        # "下一步室温变化率反推"不可行,改为记录估计值供人工分析
        err_ekf = abs(ekf_pred - t_true)
        err_ol = abs(ol_pred - t_true)
        ekf_errs.append(err_ekf)
        ol_errs.append(err_ol)

        open_loop = SystemState(
            ekf_sim.step(open_loop, ctrl, ext, STEP_SEC / 3600.0).x,
            ["T_air", "T_wall"])
        t_room = t_true
        rows.append([k * STEP_SEC / 3600.0, t_true, z, float(x_hat.x[0]),
                     float(x_hat.x[1]), err_ekf, err_ol])
        if (k + 1) % 96 == 0:
            print(f"  step {k+1}/{args.steps} T={t_true:.2f}°C "
                  f"EKF一步误差={np.mean(ekf_errs[-96:]):.3f}K "
                  f"开环一步误差={np.mean(ol_errs[-96:]):.3f}K", flush=True)

    # 汇总
    late = args.steps // 2
    m_ekf = float(np.mean(ekf_errs[late:]))
    m_ol = float(np.mean(ol_errs[late:]))
    improvement = (1.0 - m_ekf / max(m_ol, 1e-9)) * 100.0
    print("\n===== EKF 观测器试验台验证 =====")
    print(f"后半程平均一步温度预测误差: EKF={m_ekf:.3f}K vs 开环(错墙温初值)={m_ol:.3f}K")
    print(f"EKF 相对改善: {improvement:.0f}%")
    print(f"墙温估计终值: {rows[-1][4]:.2f}°C（供分析,无真值对照）")

    csv_path = os.path.join(args.output, "boptest_ekf_validation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "t_air_true", "t_air_meas", "ekf_t_air",
                    "ekf_t_wall_est", "err_ekf_1step", "err_openloop_1step"])
        w.writerows(rows)
    with open(os.path.join(args.output, "boptest_ekf_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mae_ekf_1step_K": m_ekf, "mae_openloop_1step_K": m_ol,
                   "improvement_pct": improvement,
                   "sensor_sigma": args.sensor_sigma}, f, indent=2)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
