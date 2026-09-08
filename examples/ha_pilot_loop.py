"""实楼试点闭环脚本：Home Assistant 读数 → GCCM 决策 → 写回空调设定温度。

前提（见 docs/PILOT_PLAN.md）：HA 已接入温湿度传感器、电量插座、空调万能遥控；
HA Long-Lived Access Token 放环境变量 HA_TOKEN。

    PYTHONPATH=. python3 examples/ha_pilot_loop.py \
        --ha-url http://homeassistant.local:8123 \
        --temp sensor.room_temp --power sensor.ac_power \
        --climate climate.ac --config examples/config.json

控制周期默认 15 分钟（真实等待由外部 cron/systemd 触发单步，或 --once 单步）。
每个周期追加一行 M&V 数据到 CSV（PILOT_PLAN §4 的数据凭证）。
功率 → 设定温度的映射策略按试点标定调整（此处给保守线性映射示例）。
"""
from __future__ import annotations

import argparse
import os
import time

from gccm_be.app.config import engine_from_config
from gccm_be.app.pilot_log import append_row
from gccm_be.physics.homeassistant import HABuildingAdapter
from gccm_be.physics.observer import EKFStateObserver
from gccm_be.types import SystemState


def now_hour() -> float:
    """本地时钟 → 当日小时（浮点）。"""
    lt = time.localtime()
    return (lt.tm_hour * 60 + lt.tm_min) / 60.0


def control_to_setpoint(q_cool_kw: float, q_max: float = 8.0,
                        t_lo: float = 18.0, t_hi: float = 30.0) -> float:
    """保守线性映射：制冷需求越强，设定温度越低。按试点标定调整。"""
    demand = min(max(-q_cool_kw / max(q_max, 1e-9), 0.0), 1.0)  # 0~1
    return round(t_hi - demand * (t_hi - t_lo), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL", "http://homeassistant.local:8123"))
    parser.add_argument("--temp", required=True, help="室温传感器实体")
    parser.add_argument("--power", default=None, help="空调功率/电量传感器实体（可选）")
    parser.add_argument("--climate", required=True, help="空调 climate 实体")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    parser.add_argument("--csv", default="output/pilot_data.csv", help="M&V 数据 CSV 路径")
    parser.add_argument("--once", action="store_true", help="只跑一步（cron 调度模式）")
    parser.add_argument("--interval-min", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()

    token = os.environ.get("HA_TOKEN", "")
    if not token:
        raise SystemExit("请设置环境变量 HA_TOKEN（HA Long-Lived Access Token）")

    engine = engine_from_config(args.config)
    adapter = HABuildingAdapter(
        base_url=args.ha_url, token=token,
        temp_entity=args.temp, power_entity=args.power, climate_entity=args.climate,
    )
    labels = list(engine.manifold.labels)
    adapter.reset(SystemState([26.0] * len(labels), labels))
    # 状态观测器（真楼只有 T_air 可测；墙/蓄热层等隐状态由 EKF 估计——
    # 直接把测量塞进全状态会让 MPC 建立在对 T_wall 的错误假设上）
    observer = EKFStateObserver(engine.simulator)
    last_control = None

    n = 1 if args.once else args.steps
    for k in range(n):
        measured = adapter.sync_measurements()   # 可测通道为真值
        t_air = float(measured.x[0])
        t_h = now_hour()
        ext = engine.external_provider.get(t_h, 1)[0]
        # EKF: 用上一步控制与当前预报做 predict,再用新测量 update
        if last_control is not None:
            observer.update({"T_air": t_air}, last_control, ext, engine.dt)
        state = observer.state()   # 隐状态为估计值,可测维度收敛到测量
        decision = engine.optimize(state, t_h)
        last_control = decision.control
        q = float(decision.control.u[0])
        setpoint = control_to_setpoint(q, q_max=abs(engine.simulator.hvac.q_min))
        adapter.set_target_temperature(setpoint)

        # 最小告警（评审"基础告警"项）：安全越界 / 求解失败
        safety_hi = (engine.comfort_max or 27.0) + 2.0
        if t_air > safety_hi:
            print(f"[ALERT] SAFETY: T_air {t_air:.1f} > {safety_hi:.1f}", flush=True)
        if not decision.solver_success:
            print("[ALERT] SOLVER: falling back to safe control", flush=True)

        try:
            price = float(engine.external_provider.get(t_h, 1)[0].price)
        except Exception:
            price = ""
        power = ""
        if args.power:
            try:
                power = adapter.get_measurements().get("power", "")
            except Exception:
                power = ""
        append_row(args.csv, {
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_h": f"{t_h:.4f}",
            "T_air": f"{float(state.x[0]):.3f}",
            "u_kw": f"{q:.3f}",
            "setpoint": setpoint,
            "power_w": power,
            "price": price,
            "mode": decision.mode,
            "confidence": f"{decision.confidence:.3f}",
        })
        print(f"[{k}] T={float(state.x[0]):.2f}°C Q={q:.2f}kW → 设定 {setpoint}°C "
              f"mode={decision.mode} conf={decision.confidence:.2f} → {args.csv}")
        if not args.once and k < n - 1:
            time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
