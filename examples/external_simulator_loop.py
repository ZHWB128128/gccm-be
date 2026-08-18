"""GCCM 控制外部建筑仿真器（EnergyPlus 适配器模拟）闭环示例。"""
from __future__ import annotations

import numpy as np

from gccm_be import GCCMEngine
from gccm_be.geometry.manifold import StateManifold
from gccm_be.physics.building_simulator_adapter import EnergyPlusAdapterStub
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import SystemState

from examples.two_zone_compare import TwoZoneProvider

STEP_H = 0.25
STEPS = 12


def main() -> None:
    provider = TwoZoneProvider()
    manifold = StateManifold(
        labels=["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"],
        units={l: "°C" for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        bounds={l: (15, 40) for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        scale={l: 5 for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
    )
    engine = GCCMEngine(
        simulator=Simulator(TwoZoneRCBuildingModel(),
                            HVACModel(q_min=-8, q_max=8, n_units=2,
                                      control_labels=["Q_hvac_A", "Q_hvac_B"])),
        external_provider=provider,
        manifold=manifold,
        horizon=6,
        dt=STEP_H,
        setpoints={"T_air_A": 26.0, "T_air_B": 26.0},
        comfort_min=25.0,
        comfort_max=27.0,
        comfort_weight=5.0,
        energy_weight=0.1,
        smooth_weight=0.1,
    )

    # 外部仿真器（模拟 EnergyPlus）
    adapter = EnergyPlusAdapterStub()
    adapter.reset(SystemState(np.full(5, 28.0), manifold.labels))

    t = 0.0
    prev = None
    for i in range(STEPS):
        dec = engine.optimize(adapter.get_state(), t, prev_control=prev, forced_mode="comfort")
        w = provider.get(t, 1)[0]
        state = adapter.step(dec.control, w)
        print(f"step {i}: T_A={state.x[0]:.2f}, T_B={state.x[2]:.2f}")
        prev = dec.control
        t += STEP_H


if __name__ == "__main__":
    main()
