"""架构补全模块演示：自指回路、度量张量、几何公理突变、重整化群流、哥德尔边界。"""
from __future__ import annotations

import numpy as np

from gccm_be.decision.godel_boundary import GodelBoundary
from gccm_be.decision.self_monitor import SelfMonitor
from gccm_be.decision.triggers import AxiomMutationTrigger
from gccm_be.geometry.curvature import CurvatureAnalysis
from gccm_be.geometry.landscape import EnergyLandscape
from gccm_be.geometry.manifold import StateManifold
from gccm_be.multiscale import RenormalizationFlow
from gccm_be.physics.models import HVACModel, Simulator, TwoZoneRCBuildingModel
from gccm_be.types import SystemState


def main() -> None:
    print("=" * 60)
    print("1. 自指回路 SelfMonitor")
    print("=" * 60)
    monitor = SelfMonitor()
    for err in [0.1, 0.12, 0.15, 0.2, 0.35, 0.6, 0.9, 1.2]:
        monitor.update(err)
    print(monitor.feedback())

    print("\n" + "=" * 60)
    print("2. 显式度量张量")
    print("=" * 60)
    sim = Simulator(TwoZoneRCBuildingModel(), HVACModel(q_min=-8, q_max=8, n_units=2))
    manifold = StateManifold(
        labels=["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"],
        units={l: "°C" for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        bounds={l: (15, 40) for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
        scale={l: 5.0 for l in ["T_air_A", "T_wall_A", "T_air_B", "T_wall_B", "T_partition"]},
    )
    landscape = EnergyLandscape(
        setpoints={"T_air_A": 26, "T_air_B": 26},
        weights={"comfort": 5.0, "energy": 0.1, "smooth": 0.1},
        manifold=manifold,
        hvac=sim.hvac,
        comfort_min=25,
        comfort_max=27,
    )
    state = SystemState(np.full(5, 26.0), manifold.labels)
    g = landscape.metric(state)
    print("Metric diag:", np.round(np.diag(g), 4))
    print("Kinetic term for delta [0.1,0,0.2,0,0]:", round(landscape.kinetic_term(np.array([0.1,0,0.2,0,0]), 0.25), 4))

    print("\n" + "=" * 60)
    print("3. 几何触发公理突变")
    print("=" * 60)
    curvature = CurvatureAnalysis(
        hessian=np.eye(2),
        eigenvalues=np.array([-0.1, 0.2]),
        eigenvectors=np.eye(2),
        classification="saddle",
        stability=-0.1,
    )
    trigger = AxiomMutationTrigger()
    print(trigger.suggest_landscape_mutation(curvature))

    print("\n" + "=" * 60)
    print("4. 重整化群流")
    print("=" * 60)
    rg = RenormalizationFlow()
    print(rg.analyze([26.2, 27.4, 26.0, 28.1]))

    print("\n" + "=" * 60)
    print("5. 哥德尔边界")
    print("=" * 60)
    godel = GodelBoundary()
    print(godel.evaluate(prediction_error=1.0, curvature_min_eig=0.005, self_predicted_error=0.8))


if __name__ == "__main__":
    main()
