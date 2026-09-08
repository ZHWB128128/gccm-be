"""Safe-control owner extracted from GCCMEngine.

This module is the single canonical owner of the engine's *fallback / safe*
control laws — the controllers used when the main geodesic MPC solve is
undecidable, unreliable, or has failed. It was split out of the 900-line
`GCCMEngine` God Object so the safety-critical logic has one clear home and can
be tested in isolation.

Two laws live here:

- ``worst_case`` — a conservative model-based one-step controller that, using
  identified (or pessimistic) RC parameters, computes the cooling power needed to
  keep the next step below the comfort ceiling, guarded against overcooling.
- ``feedback`` — a purely model-free PI + weather-feedforward law that does not
  trust the internal model at all (used when identification is untrusted or the
  building model is unknown / multi-zone).

The controller owns its own integral state (`_integral`), so the engine no longer
carries `_safe_integral` for this purpose.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..physics.models import RCBuildingModel, Simulator
from ..types import ControlInput, ExternalInput, SystemState


@dataclass
class SafeController:
    """Owns the engine's conservative and model-free fallback control laws."""

    simulator: Simulator
    setpoints: dict[str, float]
    comfort_min: float | None = None
    comfort_max: float | None = None
    # 每区独立舒适带 label -> (lo, hi)：降级控制按区判定，覆盖标量边界。
    # 未覆盖的区回退标量 comfort_min/comfort_max。
    zone_comfort_bounds: dict[str, tuple] = field(default_factory=dict)
    safe_kp: float = 2.0
    safe_ki: float = 0.2
    safe_feedforward_gain: float = 0.5
    safe_integral_limit: float = 2.0
    safe_weather_margin: float = 1.0
    worst_case_outdoor_temp: float = 40.0
    worst_case_solar: float = 0.8
    worst_case_occ: float = 1.2
    # collaborators supplied by the engine (identification trust + parameters)
    identification_trusted: Callable[[int], bool] = lambda _min: False
    rc_parameters: Callable[[], dict] = dict

    _integral: dict[int, float] = field(default_factory=dict, init=False)

    # ---- state management (so counterfactual rollouts can snapshot/restore) ----
    def snapshot(self) -> dict:
        return {"integral": dict(self._integral)}

    def restore(self, snap: dict) -> None:
        self._integral = dict(snap.get("integral", {}))

    def reset(self) -> None:
        self._integral = {}

    # ---- helpers ----
    def _labels(self, n_u: int):
        labels = getattr(self.simulator.hvac, "control_labels", None)
        if not labels or len(labels) != n_u:
            labels = [f"u{i}" for i in range(n_u)]
        return list(labels)

    def cool_cap(self, err: float, q_max: float) -> float:
        """Dynamically caps safe cooling power by temperature deviation."""
        if err > 2.0:
            return min(q_max, 6.0)
        if err > 1.0:
            return min(q_max, 4.0)
        return min(q_max, 2.5)

    def zone_bounds(self, label: str) -> tuple:
        """该区的有效舒适边界；每区覆盖优先，缺省回退标量（可能为 None 侧）。"""
        zone = self.zone_comfort_bounds.get(label)
        if zone is not None:
            return tuple(zone)
        return (self.comfort_min, self.comfort_max)

    # ---- worst-case (conservative model-based) ----
    def worst_case(self, state: SystemState, dt: float,
                   external: ExternalInput | None = None) -> ControlInput:
        building = self.simulator.building
        bounds = self.simulator.hvac.bounds()
        n_u = len(bounds)
        labels = self._labels(n_u)

        if not isinstance(building, RCBuildingModel):
            # multi-zone / unknown model -> fall back to model-free feedback
            return self.feedback(state, dt)

        T_air, T_wall = state.x[0], state.x[1]
        target = (self.comfort_max or 27.0) - 0.1
        setpoint = self.setpoints.get("T_air", 26.0)

        C = building.c_air
        r_air = building.r_air
        r_wall = building.r_wall
        solar_gain = building.solar_gain
        if not self.identification_trusted(30):
            C = min(getattr(building, "c_air", 0.6), 0.4)
            r_air = min(getattr(building, "r_air", 0.8), 0.3)
            r_wall = min(getattr(building, "r_wall", 2.0), 0.5)
            solar_gain = max(getattr(building, "solar_gain", 0.05), 0.1)
        else:
            try:
                params = self.rc_parameters()
                C = 1.0 / params["one_over_C_air"]
                r_air = params["R_air_times_C_air"] / C
                r_wall = params["R_wall_times_C_air"] / C
                solar_gain = params["solar_gain_over_C_air"] * C
            except Exception:
                C = min(getattr(building, "c_air", 0.6), 0.4)
                r_air = min(getattr(building, "r_air", 0.8), 0.3)
                r_wall = min(getattr(building, "r_wall", 2.0), 0.5)
                solar_gain = max(getattr(building, "solar_gain", 0.05), 0.1)

        # not hot -> gentle proportional cooling to avoid overcooling oscillation
        if T_air <= setpoint + 0.5:
            fb = self.safe_kp * (setpoint - T_air)
            q = min(0.0, float(np.clip(fb, bounds[0][0], 0.0)))
            return ControlInput(np.array([q]), labels)

        if external is not None and external.w.size >= 3:
            t_out_eff = external.w[0] + self.safe_weather_margin
            solar_eff = external.w[1] * 1.2 + 0.05
            occ_eff = external.w[2] + 0.2
        else:
            t_out_eff = self.worst_case_outdoor_temp
            solar_eff = self.worst_case_solar
            occ_eff = self.worst_case_occ
        q_none = (
            (T_wall - T_air) / r_air
            + (t_out_eff - T_air) / r_wall
            + solar_gain * solar_eff
            + occ_eff
        )
        T_nocool = T_air + dt / C * q_none
        required = C / dt * (target - T_nocool)
        fb = self.safe_kp * (setpoint - T_air)
        q = min(0.0, required + fb)
        t_low = (self.comfort_min or 25.0) + 0.2
        lower_bound = C / dt * (t_low - T_nocool)
        q = max(q, lower_bound)
        cool_cap = self.cool_cap(max(0.0, T_air - setpoint), bounds[0][1])
        q = float(np.clip(q, -cool_cap, 0.0))
        return ControlInput(np.array([q]), labels)

    # ---- model-free feedback ----
    def feedback(self, state: SystemState, dt: float,
                 external: ExternalInput | None = None) -> ControlInput:
        bounds = self.simulator.hvac.bounds()
        n_u = len(bounds)
        air_idx = [i for i, lab in enumerate(state.labels) if lab.startswith("T_air")]
        if not air_idx:
            air_idx = [0]
        safe_u = np.zeros(n_u)
        for j, idx in enumerate(air_idx[:n_u]):
            setpoint = self.setpoints.get(state.labels[idx], 26.0)
            err = state.x[idx] - setpoint
            z_lo, _ = self.zone_bounds(state.labels[idx])
            if z_lo is not None and state.x[idx] < z_lo:
                heat = self.safe_kp * (z_lo - state.x[idx])
                safe_u[j] = float(np.clip(heat, 0.0, bounds[j][1]))
                self._integral[idx] = 0.0
                continue
            ff = 0.0
            if external is not None and external.w.size > 0:
                ff = self.safe_feedforward_gain * max(0.0, external.w[0] - 26.0)
            if err <= 0.2 and ff <= 0.0:
                self._integral[idx] = 0.0
                safe_u[j] = 0.0
                continue
            integral = self._integral.get(idx, 0.0) + err * dt
            integral = float(np.clip(integral, -self.safe_integral_limit, self.safe_integral_limit))
            self._integral[idx] = integral
            cool_cap = self.cool_cap(err, max(abs(bounds[j][0]), bounds[j][1]))
            cool = float(np.clip(self.safe_kp * err + self.safe_ki * integral + ff, 0.0, cool_cap))
            safe_u[j] = -cool
        return ControlInput(safe_u, self._labels(n_u))
