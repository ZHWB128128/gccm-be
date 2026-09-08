"""Independent multi-day weather/price scenario sampler.

Rationale (paired-t independence)
---------------------------------
The first-pass experiments varied only additive process noise on ONE shared
weather+price curve, so the paired samples were not truly independent draws —
the mean weather was identical across seeds, which can make the paired-t p-value
optimistic. This module draws a genuinely DIFFERENT day per sample by randomizing
the physically meaningful degrees of freedom of a summer cooling day:

  - base outdoor temperature (climate/day-to-day variability)
  - diurnal swing amplitude
  - peak hour (advection / cloud timing)
  - solar amplitude and an intermittent-cloud multiplier
  - occupancy level
  - price-peak window placement and height (dynamic tariff variation)

Each sampled scenario is a self-contained ExternalInputProvider, so an experiment
that loops over scenarios is drawing i.i.d.-like days rather than the same day
with jitter — a defensible basis for a paired t-test across days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gccm_be.physics.external import ExternalInputProvider
from gccm_be.types import ExternalInput


@dataclass
class WeatherScenario:
    base_temp: float
    swing: float
    peak_hour: float
    solar_amp: float
    cloud: float          # 0..1 multiplier applied stochastically per step
    occ_level: float
    price_peak_start: float
    price_peak_end: float
    price_peak_val: float
    price_base: float


def sample_scenario(rng: np.random.Generator) -> WeatherScenario:
    return WeatherScenario(
        base_temp=float(rng.uniform(27.0, 33.0)),
        swing=float(rng.uniform(3.5, 7.0)),
        peak_hour=float(rng.uniform(13.0, 16.0)),
        solar_amp=float(rng.uniform(0.3, 0.6)),
        cloud=float(rng.uniform(0.0, 0.4)),
        occ_level=float(rng.uniform(0.7, 1.2)),
        price_peak_start=float(rng.uniform(10.0, 12.0)),
        price_peak_end=float(rng.uniform(15.0, 18.0)),
        price_peak_val=float(rng.uniform(1.2, 1.8)),
        price_base=float(rng.uniform(0.3, 0.7)),
    )


class ScenarioProvider(ExternalInputProvider):
    """A weather/price provider realizing one sampled day.

    `plant=True` adds intermittent-cloud measurement realism at k==0 only, so the
    controller forecast (plant=False) and the true plant differ by realistic,
    scenario-specific disturbances rather than a single global noise term.
    """

    def __init__(self, scenario: WeatherScenario, dt_h: float, seed: int = 0,
                 plant: bool = False):
        self.s = scenario
        self.dt_h = dt_h
        self.rng = np.random.default_rng(seed)
        self.plant = plant

    def get(self, time_h: float, horizon: int = 1):
        s = self.s
        out = []
        for k in range(horizon):
            t = time_h + k * self.dt_h
            hour = t % 24.0
            t_out = s.base_temp + s.swing * np.cos(2.0 * np.pi * (t - s.peak_hour) / 24.0)
            solar = 0.0
            if 6.0 <= hour <= 18.0:
                solar = s.solar_amp * max(0.0, np.sin(np.pi * (t - 6.0) / 12.0))
                if self.plant and k == 0 and s.cloud > 0.0:
                    # intermittent cloud cover: multiplicative disturbance on the plant
                    solar *= (1.0 - s.cloud * float(self.rng.random()))
            occ = s.occ_level if 8.0 <= hour <= 18.0 else 0.3 * s.occ_level
            price = s.price_peak_val if s.price_peak_start <= hour < s.price_peak_end else s.price_base
            out.append(ExternalInput(
                np.array([t_out, solar, occ, price]),
                ["T_out", "solar", "occ", "price"],
            ))
        return out
