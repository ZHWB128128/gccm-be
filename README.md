# GCCM-BE — Robust Gray-Box MPC Engine for Building Energy

> **English | [中文](README.zh-CN.md)**

[![CI](https://github.com/ZHWB128128/gccm-be/actions/workflows/test.yml/badge.svg)](https://github.com/ZHWB128128/gccm-be/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Deps: numpy/scipy](https://img.shields.io/badge/deps-numpy%2Fscipy-green.svg)](pyproject.toml)

A **robust gray-box Model Predictive Control (MPC) engine with an explicit safety degradation chain** for building HVAC and data-center cooling. Pure Python (numpy/scipy), optional CasADi backend, deployable on edge devices.

> **What it is, precisely:** rolling-horizon MPC on physically grounded gray-box RC models — per-zone comfort bands, robust multi-scenario control, online identification with physical plausibility gating and shadow-prediction validation — supervised by an explicit safety chain (solver failure / model mismatch / prediction drift → model-free fallback control). Every number in this README is reproducible from documented scripts and seeds; negative results are archived next to positives ([testbed data](docs/data/README.md)).
>
> **Naming note:** the project codename "GCCM" (Geometry-based Causal Control Model) dates from early exploration; the shipped control law is weighted nonlinear MPC, not geodesic solving or causal inference. Historical docs keep the codename for continuity with experiment history.

## ✨ Features

- **Rolling-horizon MPC** — anticipates weather and electricity prices: pre-cools in cheap hours, shaves peaks (building HVAC and data-center cooling)
- **Per-zone gray-box models** — single-zone / two-zone (with optional floor-slab thermal mass) / three-mass RC / nonlinear RC (window, dehumidification, VAV) / data-center cooling; selected by JSON config, no code changes
- **Safety degradation chain** — solver failure, model mismatch, or prediction-drift spike falls back to model-free safe control; recovery gated by hysteresis, minimum duration, and temperature; never loses control
- **Robust MPC** — multi-scenario shared control sequence; under 30 % model mismatch, violations beat classic MPC (64.6 % → 32.3 %); robustness delta calibratable from measured NWP forecast errors (`tools_local/calibrate_robust.py`)
- **Online adaptation** — RC parameter identification per zone with physical plausibility gating and shadow validation; AR(1) self-monitoring
- **Explainable decisions** — every step outputs confidence, undecidable flags, trigger list, and counterfactual comparison
- **Real forecast integration** — Open-Meteo NWP adapter (no API key, timezone-safe, mock fallback); Home Assistant adapter for real-room pilots
- **CasADi backend** — ~200× faster solves (16 ms/step) for edge real-time deployment
- **Lightweight ops** — thread-safe REST API, data-driven web dashboard, JSON config, M&V CSV logging with occupied-hours violation KPI

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────┐
│ App: web dashboard / REST API / JSON config  │  gccm_be/app
├──────────────────────────────────────────────┤
│ Decision: confidence / undecidable /         │  gccm_be/decision
│   counterfactual / triggers / self-monitor   │
├──────────────────────────────────────────────┤
│ Normative: modes / weights / context labels  │  gccm_be/normative
├──────────────────────────────────────────────┤
│ Optimization: energy landscape / weighted    │  gccm_be/geometry
│   MPC solvers (SLSQP / CasADi / robust)      │
├──────────────────────────────────────────────┤
│ Physics: RC models / registry / NWP / HA /   │  gccm_be/physics
│   online identification                      │
├──────────────────────────────────────────────┤
│ Safety: SafeController fallback laws         │  gccm_be/control
└──────────────────────────────────────────────┘
        Top-level orchestration: GCCMEngine (engine.py)
```

Data flow: physics provides state transition and forecasts → normative maps mode to weights → optimization solves the horizon → decision supervises and degrades when needed → app exposes REST/dashboard.

## 🚀 Quick Start

```bash
pip install numpy scipy            # runtime deps only
pip install -e .                   # or simply use PYTHONPATH=.

# One-command launch: web dashboard (opens browser)
python run.py                      # same as python -m gccm_be, or `gccm-be` after install
python run.py --config examples/config.json --port 8080

# Minimal demo: single-zone 24h closed loop
PYTHONPATH=. python3 examples/demo.py

# Two-zone closed loop, driven purely by JSON config (no code changes)
PYTHONPATH=. python3 examples/two_zone_config_loop.py --config examples/config_two_zone.json

# Baselines comparison: rule / PID / GCCM
PYTHONPATH=. python3 examples/compare_baselines.py --horizon 48 --no-plot

# REST API only (no page)
python -m gccm_be.app.api --config examples/config.json
# → GET /health /status /introspection   POST /control /config /simulate
```

## 📊 Results (simulation, reproducible)

All numbers re-measured after the pipeline fixes (2026-08-16); baselines, scenarios, and seeds are documented in [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md). Standard-testbed (BOPTEST/EnergyPlus) verification — ABAB drill, parameter sweeps, two-zone drills, including archived negative results — is in [docs/data/](docs/data/README.md).

### Building, single zone (fair_compare, 24 h, 25–27 °C)

| Method | Cost (¥) | Violation (%) | Peak (kW) |
|---|---:|---:|---:|
| Strict comfort PID | 28.70 | 0.0 | 2.11 |
| **GCCM** | **24.86** | **0.0** | 1.96 |

**13.4 % cost savings at 0 % violation**; Pareto-optimal config (energy=0.8, margin=0.3) reaches **18.2 %**, stable across seeds/scenarios.

### Data-center cooling (datacenter_demo, peak price 5 ¥/kWh)

| Metric | Rule control | GCCM |
|---|---:|---:|
| Daily cooling electricity | ¥1296 | **¥921 (−29.0 %)** |
| Cold-aisle violation | 0.0 % | **0.0 %** |
| Peak-hour (11–18 h) chiller power | 80.2 kW | **74.4 kW (−7.2 %)** |

### Model mismatch (controller model ≠ real building, +30 %)

| Method | Violation (%) |
|---|---:|
| Strict comfort PID | 68.8 |
| Classic MPC | 62.5 |
| **GCCM (robust MPC)** | **32.3** |

### Two-zone (two_zone_compare)

Strict comfort PID violates A 24.0 % / B 43.8 % → **GCCM 5.2 % / 14.6 %**, with 2.0 % lower cost.

> **Honest scope note:** all figures above are simulation results (internal simulators + BOPTEST/EnergyPlus standard testbed). The measured-building pilot (IPMVP protocol) is the next milestone — [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md).

## 📁 Project Structure

```text
gccm_be/
├── app/          # REST API, config (model registry), pilot CSV/KPI, reporting
├── decision/     # confidence, undecidable, self-monitoring, triggers
├── normative/    # modes, weights, context labels
├── geometry/     # energy landscape, weighted MPC solvers (scipy/CasADi/robust)
├── physics/      # RC models, model registry, NWP/HA adapters, online identification
├── causal/       # SCM, data-driven structural equations, counterfactuals
├── multiscale/   # cross-scale zone weighting (RG→MPC)
├── control/      # SafeController fallback laws
└── engine.py     # top-level orchestration
examples/         # 49 experiment scripts (see examples/README.md)
tests/            # 187 behavior-level tests (CasADi tests auto-skip when absent)
docs/             # technical report / architecture / testbed data / pilot plan
```

## 📚 Documentation

| Document | Content |
|---|---|
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | Full experiment data, configs, conclusions |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, data flow, decision pipeline, diagrams |
| [docs/BOPTEST.md](docs/BOPTEST.md) | Standard-testbed verification: methods, results, data archive |
| [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md) | Real-building pilot: hardware, M&V protocol, readiness checklist |
| [docs/CONTROL_DESIGN.md](docs/CONTROL_DESIGN.md) | Design decisions: Riemannian switches, zone weighting, solver choices |
| [docs/WEB_DASHBOARD.md](docs/WEB_DASHBOARD.md) | Dashboard usage, endpoints, sync contract |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide |

## 🗺️ Roadmap

- [x] BOPTEST standard-testbed validation (single-zone pipeline, ABAB drill, parameter sweeps, two-zone drills) — [docs/BOPTEST.md](docs/BOPTEST.md)
- [x] Two-zone configuration support (`model.type: two_zone` in JSON) — [docs/CONTROL_DESIGN.md](docs/CONTROL_DESIGN.md)
- [ ] Real-building pilot with measured data (IPMVP protocol) — [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md)
- [ ] BACnet / Modbus integration (interface stubbed)
- [ ] Hydronic direct-actuator control (channel semantics under study)
- [ ] District heating / thermal storage / battery scenarios

## 📄 License

[MIT](LICENSE)
