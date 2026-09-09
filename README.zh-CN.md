# GCCM-BE — 建筑能源鲁棒灰箱 MPC 引擎

> **中文 | [English](README.md)**

[![CI](https://github.com/ZHWB128128/gccm-be/actions/workflows/test.yml/badge.svg)](https://github.com/ZHWB128128/gccm-be/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Deps: numpy/scipy](https://img.shields.io/badge/deps-numpy%2Fscipy-green.svg)](pyproject.toml)

一套**鲁棒灰箱滚动时域模型预测控制（MPC）引擎，带显式安全降级链**，用于楼宇暖通与数据中心制冷的节能优化。纯 Python（numpy/scipy）实现，CasADi 可选加速，可在边缘设备部署。

> **精确定位**：物理接地灰箱 RC 模型上的滚动时域 MPC——每区独立舒适带、多场景鲁棒控制、带物理合理性门控与影子预测验证的在线辨识——由显式安全链监督（求解失败 / 模型失配 / 预报漂移 → 免模型兜底控制）。本 README 的每个数字都可由记录在案的脚本与 seed 复现，负结果与正结果一并归档（[试验台数据](docs/data/README.md)）。
>
> **命名说明**：项目代号"GCCM"（Geometry-based Causal Control Model）来自早期探索阶段；实际交付的控制律是加权非线性 MPC，不是测地线求解或因果推断。历史文档保留代号以与实验历史保持一致。

## ✨ 特性

- **滚动时域 MPC** —— 预知天气与电价：谷时预冷、峰时削峰（楼宇 HVAC 与数据中心制冷）
- **每区灰箱模型** —— 单区 / 双区（可选地板蓄热层）/ 三容 RC / 非线性 RC（开窗、除湿、VAV）/ 数据中心制冷；JSON 配置切换，无需改代码
- **安全降级链** —— 求解失败、模型失配、预报漂移自动切换免模型安全控制；恢复带滞回、最短时限与温度门控；**永不失守**
- **鲁棒 MPC** —— 多场景共享控制序列；30% 模型失配下违温 64.6% → 32.3%；鲁棒 delta 可用真实预报误差校准（`tools_local/calibrate_robust.py`）
- **在线自适应** —— 每区 RC 参数辨识，带物理合理性门控与影子预测验证；AR(1) 自监控
- **可解释决策** —— 每步输出置信度、不可判定标志、触发器清单、反事实对比
- **真实预报接入** —— Open-Meteo 适配器（免 key、时区安全、失败回退 mock）；Home Assistant 适配器对接真楼试点
- **CasADi 后端** —— 求解提速约 200 倍（16ms/步），边缘实时部署
- **轻量运维** —— 线程安全 REST API、数据驱动 Web 控制台、JSON 配置、M&V CSV 数据凭证与有人时段违温 KPI

## 🏗️ 架构

```text
┌──────────────────────────────────────────────┐
│ 应用: Web 控制台 / REST API / JSON 配置       │  gccm_be/app
├──────────────────────────────────────────────┤
│ 决策: 置信度 / 不可判定 / 反事实 /            │  gccm_be/decision
│   触发器 / 自监控                             │
├──────────────────────────────────────────────┤
│ 规范: 模式 / 权重 / 上下文标签                │  gccm_be/normative
├──────────────────────────────────────────────┤
│ 优化: 能量景观 / 加权 MPC 求解器              │  gccm_be/geometry
│   (SLSQP / CasADi / 鲁棒多场景)               │
├──────────────────────────────────────────────┤
│ 物理: RC 模型 / 模型注册表 / NWP / HA /       │  gccm_be/physics
│   在线辨识                                    │
├──────────────────────────────────────────────┤
│ 安全: SafeController 兜底控制律               │  gccm_be/control
└──────────────────────────────────────────────┘
        顶层编排: GCCMEngine (engine.py)
```

数据流：物理层提供状态转移与预报 → 规范层按模式映射权重 → 优化层求解时域 → 决策层监督并在必要时降级 → 应用层暴露 REST/控制台。

## 🚀 快速开始

```bash
pip install numpy scipy            # 运行时依赖仅此两项
pip install -e .                   # 或直接 PYTHONPATH=.

# 一键启动 Web 控制台（自动开浏览器）
python run.py                      # 等价 python -m gccm_be，或安装后 gccm-be
python run.py --config examples/config.json --port 8080

# 最小演示：单区 24h 闭环
PYTHONPATH=. python3 examples/demo.py

# 双区闭环：纯 JSON 配置切换模型，无需改代码
PYTHONPATH=. python3 examples/two_zone_config_loop.py --config examples/config_two_zone.json

# 基线对比：规则 / PID / GCCM
PYTHONPATH=. python3 examples/compare_baselines.py --horizon 48 --no-plot

# 仅启动 REST API（无页面）
python -m gccm_be.app.api --config examples/config.json
# → GET /health /status /introspection   POST /control /config /simulate
```

## 📊 结果（仿真，可复现）

所有数字均为管线修复后复测（2026-08-16），基线/场景/seed 记录在
[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)。标准试验台（BOPTEST/EnergyPlus）
验证——ABAB 演练、参数扫描、双区演练（含归档的负结果）——见
[docs/data/](docs/data/README.md)。

### 楼宇单区（fair_compare，24h，25~27°C）

| 方法 | 电费(元) | 违温(%) | 峰值(kW) |
|---|---:|---:|---:|
| 严格舒适 PID | 28.70 | 0.0 | 2.11 |
| **GCCM** | **24.86** | **0.0** | 1.96 |

**省电 13.4% 且 0% 违温**；Pareto 最优配置（energy=0.8, margin=0.3）达 **18.2%**，多 seed/多场景稳定。

### 数据中心制冷（datacenter_demo，尖峰电价 5 元/kWh）

| 指标 | 规则控制 | GCCM |
|---|---:|---:|
| 日制冷电费 | ¥1296 | **¥921（−29.0%）** |
| 冷通道违温 | 0.0% | **0.0%** |
| 峰时(11~18h)机组功率 | 80.2 kW | **74.4 kW（−7.2%）** |

### 模型失配（控制模型 ≠ 真实建筑，+30%）

| 方法 | 违温(%) |
|---|---:|
| 严格舒适 PID | 68.8 |
| 经典 MPC | 62.5 |
| **GCCM（鲁棒 MPC）** | **32.3** |

### 双区（two_zone_compare）

严格舒适 PID 违温 A 24.0% / B 43.8% → **GCCM 5.2% / 14.6%**，电费再降 2.0%。

> **诚实的范围声明**：以上数字均为仿真结果（内部仿真器 + BOPTEST/EnergyPlus
> 标准测试台）。实测建筑试点（IPMVP 口径）是下一个里程碑——
> [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md)。

## 📁 项目结构

```text
gccm_be/
├── app/          # REST API、配置（模型注册表）、试点 CSV/KPI、报告
├── decision/     # 置信度、不可判定、自监控、触发器
├── normative/    # 模式、权重、上下文标签
├── geometry/     # 能量景观、加权 MPC 求解器（scipy/CasADi/鲁棒）
├── physics/      # RC 模型、模型注册表、NWP/HA 适配器、在线辨识
├── causal/       # SCM、数据驱动结构方程、反事实
├── multiscale/   # 跨尺度分区权重（RG→MPC）
├── control/      # SafeController 兜底控制律
└── engine.py     # 顶层编排
examples/         # 49 个实验脚本（见 examples/README.md）
tests/            # 187 项行为级测试（CasADi 缺失自动跳过）
docs/             # 技术报告 / 架构 / 试验台数据 / 试点方案
```

## 📚 文档

| 文档 | 内容 |
|---|---|
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | 完整实验数据、配置与结论 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层、数据流、决策管线、示意图 |
| [docs/BOPTEST.md](docs/BOPTEST.md) | 标准试验台验证：方法、结果、数据归档 |
| [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md) | 真楼试点：硬件、M&V 协议、就绪清单 |
| [docs/CONTROL_DESIGN.md](docs/CONTROL_DESIGN.md) | 设计决策：黎曼开关、分区权重、求解器选型 |
| [docs/WEB_DASHBOARD.md](docs/WEB_DASHBOARD.md) | 控制台用法、端点、同步契约 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 |

## 🗺️ Roadmap

- [x] BOPTEST 标准试验台验证（单区管线、ABAB 演练、参数扫描、双区演练）—— [docs/BOPTEST.md](docs/BOPTEST.md)
- [x] 双区配置化支持（JSON `model.type: two_zone`）—— [docs/CONTROL_DESIGN.md](docs/CONTROL_DESIGN.md)
- [ ] 真楼试点与实测数据（IPMVP 口径）—— [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md)
- [ ] BACnet / Modbus 接入（接口已预留）
- [ ] 水暖直接执行机构控制（通道语义研究中）
- [ ] 供热 / 蓄能 / 电池储能场景

## 📄 许可证

[MIT](LICENSE)
