# GCCM-BE 建筑能源几何因果推理引擎

> **中文 | [英文版](README.md)**

**Geometry-based Causal Control Model for Building Energy**

一套**带安全兜底的滚动时域模型预测控制（MPC）引擎**，用于楼宇暖通与数据中心制冷的节能优化。
纯 Python（numpy/scipy）实现，CasADi 可选，可在边缘设备部署。

> 定位说明：项目名中的"几何/因果"指**离散时间最小作用量路径近似 + 因果启发的自适应控制**
> （详见 [docs/RIEMANNIAN_CONTROL_THEORY.md](docs/RIEMANNIAN_CONTROL_THEORY.md) 与"数学定位声明"），
> 非严格黎曼测地线/统计因果推断——我们对宣称保持诚实与可复现。

## ✨ 特性

- **滚动时域 MPC**：前瞻天气与电价，谷时预冷蓄冷、峰时削峰（建筑 HVAC + 数据中心冷却双场景）
- **安全降级链**：求解失败 / 模型失配 / 预测误差超标时自动降级到无模型安全控制，不失控
- **可解释决策**：每次控制输出置信度、不可判定判定、反事实对比报告
- **在线自适应**：RC 参数在线辨识（带物理合理性门控 + 影子预测验证）、自监控（AR(1) 自预测）
- **鲁棒 MPC**：多模型场景共享控制，模型失配下显著优于经典 MPC（违规率 64.6% → 32.3%）
- **储能价值项**：蓄冷罐/电池类储能系统的防短视排空机制（通用储能调度）
- **轻量部署**：REST API（线程安全）+ JSON 配置化 + Dockerfile

## 🏗️ 架构

```text
┌─────────────────────────────────────────────┐
│ 应用层：REST API / 配置 / 报告 / 演示图表        │  gccm_be/app
├─────────────────────────────────────────────┤
│ 决策与诊断层：置信度 / 不可判定 / 反事实 / 触发器   │  gccm_be/decision
├─────────────────────────────────────────────┤
│ 规范层：模式 / 权重 / 上下文标签                 │  gccm_be/normative
├─────────────────────────────────────────────┤
│ 几何推理层：能量景观 / 度量 / 测地线求解 / 曲率    │  gccm_be/geometry
├─────────────────────────────────────────────┤
│ 物理层：RC 模型 / 数据中心冷却 / 在线辨识        │  gccm_be/physics
└─────────────────────────────────────────────┘
            顶层编排：GCCMEngine（engine.py）
```

数据流：物理层供状态转移 → 规范层供参数 → 几何层供优化（MPC）→ 决策层供监督/降级 → 应用层供接口。

## 🚀 快速开始

```bash
pip install numpy scipy          # 运行时依赖仅这两个
pip install -e .                 # 或直接 PYTHONPATH=. 使用

# 最小演示：单区域 24h 闭环
PYTHONPATH=. python3 examples/demo.py

# 对比实验：规则 / PID / GCCM
PYTHONPATH=. python3 examples/compare_baselines.py --horizon 48 --no-plot

# 启动 REST API
python -m gccm_be.app.api --config examples/config.json
# → GET /health  /status   POST /control  {"state":[...], "labels":[...]}
```

## 📊 实测结果

> 所有数字均来自修复管线后的复测（2026-08-16），基线/场景/seed 已注明，可复现。

### 建筑单区域（fair_compare, 24h, 25~27°C）

| 方法 | 电费(元) | 违温(%) | 峰值(kW) |
|---|---:|---:|---:|
| 严格舒适 PID | 28.70 | 0.0 | 2.11 |
| **GCCM** | **24.86** | **0.0** | 1.96 |

**省电 13.4% 且 0% 违温**；Pareto 最优配置（energy=0.8, margin=0.3）进一步到 **18.2%**，多 seed/多场景稳定。

### 数据中心冷却（datacenter_demo, 尖峰电价 5 元）

| 指标 | 规则控制 | GCCM |
|---|---:|---:|
| 日制冷电费 | 1231 元 | **801 元（省 34.9%）** |
| 冷通道违温 | 0.0% | **0.0%** |
| 峰时(11-18h)机组电功率 | 18.7 kW | **13.9 kW（削峰 26%）** |

### 模型失配（控制器模型 ≠ 真实建筑）

| 方法 | 违温(%) |
|---|---:|
| 严格舒适 PID | 68.8 |
| 经典 MPC | 62.5 |
| **GCCM（鲁棒 MPC）** | **32.3** |

### 两区域（two_zone_compare）

严格舒适 PID 违温 A 24.0% / B 43.8% → **GCCM 5.2% / 14.6%**，且电费低 2.0%。

## 🖼️ 演示图表

![温度对比](docs/images/compare_temperature.png)
![功率与电价](docs/images/compare_power_price.png)
![指标对比](docs/images/compare_metrics.png)
![诊断时间线](docs/images/diagnosis_timeline.png)
![数据中心冷却](docs/images/datacenter_cooling.png)

重新生成：`PYTHONPATH=. python3 examples/demo_report.py` 与 `examples/datacenter_demo.py`。

## 📁 项目结构

```text
gccm_be/
├── app/          # REST API、配置加载、报告
├── decision/     # 置信度、不可判定、自监控、触发器
├── normative/    # 模式、权重、上下文标签
├── geometry/     # 能量景观、度量、Christoffel、scipy/CasADi 求解器
├── physics/      # RC 模型、数据中心冷却模型、在线辨识
├── causal/       # 确定性 SCM、数据驱动结构方程、反事实分析
├── multiscale/   # 多尺度粗粒化
└── engine.py     # 顶层编排引擎（滚动时域 MPC + 安全降级）
examples/         # 28 个示例/实验脚本（见 examples/README.md）
tests/            # 38 个测试（行为级断言，CasADi 缺失自动跳过）
docs/             # 架构 / 技术报告 / 数学定位 / 图表
```

## 📚 文档

- [docs/architecture.md](docs/architecture.md) — 分层架构
- [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) — 技术报告（含完整实验数据）
- [docs/RIEMANNIAN_CONTROL_THEORY.md](docs/RIEMANNIAN_CONTROL_THEORY.md) — 数学定位
- [docs/ENERGYPLUS_INTEGRATION.md](docs/ENERGYPLUS_INTEGRATION.md) — EnergyPlus/BOPTEST 接入指南
- [docs/multi_zone_and_solver.md](docs/multi_zone_and_solver.md) — 两区域模型与求解器演进
- [CONTRIBUTING.md](CONTRIBUTING.md) — 贡献指南

## 🗺️ Roadmap

- [ ] 真楼试点与实测数据（IPMVP 口径）
- [ ] BACnet / Modbus 真实接入（接口已预留）
- [ ] 两区域配置化支持
- [ ] 供热 / 蓄能 / 电池储能场景
- [ ] 主动放冷削峰模型的容量优化

## 📄 License

[MIT](LICENSE)
