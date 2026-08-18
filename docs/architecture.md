# GCCM-BE 架构落地映射

| 架构层 | 代码模块 | 核心职责 |
| --- | --- | --- |
| 应用层 | `gccm_be/app` | 轻量 REST API、报告生成 |
| 决策与诊断层 | `gccm_be/decision` | 公理突变触发、置信度评估、不可判定输出 |
| 规范层 | `gccm_be/normative` | 上下文标签、能量景观权重、多模式管理 |
| 几何推理核心层 | `gccm_be/geometry` | 状态流形、能量景观、测地线求解、曲率分析 |
| 物理世界仿真层 | `gccm_be/physics` | RC 建筑热工、HVAC、外部输入、仿真器 |

## 数据流

1. 物理层提供 `Simulator.step(state, control, external)` 作为状态转移函数。
2. 规范层根据外部情境生成 `EnergyLandscapeParams`。
3. 几何层构造 `EnergyLandscape` 并调用 `GeodesicSolver` 求最优控制序列。
4. 决策层基于曲率/误差计算置信度，必要时切换模式或输出不可判定。
5. 应用层暴露 `/control` 接口给楼宇自控系统。

## 扩展点

- 物理模型：替换/新增 `RCBuildingModel` 或接入真实传感器。
- 求解器：在 `GeodesicSolver` 内替换为 iLQR、SQP 等算法。
- 模式：扩展 `ModeManager.allowed_modes` 与 `WeightMapper.BASE_WEIGHTS`。
- 触发规则：扩展 `AxiomMutationTrigger` 为学习式触发。
- 通信：将 `app.api` 替换为 BACnet/Modbus/MQTT 适配器。
