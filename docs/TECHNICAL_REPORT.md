# GCCM-BE 技术报告

> 本文是项目的**终态技术报告**：模型、控制、辨识、验证与结论的完整快照。
> 每个数字可由 `examples/` 中记录在案的脚本与 seed 复现；标准试验台（BOPTEST/
> EnergyPlus）验证的方法与原始数据见 [docs/BOPTEST.md](docs/BOPTEST.md) 与
> [docs/data/](docs/data/README.md)；设计决策的推导见
> [docs/CONTROL_DESIGN.md](docs/CONTROL_DESIGN.md)。

## 1. 系统概述

GCCM-BE 是面向楼宇 HVAC 与数据中心制冷的**鲁棒灰箱 MPC 引擎**，核心是一个
五层流水线：

```text
物理层(gccm_be/physics)      RC 模型×5(注册表分发) + NWP/HA 适配器 + 在线辨识
优化层(gccm_be/geometry)     能量景观 + 加权 MPC 求解器(SLSQP/CasADi/鲁棒)
规范层(gccm_be/normative)    模式(comfort/balanced/energy/demand_response)→权重
决策层(gccm_be/decision)     置信度 / 不可判定门控 / 反事实 / 触发器 / 自监控
应用层(gccm_be/app)          Web 控制台(introspection 驱动) / REST API / 配置 / CSV+KPI
安全层(gccm_be/control)      SafeController 兜底控制律(worst_case / feedback)
顶层: engine.py 编排 + 降级恢复(hysteresis/最短时限/温度门控)
```

运行时依赖仅 numpy/scipy；CasADi 可选（求解提速 ~250×，16ms/步）。

## 2. 模型

| 模型 | 状态 | 适用 | 配置 |
|---|---|---|---|
| RCBuildingModel | 2（空气/墙） | 单区分体空调 | `model.type: single_zone` |
| TwoZoneRCBuildingModel | 5（可含蓄热层 7） | 双区（`with_slab` 支持地暖热惯性） | `model.type: two_zone` |
| ThreeRCBuildingModel | 3（空气/墙/家具） | 单区太阳蓄热 | `model.type: three_rc` |
| NonlinearRCBuildingModel | 3 | 开窗通风/除湿/VAV 非线性 | `model.type: nonlinear_rc` |
| DataCenterCoolingModel | 2（冷通道/蓄冷罐） | 机房+蓄冷罐 | `model.type: datacenter` |

全部模型经 `physics/registry.py` 协议注册（state/control/external 标签自描述 +
`step`），config 按 `model.type` 分发：single_zone/two_zone 走严格校验专属
builder，其余走通用路径。**新 HVAC 类型 = 一个注册调用 + CasADi 分支**，
不再逐类写 config 代码。

HVAC 层支持每台独立容量（`unit_bounds`）、部分负荷 COP（额定基准按方向取：
制冷 |q_min| / 制热 q_max）、多设备电功率合计。

## 3. 控制

### 3.1 目标函数（能量景观）

每步代价 = 舒适（每区独立舒适带硬约束可开 + 带外二次惩罚）+ 电费
（分时电价×峰时惩罚）+ 控制平滑 + 储能价值项（防短视排空，适用蓄冷/电池）。
分区舒适权重由跨尺度序参量相关度驱动（RG→MPC 耦合）。

### 3.2 求解器

| 后端 | 单步耗时 | 场景 |
|---|---:|---|
| scipy SLSQP（软约束）/ L-BFGS-B | 0.5–4s | 单区原型、边缘部署 |
| CasADi + IPOPT | **16ms** | 多区、硬约束、实时 |
| 鲁棒（scipy/CasADi） | — | 多场景共享控制序列，`robust.delta` 可由实测 NWP 误差校准 |

鲁棒 delta 校准：试点位置真实 NWP 误差实测（温度 P95 4.6K、辐射 P90 66%），
30% 失配场景扫描结论 **0.15 最优**（违温 6.2%→1.0% 且能耗不增；加大到 0.3/0.4
违温不变、纯付保险费）。工具：`tools_local/calibrate_robust.py`。

### 3.3 硬约束与权重的分工

试验台 cw 扫描结论：硬舒适约束开启时 comfort_weight 不影响行为（被约束覆盖）；
关硬约束后权重生效（cw 500 使超温 1.4%→0.3%）。**默认：硬约束开启 + cw=15**。

## 4. 辨识与在线自适应

| 能力 | 实现 | 门控 |
|---|---|---|
| 单区 RC 在线辨识 | RLS 6 参数（a1..a6） | 物理合理性范围 + 影子预测验证 |
| 双区逐区辨识 | `RCOnlineIdentifier(zone=...)` ×2 | 任一区失稳 → 整体关闭信任 |
| 参数换算 | R = param/C（修正过 1/(param·C) 的失真 Bug） | — |
| 自监控 | AR(1) 残差自预测，驱动预防性降级与噪声自适应裕度 | — |

bestest_air 灰盒辨识演进（诚实记录）：默认参数 −17% → 退化拟合 +15.8% →
ARX 单状态 −56% → 灰盒 2 状态 RMSE 1.18°C → **3 状态 RMSE 1.047°C**
（增加家具热容）。已知残留：c_furn 顶辨识上界，墙/家具时间常数未完全分离。

## 5. 安全降级链

| 触发 | 响应 |
|---|---|
| 求解失败/异常/空轨迹 | 预测状态置空 → `SafeController.feedback` 免模型 PI 控制 |
| 求解成功但不可靠 | 告警入诊断；置信度下降 |
| 预报漂移/自指误差高 | 预防性反馈降级（`preemptive_feedback`） |
| 三重条件共识（误差+曲率+自预测同坏） | 不可判定 → 安全模式 |
| 恢复 | 滞回 + 最短降级期 + 温度门控 |

**可行性定位（诚实声明）**：本引擎不提供递归可行性的形式化保证——硬约束模式下
SLSQP 不可行时会落入安全降级链兜底。工程上等价（失效模式被系统性覆盖并实测），
但不应被解读为带 formal guarantee 的 MPC。

试验台失配压力测试（+30% 失配模型）：裸 MPC 室温失控 31.8°C；
**GCCM 安全链封顶 27.9°C**，违温 40.1%→23.1%，安全代价 0.05 元。
该测试还暴露并修复了制冷型 HVAC（q_max=0）兜底从不制冷的真 Bug——
内部双向 HVAC 仿真是测不出来的。

## 6. 试点工具链

| 能力 | 实现 |
|---|---|
| 真实天气预报 | Open-Meteo 适配器（免 key、`timezone=auto`、失败回退 mock） |
| 真楼数据通道 | Home Assistant 适配器（读传感器/写设定温度，传输可注入） |
| 试点闭环 | `ha_pilot_loop.py`（`--once` 配 cron 15 分钟） |
| M&V 数据凭证 | `pilot_log.append_row`（逐周期 CSV，表头一次写入） |
| KPI 口径 | `violations_occupied()` 有人时段违温率（全时段口径 ~90% 违温来自夜间无人漂移） |
| ABAB 协议 | `boptest_abab_drill.py` 已在试验台全流程验证 |

## 7. 验证结果汇总

### 7.1 内部仿真

| 场景 | 结果 |
|---|---:|
| 单区 fair_compare（24h） | 省电 13.4%、0% 违温；Pareto 最优 18.2% |
| 数据中心（尖峰 5 元） | 省电 **29.0%**、0% 违温、峰时 −7.2%（含泵功+罐损的诚实物理；此前的 34.9% 缺失了这两项能耗） |
| 模型失配（+30%） | 鲁棒 MPC 违温 32.3%（经典 MPC 62.5%） |
| 双区 two_zone_compare | 违温 A 24.0→5.2%、B 43.8→14.6%，电费 −2.0% |
| **经典线性 MPC 锚点**（独立实现，工厂模型预测，无几何层/无安全链/无辨识） | 同场景对比：违温 **87.5%→1.0%**、电耗 7.58→6.27 kWh（**GCCM 省 17% 且舒适大幅改善**）；经典 MPC 求解成功率 92/96（SLSQP 不可行时无兜底） |
| 因果归因（Shapley） | 可加性残差浮点级；样例：天气 73%、策略 34%、电价 −7.5% |

### 7.2 BOPTEST 标准试验台

| 验证 | 结果 |
|---|---|
| 常规对比（bestest_air 48h） | 省电 13.8%、违温更优 |
| 模型失配压力 | 安全链封顶 27.9°C（裸 MPC 31.8°C 失控） |
| AI 机房多日 | 省电 50–64%、0% 违温（峰值更高为谷时蓄冷代价，已标注） |
| ABAB 演练 | 配对节省 46.1%/26.3%（天气未配平限制已标注），协议全流程验证 |
| cw 扫描 | 硬约束开启下权重无效 → 配置结论固化 |
| 双区水暖 | 机制通过；蓄热层修辨识（RMSE 1.154）；执行器权限不足（+182%）如实归档 |
| 双区空气 | 480ms/步零崩溃；本窗口场景无差异（如实报告） |
| 不确定度压测 | 阴性结果归档（该扰动未压出真实压力，不夸大鲁棒层） |

详见 [docs/BOPTEST.md](docs/BOPTEST.md) 与 [docs/data/](docs/data/README.md)。

### 7.3 设计决策的统计证据

黎曼动能项（平滑正则）在正确 KPI 上 24 seeds 显著（温度爬升 RMS −0.0364、
违温 −7.29pp，p<0.0001），代价为电费 +0.45（Pareto 前沿点）；Christoffel 修正
在近线性系统上无显著影响（p≥0.22，可证伪负结论）。RG→MPC 分区权重在峰值
KPI 上 24 seeds 显著（−0.41°C，p<0.0001），gain 单调饱和结构可解释。详见
[docs/CONTROL_DESIGN.md](docs/CONTROL_DESIGN.md)。

## 8. 工程质量

| 维度 | 状态 |
|---|---|
| 测试 | 171 项行为级（168 passed + 3 CasADi skip），Windows/Linux 双平台 |
| 契约 | introspection 注册表 ↔ 控制台由 CI 双向校验（漏登记直接挂） |
| 结构 | engine God-Object 拆分完成（SafeController/RG→MPC/robust builder owner 化） |
| 模型扩展 | registry 协议（新 HVAC 类型 = 注册调用 + CasADi 分支） |
| 基线锚点 | 独立经典线性 MPC（`examples/classic_mpc_baseline.py`）：无 GCCM 依赖的同模型 SLSQP 滚动优化，作为"架构差异 vs 调参差异"的公平对照 |

## 9. 已知限制与路线

1. **实测数据为零**——全部数字来自仿真器；实楼试点（IPMVP）是下一里程碑，
   就绪清单见 [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md) §10。
2. **状态观测已过试验台验证**——`physics/observer.py` EKF 在 BOPTEST
   bestest_air 上完成闭环验证（`docs/data/boptest_ekf_summary.json`）：仅用
   带噪 T_air 测量（σ=0.2K），后半程一步温度预测误差 **1.047K**，相比错墙温
   初值的开环预测（10.843K）**改善 90%**；试点前剩余工作仅剩与真实传感器联调。
3. **供暖场景性能未达标**——水暖设定点-only 执行器权限不足（能耗 +182%），
   直接执行机构通道语义需读 Modelica 源码专门研究。
4. **HVAC 类型适配成本线性**——registry 收敛了 config 层，CasADi 分支/辨识结构/
   执行器映射仍每类一写。
5. **HVAC 系统层能耗仍有缺口**——风机能耗（VAV 占 20–30%）未计入
   `electrical_power`；数据中心泵功/罐损已加入（省电 34.9%→29.0%）。
6. **舒适口径边界**——仅空气温度，无 PMV/操作温度/辐射不对称（地暖场景辐射
   主导），无 ASHRAE 62.1 新风/CO₂ 约束；自然通风门控未考虑 IAQ。
7. **真楼预报误差的鲁棒性**——校准工具已就绪，长期需以滚动实测误差分布更新。
