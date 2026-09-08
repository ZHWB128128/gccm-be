# BOPTEST 试验台测试数据（2026-08-30）

运行环境：私有 Linux 测试设备（Debian/Ubuntu 系），项目部署于 /opt/gccm-pilot。
本目录数据为标准试验台验证的原始摘要，生成脚本见 examples/。

## 1. ABAB M&V 演练（bestest_air，每期 24h，`boptest_abab_drill.csv`）

| 期间 | 控制器 | 电费(元) | 违温 | 峰时均功率 | 室外均温 |
|---|---|---:|---:|---:|---:|
| A1 | 恒温器 | 0.40 | 25.0% | 1.113 kW | 15.2°C |
| B1 | GCCM | 0.21 | 26.4% | 0.779 kW | 6.6°C |
| A2 | 恒温器 | 0.34 | 38.9% | 1.021 kW | 10.3°C |
| B2 | GCCM | 0.25 | 28.1% | 0.904 kW | 12.6°C |

配对节省 A1→B1 = 46.1%，A2→B2 = 26.3%。**注意**：24h 期间天气漂移大
（A1/B1 室外均温差 8.5°C），46.1% 偏乐观；正式数字需 5~7 天期间 + 回归归一化。

## 2. comfort_weight 扫描（bestest_air，同起点 24h，`boptest_cw_sweep_*.json`）

硬约束开启时 cw∈{15,50,150,500} 轨迹逐位相同（权重被硬约束覆盖，无效旋钮）；
关硬约束后权重生效（cw 15→500 使超温 1.4%→0.3%，电费略增）。
违温主导项是夜间低温（18.8%，制冷-only 执行器无法修复）。
**结论：试点保持 enforce_comfort_constraints=true，cw=15 即可。**

## 3. 双区试验台演练（twozone_apartment_hydronic，`boptest_twozone_summary.json`）

### 3.1 机制验证

双区引擎全链路（REST 对接/每区设定点/每区舒适带硬约束/限速映射/带内钳制/CSV）
端到端跑通。

### 3.2 性能与模型结构

默认参数下 GCCM 能耗 +49%（模型预测散热远大于实际 → 持续过度供暖）。
为此 `TwoZoneRCBuildingModel` 新增可选地板蓄热层（`with_slab`，7 状态：每区
T_slab，Q 先注入蓄热层再经 r_slab 缓释进空气），CasADi/config/引擎全链路兼容。
带蓄热层辨识重跑：RMSE 1.154°C（结构退化解决），但 GCCM 能耗仍 +182%——
剩余根因是**设定点-only 执行器对地暖延迟的控制权限不足**。直接执行机构
（区阀门 0~1 / 泵 kg/s / 供水温度通道）三轮迭代：量纲错误两轮（通道规格
须逐字读 BOPTEST doc，`oveM*Z` 是阀门开度不是流量），第三轮量纲正确后
电耗读数仍超物理范围（3.7 万 kW，水环路内部与覆盖输入存在耦合）。
**结论**：①供暖场景的控制权限问题确认存在；②直接执行机构需要针对该
FMU 水环路结构专门研究（下一步：读 Source/Modelica 源码或换
bestest_hydronic_heat_pump）；③空气系统测试台 multizone_office_simple_air
模型假设匹配，可直接使用。


## 4. 空气系统两区演练（multizone_office_simple_air，Sou/Nor）

双区引擎驱动 5 区标准测试台的两区 24h×2 臂，480ms/步零崩溃——机制验证通过。
本窗口楼栋不过热（AHU 0.4 风量 + 免费冷却），配对差异不显著
（有人时段违温 基线 35.4/62.5% vs GCCM 37.5/60.4%）——如实报告"场景无差异"。

## 5. KPI 口径：有人时段违温率

`gccm_be/app/pilot_log.violations_occupied()`（7~19h）。全时段口径下约 90% 的
"违温"来自夜间无人漂移（ABAB 与 cw 扫描两轮演练反复证实），正式 M&V 以
有人时段口径为准。

## EKF 状态观测器试验台验证（bestest_air）

`boptest_ekf_validation.csv` / `boptest_ekf_summary.json`：仅用带噪室温
（σ=0.2K）驱动 EKF 估计墙温等隐状态，后半程一步温度预测误差 1.047K，
相比错墙温初值的开环预测（10.843K）改善 90%。试点闭环的状态来源由此验证。
