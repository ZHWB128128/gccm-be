# 示例脚本清单

统一运行方式：`PYTHONPATH=. python3 examples/<脚本名>.py`

## 核心演示

| 脚本 | 说明 |
|---|---|
| `demo.py` | 最小演示：单区域单步优化 + 24 步闭环 + Markdown 报告 |
| `demo_report.py` | 一键生成对比图表（温度/功率/指标/诊断时间线），存 `output/` |
| `datacenter_demo.py` | 数据中心冷却 MPC：谷时充冷、峰时放冷（`--peak-price` 控制价差） |
| `external_simulator_loop.py` | 通过外部仿真器适配器闭环（EnergyPlus 接口演示） |

## 对比实验

| 脚本 | 说明 |
|---|---|
| `compare_baselines.py` | 规则 / PID / GCCM 对比（被 8+ 脚本复用的公共基础设施） |
| `fair_compare.py` | 公平基线：规则、PID、严格规则、严格舒适 PID、GCCM |
| `hard_benchmark.py` | 固定协议硬基准（4 方法 × 场景，`--q-max` 全方法生效） |
| `parameter_scan.py` | energy_weight × peak_penalty 扫描 |
| `pareto_sweep.py` | energy_weight × comfort_margin Pareto 前沿 |
| `pareto_validation.py` | 最优 Pareto 点多 seed/场景验证（`--energy-weight` 可指定） |
| `two_zone_compare.py` | 两区域 GCCM vs 严格 PID |
| `two_zone_casadi.py` | 两区域 CasADi+IPOPT 硬约束对比 |
| `two_zone_baselines.py` | 两区域：GCCM vs 集中式舒适优先 MPC vs 分散式独立 MPC |
| `scenario_sweep.py` | 多场景鲁棒性（极端天气/噪声/高电价） |
| `model_mismatch_experiment.py` | 模型失配实验（`--comfort-margin` 扫描） |
| `complex_realistic_test.py` | 综合：失配+噪声+尖峰电价+鲁棒 MPC |
| `online_decision_test.py` | 预报噪声下在线误差反馈决策 |
| `riemannian_compare.py` | Riemannian 修正开关对比 |

## 鲁棒 MPC 实验链

| 脚本 | 说明 |
|---|---|
| `robust_mismatch_loop.py` | RobustGeodesicSolver 失配闭环（几何层） |
| `casadi_robust_mismatch_loop.py` | CasADi 鲁棒 MPC 失配闭环（求解器级） |
| `engine_robust_mismatch_loop.py` | GCCMEngine + CasADi 鲁棒后端失配闭环 |
| `robust_backend_multiseed.py` | CasADi 鲁棒后端多 seed |
| `two_zone_robust_test.py` | 两区域 CasADi 鲁棒 MPC 闭环 |
| `two_zone_robust_multiseed.py` | 两区域鲁棒 MPC 多 seed |

## 方法演示

| 脚本 | 说明 |
|---|---|
| `scm_demo.py` | 确定性 SCM + do-干预演示 |
| `validate_data_driven_scm.py` | 数据驱动 SCM 因果效应 vs 真实物理 SCM |
| `riemannian_demo.py` | 黎曼测地线基础（Christoffel 符号） |
| `riemannian_theory_check.py` | AD 梯度 vs 数值梯度理论验证（需 casadi） |
| `axiom_test.py` | 公理突变三场景（电价尖峰/性能衰减/误差过大） |
| `philosophy_layers_demo.py` | 自监控/哥德尔/重整化等模块演示 |
