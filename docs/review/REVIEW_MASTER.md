# GCCM-BE 代码审查总报告（Master Review）

> 审查对象：`/root/GCCM`（GCCM-BE，建筑能源几何因果推理引擎 v0.1.0）
> 审查方式：5 路并行深度审查（核心引擎 / 物理·几何层 / 决策·规范·因果·多尺度层 / API·测试·示例 / 文档），代码已同步至本地 `E:\deepseek\gccm`
> 配套报告：`docs/review_physics_geometry.md`（物理几何专项，20 项问题）
> 实测验证：pytest 收集 16 个测试，**15 通过、1 失败**（casadi 缺失且无 skip 守卫）

---

## 一、项目定位与架构总评

**定位**：楼宇 HVAC 的滚动时域 MPC 控制器 + 决策诊断框架（研究型 MVP）。在 25~27°C 舒适约束下最小化电费，并在模型失配/预报噪声下提供"可解释的"安全降级。五层架构（app/decision/normative/geometry/physics）+ 顶层 `GCCMEngine` 编排，类型集中在 `types.py`，无循环导入，分层清晰。

**总体评价**：**骨架好、防御性设计思路成熟，但存在三类系统性问题——(1) 命名与方法论的过度承诺；(2) 引擎实例共享可变状态导致的状态污染；(3) 语义约定不统一（索引/时间步长/单位/阈值多套并存）。** 所有严重 bug 几乎都源于"800 行上帝 dataclass 把规划/辨识/诊断/安全/反事实状态全放同一实例"这一设计决策。

---

## 二、核心发现（按主题）

### 主题 1：方法论命名过度承诺（重点）

| 概念 | 实际实现 | 评价 |
|---|---|---|
| 黎曼测地线/Christoffel 联络 | 动能项=状态增量 L2 正则；**度量矩阵不正定**（coupling=1.0 时 det=-0.96<0，"动能"可为负）；解析 Christoffel 公式对 i≠j 产生虚假分量（正确应为 0 却得 ±α/2）且为默认启用版本；`metric_state_dependence` 从未传入引擎 → 默认管线修正恒为 0 空转 | 🔴 比喻包装。文档自己承认"离散时间最小作用量近似，非严格黎曼测地线"（这点诚实） |
| 哥德尔边界 | 三个布尔阈值 AND（err>0.8 ∧ eig<0.01 ∧ 自指误差>0.5），与哥德尔不完备定理无数学关联；且因诊断器 OR 结构，实际几乎总由置信度分支触发，哥德尔门近乎装饰 | 🔴 概念包装（功能上是个合理的保守三重门控） |
| 重整化群流 | 一次均值/标准差聚合 + 硬编码阈值（27.0/26.0/0.5），无 RG 迭代/不动点/跨尺度自相似；输出 action 无消费方 | 🔴 概念包装（实质是"两级聚合报警器"） |
| SCM 因果推断/do-干预 | do 算子语义在拓扑序下正确，但零噪声确定性（两次求值之差）、系数手拍（0.5 无出处）、temperature 方程不含室外温度/太阳辐射/人员热源（与声称来源的 RC 模型脱节）；`DataDrivenSCM` 只是"列序即因果序"的顺序回归，含中介控制/混杂问题 | 🔴 玩具模型。引擎把玩具 `causal_effect` 写进实际输出 `diagnosis.details`，无标注易被误读 |
| 自监控 SelfMonitor | AR(1) 最小二乘预测自身误差 + 滑动窗口发散检测，已接入主循环（自适应裕度/黎曼耦合/降级恢复） | ✅ 真数学，四个"哲学概念"里唯一名副其实 |
| 反事实分析 | 引擎私有 `_run_counterfactual` 通过 `_rollout_mode` 真实滚动仿真（语义正确）；但公开的 `CounterfactualAnalyzer` 是**死代码**（仅测试引用），README 却声称它"自动运行"——文档失实 | ⚠️ 真东西 + 双轨 + 文档失实 |

### 主题 2：引擎状态污染（最本质的设计缺陷）

`GCCMEngine` 所有状态（`_warm_start`、`_safe_integral`、`_degraded`、`mode_manager.current_mode`、`_last_curvature_min` 等）都是实例可变字段，导致：

- 🔴 **反事实/降级滚动污染真实控制**：`_rollout_mode` 在真实步进前直接调 `self.optimize`/`_feedback_safe_control`，假设轨迹写入 `_warm_start`、污染 PI 积分、可 `mode_manager.switch` 并置 `_degraded`——**一次"假设性"滚动能让真实引擎永久进入降级模式**
- 🔴 **降级恢复失效**：不可判定分支先 switch 到 safe_mode 再保存 `_pre_degradation_mode = mode`（此时已是 safe_mode），原模式永久丢失，恢复逻辑变无操作（笔误级 bug，engine.py:358-362）
- 🔴 **API 线程安全**：ThreadingHTTPServer 每请求一线程但 `optimize` 无锁写共享状态，并发 POST /control 竞态；模式切换跨请求泄漏无会话隔离
- 修复方向：把可变状态收敛为可快照/恢复的会话状态对象，或反事实用克隆引擎执行

### 主题 3：求解器健壮性

- 🔴 **CasADi 失败崩溃**：IPOPT 异常返回空 `controls`，引擎 `trajectory.controls[0]` 无守卫 → IndexError，无降级（engine.py:330, casadi_solver.py:272-289）
- 🔴 **CasADi dt=None 崩溃**：`_casadi_step` 执行 `None*MX`（引擎默认 dt=None）
- 🟠 **求解失败静默吞掉**：`nit==0` 一律算成功（x0 即最优与提前退出无法区分）；`ControlDecision` 无 success 字段；SLSQP 失败无回退
- 🟠 **CasADi 指标失真**：返回 total_cost 恒 0、costs 空
- 🟠 **两后端目标不一致**：scipy 用部分负荷 COP vs CasADi 用线性 COP → 对比不可比
- 🟠 **鲁棒模式空场景静默降级**：`use_casadi_robust=True` 但 `robust_scenarios` 为空时无告警走普通 MPC

### 主题 4：语义约定不统一（多区域支持缺陷）

- 🔴 **电价索引硬编码 `w[3]`**：engine.py:389/416/673、landscape.py:115、context.py:56、triggers.py:29、casadi_robust_solver.py:121 全部硬编码；两区域 external 布局 `[T_out, solar_A, solar_B, occ_A, occ_B, price]` 下 price 在索引 5——**把 occ_A 当电价**。casadi_solver.py:229 已正确处理（`w[5] if numel()>5 else w[3]`），说明作者知道但未统一
- 🟠 **时间步长不一致**：`self.dt or 0.25` vs 模型默认 1/12 → 反事实指标与真实系统不同时间尺度，对比结论失真
- 🟠 **solar 双重增益**：docstring 称 solar 已是"等效热功率 kW"，代码却再乘 solar_gain=0.05/0.08 → 太阳得热 0.015~0.024 kW 被 occ 淹没
- 🟠 **阈值体系多套**：电价 3 个阈值（0.9/1.0/1.2）、温度 6 个（18/22/25/26/27/28）并存；27.5°C 已超舒适上限却不触发任何切换
- 🟠 **多区域决策盲区**：triggers/context 只检查 `state.x[0]`（A 区），B 区越限不触发；scipy 硬约束也只约束 x[0] 而 casadi 约束所有 T_air*

### 主题 5：物理模型

- 🟠 **显式 Euler 无稳定性保护**：辨识门限允许 r_air=0.05, c_air=0.1 → λ·dt≈17 指数发散无检测
- 🟠 **RLS 数值健壮性**：遗忘因子 λ<1 下 P 无对称保持/正则，长期可能非正定
- 🟠 **MockExternalInputProvider 固定 4 维**，两区域模型需 6 维 → IndexError
- ✅ 二阶/五阶 RC 模型结构正确、HVAC 电功率换算（|Q|/COP + 部分负荷惩罚）合理

### 主题 6：控制逻辑正确性

- 🟠 **worst_case 反馈项符号反了**（engine.py:610-611）：`fb = kp*(T_air - setpoint)` 过热时 >0，`q = min(0, required + fb)` → **越热制冷越少**；与 `_feedback_safe_control` 方向相反
- 🟠 **comfort_margin 约束层失效**：margin 只传给 landscape 影响代价，硬约束仍用未收紧的 comfort_min/max——"约束收紧"名存实亡，且两套目标互相矛盾
- 🟠 **模式决策 4 处竞争**：forced_mode > 曲率自适应 mode_override（绕过决策层）> diagnoser 建议 > 当前模式；`ModeManager.suggest_mode` 是死代码
- 🟠 **自监控/降级链路多处断**：`model_mismatch` 通道死输入（引擎从不填充）；`run_gccm`（compare_baselines.py:270）不传 prediction_error → 8 个复用它实验的降级/哥德尔路径不可达；API 也无法传 prediction_error

### 主题 7：测试与验证（实测）

- **15 过 1 败**：`test_engine_casadi_robust_backend` 无 skip 守卫且 casadi 未在 pyproject 声明（连可选依赖都没有）→ 全新环境必红
- **行为级断言为零**：只有形状/结构断言，没有任何"高温制冷、温度守带、省电、削峰"数值验证——能证明"不崩"，不能证明"控制正确"
- **盲区**：API 层、报告器、verify 脚本、示例全部零测试；错误路径/退化输入无测试
- **verify 三脚本**（verify_dist/fbpow/peak）：验证温度分布/反馈功率/峰值行为，但全部纯 print 无断言无退出码，不能当 CI 门禁；三份复制同一闭环循环，维护发散；`verify_fbpow.py:15` 把模块对象赋给 rc_identifier（调试残留）

### 主题 8：API 层

- 极简可用（GET /health、/status、POST /control），无第三方依赖
- 问题：异常全映射 400 + 内部异常字符串回显（服务端错误应 500）；Content-Length 解析在 try 外（畸形头线程无响应）；无输入校验（缺键裸 KeyError）；无请求体上限/CORS/日志；version 硬编码与包重复

### 主题 9：文档（README 14 处矛盾）

- 最典型 7 处：①"测地线未接入主循环"（README:238）vs"推荐启用 use_riemannian=True"（TR:183）；②"无 SCM/do-calculus"（README:188）vs"引擎默认 SCM 输出 do-干预效应"（README:222）；③模型失配违规 63.5% vs 52.1% 两套数字；④两区域 CasADi 电费 59.84 vs 58.94（B 区 0% vs 6.2%）三套数字；⑤省电 13.4%/14.3%/17.1% 口径不一；⑥两区域求解 0.67s 快于单区域 1.11s 与预估矛盾；⑦"三大创新支柱均已进入主循环"（TR:118）vs"未接入"（README:238）
- 优点：数据诚实（主动披露噪声下违规 17~24%、失配 63.5%、Riemannian 对噪声不利），实验记录详实，有"数学定位声明"自我降格
- 缺点：过程日志属性重、缺最终状态总览、无依赖清单、目录树漏 causal/multiscale 两包

---

## 三、严重度分级问题总表（合并去重后）

### P0（会导致崩溃/错误结果/物理不成立）
| # | 问题 | 位置 |
|---|---|---|
| 1 | 度量矩阵不正定，黎曼几何不成立（det<0，"动能"无下界） | metric_tensor.py:50-54, landscape.py:52-54 |
| 2 | 解析 Christoffel 公式错误（i≠j 虚假分量），且为默认启用版本 | riemannian.py:104-130 |
| 3 | CasADi 失败返回空 controls，引擎 `controls[0]` 崩溃无降级 | casadi_solver.py:272-289, engine.py:330 |
| 4 | CasADi dt=None 时 `None*MX` 崩溃（引擎默认 dt=None） | casadi_solver.py:78,60 |
| 5 | 反事实滚动污染真实引擎状态（warm_start/积分/降级/模式） | engine.py:402-423, 760-779 |
| 6 | solar 单位矛盾：标称 kW 再乘增益，太阳得热被淹没 | models.py:32,57,94,127 |
| 7 | 降级前模式被 safe_mode 覆盖，恢复永久失效 | engine.py:358-362 |

### P1（行为偏差/配置失效/对比失真）
| # | 问题 | 位置 |
|---|---|---|
| 8 | 电价索引硬编码 w[3]，两区域读到 occ_A | engine.py:389, landscape.py:115, triggers.py:29, casadi_robust_solver.py:121 等 6 处 |
| 9 | worst_case 反馈项符号反：越热制冷越少 | engine.py:610-611 |
| 10 | comfort_margin 不作用于硬约束，"约束收紧"名存实亡 | engine.py:238-253 |
| 11 | 求解失败静默（nit==0 误报成功、无 success 透传） | geodesic.py:201, engine.py:301-307 |
| 12 | 时间步长 0.25 vs 1/12 不一致，反事实结论失真 | engine.py:326/413/670, models.py:33 |
| 13 | 显式 Euler 无稳定性保护（λ·dt≈17 发散） | models.py:67, engine.py:457-463 |
| 14 | 两后端 COP 模型不一致，scipy vs CasADi 对比不可比 | models.py:199, casadi_solver.py:227-228 |
| 15 | CasADi total_cost 恒 0、costs 空，引擎指标失真 | casadi_solver.py:291-298 |
| 16 | SCM 玩具化（零噪声/手拍系数/无外部输入），效应值无物理意义却进输出 | scm.py:38-62, engine.py:691-702 |
| 17 | API 线程安全缺失 + 模式跨请求泄漏 | api.py:27, engine.py:307/348/364 |
| 18 | 自监控/降级链路断：run_gccm 不传 prediction_error（8 实验失效）、model_mismatch 死输入、API 无法传入 | compare_baselines.py:270, engine.py:155 |
| 19 | casadi 测试无 skip 守卫 + 依赖未声明，新环境整套必红 | tests/test_integrated_loop.py:280-304, pyproject.toml |
| 20 | hard_benchmark --q-max 对 MPC/GCCM 静默失效，基线口径不一致 | hard_benchmark.py:104-127 |
| 21 | 阈值体系多套（电价 0.9/1.0/1.2，温度 6 值并存），27.5°C 不触发切换 | triggers.py:16/35, context.py:28-30, engine.py:51 |
| 22 | 多区域决策只看 x[0]（B 区盲区），scipy 硬约束同病 | triggers.py:34, context.py:44, geodesic.py:150-159 |

### P2（中等）
| # | 问题 |
|---|---|
| 23 | 模式决策 4 处竞争 + ModeManager.suggest_mode 死代码 + 曲率 mode_override 绕过决策层 |
| 24 | CounterfactualAnalyzer 死代码 vs README 声称自动运行（文档失实） |
| 25 | 多尺度/公理突变闭环未接通（action 无消费方） |
| 26 | 哥德尔门几乎不触发（OR 结构下置信度分支抢先）+ severity 未消费 |
| 27 | 两套度量实现公式不同（MetricTensor vs EnergyLandscape.metric），引擎只用后者 |
| 28 | metric_state_dependence 从未传入引擎，Riemannian 修正恒为 0 空转 |
| 29 | 曲率自适应阈值过敏感（-1e-3 + 非光滑 Hessian），时域/mode 抖动 |
| 30 | DataDrivenSCM 空数据 IndexError、欠定静默 |
| 31 | 测试行为级断言为零；API/报告器/verify/示例零测试 |
| 32 | verify 三脚本纯 print 无断言无退出码 + 三份复制循环 + 私有 API 依赖 + verify_fbpow.py:15 模块对象赋值 |
| 33 | API 异常全 400 + 异常字符串回显、Content-Length 在 try 外、无输入校验、无请求体上限 |
| 34 | 启发式初值仅首次生效：horizon 变化后既无 warm start 也无启发式 → x0=0 |
| 35 | 恢复温度门控用未收紧边界（与 effective_comfort_max 不一致） |
| 36 | RLS 遗忘因子下 P 无对称保持/正则，长期可能非正定 |

### P3（轻微/清理项）
- 死参数 `safe_max_cooling`（_safe_cool_cap 硬编码 2.5/4/6，用户设置无效）、死字段 `_safe_prev_error`
- 4 处未用导入（engine.py:11 CounterfactualAnalyzer、scm.py:7 numpy、context.py:7 numpy、modes.py:5 Dict）
- 死代码：counterfactual.py:70 两分支相同、scm.py:51 r_wall、fair_compare.py:45 立即覆盖、run_strict_rule
- 重复代码：apply_rc_identification 与 identification_trusted 校验重复、滚动统计循环重复、running/terminal cost 舒适项重复
- 魔法数字泛滥（26.5/27.0/25.0/0.3/0.5/0.8/2.5/4/6/1.2/0.05 等无命名常量）
- 示例导入风格混用（裸导入 vs examples. 包式）互斥失败；两区域模型 6 维外部输入 vs 默认数据源 4 维
- `SystemState(x=标量)` 生成 0 维数组后续 x[0] 越界；Trajectory.success 三条路径语义不一致
- 文档 7 处自相矛盾 + 目录树漏 causal/multiscale + 无依赖清单

---

## 四、概念真实性总评

```
真东西（✅）：RC 物理模型、AR(1) 自监控、反事实滚动仿真、影子预测验证、
           物理合理性门控、滞回降级恢复、安全降级链（feedback/worst-case）
比喻包装（🔴）：黎曼测地线/Christoffel（=加权 L2 正则 + 错误公式 + 空转）、
             哥德尔边界（=三重阈值 AND）、重整化群流（=两级聚合报警）、
             SCM 因果推断（=确定性玩具结构方程）、"曲率分析"（=代价 Hessian）
死代码/断链（⚠️）：CounterfactualAnalyzer、ModeManager.suggest_mode、公理突变 action、
                  model_mismatch 通道、safe_max_cooling、MetricTensor 双实现之一
```

**一句话总评：数据诚实、命名浮夸、闭环多处未接通、文档前后欠调和。**

---

## 五、改进路线图（按优先级）

**P0 修复（控制正确性与崩溃）**
1. 求解结果有效性统一检查：空 controls / success=False → 强制安全控制分支 + 记录 `solver_failure`；`ControlDecision` 增加 `solver_success` 字段
2. 引擎状态快照/恢复（或克隆引擎）跑反事实，保证纯"旁观"；修正 `_pre_degradation_mode` 保存时序
3. 修复 worst_case 反馈符号（与 `_feedback_safe_control` 对齐）+ 高温场景单测
4. 度量正定性约束（|coupling| ≤ √(g_ii·g_jj) 断言）；重写或删除解析 Christoffel 并补一致性单测；`metric_state_dependence` 接入引擎或从文档摘牌
5. 统一 dt 解析（`self.dt or simulator.building.dt` 一处）、修复 CasADi dt=None 崩溃与指标返回、统一两后端 COP

**P1 语义统一**
6. ExternalInput 提供 `price`/`t_out` 属性（按 labels 定位），消灭 w[3] 硬编码
7. 阈值收敛为引擎级配置注入各层（电价 1 个、温度以舒适带为唯一标准）
8. comfort_margin 同时收紧硬约束；多区域决策/约束遍历所有 T_air*
9. solar 单位语义统一；Euler 稳定性检查；RLS 正则化
10. 自监控链路接通：run_gccm 透传 prediction_error、API 支持、model_mismatch 通道填充

**P2 收尾与测试**
11. casadi 测试加 `pytest.importorskip` + pyproject 声明可选依赖；补行为级断言（高温制冷、守带、省电、削峰）
12. API 层补测试（并发/畸形请求/序列化）；verify 脚本改为断言 + 退出码 + 复用 run_mpc
13. 清理死代码（未用导入、死分支、死参数、verify_fbpow.py:15 调试残留）；删除或接入 CounterfactualAnalyzer/公理突变闭环
14. 文档更新：调和 7 处矛盾、补依赖清单、更新目录树

---

## 附：审查产物清单

| 产物 | 路径 |
|---|---|
| 项目代码（同步自远程） | `E:\deepseek\gccm\`（71 文件，~297KB） |
| 物理几何专项报告 | `E:\deepseek\gccm\docs\review_physics_geometry.md` |
| 本总报告 | `E:\deepseek\gccm\docs\REVIEW_MASTER.md` |
| 远程项目路径 | `/root/GCCM`（Ubuntu 26.04，root@192.168.31.44） |
