# GCCM-BE 完整审查报告 v2（当前状态）

> 审查时间：2026-08-16（第二轮全量复查）
> 覆盖：全部 79 个源文件（engine 869 行逐行、物理 5 文件、几何 10 文件、决策/规范/因果/多尺度 16 文件、API/示例/测试/verify、7 份文档）
> 实测：本地 pytest **24 passed, 1 skipped**；远程（含 casadi）**25 passed**
> 本轮新增审查对象：`building_simulator_adapter.py`、`external_simulator_loop.py`、`riemannian_theory_check.py`、`RIEMANNIAN_CONTROL_THEORY.md`、`ENERGYPLUS_INTEGRATION.md`

---

## 一、本轮修复清单（11 项，全部远程验证）

### 上轮 4 个 P0（本轮逐行验证：全部正确）

| # | 修复 | 验证结论 |
|---|---|---|
| 1 | 求解失败回退（空轨迹→反馈安全控制 + `solver_success` 字段） | ✅ 正确；`current_external` 恒可用；附带兜住 CasADi dt=None 崩溃 |
| 2 | `_pre_degradation_mode` 保存时序 | ✅ 核心正确（先保存后 switch + 已降级不覆盖）；本轮又修复了 mode_override 污染（见 #8） |
| 3 | worst_case 反馈符号（两处） | ✅ 正确，与 `_feedback_safe_control` 方向一致 |
| 4 | 反事实快照/恢复（11 字段） | ✅ 覆盖完备，引用级测试通过 |

### 本轮新修复（7 项，含对抗性测试发现）

| # | 问题 | 修复 |
|---|---|---|
| 5 | 🔴 **`ExternalInput` 导入缺失**：model_bias≠0 时 `_apply_model_bias` NameError 崩溃（对抗测试抓出） | engine.py 导入补上 `ExternalInput` |
| 6 | 🔴 **求解器抛异常仍崩溃**（回退只覆盖"空轨迹"不覆盖"异常"） | 所有 `solver.solve` 调用包 try/except，异常与空轨迹同等回退，异常信息入 `details["solver_failure"]` |
| 7 | 🔴 **求解失败步污染预测误差**：`predicted_next_state=state.copy()` 把状态变化误当模型误差喂给自监控/辨识器 | 失败步置 `None`（run_closed_loop 已判空跳过） |
| 8 | 🟠 **降级模式保存被 mode_override 污染**：曲率奇异时保存 balanced 而非管理器真实模式，恢复后永久改写 | 两处（preemptive + undecidable）改存 `mode_manager.current_mode` |
| 9 | 🟠 **反事实异常杀死整个闭环** | run_closed_loop 反事实块 try/except + `_run_counterfactual` 两路对称保护 |
| 10 | 🔴 **CasADi 硬约束 None 崩溃**（comfort_min/max 默认 None + use_casadi=True 即炸） | 仅当上下界非 None 时施加约束 |
| 11 | 🟠 **度量不正定**（coupling=1.0 时 det<0） | `metric()`/`metric_casadi()` 加正定性钳制 \|coupling\| < 0.999·√(g00·g11)，实测严格正定 |

**新增测试**：求解器抛异常回退、失败步预测不污染、降级保存+恢复、反事实零污染、越热制冷越强、高温必制冷——共 9 个行为级测试，全部通过。

---

## 二、各层完整审查结论

### 1. 引擎层（engine.py 869 行逐行）

- **架构**：五层组合根 + 滚动 MPC 编排，类型集中，无循环导入
- **健壮性**：修复后正常路径（求解成功/失败/不可判定/反事实）均不崩溃；**但"异常传播"与"错误语义传播"两链已闭环**（本轮 #6/#7）
- **残留**：`solver_unreliable` 只告警仍使用不可靠解（建议升级为强制安全回退）；鲁棒失败被名义解成功掩盖；`undecidable` 分支 `decision.mode=safe_mode` 但 weights/trajectory 仍来自原模式；`self_monitor.update` 判空不一致（861 行未判、321 行判了）；`_safe_prev_error` 死字段、`safe_max_cooling` 死参数

### 2. 物理层（models / external / online_id / 新增 adapter）

- **模型正确**：二阶/五阶 RC、HVAC 电功率换算合理；**Euler 显式积分无稳定性保护**（实测 dt<0.634h 才稳定，dt=1.0h 谱半径 2.156 发散，零检查）
- **solar 双重增益**（遗留）：标称"等效热功率"却再乘 0.05，净贡献 ~0.015kW 被 occ 淹没
- **新增适配器**：接口契约清晰（reset 前置校验、标签化测量），但 `step` 硬编码 dt=0.25 与内部默认 1/12 解耦、无外部输入维度校验（4 维输入 IndexError）、示例 plant 与模型零失配（演示价值有限）
- **online_id**：RLS 公式标准；`history` 无界增长（内存泄漏）；默认 dt=0.25 与模型 1/12 不一致

### 3. 几何层（10 文件全量）

**数学真实性结论（更新版）**：
- 公式是"真数学"（AD 版 Christoffel 正确），但**默认配置下整层空转**：`metric_state_dependence` 从未传入引擎（engine.py:67 死配置）→ Γ≡0 → 黎曼修正/测地线罚恒为 0
- 🔴 **数值 Christoffel 第三项沿 k 而非 l 扰动**（riemannian.py:39-41）：coupling≠0 时全部算错（实测偏差 0.088），对角度量下侥幸正确掩盖
- 🔴 **解析 Christoffel 对非 setpoints 维度赋非零导数**（riemannian.py:125-127）：实测 Γ[1,1,1]=0.1 vs 真值 0
- 🔴 **use_casadi 静默丢弃全部 riemannian 选项**（geodesic.py:84-106）——配置看似生效实则被忽略
- ✅ **度量正定性已修复**（本轮 #11）
- "曲率分析"= 代价函数 Hessian，**非黎曼曲率**（命名包装）
- 新 `riemannian_theory_check.py` 验证的是 AD 梯度 vs 数值梯度（1.79e-08 PASS）——只证明目标可微，不验证正定性（已由 #11 补齐）
- `geodesic.py:212` `success = result.success or result.nit==0` 误报成功（遗留）

### 4. 决策/规范/因果/多尺度层（16 文件）

（grep 交叉验证：与上轮结论一致，无新增变化）
- **自监控** AR(1) 最小二乘：真数学，唯一名副其实
- **哥德尔边界**：三重布尔 AND 包装；OR 结构下置信度分支抢先，哥德尔门近乎装饰；severity 未消费
- **SCM**：零噪声确定性玩具（系数手拍、温度方程无外部输入）；`DataDrivenSCM` 列序回归 + 空数据 IndexError
- **重整化流**：粗粒化+阈值，action 无消费方
- **模式**：4 处竞争入口；`ModeManager.suggest_mode` 死代码；曲率 mode_override 绕过决策层
- **阈值**：电价 0.9/1.0/1.2、温度 18/22/25/26/27/28 多套并存
- **多区域盲区**：triggers/context 只看 `x[0]`（B 区不触发）

### 5. 示例/测试/API（27 示例 + 24 测试 + API）

- **测试现状**：本地 24 passed 1 skipped、远程 25 passed；**本轮新增 9 个行为级测试**（高温制冷、越热越强、降级恢复、反事实零污染、失败回退×3）；但仍缺 API/报告器/verify 脚本测试
- **verify 三脚本**：纯 print 无断言无退出码；`verify_fbpow.py:15` 模块对象赋值调试残留
- **API**：无锁（线程不安全）、Content-Length 在 try 外、全 400+异常回显、无输入校验、prediction_error 无法传入
- **示例共性问题**：`run_gccm` 不传 prediction_error（8+ 实验降级路径不可达）；hard_benchmark `--q-max` 对 MPC/GCCM 静默失效；导入风格混用
- **新增示例实测**：`external_simulator_loop.py` 运行正常（两区收敛 27.2°C）；`riemannian_theory_check.py` 需 casadi

### 6. 文档（7 份）

- **旧 7 处矛盾仍未修**：测地线接入状态、SCM 定位、失配两套数字、两区域 CasADi 三套数字、省电口径、三大支柱
- **README 两处失实声明**："MetricTensor 已统一消除双实现"（实际 MetricTensor 已无人引用）、"CounterfactualAnalyzer 自动运行"（实际死代码）
- **新文档**：`ENERGYPLUS_INTEGRATION.md` 诚实（明确 Stub 状态+接入指南）；`RIEMANNIAN_CONTROL_THEORY.md` 数学推导成立但验证范围有限（未覆盖正定性，已修）
- 优点：数据诚实（主动披露负结果），实验记录详实

---

## 三、遗留问题总表（修复后）

### P1（语义/配置类，建议优先）
| # | 问题 | 位置 |
|---|---|---|
| 1 | **w[3] 电价索引**：两区域下读到 occ_A（6 处） | engine.py:411/438/701, landscape.py:115, context.py:56, triggers.py:29, casadi_robust_solver.py:123 |
| 2 | **dt 三套并存**：0.25 / 1/12 / 1.0，反事实指标与真实系统不同尺度；observe_step 记录 dt 错位致辨识标度错 3 倍 | engine.py 多处, geodesic.py:141-153, online_id.py:88 |
| 3 | solar 双重增益 | models.py:23/32/57 |
| 4 | Euler 无稳定性保护（dt≥0.634h 发散无检测） | models.py:67 |
| 5 | 两后端 COP 不一致 + total_cost 恒 0（3 个非 scipy 后端） | casadi_solver.py:223-228/294-295, casadi_robust_solver.py |
| 6 | SCM 玩具化 + 引擎输出未标注 | scm.py:38-62, engine.py:691-702 |
| 7 | API 线程安全 + 模式跨请求泄漏 | api.py:27 |
| 8 | run_gccm 不传 prediction_error（8+ 实验降级不可达） | compare_baselines.py:274 |
| 9 | hard_benchmark --q-max 对 MPC/GCCM 静默失效 | hard_benchmark.py:105/117 |
| 10 | 阈值多套并存（电价 3 个、温度 6 个） | triggers/context/engine |
| 11 | 多区域决策与 scipy 硬约束只看 x[0] | triggers.py:34, context.py:44, geodesic.py:168 |
| 12 | solver_unreliable 只告警不降级；鲁棒失败被名义解掩盖 | engine.py:343-344, 299-305 |
| 13 | 数值/解析 Christoffel 两处公式错误（coupling≠0 时） | riemannian.py:39-41, 125-127 |
| 14 | use_casadi 静默丢弃 riemannian 选项；--state-dep 参数静默无效 | geodesic.py:84-106, engine.py:67 |

### P2（收尾类）
- 模式 4 处竞争 + suggest_mode 死代码；CounterfactualAnalyzer 死代码 + README 失实；哥德尔门/severity；DataDrivenSCM 空数据；identifier history 无界；SCM 辨识后过期；适配器 dt 硬编码/无维度校验；API 错误语义/校验缺失；verify 脚本无断言；测试仍缺 API/报告器/verify 覆盖；魔法数字与死代码清理

---

## 四、整体评估

**修复后状态：演示级已稳固，生产级仍不足。**

- ✅ 正常路径与失败路径（空轨迹/异常/不可判定/反事实）全部不崩溃，有行为级测试守护
- ✅ 度量正定、求解状态可观测（solver_success）、降级恢复语义正确
- ⚠️ 遗留问题集中在**语义约定**（索引/时间尺度/阈值/单位）与**实验口径**（COP/指标），是两区域场景和真实部署的主要隐患
- ⚠️ "几何因果"命名溢价依旧（黎曼层默认空转、曲率非黎曼），文档 9 处矛盾/失实未清理

**下阶段优先级建议**：
1. **P1-1 w[3] 电价索引**：`ExternalInput` 增加按标签取数（`get("price")`），替换全部位置索引——30 分钟，收益最大
2. **P1-2 dt 统一**：解析一处 `self.dt or building.dt`，消除 0.25/1/12/1.0 三套并存
3. **P1-13 Christoffel 两处公式**（数值第三项沿 l、解析非 setpoints 置 0）+ AD 版一致性单测
4. **P1-8/9 实验口径**：run_gccm 透传 prediction_error、hard_benchmark 透传 sim
5. **文档清理**：修正 9 处矛盾/失实声明（竞赛/面试前必做）
6. **测试补强**：API 层测试、verify 脚本断言化、两后端代价一致性测试

---

## 六、追加更新（本轮收尾修复，远程 25 passed）

**① P1-1 w[3] 电价索引已修复**（收益最大项）
- `ExternalInput` 新增 `get(label)` 与 `price` 属性（优先按 `price` 标签定位；无标签时按约定回退 w[3]/w[5]）
- 替换 6 处 numpy 位置索引（engine.py×3、landscape.py、context.py、triggers.py）
- 两个 CasADi 求解器改为从 labels 计算 `price_idx`（替代 w[3]/w[5] 硬编码）
- 实测：单区域 1.5 ✓、两区域 1.8 ✓（旧代码读到 occ_A=1.0）、无标签回退 1.2 ✓

**② README 失实/矛盾清理（5 处）**
- 数学定位声明：明确 SCM 为确定性结构模型（示意性数值、非统计因果），消除"无 SCM"与"默认 SCM"矛盾
- 删除失实声明"MetricTensor 已统一消除双实现"→ 改为"统一由 EnergyLandscape.metric() 提供（含正定性钳制），MetricTensor 为历史参考"
- 删除失实声明"CounterfactualAnalyzer 自动运行"→ 改为"引擎内部 `_run_counterfactual` 真实滚动仿真，CounterfactualAnalyzer 为独立参考实现"
- "三大创新支柱"：修正"测地线尚未接入主循环"矛盾 → 明确 use_riemannian 已支持但 metric_state_dependence 未接入（默认修正为零）、噪声下不利建议关闭；"严格因果推断"降格为"确定性结构模型+反事实仿真"
- 模型失配两套数字：标注口径差异（margin=0 vs margin=0.9）

**遗留仍建议处理**：dt 三套并存统一（P1-2）、Christoffel 数值/解析两处公式（P1-13）、run_gccm 透传 prediction_error（P1-8）、两区域 CasADi 三套实验数字标注。

**③ 第一档（数字可信度）修复完成**（远程 25 passed）
- **run_gccm 闭环化**（compare_baselines.py）：透传 `prediction_error` 给 optimize；补全 self_monitor/observe_step/apply_rc_identification/online_identifier/model_bias 更新——**8+ 个实验的降级/自监控路径从此真正激活**（此前 GCCM 在这些实验里实际只是普通 MPC）；失败步（predicted=None）不污染误差通道；`w.w[3]`→`w.price`
- **hard_benchmark 口径统一**：classic_mpc/gccm 现在透传 `simulator=sim`——`--q-max` 对全部方法生效（此前 GCCM 锁死 Q_MAX=8，基线却在 Q_MAX=15，对比不公平）
- **dt 三套并存消除**：`GCCMEngine.__post_init__` 与 `GeodesicSolver.__post_init__` 统一解析 `self.dt = self.dt or building.dt`（默认 1/12h）；删除全部 `self.dt or 0.25` / `self.dt or 1.0` 回退链；`observe_step` 记录真实 `step_h`（修复 RC 辨识标度错 3 倍问题）
- **注意**：以上修复改变了实验行为，README 中的历史数字（省电 13~17% 等）需重新跑基准更新

**遗留仍建议处理**：Christoffel 数值/解析两处公式（P1-13）、两区域 CasADi 三套实验数字标注、API 线程安全、verify 脚本断言化。

---

## 五、本轮产物

| 产物 | 路径 |
|---|---|
| 修复后的核心代码（已同步远程并 25 passed） | `E:\deepseek\gccm\` ↔ `/root/GCCM` |
| 本报告 | `E:\deepseek\gccm\docs\REVIEW_MASTER_v2.md` |
| 物理几何专项（含实测数据） | `E:\deepseek\gccm\docs\review_physics_geometry.md` |
| 自动重连隧道（3090 端口） | `E:\deepseek\tunnel_3090.py`（任务 pwsh-128，HTTP 200） |
