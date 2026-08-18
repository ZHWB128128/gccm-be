# GCCM-BE 物理层与几何层代码审查报告

审查范围：`gccm_be/physics/`（models / external / online_id / `__init__`）与 `gccm_be/geometry/`（manifold / metric_tensor / landscape / geodesic / riemannian / curvature / robust_solver / casadi_solver / casadi_robust_solver / `__init__`），并交叉核对了 `types.py`、`engine.py`、`pyproject.toml`、README 与 `docs/`。

---

## 0. 结论摘要

- **物理层**：二阶/五阶 RC 模型结构合理、量纲基本一致，但存在太阳得热单位不一致（被双重折扣到可忽略）、显式 Euler 无稳定性保护、两区域外部输入维度不匹配等隐患。
- **几何层（重点）**：大部分"黎曼几何"是**比喻包装 + 正确的惰性实现**。默认管线中度量是常对角阵，Christoffel 联络恒为 0，曲率为 0，"测地线修正"空转；真正起作用的是"动能项"——一个带度量权重的状态增量 L2 正则项。耦合开启时度量**不正定**（存在负特征值），不再是黎曼度量；解析 Christoffel 公式存在数学错误（对非对角指标产生虚假分量）。
- **数值层**：scipy 路径用无梯度的有限差分优化非光滑目标，慢且无收敛保证；CasADi/IPOPT 路径健壮但存在 `dt=None` 崩溃、返回 cost 恒 0、与 scipy 后端目标函数不一致等问题。
- **依赖**：项目**没有**宣称整体"纯标准库"；pyproject 声明 numpy+scipy 硬依赖，casadi 为可选（5 处 import，均带守卫）。README 中"标准库"仅指 app 层 REST API（`http.server`），属实。

---

## 1. 各模块职责

| 文件 | 职责 |
|---|---|
| `physics/models.py` | 二阶单区域 RC 模型、五阶两区域 RC 模型、HVAC 电功率换算、Simulator 封装 |
| `physics/external.py` | 外部输入抽象接口 + 确定性模拟数据源 |
| `physics/online_id.py` | 通用 RLS 辨识器 + 面向单区域 RC 的在线辨识器 |
| `geometry/manifold.py` | 状态标签/单位/归一化/简单对角度量 |
| `geometry/metric_tensor.py` | 由规范层权重构造的对角度量（独立第二套度量实现） |
| `geometry/landscape.py` | 能量函数 E(state, control, external)、显式度量 g(z) 及其 CasADi 版本 |
| `geometry/geodesic.py` | 主求解器：scipy（L-BFGS-B/SLSQP）与 CasADi 双后端、Riemannian 修正步 |
| `geometry/riemannian.py` | Christoffel 联络三种实现（数值/自动微分/解析）+ 测地线方程欧拉积分 |
| `geometry/curvature.py` | 对 running_cost 求状态 Hessian 的有限差分二阶分析 |
| `geometry/robust_solver.py` | scipy SLSQP 多场景共享控制鲁棒 MPC（单区域） |
| `geometry/casadi_solver.py` | CasADi+IPOPT 硬约束非线性 MPC（单/两区域） |
| `geometry/casadi_robust_solver.py` | CasADi+IPOPT 多场景鲁棒 MPC（单区域） |

---

## 2. 物理模型细节分析

### 2.1 `RCBuildingModel`（二阶 RC，models.py:12-71）

- 状态 `[T_air, T_wall]`，控制 `Q_hvac`，外部 `[T_out, solar, occ, price]`。动态：

  dT_air = ((T_wall−T_air)/r_air + (T_out−T_air)/r_wall + solar_gain·solar + occ + Q_hvac)/c_air
  dT_wall = ((T_air−T_wall)/r_air + (T_out−T_wall)/r_wall)/c_wall

- **量纲**：c 为 kWh/K，r 为 K/kW，功率项为 kW → dT 单位 K/h，正确。
- **物理合理性**：空气节点经 r_wall 与室外直连（模拟窗墙得热），墙体经同一 r_wall 与室外相连——`r_wall` 被两个支路共用属于简化，可接受但会低估墙体热阻的作用。
- **单位不一致（重点）**：docstring（models.py:22-23）声明 `solar` 已是"等效热功率 (kW)"，却又乘以 `solar_gain=0.05`（models.py:32,57）。默认数据源 solar 峰值 0.3 kW，有效太阳得热仅 0.015 kW，而 occ 为 0.5–0.75 kW——太阳得热被削弱到可忽略，夏季制冷场景物理上不合理（要么 solar 用原始辐照 W/m²、solar_gain 用真实 g·A/1000 系数，要么去掉这重增益）。
- **积分格式**：显式 Euler（models.py:67）。默认参数下空气时间常数 τ≈0.48 h，dt=1/12 h，λ·dt≈0.24，稳定。但见问题 P1-1（无稳定性保护）。
- 默认 `dt=1/12`（5 分钟），与 README 中"15 分钟步长"（dt=1/4）的表述不一致（README 为 1/4 h 时是 15 分钟，代码默认是 5 分钟）。

### 2.2 `TwoZoneRCBuildingModel`（五阶 RC，models.py:75-166）

- 5 状态 `[T_air_A, T_wall_A, T_air_B, T_wall_B, T_partition]`，隔墙耦合通过 r_partition 双向传导，符号正确；隔墙方程两股热流求和除以 r_partition 一次，正确。
- 与单区域相同的 solar 单位问题（solar_gain_a=0.08/b=0.02 二次折扣）。
- `initial_state(t=28.0)`（models.py:165-166）默认 28°C，与单区域默认 24°C 不一致。
- 外部输入需要 6 维 `[T_out, solar_A, solar_B, occ_A, occ_B, price]`，而 `MockExternalInputProvider` 只产生 4 维（external.py:37,47），直接用会 IndexError（无维度防护）。

### 2.3 `HVACModel`（models.py:169-204)

- `electrical_power`：P_elec = |Q|/COP，COP 按制热/制冷区分，并带部分负荷惩罚 `cop_eff = cop·(1 − 0.15·(1−load_ratio)²)`。量纲：热 kW / COP → 电 kW，正确；低负荷 COP 下降对定频机合理。
- 注意：该部分负荷模型**只在 scipy 求解路径中使用**，CasADi 路径用线性 COP（见问题 P1-3，两后端目标不一致）。

### 2.4 在线辨识（online_id.py)

- 通用 RLS 与 RC 专用 RLS 的协方差更新 `P = (P − outer(gain, φᵀP))/λ` 对对称 P 等价于标准秩 1 更新，数学正确。
- 特征构造（online_id.py:78-85）与模型公式一一对应，`parameters()` 反解参数正确。
- 隐患：观测用后向差分 `y = (T_air(k+1)−T_air(k))/dt`，放大量测噪声；`λ<1` 下 P 可能协方差膨胀（特征含常数项 1.0，激励不足时 P 无界增长）；特征间存在共线性（T_out−T_air 与 T_wall−T_air 强相关），参数可辨识性一般。`a4` 与 `a5` 真值相等（都=1/C_air）却被当成两个独立参数估计，RLS 会把误差分摊，`parameters()` 只用 a4。

---

## 3. 几何/数学方法真实性评估（重点）

### 3.1 "流形"与度量

- `StateManifold` 只是一个标签/单位/缩放表（manifold.py），没有流形结构（无图册、无坐标卡），"流形"是命名包装。
- 度量有两种**独立实现且公式不同**：
  - `MetricTensor.matrix`（metric_tensor.py:40-54）：按标签前缀 `T_air*` 给 `comfort_weight/temperature_scale²`，其余 1.0；
  - `EnergyLandscape.metric`（landscape.py:37-55）：按"标签 ∈ setpoints"给 `comfort/scale²`，其余 1.0。
  - 单区域默认下两者一致（T_air→0.04，T_wall→1.0）；**两区域下行为不同**（metric_tensor 会给 T_air_A、T_air_B 都加权，landscape 只给出现在 setpoints 字典里的键加权）。引擎只用 landscape 版本（engine.py:231-243），MetricTensor 只在测试里用——两套实现有漂移风险。
- **度量不正定（核心数学缺陷）**：`metric_coupling=1.0`（技术报告推荐配置）时，单区域度量矩阵为
  G = [[0.04, 1.0], [1.0, 1.0]]，det = 0.04 − 1.0 = **−0.96 < 0**，特征值 {+1.63, **−0.59**}。负特征值意味着：
  1. 这不是黎曼度量（黎曼度量必须正定）；至多是伪黎曼（类时/类空混合），且其"距离"无下界；
  2. "动能" ½·dzᵀG·dz/dt² 可以为负，作用量泛函无下界，优化器存在被负动能"吸走"的理论风险；
  3. `np.linalg.inv` 仍可逆，Christoffel/测地线数值上"能算"，但几何意义已破坏。
  正确做法：保证 `|coupling| < √(g11·g22)`（如取 coupling=0.19 时 det>0），或对度量做 G = G0 + β·(低秩正半定) 保持正定。
- 度量与物理/代价结构**无任何联系**：它由任意权重（comfort_weight/scale²、耦合常数）拼成，既不是 cost 的 Hessian，也不编码动力学。T_wall 的 g=1.0 是 T_air 的 25 倍，导致"动能"对墙体温度变化惩罚远大于空气——会抑制需要墙体蓄热的预冷策略，属任意选择。

### 3.2 Christoffel 联络（riemannian.py）

- 数值实现 `christoffel_symbols`（riemannian.py:13-46）：中心差分 + 标准公式 Γᵏ_ij = ½gᵏˡ(∂ᵢg_jl + ∂ⱼg_il − ∂ₗg_ij)，**数学正确**。
- 自动微分实现（riemannian.py:68-101）：用 CasADi 的 `ca.jacobian` 求 ∂g，公式正确（gⁱʲ 作为系数不求导，正确）。
- **解析实现 `christoffel_symbols_analytic`（riemannian.py:104-130）存在公式错误**：推导如下。
  对对角指数度量 g_ll = base_l·exp(α(z_l − sp_l))，正确有 ∂ᵢg_jl = g_ll·α·δᵢⱼ·δ_jl（仅当 i=j=l 非零）。而代码写的是：
  `dg_jl = g[l,l]·α if l==j`、`dg_il = g[l,l]·α if l==i`、`dg_ij = g[i,i]·α if i==j`，
  即对 i≠j 也产生非零贡献。具体算例（n=2, g=diag(e^{αz₀},1)）：
  - 正确 Γ⁰₀₁ = 0，代码给出 α/2；
  - 正确 Γ⁰₁₁ = 0，代码给出 −α/2·e^{−αz₀}。
  因此解析版本**只有 i==j==k 的对角分量正确，其余分量全是虚假值**。且 docstring 声称处理常数非对角耦合，实际完全忽略耦合项（注释自认"近似"）。默认 `use_analytic_christoffel=True`（geodesic.py:35）。
- **讽刺的是**：默认配置下 α=0（`metric_state_dependence` 从未被引擎传入，engine.py:231-243 未传该参数），三种实现都给出 Γ≡0——所以错误在默认路径不发作，但一旦有人按技术报告 §11"状态相关度量已接入"启用状态依赖，就会用上错误的联络。

### 3.3 测地线（riemannian.py:49-65, geodesic.py:39-70）

- `geodesic_step`：对测地线方程 d²z/dτ² = −Γ vv 做半隐式欧拉，数值上正确。
- `_riemannian_corrected_step`（geodesic.py:39-70）：在物理 Euler 步上叠加 `−½·Γ vv·dt²` 修正。对**常度量** Γ≡0，修正恒为 0，"黎曼修正"空转——这解释了技术报告 §7.1"Christoffel 修正对结果影响很小"（两个配置输出完全相同）。即便 Γ≠0，把物理轨迹往测地线上"掰"也缺乏物理依据：状态轨迹本来就不是该度量的测地线。
- 真正影响结果的是 `use_kinetic` 动能项（默认开启，engine.py:68）：目标函数中加入 ½·dzᵀG·dz/dt²。当度量是常量对角阵时，这就是**带权重的状态增量 L2 正则**（一种平滑/鲁棒化正则项），与"最小作用量路径"类比相关，但完全不涉及联络、平行移动或距离。README 宣称的"GCCM 省电 17%"对比的实质是"经典 MPC vs 经典 MPC+状态增量正则"。
- **结论**：几何层的实质计算 = ①加权二次正则（动能项，有效）；②一个几乎恒为零的 Christoffel 修正（空转）；③一个只在非默认配置下启用、且解析实现有错的状态相关度量。**"黎曼几何/测地线/流形"属于比喻包装**：包装之下是加权二次正则化 + 有限差分二阶分析。技术报告第 1 节自己也承认"不是严格黎曼测地线推理"。

### 3.4 曲率（curvature.py）

- `CurvatureAnalyzer` 计算的是 **running_cost 关于状态的有限差分 Hessian**（固定控制与外部输入），然后做特征分解、分类（stable/unstable/saddle/flat）。中心差分公式（含混合偏导四角公式）正确。
- 这是对**代价景观**的二阶分析，不是黎曼曲率（截面曲率/Ricci），两者被文档混称。作为"局部代价凸性/稳定性"的启发式（引擎用它做自适应时域收缩，engine.py:161-175）是合理且有效的，只是命名误导。

---

## 4. 数值方法评估

### 4.1 scipy 路径（geodesic.py, robust_solver.py）

- 默认 `L-BFGS-B` + 边界，无解析梯度 → 每次迭代对**整个时域 rollout 做有限差分**（horizon=12 时每轮 ≥13 次全时域仿真），且目标非光滑（`max(0, dev−band)`、COP 的 if-else），FD 梯度误差与收敛性无保证；`maxiter=200` 常到限而非收敛。
- 约束路径 `SLSQP` 同样无解析雅可比；多场景约束可能不可行，`success=False` 时**无回退**，仍用返回的（可能不可行）控制量。
- `success = result.success or result.nit == 0`（geodesic.py:201）：SLSQP 立即不可行（nit=0）时被误判为成功。
- 结果总代价 `total` 在 scipy 路径是**事后重算**的（与优化目标一致），可作指标；但 CasADi 路径 total_cost 恒 0（见下）。

### 4.2 CasADi/IPOPT 路径（casadi_solver.py, casadi_robust_solver.py）

- 构造 `ca.Opti` 问题、硬约束（控制边界 + 各场景空气温度舒适约束）、两级热启动（舒适优先 MPC 提供初值）、失败时用 `opti.debug.value` 回退——工程上做得不错。
- **`dt=None` 崩溃**：`_casadi_step` 直接 `x + dt*ca.vertcat(...)`（casadi_solver.py:78,60），dt=None → `None*MX` TypeError。引擎默认 `dt=None`（engine.py:43），`use_casadi=True` 且不设 dt 即崩溃（示例 two_zone_casadi.py 显式传了 dt=STEP_H 才不触发）。
- **指标失真**：返回的 Trajectory `costs=[]`、`total_cost=0.0`（casadi_solver.py:291-298；casadi_robust_solver.py 同），引擎的 `decision.trajectory.total_cost` 恒 0。
- 舒适代价只对 `setpoints` 中存在的标签计算：两区域下若 setpoints 只有 "T_air"（引擎默认 engine.py:100），T_air_A/T_air_B 无舒适代价，靠硬约束兜底——代价面不连续。
- `casadi_robust_solver.py:121` 电价取 `w[3]`：对 6 维两区域外部输入会把 occ_A 当电价（该求解器文档限定单区域，属潜在陷阱）。

### 4.3 两后端目标不一致（可复现性问题）

- scipy 路径：`running_cost` → `HVACModel.electrical_power`（含部分负荷 COP，models.py:199）；
- CasADi 路径：`elec = Σ|q|/cop`（线性 COP，casadi_solver.py:224-228；casadi_robust_solver.py:116-120）。
  两者**解的是不同的优化问题**，README/技术报告里两后端的对比在数学上不可比。

### 4.4 显式 Euler 稳定性无保护

- 在线辨识门限允许 `r_air∈[0.05,5]`、`c_air∈[0.1,10]`（engine.py:457-463, 535-542）。取 r_air=0.05、c_air=0.1 时空气时间常数 τ≈0.005 h << dt=1/12 h≈0.083 h，λ·dt≈17 > 2，**显式 Euler 指数发散**。模型步进（models.py:67）没有任何稳定性检查或 dt 自适应。

---

## 5. 问题清单（按严重程度排序）

### P0 —— 数学/物理错误

1. **度量不正定**：`metric_coupling=1.0` 时 `G=[[0.04,1],[1,1]]` det<0、含负特征值 −0.59 → 不是黎曼度量，"动能"可负、作用量无下界，全部黎曼语义被破坏。位置：metric_tensor.py:50-54、landscape.py:52-54（metric_tensor.py:44-48 与 landscape.py:45-51 的度量公式同因）。
2. **解析 Christoffel 公式错误**：对 i≠j 指标产生虚假分量（Γ⁰₀₁、Γ⁰₁₁ 应为 0 却得 ±α/2 量级），仅对角 i==j==k 正确；且 docstring 声称支持的非对角耦合被完全忽略。默认启用该版本（geodesic.py:35）。位置：riemannian.py:104-130。
3. **太阳得热单位不一致**：solar 标称 kW 却又乘 solar_gain=0.05/0.08，有效得热 0.015–0.024 kW，被 occ 淹没，物理不合理。位置：models.py:32,57,94,127,141 与 docstring models.py:22-23。
4. **CasADi `dt=None` 崩溃**：`None*MX` TypeError；引擎默认 dt=None 且 `use_casadi=True` 即触发。位置：casadi_solver.py:78,60（engine.py:249 传入 self.dt）。

### P1 —— 数值/稳定性/一致性问题

5. **显式 Euler 无稳定性保护**：在线辨识门限可产生 λ·dt>2 的参数，模型发散无检测。位置：models.py:67, engine.py:457-463,535-542。
6. **scipy 求解器无解析梯度、非光滑目标、无失败回退**；SLSQP 立即不可行时 `nit==0` 被误报成功（geodesic.py:201）；robust_solver 失败仍返回不可行控制（robust_solver.py:73-75）。
7. **两后端目标不一致**（部分负荷 COP vs 线性 COP），对比结果不可比。位置：models.py:198-200 vs casadi_solver.py:224-228、casadi_robust_solver.py:116-120。
8. **CasADi 路径指标恒 0**：`costs=[]`、`total_cost=0.0`，引擎层指标失真。位置：casadi_solver.py:291-298、casadi_robust_solver.py:146。
9. **`dt=None` 时物理步长与几何修正步长不一致**：物理步用模型默认 1/12，修正项用 1.0（geodesic.py:45-46）。
10. **`casadi_robust_solver` 电价索引**：`w[3]` 在 6 维外部输入（两区域）下读到 occ_A。位置：casadi_robust_solver.py:121。

### P2 —— 代码质量/一致性

11. **两套度量实现公式不同、漂移风险**：MetricTensor.matrix（按标签前缀）vs EnergyLandscape.metric（按 setpoints 键），两区域下行为不一致；引擎只用后者。位置：metric_tensor.py:40-54 vs landscape.py:37-55。
12. **`metric_state_dependence` 未接入引擎**：engine.py:231-243 未传参，默认管线 Γ≡0、Riemannian 修正空转；与技术报告 §11"状态相关非对角度量已接入"不符。
13. **动能项权重任意**：T_wall 的 g=1.0 是 T_air 的 0.04 的 25 倍，抑制墙体蓄热/预冷；与 smooth 控制正则功能重叠。位置：landscape.py:47,71。
14. **两区域外部输入维度不匹配无防护**：MockExternalInputProvider 固定 4 维，两区域模型需 6 维 → IndexError。位置：external.py:37,47 vs models.py:102-104。
15. **类型注解错误**：`prev_control: ControlInput = None`（应为 Optional）。位置：robust_solver.py:35-36、casadi_robust_solver.py:54-55。
16. **`StateManifold.normalize` 只取下界、不用上界**（manifold.py:30-35），bounds 语义半实现；normalize/denormalize 实际未被引擎使用（死代码）。
17. **`_feedback_safe_control` 制冷上限用 `bounds[j][1]`（q_max 正值）而非 |q_min|**：非对称边界下可能超出制冷下限。位置：engine.py:652-654。

### P3 —— 细节

18. `r_wall` 同时用于空气–室外与墙体–室外两支路（简化，可接受）；墙体无太阳得热（models.py:62-65）。
19. 两区域 `initial_state` 默认 28°C 与单区域 24°C 不一致（models.py:165-166）。
20. 模型默认 dt=1/12（5 分钟）与 README"15 分钟步长"表述不一致（models.py:33 vs README 用例）。

---

## 6. 依赖情况

| 文件 | 第三方依赖 | 是否可选 |
|---|---|---|
| physics/*（4 个） | numpy | 硬依赖 |
| geometry/manifold.py, metric_tensor.py, curvature.py | numpy | 硬依赖 |
| geometry/geodesic.py | numpy + scipy（`scipy.optimize.minimize`，geodesic.py:8） | scipy 硬依赖 |
| geometry/robust_solver.py | numpy + scipy（robust_solver.py:8） | scipy 硬依赖 |
| geometry/landscape.py | casadi（方法内 import，landscape.py:59） | 可选 |
| geometry/riemannian.py | casadi（方法内 import，riemannian.py:74） | 可选 |
| geometry/casadi_solver.py | casadi（模块级 try/except 守卫，casadi_solver.py:13-18） | 可选 |
| geometry/casadi_robust_solver.py | casadi（同上，casadi_robust_solver.py:13-18） | 可选 |

- `pyproject.toml:11-14` 明确声明 `numpy>=1.23`、`scipy>=1.9` 为**硬依赖**，casadi 未列入任何 extra（安装时若需要需手动装）。
- 项目**没有**宣称整体"纯标准库"：README.md:375 的"标准库实现的最小 REST API"仅指 app 层（`app/api.py` 用 `http.server`，属实）。
- **若对外宣称"纯标准库/零依赖"则与事实矛盾**：物理层即依赖 numpy；几何层核心求解依赖 scipy；casadi 可选但被 4 个文件引用（其中 landscape/riemannian 是方法内 import，没装只在这些方法被调用时抛错；casadi_solver/casadi_robust_solver 有守卫，但构造实例会直接抛 RuntimeError）。

---

## 7. 改进建议

1. **度量正定化**：对耦合加约束 `|coupling| ≤ √(g_ii·g_jj) − ε`（默认 temperature_scale=5 时 coupling ≤ ~0.19），或改用 G = diag + β·vvᵀ 半正定修正；在 `metric()`/`matrix()` 内加正定性断言；并把"这是加权正则化权重，不是黎曼度量"写进文档，避免论文级误用。
2. **Christoffel**：删除或重写 `christoffel_symbols_analytic`（要么推导含耦合的完整解析式并加单测，要么默认走数值/AD 版本）；为三种实现添加对已知度量（如球面/双曲空间）的一致性单元测试。
3. **打通或摘牌状态相关度量**：把 `metric_state_dependence` 传入引擎的 EnergyLandscape 构造（engine.py:231-243），否则从文档中删除"已接入"表述。
4. **物理模型**：统一 solar 单位语义（kW 等效得热 vs 辐照×系数），修正双重增益；给 `step()` 加 Euler 稳定性检查（`dt < 2/λ_max`，必要时自动缩小 dt 或切隐式/半隐式）；两区域与单区域默认初温统一；ExternalInputProvider 增加维度校验。
5. **求解器**：统一两后端 COP 模型（或在文档标注差异）；CasADi 路径 `dt or 模型默认dt` 防御；给 scipy 路径提供解析梯度（或复用 CasADi 生成的符号导数）并加 SLSQP 失败回退（上次可行解/安全控制）；修正 `nit==0` 误报；让 CasADi 路径返回真实 costs/total_cost。
6. **命名诚实化**：将"动能项"在文档中明确为"状态增量加权正则"；"曲率"明确为"代价景观局部 Hessian"；避免黎曼几何术语带来的认知负担（技术报告已自认，建议贯彻到 README 与 API 文档）。
7. **测试补强**：加回归测试覆盖 metric 正定性（coupling 边界）、analytic vs numeric Christoffel 一致性、`dt=None` 各后端路径、两区域外部输入维度、Euler 稳定性边界参数。
