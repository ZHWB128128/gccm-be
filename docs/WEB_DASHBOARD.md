# GCCM-BE Web 控制台（操作 / 管理 / 监控）

> Owners: `gccm_be/app/api.py`（HTTP 服务 + 路由）、`gccm_be/app/introspection.py`
> （模型能力注册表，**页面的唯一数据源**）、`gccm_be/app/dashboard.py`（数据驱动的
> 前端 HTML）。Tests: `tests/test_dashboard_api.py`。

## 1. 启动

零外部依赖，纯标准库 HTTP 服务：

```bash
# 默认引擎
python -m gccm_be.app.api --host 127.0.0.1 --port 8080
# 或从配置文件构建引擎
python -m gccm_be.app.api --config examples/config.json --port 8080
```

浏览器打开 `http://127.0.0.1:8080/` 即为控制台页面。

## 2. 功能

- **操作/管理（左栏）**：黎曼四开关（含强度）、目标权重、控制器开关（硬约束/曲率自适应/
  协变 Hessian/重整化削峰/预防性反馈/噪声自适应裕度）、物理模型清单。改完点"应用配置"
  即热更新到运行引擎。
- **监控（右栏）**：选模式/初温/步数，"运行闭环仿真"绘制室温轨迹（叠加舒适带）+ 控制量/
  电价曲线，展示峰值/均温/越界步数/总电费 KPI；"单步决策"看一次决策输出；诊断面板显示
  求解状态、是否不可判定、触发器、曲率几何等。

## 3. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 控制台 HTML（数据驱动） |
| GET | `/health` | 健康检查 |
| GET | `/status` | 当前模式/时域/版本 |
| GET | `/introspection` | **模型能力全量描述 + 实时值**（页面据此渲染） |
| POST | `/config` | 设置白名单引擎属性（热更新） |
| POST | `/control` | 单步滚动决策 |
| POST | `/simulate` | 闭环仿真轨迹（供绘图；多区模型返回逐区/逐设备序列） |

`/simulate` 返回体（多区兼容）：

- `temps` / `controls`：兼容字段，指向第一个温度区 / 第一台设备（单区模型即唯一序列）；
- `zones`：`[{label, temps, comfort_min, comfort_max}, ...]`，每个 `T_air*` 状态一条
  室温曲线，`comfort_min/max` 为**该区有效舒适带**（每区覆盖 > 标量回退），可据此逐区画带；
- `unit_controls`：`[{label: "Q_hvac_A", data: [...]}, ...]`，每台设备一条控制序列；
- `violations`：按**每区有效边界**统计的全部越界步数合计（不是只算第一区）。

双区引擎启动（配置化，页面自动画两条区曲线 + 两台设备控制量）：

```bash
python -m gccm_be.app.api --config examples/config_two_zone.json --port 8080
```

`/introspection` 另返回 `active_model`（当前引擎实际使用的物理模型类名，对应
`physics_models` 注册表中的 `id`），页面据此标注当前模型。

## 4. 安全说明（诚实标注）

该服务**无鉴权**，仅供本地/内网 demo 与调试；`/config` 可热改控制器参数、`/simulate`
会跑引擎计算。**不要**直接暴露到公网。如需生产部署，应在前置反向代理加认证/限流，或
在 `start_api` 外包一层鉴权中间件。默认绑定 `127.0.0.1`，仅本机可访问。

## 5. 长期维护契约（关键）

> **每次模型改动，页面必须一起改。**（契约内容见本文 §5，由 CI 双向校验强制）

页面是**数据驱动**的：前端不硬编码模型能力，而是运行时拉取 `/introspection`，据此动态
渲染所有开关/权重/模式/物理模型/诊断项。因此"页面跟着模型变"这条要求，收敛为一个可执行
动作：**改了引擎的开关/权重/物理模型/模式/诊断输出，就同步更新
`gccm_be/app/introspection.py` 里的对应条目。**

`tests/test_dashboard_api.py::test_registry_fields_exist_on_engine` 与
`test_registry_covers_known_switch_weight_fields` 双向校验：注册表引用的字段必须真实存在
于 `GCCMEngine`；且引擎里任何 `use_*` 开关或 `*_weight` 权重，要么出现在注册表、要么显式
登记在 `NON_SURFACED_FIELDS`（附不开放理由）。忘记同步会导致测试失败——契约由 CI 强制。
