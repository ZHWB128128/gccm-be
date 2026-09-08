# 贡献指南

## 快速开始

```bash
pip install numpy scipy pytest
pip install -e .            # 或 PYTHONPATH=.
PYTHONPATH=. python -m pytest tests -q   # 171 项(168 passed + 3 CasADi skip)
```

## 文档规则（硬性）

1. **原文即终稿**：修改文档一律在原文处直接改，不做文末追加/补录/后记。
2. **中英同步**：`README.md` 与 `README.zh-CN.md` 结构镜像，改动必须双写。
3. **数据归档**：试验台/仿真结果写入 `docs/data/`（摘要 JSON/CSV + README 结论），
   结论与限制同段呈现。
4. **控制台同步契约**：引擎开关/权重/物理模型/诊断输出变更 → 同步
   `app/introspection.py`（`tests/test_dashboard_api.py` CI 双向校验，漏登记直接挂）。

## 代码规则

- 行为级测试优先：新能力必须带 `tests/` 用例（含边界条件）；
- 模型新类型：实现 `state_labels/control_labels/step` 协议并在
  `physics/registry.py` 注册；CasADi 分支在 `geometry/casadi_solver.py`；
- 依赖保持最小：运行时仅 numpy/scipy（CasADi 可选）；
- 负结果与正结果同等记录（试验台演练先例见 docs/data/）。
