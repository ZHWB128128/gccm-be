# 贡献指南

欢迎贡献！本项目是一个研究型控制引擎，任何改进都欢迎。

## 开发环境

```bash
pip install -e ".[dev]"
```

## 运行测试

```bash
python -m pytest tests/ -q
```

> CasADi 相关测试（鲁棒 MPC、自动微分 Christoffel）在未安装 casadi 时自动跳过；
> 完整测试请 `pip install casadi`。

## 代码约定

- 保持**纯 Python 标准库 + numpy/scipy**（CasADi 为可选依赖，import 需带守卫）
- 新增功能必须配套测试（行为级断言，非仅形状断言）
- 所有实验数字必须可复现（固定 seed + 注释运行命令）

## 提交 PR 前检查

1. `python -m pytest tests/ -q` 全绿
2. 新文件不引入 `__pycache__` / `output/` 产物
3. 若修改控制行为，更新 README 中的对应数字并标注复测日期
