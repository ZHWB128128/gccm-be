"""数据驱动因果推断：从干预实验数据估计 SCM 参数。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .scm import StructuralCausalModel


@dataclass
class DataDrivenSCM:
    """从多组 do-干预实验数据估计线性 SCM。"""

    equations: Dict[str, Dict[str, float]] = None  # type: ignore[assignment]
    intercepts: Dict[str, float] = None  # type: ignore[assignment]

    def fit(self, data: List[Dict[str, float]]) -> "DataDrivenSCM":
        """data 是观测/干预样本列表，每个样本包含所有变量。"""
        variables = list(data[0].keys())
        self.equations = {}
        self.intercepts = {}
        for idx, var in enumerate(variables):
            parents = variables[:idx]  # 只使用因果顺序之前的变量作为父节点
            X = np.array([[d[p] for p in parents] for d in data])
            y = np.array([d[var] for d in data])
            A = np.column_stack([X, np.ones(len(y))])
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            self.equations[var] = {p: float(c) for p, c in zip(parents, coef[:-1])}
            self.intercepts[var] = float(coef[-1])
        return self

    def to_scm(self, noise: Dict[str, float] = None) -> StructuralCausalModel:
        """转换为可执行 SCM。"""
        eqs = {}
        for var, parent_coef in self.equations.items():
            parents = list(parent_coef.keys())
            coefs = list(parent_coef.values())
            intercept = self.intercepts[var]
            def make_func(ps=parents, cs=coefs, b=intercept):
                return lambda v: b + sum(c * v[p] for c, p in zip(cs, ps))
            eqs[var] = make_func()
        return StructuralCausalModel(equations=eqs, noise=noise or {})
