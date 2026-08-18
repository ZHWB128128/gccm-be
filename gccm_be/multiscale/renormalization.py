"""重整化群流：多尺度粗粒化与异常放大检测。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def coarse_grain(room_temps: Sequence[float]) -> dict:
    """房间级 -> 建筑级粗粒化。"""
    arr = np.asarray(room_temps, dtype=float)
    return {
        "building_mean": float(np.mean(arr)),
        "building_max": float(np.max(arr)),
        "building_std": float(np.std(arr)),
        "n_rooms": int(arr.size),
    }


@dataclass
class RenormalizationFlow:
    """两个尺度：微观房间异常 -> 宏观建筑指标。"""

    micro_threshold: float = 27.0
    macro_amplification_ratio: float = 0.5

    def analyze(self, room_temps: Sequence[float]) -> dict:
        micro = np.asarray(room_temps, dtype=float)
        macro = coarse_grain(micro)

        micro_anomalies = int(np.sum(micro > self.micro_threshold))
        micro_anomaly_ratio = float(np.mean(micro > self.micro_threshold))

        # 宏观放大指标：建筑均值偏离舒适中值的程度
        macro_deviation = abs(macro["building_mean"] - 26.0)
        amplified = bool(
            micro_anomaly_ratio > 0.0
            and macro_deviation > self.macro_amplification_ratio
        )

        return {
            "micro": {
                "anomaly_count": micro_anomalies,
                "anomaly_ratio": micro_anomaly_ratio,
                "max_temp": float(np.max(micro)),
            },
            "macro": macro,
            "amplified": amplified,
            "action": "trigger_axiom_mutation" if amplified else "observe",
        }
