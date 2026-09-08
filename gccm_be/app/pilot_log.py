"""Pilot CSV logger: append M&V data rows (header written on first row).

实楼试点的数据凭证（IPMVP Option B 口径，见 docs/PILOT_PLAN.md）：每个控制
周期追加一行 `(时间, 室温, 控制功率, 设定温度, 功率, 电价, 模式, 置信度)`。
表头只在文件为空时写一次；字段固定，多余键忽略，缺失键留空。
"""
from __future__ import annotations

import csv
import os

FIELDNAMES = (
    "time_iso", "time_h", "T_air", "u_kw", "setpoint",
    "power_w", "price", "mode", "confidence",
)


def append_row(path: str, row: dict[str, object]) -> None:
    """追加一行 M&V 数据；文件不存在/为空时先写表头。"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDNAMES), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def violations_occupied(rows, bands, occupied_hours=(7.0, 19.0),
                        hour_index: int = 0) -> dict:
    """有人时段违温率（PILOT_PLAN §5.4 的正确口径）。

    夜间无人时的温度漂移（供暖停机、免费冷却）不应计入控制质量 KPI——
    BOPTEST 演练实测：全时段口径下 90% 的"违温"来自无人时段。

    rows: 每行 [hour_of_day, t_zone_0, t_zone_1, ...]；bands: {区名: (lo, hi)}，
    区顺序与 rows 中的温度列对应。
    """
    counters = {z: [0, 0] for z in bands}
    for row in rows:
        hour = float(row[hour_index]) % 24.0
        if not (occupied_hours[0] <= hour < occupied_hours[1]):
            continue
        for i, (z, (lo, hi)) in enumerate(bands.items()):
            t = float(row[hour_index + 1 + i])
            counters[z][1] += 1
            counters[z][0] += int(t < lo or t > hi)
    return {z: 100.0 * c[0] / max(c[1], 1) for z, c in counters.items()}
