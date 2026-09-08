"""Renormalization-group flow: multi-scale coarse graining, order-parameter
identification and cross-scale anomaly amplification.

Scales
------
- micro : individual zone air temperatures (and their local anomalies)
- macro : building-level aggregates (mean / max / peak-driving indicators)

Two capabilities beyond a plain average:

1. Order-parameter identification (`identify_order_parameters`): which micro
   zone actually drives a macro indicator (e.g. building peak / mean). This is a
   cross-scale *sensitivity*: how much does a small perturbation of zone i move
   the macro observable, weighted by how anomalous zone i already is. Zones with
   the largest coupling are the "relevant" order parameters in the RG sense;
   near-noise zones are "irrelevant" and can be down-prioritized.

2. Cross-scale amplification (`analyze`): a micro anomaly that will be *amplified*
   at the macro scale (e.g. a top-floor sunlit zone dragging the whole floor's
   peak up) is flagged early, with a tiered action recommendation:
     - local      : micro-only deviation, handle with local adjustment
     - mode_switch : a zone anomaly with moderate macro coupling
     - axiom_mutation : macro indicator already amplified -> rebuild the objective
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
class OrderParameter:
    """A micro zone's cross-scale relevance."""

    index: int
    label: str
    sensitivity: float       # d(macro observable)/d(zone i) — the coupling
    anomaly: float           # how far zone i is beyond the comfort ceiling
    relevance: float         # sensitivity * (1 + anomaly): RG relevance score

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "sensitivity": self.sensitivity,
            "anomaly": self.anomaly,
            "relevance": self.relevance,
        }


@dataclass
class RenormalizationFlow:
    """Two scales: micro zone anomalies -> macro building indicators.

    Adds order-parameter identification and tiered cross-scale action.
    """

    micro_threshold: float = 27.0
    macro_amplification_ratio: float = 0.5
    comfort_center: float = 26.0
    # macro observable used for order-parameter coupling: "max" (peak-driving)
    # or "mean" (average-driving)
    macro_observable: str = "max"
    # softness of the soft-max used to make d(max)/d(zone) differentiable
    softmax_beta: float = 6.0
    labels: list[str] | None = None

    def _macro_value(self, temps: np.ndarray) -> float:
        if self.macro_observable == "mean":
            return float(np.mean(temps))
        # smooth (soft) max so the peak indicator has a well-defined per-zone
        # sensitivity instead of a hard argmax with zero/one gradient
        b = self.softmax_beta
        w = np.exp(b * (temps - np.max(temps)))
        w = w / np.sum(w)
        return float(np.sum(w * temps))

    def _macro_sensitivity(self, temps: np.ndarray) -> np.ndarray:
        """d(macro observable)/d(zone_i)."""
        n = temps.size
        if self.macro_observable == "mean":
            return np.full(n, 1.0 / n)
        # analytic gradient of the soft-max value:
        #   V = sum_i w_i T_i,  w_i = softmax(b*T)_i
        #   dV/dT_j = w_j + b * w_j * (T_j - V)
        b = self.softmax_beta
        w = np.exp(b * (temps - np.max(temps)))
        w = w / np.sum(w)
        V = float(np.sum(w * temps))
        return w + b * w * (temps - V)

    def identify_order_parameters(self, room_temps: Sequence[float]) -> list[OrderParameter]:
        """Rank zones by cross-scale relevance to the macro observable.

        Relevance = macro-sensitivity * (1 + anomaly_above_threshold). A zone
        that both strongly moves the building peak AND is already hot is the
        dominant order parameter; a cool zone with tiny coupling is irrelevant.
        """
        temps = np.asarray(room_temps, dtype=float)
        n = temps.size
        sens = self._macro_sensitivity(temps)
        anomalies = np.maximum(0.0, temps - self.micro_threshold)
        params: list[OrderParameter] = []
        for i in range(n):
            label = self.labels[i] if self.labels and i < len(self.labels) else f"zone_{i}"
            relevance = float(sens[i] * (1.0 + anomalies[i]))
            params.append(OrderParameter(
                index=i,
                label=label,
                sensitivity=float(sens[i]),
                anomaly=float(anomalies[i]),
                relevance=relevance,
            ))
        params.sort(key=lambda p: p.relevance, reverse=True)
        return params

    def zone_comfort_weights(self, room_temps: Sequence[float],
                             gain: float, labels: list[str] | None = None) -> dict[str, float]:
        """Map cross-scale relevance to per-zone comfort penalty multipliers.

        This is the canonical owner of the RG->MPC coupling used by the engine's
        ``renormalization_weighting`` mode. A zone whose relevance exceeds the
        mean gets a multiplier > 1, focusing the limited cooling on the
        peak-driving order parameter:

            mult_i = 1 + gain * max(0, (rel_i - mean_rel) / mean_rel)

        Returns an empty dict when there are fewer than two zones (nothing to
        prioritize) so callers can treat it as a no-op.
        """
        if labels is not None:
            self.labels = list(labels)
        temps = list(room_temps)
        if len(temps) < 2:
            return {}
        params = self.identify_order_parameters(temps)
        rels = np.array([p.relevance for p in params], dtype=float)
        mean_rel = float(np.mean(rels)) if rels.size else 0.0
        weights: dict[str, float] = {}
        for p in params:
            if mean_rel > 1e-12:
                mult = 1.0 + gain * max(0.0, (p.relevance - mean_rel) / mean_rel)
            else:
                mult = 1.0
            weights[p.label] = mult
        return weights

    def analyze(self, room_temps: Sequence[float]) -> dict:
        micro = np.asarray(room_temps, dtype=float)
        macro = coarse_grain(micro)

        micro_anomalies = int(np.sum(micro > self.micro_threshold))
        micro_anomaly_ratio = float(np.mean(micro > self.micro_threshold))

        # 宏观放大指标：建筑均值偏离舒适中值的程度
        macro_deviation = abs(macro["building_mean"] - self.comfort_center)
        amplified = bool(
            micro_anomaly_ratio > 0.0
            and macro_deviation > self.macro_amplification_ratio
        )

        order_params = self.identify_order_parameters(micro)
        dominant = order_params[0] if order_params else None

        # 分级动作：宏观已放大 -> 公理突变；单区异常且耦合中等 -> 模式切换；否则局部
        if amplified:
            action = "axiom_mutation"
            scale = "macro"
        elif dominant is not None and dominant.anomaly > 0.0 and dominant.sensitivity > (1.5 / max(micro.size, 1)):
            action = "mode_switch"
            scale = "meso"
        elif micro_anomalies > 0:
            action = "local"
            scale = "micro"
        else:
            action = "observe"
            scale = "micro"

        return {
            "micro": {
                "anomaly_count": micro_anomalies,
                "anomaly_ratio": micro_anomaly_ratio,
                "max_temp": float(np.max(micro)),
            },
            "macro": macro,
            "amplified": amplified,
            "order_parameters": [p.as_dict() for p in order_params],
            "dominant_zone": dominant.as_dict() if dominant is not None else None,
            "scale": scale,
            "action": action,
        }
