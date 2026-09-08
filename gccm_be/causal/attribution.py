"""Shapley-based cost attribution: decompose an energy-cost difference into the
additive contributions of independent causal factors (control policy, weather,
electricity price).

Motivation
----------
A raw A/B number ("GCCM saved 13%") cannot by itself prove the saving came from
the *control strategy* rather than a lucky cooler day or a different tariff.
This module answers that question defensibly: it treats policy, weather and
price as toggleable factors between a *baseline* world and a *treatment* world,
runs the real physics simulator for every factor coalition, and assigns each
factor its Shapley value. Shapley values are the unique additive attribution
that satisfies efficiency (contributions sum exactly to the total difference),
symmetry and the null-player property — the standard game-theoretic footing a
reviewer will accept.

The value function is the closed-loop electricity cost produced by the actual
`Simulator`, so this is a counterfactual decomposition on the *true* structural
model, not on a hand-tuned surrogate.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

from ..physics.models import Simulator
from ..types import ControlInput, ExternalInput, SystemState

# A policy maps (state, time_h) -> control, matching CounterfactualAnalyzer.
Policy = Callable[[SystemState, float], ControlInput]


def shapley_values(
    factor_names: Sequence[str],
    value_fn: Callable[[frozenset[str]], float],
) -> tuple[dict[str, float], dict[frozenset[str], float]]:
    """Exact Shapley decomposition over a small set of factors.

    Args:
        factor_names: the factors to attribute. `value_fn(S)` must return the
            outcome when exactly the factors in `S` are set to *treatment* and
            the rest to *baseline*.
        value_fn: coalition value function; called once per subset (2**n calls).

    Returns:
        (phi, cache) where `phi[name]` is the Shapley value of each factor and
        `cache[S]` is the memoized coalition value. By efficiency,
        `sum(phi.values()) == cache[full] - cache[empty]` up to float error.
    """
    names = list(factor_names)
    n = len(names)
    if n == 0:
        return {}, {frozenset(): value_fn(frozenset())}

    # Memoize every coalition value (2**n evaluations — n is tiny, e.g. 3).
    cache: dict[frozenset[str], float] = {}
    for r in range(n + 1):
        for combo in combinations(names, r):
            S = frozenset(combo)
            cache[S] = float(value_fn(S))

    phi: dict[str, float] = {name: 0.0 for name in names}
    for name in names:
        others = [x for x in names if x != name]
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                S = frozenset(combo)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi[name] += weight * (cache[S | {name}] - cache[S])
    return phi, cache


@dataclass
class AttributionResult:
    """Result of a Shapley cost attribution.

    All costs are in the same monetary unit as `price * power * dt`.
    """

    baseline_cost: float           # v(empty): all factors at baseline
    treatment_cost: float          # v(full): all factors at treatment
    total_difference: float        # treatment_cost - baseline_cost
    contributions: dict[str, float]  # factor -> Shapley contribution to the difference
    residual: float                # additivity residual (should be ~0)
    coalition_values: dict[frozenset[str], float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "baseline_cost": self.baseline_cost,
            "treatment_cost": self.treatment_cost,
            "total_difference": self.total_difference,
            "contributions": dict(self.contributions),
            "contribution_fractions": self.contribution_fractions(),
            "residual": self.residual,
        }

    def contribution_fractions(self) -> dict[str, float]:
        """Each factor's share of the total difference (fractions sum to ~1)."""
        denom = self.total_difference
        if abs(denom) < 1e-12:
            return {k: 0.0 for k in self.contributions}
        return {k: v / denom for k, v in self.contributions.items()}

    def summary(self) -> str:
        lines = [
            f"baseline cost : {self.baseline_cost:.4f}",
            f"treatment cost: {self.treatment_cost:.4f}",
            f"total diff    : {self.total_difference:+.4f}",
            "attribution (Shapley):",
        ]
        fracs = self.contribution_fractions()
        for name, val in self.contributions.items():
            lines.append(f"  {name:<10}: {val:+.4f}  ({fracs[name]*100:+.1f}%)")
        lines.append(f"residual      : {self.residual:.2e}")
        return "\n".join(lines)


class CostAttributor:
    """Attribute a closed-loop electricity-cost difference to policy, weather and price.

    Each factor has a *baseline* and a *treatment* setting:
      - policy  : baseline_policy   vs treatment_policy
      - weather : baseline_weather  vs treatment_weather  (T_out/solar/occ ...)
      - price   : baseline_price[]  vs treatment_price[]

    The value function runs the real simulator closed-loop under the chosen
    combination and returns total electricity cost. Because the electricity cost
    couples power (policy+weather) with price multiplicatively, the factors
    interact — which is exactly why a Shapley decomposition (not naive one-at-a-time
    subtraction) is required for an exact additive answer.
    """

    def __init__(
        self,
        simulator: Simulator,
        initial_state: SystemState,
        dt: float,
        baseline_policy: Policy,
        treatment_policy: Policy,
        baseline_weather: Sequence[ExternalInput],
        treatment_weather: Sequence[ExternalInput],
        baseline_price: Sequence[float],
        treatment_price: Sequence[float],
    ) -> None:
        n = len(treatment_weather)
        if not (len(baseline_weather) == len(baseline_price) == len(treatment_price) == n):
            raise ValueError(
                "baseline/treatment weather and price sequences must all have equal length"
            )
        self.simulator = simulator
        self.initial_state = initial_state
        self.dt = dt
        self.baseline_policy = baseline_policy
        self.treatment_policy = treatment_policy
        self.baseline_weather = list(baseline_weather)
        self.treatment_weather = list(treatment_weather)
        self.baseline_price = [float(p) for p in baseline_price]
        self.treatment_price = [float(p) for p in treatment_price]
        self.horizon = n

    def _run_cost(
        self,
        policy: Policy,
        weather: Sequence[ExternalInput],
        price: Sequence[float],
    ) -> float:
        """Closed-loop electricity cost under a specific (policy, weather, price)."""
        state = self.initial_state.copy()
        t = 0.0
        total = 0.0
        for k in range(self.horizon):
            control = policy(state, t)
            w = weather[k]
            total += self.simulator.hvac.electrical_power(control) * price[k] * self.dt
            state = self.simulator.step(state, control, w, self.dt)
            t += self.dt
        return total

    def _coalition_value(self, treatment_set: frozenset[str]) -> float:
        policy = self.treatment_policy if "policy" in treatment_set else self.baseline_policy
        weather = self.treatment_weather if "weather" in treatment_set else self.baseline_weather
        price = self.treatment_price if "price" in treatment_set else self.baseline_price
        return self._run_cost(policy, weather, price)

    def attribute(self) -> AttributionResult:
        factors = ["policy", "weather", "price"]
        phi, cache = shapley_values(factors, self._coalition_value)
        empty = cache[frozenset()]
        full = cache[frozenset(factors)]
        total_diff = full - empty
        residual = total_diff - sum(phi.values())
        return AttributionResult(
            baseline_cost=empty,
            treatment_cost=full,
            total_difference=total_diff,
            contributions=phi,
            residual=residual,
            coalition_values=cache,
        )
