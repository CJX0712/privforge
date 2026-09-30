"""Utility@epsilon curve helpers (SPEC.md section 6.4).

Given a set of `EvalResult`s for one (method, dataset) across several epsilon
values, recover the curve and the smallest epsilon at which the method reaches a
utility target (DoD-5 / DoD-7).  `U_nodp` is recovered from
`accuracy + utility_gap` so no extra field is needed.

Author: 晨星
"""

from __future__ import annotations

from privforge.core.types import EvalResult


def _u_nodp(r: EvalResult) -> float:
    return float(r.accuracy + r.utility_gap)


def eps_min_at_target(results: list[EvalResult], target_gap: float = 0.05) -> float | None:
    """Smallest epsilon whose accuracy reaches `U_nodp - target_gap`.

    Returns None if no result meets the target.
    """
    best: float | None = None
    for r in sorted(results, key=lambda x: x.epsilon):
        if r.accuracy >= _u_nodp(r) - target_gap and (best is None or r.epsilon < best):
            best = float(r.epsilon)
    return best


def utility_curve(results: list[EvalResult]) -> list[tuple[float, float]]:
    """(epsilon, accuracy) points sorted by epsilon."""
    return [(float(r.epsilon), float(r.accuracy)) for r in sorted(results, key=lambda x: x.epsilon)]


def eps_efficiency(
    flagship: list[EvalResult], best_base: list[EvalResult], target_gap: float = 0.05
) -> float | None:
    """DoD-5 ratio eps_min_at_target(flagship) / eps_min_at_target(best_base).

    Returns None if either side has no meeting point.  A value <= 0.8 means the
    flagship reaches the target at <= 80% of the baseline epsilon (>= 20% saved).
    """
    ef = eps_min_at_target(flagship, target_gap)
    eb = eps_min_at_target(best_base, target_gap)
    if ef is None or eb is None or eb <= 0.0:
        return None
    return float(ef / eb)
