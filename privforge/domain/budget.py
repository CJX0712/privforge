"""Budget allocation and validation (SPEC.md section 2.2).

Two spend modes are supported:

* ERM single-release (OP / ObjP): ``eps = eps_clip + eps_train + eps_select``
  with the default (0.05, 0.92, 0.03) split.
* DP-SGD stepwise (flagship): the clip selection consumes ``eps_q`` per step via
  ``step_pure``; the Gaussian steps consume the rest. The recommended split is
  ``eps_q_total = 0.1 * epsilon`` (the clipping selection takes 10% of the
  total budget, spread uniformly over the T steps).

Author: 晨星
"""

from __future__ import annotations

from privforge.core.types import Budget

CLIP_BUDGET_FRACTION: float = 0.1  # clipping selection takes 10% of the budget


def make_budget(
    epsilon: float, delta: float, fractions: tuple[float, float, float] | None = None
) -> Budget:
    """Build a Budget from the policy defaults (architecture.md section 4.1)."""
    if fractions is None:
        return Budget.from_fractions(epsilon=epsilon, delta=delta)
    return Budget.from_fractions(epsilon=epsilon, delta=delta, fractions=fractions)


def clip_epsilon_total(epsilon: float) -> float:
    """Total epsilon spent on DP clip-norm selection across all steps."""
    return CLIP_BUDGET_FRACTION * float(epsilon)


def per_step_clip_epsilon(epsilon: float, n_steps: int) -> float:
    """Per-step pure-DP budget for the Report-Noisy-Max clip selection."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    return clip_epsilon_total(epsilon) / float(n_steps)
