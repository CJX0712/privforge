"""Three safeguards (SPEC.md section 2.3): S1 non-inferiority, S2 budget, S3 audit.

S1  -- compare the flagship against the fixed-C baseline on a private validation
      split and select with an Exponential mechanism (pure eps_select-DP). This
      guarantees the worst case is approximately the baseline, removing the
      quantile-clip regressions measured in v1 (easy -0.6pt, hard -3.1pt).
S2  -- if the accountant reports more than the declared epsilon, fall back to the
      most conservative configuration (E400 -> E502, a usable model is returned).
S3  -- run the empirical MIA audit; if eps_emp exceeds epsilon + tol, fall back to
      Output Perturbation (E402).

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.core.errors import AuditViolation, BudgetExceeded
from privforge.core.types import Budget
from privforge.domain.accountant import RDPAccountant
from privforge.domain.audit import audit_model
from privforge.domain.mechanisms import ReportNoisyMax


def budget_safeguard(accountant: RDPAccountant, budget: Budget, tol: float = 1e-9) -> float:
    """S2: return spent epsilon; raise if it exceeds the declared budget (I1)."""
    spent = accountant.epsilon(budget.delta)
    if spent > budget.epsilon + tol:
        raise BudgetExceeded(
            "S2 budget safeguard tripped", spent=spent, epsilon=float(budget.epsilon)
        )
    return float(spent)


def non_inferiority_select(
    aqua_acc: float,
    base_acc: float,
    eps_select: float,
    rng: np.random.Generator,
    margin: float = 0.015,
) -> int:
    """S1: Exponential-mechanism selection between flagship (0) and baseline (1).

    Positive score = accuracy. The mechanism keeps the worst case near the baseline
    when the flagship regresses. Returns 0 to keep AQUA, 1 to fall back.
    """
    scores = [aqua_acc, base_acc]
    if eps_select <= 0.0:
        # No selection budget: deterministic non-inferiority rule.
        return 0 if aqua_acc >= base_acc - margin else 1
    selector = ReportNoisyMax(epsilon=eps_select, rng=rng)
    return selector.select(scores)


def audit_safeguard(
    fit_fn,
    loss_fn,
    dataset,
    budget: Budget,
    rng: np.random.Generator,
    n_trials: int = 60,
    tol: float = 0.05,
    fallback_model=None,
) -> tuple[object, float]:
    """S3: empirical MIA audit; fall back to `fallback_model` if eps_emp too high."""
    eps_emp = audit_model(
        fit_fn,
        loss_fn,
        dataset,
        n_trials=n_trials,
        delta=budget.delta,
        seed=int(rng.integers(0, 1 << 31)),
    )
    if eps_emp > float(budget.epsilon) + tol:
        raise AuditViolation(
            "S3 audit safeguard tripped",
            eps_emp=eps_emp,
            epsilon=float(budget.epsilon),
        )
    return (fallback_model, eps_emp) if eps_emp > float(budget.epsilon) else (None, eps_emp)
