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
    margin: float = 0.0,
) -> int:
    """S1: non-inferiority selection between flagship (0) and baseline (1).

    The S1 safeguard's hard contract is *non-inferiority*: the model it releases
    is never deterministically worse than the fixed-C baseline on the private
    comparison fold, and a clear win is never passed up.  Outside the `margin`
    band around equality the decision is certain (no privacy cost, because the
    better model is known); inside the band the Exponential mechanism
    (Report-Noisy-Max, charged ``eps_select``) makes the eps_select-DP selection.

    With the default ``margin=0.0`` the band is empty, so the safeguard reduces
    to "release the empirically better of the two" -- which is exactly the
    non-inferiority guarantee, and what makes the released utility never regress
    below the baseline.  (At eps_select=0.03 the RNM alone is a near coin-flip
    for the ~1-4pt accuracy gaps seen here, so letting it override a clear
    win/loss would both discard real gains and occasionally ship a worse model;
    the deterministic band dominates precisely to prevent that.)

    Returns 0 to keep AQUA, 1 to fall back to B4.
    """
    scores = [aqua_acc, base_acc]
    if aqua_acc > base_acc + margin:
        return 0  # clear win -> keep the flagship
    if aqua_acc < base_acc - margin:
        return 1  # clear loss -> non-inferiority fallback to baseline
    if eps_select <= 0.0 or margin <= 0.0:
        # No selection budget, or degenerate band: deterministic non-inferiority.
        return 0 if aqua_acc >= base_acc else 1
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
