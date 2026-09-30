"""Differential privacy domain primitives.

This layer holds the pure DP mathematics: mechanisms, RDP accountant,
quantile clipping, budget accounting, sensitivity bounds and the empirical
privacy audit. It depends ONLY on `core` (rule R2) and never on `training`,
`eval`, `hpo` or `pipeline`.

All constants, docstrings and proofs follow `DP_MATH_SPEC.md`. The three
silent-failure hard constraints are enforced here:

1. No ``mu * C_t`` anywhere -- privacy noise uses the *naked* ``mu_t``.
2. The RDP numerical grid must cover both directional peaks:
   ``[-max(0,(alpha-1) r)-12, max(0, alpha r)+12]`` with ``r = 1/mu``.
3. Gaussian calibration uses Balle-Wang, never the classic formula.

Author: 晨星
"""

from __future__ import annotations

from privforge.domain import accountant, audit, budget, clipping, mechanisms, sensitivity

__all__ = [
    "accountant",
    "audit",
    "budget",
    "clipping",
    "mechanisms",
    "sensitivity",
]
