"""RDP accountant (DP_MATH_SPEC.md section 2.7).

@adjacency: add-remove-one

The ledger only ever consumes ``mu`` (the normalized noise scale) -- never the
clip norm ``C_t``. This is what keeps the privacy book *decoupled* from the
adaptive clip trajectory (INV-13 / I10). A pure-epsilon mechanism contributes a
constant to every alpha (Lemma, sec 1.4).

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.core.errors import BudgetExceeded
from privforge.domain.rdp import ORDERS, rdp_poisson_gaussian, rdp_to_dp


class RDPAccountant:
    """Cumulative RDP ledger with exact (non-relaxed) composition."""

    def __init__(self, orders: np.ndarray = ORDERS) -> None:
        self.orders = np.asarray(orders, dtype=np.float64)
        if not (self.orders > 1.0).all():
            raise ValueError("alpha grid must satisfy alpha > 1 (I13)")
        self.eps_alpha = np.zeros_like(self.orders)
        self.history: list[np.ndarray] = []

    def step(self, q: float, mu: float) -> None:
        """Charge one Poisson-sampled Gaussian mechanism release (unit sensitivity)."""
        if mu <= 0.0:
            raise ValueError("mu must be positive")
        if q < 0.0 or q > 1.0:
            raise ValueError("q must be in [0, 1]")
        self.eps_alpha = self.eps_alpha + rdp_poisson_gaussian(q, mu, self.orders)
        self.history.append(self.eps_alpha.copy())

    def step_pure(self, eps: float) -> None:
        """Charge a pure epsilon-DP mechanism (Exponential / Report-Noisy-Max).

        Pure epsilon-DP implies (alpha, epsilon)-RDP for every alpha (sec 1.4).
        """
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        self.eps_alpha = self.eps_alpha + eps
        self.history.append(self.eps_alpha.copy())

    def epsilon(self, delta: float) -> float:
        """Convert the current ledger to (epsilon, delta)-DP."""
        return rdp_to_dp(self.eps_alpha, self.orders, delta)

    def best_order(self, delta: float) -> float:
        """The alpha that attains the min in the RDP->DP conversion."""
        eps = self.eps_alpha + np.log(1.0 / delta) / (self.orders - 1.0)
        return float(self.orders[int(np.argmin(eps))])

    def assert_within(self, epsilon: float, delta: float, tol: float = 1e-9) -> None:
        """Raise E400 if the spent epsilon exceeds the declared budget (I1)."""
        if self.epsilon(delta) > epsilon + tol:
            raise BudgetExceeded(
                "privacy budget exceeded",
                spent=self.epsilon(delta),
                epsilon=epsilon,
                delta=delta,
            )
