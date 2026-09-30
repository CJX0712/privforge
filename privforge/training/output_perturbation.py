"""Output Perturbation baseline (SPEC.md baseline B2, DP_MATH_SPEC.md sec 3.1).

Solve the (non private) ERM, then add Laplace noise scaled by the L1 sensitivity
of the solution. Pure epsilon-DP (no delta). The noise carries a sqrt(d) factor
(high-dimensional curse); this is exactly why DP-SGD is preferred in high dim.

Inverse limit (INV-20): as epsilon -> inf the noise vanishes and w -> theta*.
Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.core.types import Budget
from privforge.domain.mechanisms import Laplace
from privforge.domain.sensitivity import output_perturbation_l1_sensitivity
from privforge.training.base import DPLogisticModel
from privforge.training.logistic import fit_logistic


class OutputPerturbation(DPLogisticModel):
    """Baseline B2: ERM + Laplace output perturbation (pure epsilon-DP)."""

    def __init__(self, lam: float = 1e-2, max_iter: int = 300) -> None:
        super().__init__()
        self.lam = lam
        self.max_iter = max_iter
        self.eps_used = 0.0

    def fit(self, dataset, budget: Budget, rng: np.random.Generator) -> OutputPerturbation:
        X = np.asarray(dataset.X, dtype=np.float64)
        y = np.asarray(dataset.y, dtype=np.float64)
        n, d = X.shape
        R = float(dataset.bounds.x_norm_max)
        lam = self.lam

        theta_star = fit_logistic(X, y, lam=lam, max_iter=self.max_iter)
        eps = float(budget.eps_train)
        delta1 = output_perturbation_l1_sensitivity(R, lam, n, d)
        b = delta1 / eps
        mech = Laplace(sensitivity=delta1, epsilon=eps, rng=rng)
        w = mech.release(theta_star)
        self.R = R
        self.w = w
        self.eps_used = eps
        self.sigma = b
        self.theta_star = theta_star
        return self
