"""Objective Perturbation baseline (SPEC.md baseline B3, DP_MATH_SPEC.md sec 3.2).

Perturb the *objective* with a K-norm (multidimensional Laplace) vector ``b`` and
solve the argmin once (post-processing immunity makes the argmin free). Pure
epsilon-DP. No sqrt(d) curse, and CMS11 variants do not require strong convexity.

Deterministic bound (INV-21): ||theta_priv - theta*|| <= ||b|| / (lambda n).
Author: 晨星
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from privforge.core.types import Budget
from privforge.domain.sensitivity import objective_perturbation_c
from privforge.training.base import DPLogisticModel
from privforge.training.logistic import fit_logistic, logistic_loss


class ObjectivePerturbation(DPLogisticModel):
    """Baseline B3: objective perturbation (Chaudhuri-Monteleoni-Sarwate 2011)."""

    def __init__(self, lam: float = 1e-2, max_iter: int = 300) -> None:
        super().__init__()
        self.lam = lam
        self.max_iter = max_iter
        self.b_norm: float = 0.0
        self.eps_used = 0.0

    def _perturbed_obj(self, X, y, b, lam, n):
        def obj(theta):
            return logistic_loss(theta, X, y, lam) + float(b @ theta) / n

        def grad(theta):
            from privforge.training.logistic import logistic_grad_avg

            return logistic_grad_avg(theta, X, y, lam) + b / n

        return obj, grad

    def fit(self, dataset, budget: Budget, rng: np.random.Generator) -> ObjectivePerturbation:
        X = np.asarray(dataset.X, dtype=np.float64)
        y = np.asarray(dataset.y, dtype=np.float64)
        n, d = X.shape
        R = float(dataset.bounds.x_norm_max)
        lam = self.lam
        eps = float(budget.eps_train)

        c = objective_perturbation_c(R, lam, eps, n)
        # K-norm mechanism: direction uniform on sphere, radius Gamma(d, 1/c).
        u = rng.normal(size=d)
        u = u / np.linalg.norm(u)
        r = rng.gamma(shape=d, scale=1.0 / c)
        b = r * u
        self.b_norm = float(np.linalg.norm(b))

        theta_star = fit_logistic(X, y, lam=lam, max_iter=self.max_iter)
        obj, grad = self._perturbed_obj(X, y, b, lam, n)
        res = minimize(
            obj, theta_star, jac=grad, method="L-BFGS-B", options={"maxiter": self.max_iter}
        )
        w = res.x

        self.R = R
        self.w = w
        self.eps_used = eps
        self.theta_star = theta_star
        self.sigma = self.b_norm / (lam * n)  # INV-21 bound
        return self
