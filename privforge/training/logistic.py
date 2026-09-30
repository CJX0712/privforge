"""Logistic regression primitives shared by every training algorithm.

Labels are in {-1, +1}. The trainer re-scales features by the public row-norm
bound ``R`` so the working per-example gradient norm is <= 1 (unit sensitivity,
sec 4.3). The returned coefficient is mapped back to the original feature space.

  loss_i(theta) = log(1 + exp(-y_i x_i^T theta))
  grad_i        = -y_i x_i * sigma(-y_i x_i^T theta)      ||grad_i|| <= 1 after re-scaling

Author: 晨星
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def sigmoid(t: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    pos = t >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def per_example_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example logistic gradient (n, d)."""
    z = X @ theta
    s = sigmoid(-y * z)  # (n,)
    return (-y * s)[:, None] * X


def logistic_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float = 0.0) -> float:
    """Average logistic loss + (lam/2)||theta||^2."""
    z = X @ theta
    nll = np.mean(np.logaddexp(0.0, -y * z))
    return float(nll + 0.5 * lam * float(theta @ theta))


def logistic_grad_avg(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float = 0.0
) -> np.ndarray:
    """Average logistic gradient + lam*theta (regulariser)."""
    g = per_example_grad(theta, X, y).mean(axis=0)
    return g + lam * theta


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    max_iter: int = 200,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the (non private) ERM by L-BFGS-B. Used by OP/ObjP baselines."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = X.shape[1]
    if x0 is None:
        x0 = np.zeros(d, dtype=np.float64)

    def obj(theta: np.ndarray) -> float:
        return logistic_loss(theta, X, y, lam)

    def grad(theta: np.ndarray) -> np.ndarray:
        return logistic_grad_avg(theta, X, y, lam)

    res = minimize(obj, x0, jac=grad, method="L-BFGS-B", options={"maxiter": max_iter})
    return res.x


def rescale_R(X: np.ndarray, R: float) -> np.ndarray:
    """Re-scale features by the public bound so working norms are <= 1."""
    return np.asarray(X, dtype=np.float64) / float(R)
