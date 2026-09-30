"""Empirical privacy audit: lower-bound epsilon via membership inference.

DP_MATH_SPEC.md section 6. The audit CANNOT prove the declared epsilon is
correct; it only gives a lower bound ``eps_emp <= eps_true <= eps_declared``. If
``eps_emp > eps_declared`` something is wrong (INV-18 / DoD-6 / S3 safeguard).

We use the Jagielski et al. (2020) "neighbouring training pair" protocol: for
each trial build D0 (n-1 points) and D1 = D0 u {z}, fit the algorithm on both,
and record the loss on ``z`` under each. "in" = z was in the training set
(theta1), "out" = z was out (theta0). A membership-inference attacker guesses
"in" when the loss is below a threshold. Sweeping the threshold yields a ROC; the
trade-off-function bound then yields ``eps_emp``.

This module depends only on numpy + core (rule R2). The caller supplies the
fit/loss closures so no training code is imported here.

Author: 晨星
"""

from __future__ import annotations

import numpy as np


def _wilson_ci(p: np.ndarray, n: int, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Clopper-Pearson-style normal interval, clamped to [0, 1]."""
    p = np.clip(p, 0.0, 1.0)
    se = np.sqrt(p * (1.0 - p) / max(n, 1))
    lo = np.clip(p - z * se, 0.0, 1.0)
    hi = np.clip(p + z * se, 0.0, 1.0)
    return lo, hi


def roc_from_losses(
    loss_in: np.ndarray, loss_out: np.ndarray, n_thr: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """TPR/FPR of the loss-threshold membership attacker (higher loss -> "out")."""
    loss_in = np.asarray(loss_in, dtype=np.float64)
    loss_out = np.asarray(loss_out, dtype=np.float64)
    thr = np.linspace(
        float(np.minimum(loss_in.min(), loss_out.min())),
        float(np.maximum(loss_in.max(), loss_out.max())),
        n_thr,
    )
    # guess "in" when loss < tau
    tpr = np.array([float(np.mean(loss_in < t)) for t in thr])
    fpr = np.array([float(np.mean(loss_out < t)) for t in thr])
    return tpr, fpr


def empirical_epsilon(
    tpr: np.ndarray,
    fpr: np.ndarray,
    n_in: int,
    n_out: int,
    delta: float = 1e-5,
) -> float:
    """Empirical lower bound epsilon from a TPR/FPR ROC (sec 6.3).

    ``tpr``/``fpr`` are arrays (one rate per ROC threshold).  For each threshold
    the trade-off bound is ``eps >= log((TPR_lo - delta) / FPR_hi)``.  When a
    rate's lower Wilson CI bound sits at/below the ``delta`` floor the attacker
    cannot beat chance on that side, the log term is undefined, and the empirical
    epsilon contribution for that threshold is zero -- never NaN (a NaN would
    silently fail the whole audit).
    """
    tpr = np.clip(np.asarray(tpr, dtype=np.float64), 0.0, 1.0)
    fpr = np.clip(np.asarray(fpr, dtype=np.float64), 0.0, 1.0)
    tpr_lo, tpr_hi = _wilson_ci(tpr, n_in)
    fpr_lo, fpr_hi = _wilson_ci(fpr, n_out)

    def _term(num_lo: np.ndarray, den_hi: np.ndarray) -> np.ndarray:
        num = num_lo - delta
        den = np.maximum(den_hi, delta)
        x = num / den
        term = np.zeros_like(x)
        ok = (num > 0.0) & (x > 0.0)
        term[ok] = np.log(x[ok])  # only evaluate the well-defined log domain
        return term

    cand = np.stack([_term(tpr_lo, fpr_hi), _term(fpr_lo, tpr_hi)], axis=0)
    return float(np.clip(float(np.nanmax(cand)), 0.0, None))


def audit_model(
    fit_fn,  # callable(Dataset, rng) -> model
    loss_fn,  # callable(model, X, y) -> float (scalar loss on one point)
    dataset,
    n_trials: int = 200,
    delta: float = 1e-5,
    seed: int = 12345,
) -> float:
    """Run the in-out training-pair audit and return eps_emp.

    `dataset` is split: each trial draws n-1 points as D0 and one held-out z as
    the add/remove target (add/remove-one adjacency).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(dataset.X, dtype=np.float64)
    y = np.asarray(dataset.y, dtype=np.float64)
    n = X.shape[0]
    loss_in_list: list[float] = []
    loss_out_list: list[float] = []
    for _ in range(int(n_trials)):
        perm = rng.permutation(n)
        z_idx = perm[0]
        rest = perm[1:]
        d0_idx = rest[: n - 1]
        # D0 = n-1 points; D1 = D0 + z
        X0, y0 = X[d0_idx], y[d0_idx]
        X1 = np.vstack([X0, X[z_idx : z_idx + 1]])
        y1 = np.concatenate([y0, y[z_idx : z_idx + 1]])
        m0 = fit_fn(_sub_dataset(dataset, X0, y0), rng)
        m1 = fit_fn(_sub_dataset(dataset, X1, y1), rng)
        loss_in_list.append(float(loss_fn(m1, X[z_idx : z_idx + 1], y[z_idx : z_idx + 1])))
        loss_out_list.append(float(loss_fn(m0, X[z_idx : z_idx + 1], y[z_idx : z_idx + 1])))
    tpr, fpr = roc_from_losses(np.array(loss_in_list), np.array(loss_out_list))
    return empirical_epsilon(tpr, fpr, len(loss_in_list), len(loss_out_list), delta)


def _sub_dataset(dataset, X, y):
    """Build a tiny Dataset view for the audit (shares bounds/meta)."""
    from privforge.core.types import Dataset

    return Dataset(name=dataset.name, X=X, y=y, bounds=dataset.bounds, meta=dataset.meta)
