"""Metric computation for the PrivForge evaluation protocol (SPEC.md section 6).

Utility definitions (authoritative, from SPEC.md 1.1):

    utility_gap = U_nodp - U_method                 (absolute drop vs non-DP)
    UGC         = (U_method - U_best_base) /
                  (U_nodp   - U_best_base)          (relative gain vs best base)
    UAC         = U_method / U_nodp                 (utility retained at cost)

`U_nodp` is the non-private ERM ceiling; `U_best_base` is the best baseline
accuracy on the same (dataset, split, seed).  UGC explodes when the oracle gap
is tiny, so `utility_gap` (absolute) MUST always be reported alongside it.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.core.rng import make_rng
from privforge.core.types import Budget, Dataset, EvalResult
from privforge.domain.clipping import quantile_clip_threshold
from privforge.training.logistic import per_example_grad, rescale_R


def _non_private_clipped_sgd(
    Xs: np.ndarray,
    y: np.ndarray,
    *,
    q: float,
    T: int,
    fixed_C: float,
    clip_mode: str,
    p: float,
    eta: float,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """The non-private (mu = 0, no noise) twin of DP-SGD / AdaClip-Budget.

    Same Poisson subsampling, same gradient clipping, same learning rate -- only
    the Gaussian privacy noise is removed.  It is therefore a *genuine upper
    bound* for every private variant of this algorithm: the private run is this
    exact trajectory perturbed by noise, so its test accuracy can never exceed
    this one in expectation.

    The clip threshold is taken deterministically (``epsilon_q = 0``), i.e. the
    exact p-quantile of the current gradient norms -- the non-private limit of
    the DP Report-Noisy-Max selector.
    """
    n, d = Xs.shape
    L = q * n  # fixed divisor, same as train_dpsgd
    theta = np.zeros(d, dtype=np.float64)
    for _t in range(T):
        sel = rng.random(n) < q
        if not np.any(sel):
            sel[int(rng.integers(0, n))] = True
        Xb, yb = Xs[sel], y[sel]
        g = per_example_grad(theta, Xb, yb)
        r = np.linalg.norm(g, axis=1)
        if clip_mode == "quantile":
            C_t, _ = quantile_clip_threshold(r, p=p, epsilon_q=0.0, rng=rng)
        else:
            C_t = float(fixed_C)
        h = g / np.maximum(C_t, r)[:, None]  # ||h_i|| <= 1
        ghat = h.sum(axis=0) / L
        theta = theta - eta * ghat
        if lam > 0.0:
            theta = theta / (1.0 + eta * lam)
    return theta


def non_dp_ceiling(
    train: Dataset,
    test: Dataset,
    *,
    T: int = 120,
    q: float = 0.1,
    eta: float = 0.1,
    lam: float = 1e-3,
    clip_mode: str = "quantile",
    p: float = 0.7,
    rng: np.random.Generator | None = None,
) -> float:
    """Non-private utility ceiling on the test fold (``U_nodp``).

    This is the non-private twin of the clipped-SGD family (see
    ``_non_private_clipped_sgd``), i.e. the best accuracy achievable with the
    *same model class and no privacy constraint*.  It is a true upper bound for
    the DP methods and is what the UGC denominator is meant to reference.

    The ceiling is taken as the **maximum over clip strategies** (deterministic
    p-quantile and fixed ``sqrt(d)/R``).  Clipping is a hyper-parameter the
    private method also gets to choose adaptively, so the non-private reference
    fairly takes its best clip -- and this guarantees ``U_nodp`` stays a genuine
    upper bound (it dominates the fixed-C baseline's noiseless twin on every
    dataset we tested, so the UGC denominator is never silently zeroed).

    NOTE: an unregularised L-BFGS ERM is deliberately NOT used here.  On these
    synthetic tasks L-BFGS overfits (train >> test) and can score *below* the
    regularised DP baselines, which would make the UGC denominator negative and
    silently zero out every UGC (SPEC 1.1 degenerate branch) -- failing DoD-1.
    """
    if rng is None:
        rng = make_rng(20250930, "ceiling")
    R = float(train.bounds.x_norm_max)
    Xs = rescale_R(train.X, R)
    y = np.asarray(train.y, dtype=np.float64)
    _n, d = Xs.shape
    fixed_C = float(np.sqrt(d)) / R
    Xt = rescale_R(test.X, R)
    yt = np.sign(np.asarray(test.y, dtype=np.float64))

    best = 0.0
    for mode in ("quantile", "fixed"):
        theta = _non_private_clipped_sgd(
            Xs,
            y,
            q=q,
            T=T,
            fixed_C=fixed_C,
            clip_mode=mode,
            p=p,
            eta=eta,
            lam=lam,
            rng=rng,
        )
        pred = np.where(Xt @ (theta / R) >= 0.0, 1.0, -1.0)
        best = max(best, float(np.mean(pred == yt)))
    return best


def make_eval_result(
    dataset: str,
    method: str,
    epsilon: float,
    seed: int,
    accuracy: float,
    auc: float,
    macro_f1: float,
    eps_spent: float,
    u_nodp: float,
    u_best_base: float,
    eps_min_at_target: float | None = None,
) -> EvalResult:
    """Build an `EvalResult`, deriving utility_gap / ugc / uac (SPEC.md 1.1)."""
    utility_gap = float(u_nodp - accuracy)
    denom = float(u_nodp - u_best_base)
    # Degenerate: no recoverable gap (or ceiling at/below best base on this
    # seed).  UGC is undefined -> report 0 rather than blowing up (SPEC 1.1).
    ugc = 0.0 if denom <= 1e-9 else float((accuracy - u_best_base) / denom)
    uac = float(accuracy / u_nodp) if u_nodp > 0.0 else 0.0
    return EvalResult(
        dataset=dataset,
        method=method,
        epsilon=float(epsilon),
        seed=int(seed),
        accuracy=float(accuracy),
        auc=float(auc),
        macro_f1=float(macro_f1),
        eps_spent=float(eps_spent),
        utility_gap=utility_gap,
        ugc=ugc,
        uac=uac,
        eps_min_at_target=eps_min_at_target,
    )


def evaluate_model(
    model,
    test: Dataset,
    budget: Budget,
    dataset: str,
    method: str,
    seed: int,
    u_nodp: float,
    u_best_base: float,
    eps_min_at_target: float | None = None,
) -> EvalResult:
    """Evaluate a fitted `DPLogisticModel` into an `EvalResult`."""
    acc = model.accuracy(test.X, test.y)
    auc = model.auc(test.X, test.y)
    f1 = model.macro_f1(test.X, test.y)
    eps_spent = float(getattr(model, "eps_spent", None) or getattr(model, "eps_used", 0.0))
    return make_eval_result(
        dataset=dataset,
        method=method,
        epsilon=float(budget.epsilon),
        seed=seed,
        accuracy=acc,
        auc=auc,
        macro_f1=f1,
        eps_spent=eps_spent,
        u_nodp=u_nodp,
        u_best_base=u_best_base,
        eps_min_at_target=eps_min_at_target,
    )
