"""RDP accountant core (DP_MATH_SPEC.md section 2).

@adjacency: add-remove-one
@sensitivity: unit (sensitivity of the summed clipped signal is exactly 1)

Everything here is pure numpy. The single-step Rényi divergence of the
Poisson-subsampled Gaussian mechanism is computed by *numerical integration*
with a grid that provably covers both directional peaks (INV-15 / I11). The
conversion ``RDP -> (epsilon, delta)-DP`` takes the pointwise minimum over the
alpha grid, and the inverse calibration is done with Brent's method (never
Newton, the minimum is non-differentiable at grid switches).

Author: 晨星
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

# --- alpha grid (DP_MATH_SPEC.md section 2.5) --------------------------------
# Fractional band (delta-dominated), integer band (numerically stable), large
# band (covers the phase transition of subsampled Gaussian RDP). Never includes
# alpha == 1 (would make log(1/delta)/(alpha-1) explode -> I13).
ORDERS: np.ndarray = np.unique(
    np.concatenate(
        [
            np.linspace(1.01, 2.00, 20),
            np.arange(2.0, 65.0),
            np.logspace(np.log10(65.0), np.log10(512.0), 24),
        ]
    )
)


def _log_integrate_rows(L: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Trapezoidal integral of exp(L) over the common grid z, in log space.

    ``L`` has shape (K, M), ``z`` has shape (M,). Returns log integral (K,).
    Subtracting the row maximum before exponentiating avoids overflow at large
    alpha (pitfall #12).
    """
    m = L.max(axis=1, keepdims=True)
    e = np.exp(L - m)
    integral = np.trapezoid(e, z, axis=1)
    integral = np.maximum(integral, 0.0)
    return np.log(integral) + m[:, 0]


def rdp_poisson_gaussian(
    q: float,
    mu: float,
    orders: np.ndarray = ORDERS,
    n_points: int = 4001,
) -> np.ndarray:
    """Rényi divergence eps_alpha of the Poisson-subsampled Gaussian mechanism.

    Parameters
    ----------
    q : Poisson sampling rate (expected batch size / n). q >= 1 degenerates to
        the closed form alpha/(2 mu^2); q <= 0 gives 0.
    mu : normalized noise scale (sigma_abs = mu * sensitivity).
    orders : array of alpha > 1.

    Returns
    -------
    eps_alpha : array, same length as ``orders``.

    The integrand peaks at z = alpha*r and z = -(alpha-1)*r with r = 1/mu.
    The grid therefore spans [-max(0,(alpha-1)r)-12, max(0,alpha r)+12] to
    cover BOTH peaks (silent-failure pitfall #11, INV-15).
    """
    orders = np.asarray(orders, dtype=np.float64)
    if q >= 1.0:
        return orders / (2.0 * mu * mu)
    if q <= 0.0:
        return np.zeros_like(orders)

    r = 1.0 / mu
    lo = float(np.min(-np.maximum(0.0, (orders - 1.0) * r) - 12.0))
    hi = float(np.max(np.maximum(0.0, orders * r) + 12.0))
    # Cap the number of points; dz never larger than 0.05 so the peak is
    # always resolved (coarser only when the span is enormous, which occurs
    # only for tiny mu far outside our calibrated range).
    span = hi - lo
    nz = int(min(n_points, max(401, math.ceil(span / 0.05) + 1)))
    z = np.linspace(lo, hi, nz)

    lq = norm.logpdf(z, 0.0, 1.0)  # (M,)
    lp = logsumexp(
        np.stack([np.log1p(-q) + lq, np.log(q) + norm.logpdf(z, r, 1.0)], axis=0),
        axis=0,
    )  # (M,)

    a = orders[:, None]  # (A,1)
    L1 = a * lp[None, :] + (1.0 - a) * lq[None, :]  # P || Q
    L2 = a * lq[None, :] + (1.0 - a) * lp[None, :]  # Q || P
    both = np.concatenate([L1, L2], axis=0)
    integ = _log_integrate_rows(both, z)
    integ = integ.reshape(2, -1)
    result = np.max(integ, axis=0)  # (A,)
    return result / (orders - 1.0)


def rdp_to_dp(eps_alpha: np.ndarray, orders: np.ndarray, delta: float) -> float:
    """Convert an RDP curve to (epsilon, delta)-DP by pointwise min (sec 2.3).

    epsilon(alpha) = eps_alpha + log(1/delta)/(alpha-1); return min over alpha.
    """
    eps_alpha = np.asarray(eps_alpha, dtype=np.float64)
    orders = np.asarray(orders, dtype=np.float64)
    if not (delta > 0.0 and delta < 1.0):
        raise ValueError("delta must satisfy 0 < delta < 1")
    eps = eps_alpha + np.log(1.0 / delta) / (orders - 1.0)
    return float(np.min(eps))


def _eps_total(
    mu: float,
    q: float,
    orders: np.ndarray,
    n_steps: int,
    eps_q_total: float,
    shape: np.ndarray | None,
) -> float:
    """Total epsilon for a given base mu (used by calibrate_mu)."""
    if shape is None:
        step = rdp_poisson_gaussian(q, mu, orders)
        eps_alpha = eps_q_total + n_steps * step
    else:
        # mu_t = mu * shape[t] ; sum the per-step RDP (exact RDP composition).
        eps_alpha = eps_q_total + sum(
            rdp_poisson_gaussian(q, float(mu) * float(s), orders) for s in shape
        )
    return rdp_to_dp(eps_alpha, orders, _DELTA_REF)


_DELTA_REF: float = 1e-5


def calibrate_mu(
    target_epsilon: float,
    target_delta: float = 1e-5,
    q: float = 0.1,
    n_steps: int = 1,
    eps_q_total: float = 0.0,
    shape: Sequence[float] | None = None,
    orders: np.ndarray = ORDERS,
) -> float:
    """Calibrate the (base) normalized noise scale ``mu`` to hit a budget.

    Single release (``n_steps == 1``) uses the Balle-Wang analytic
    calibration; multi-step uses the RDP composition + Brent inversion.

    Parameters
    ----------
    target_epsilon, target_delta : the (epsilon, delta)-DP budget to match.
    q : Poisson sampling rate (ignored when n_steps == 1).
    n_steps : number of composed Gaussian mechanisms.
    eps_q_total : total epsilon spent on pure-DP clipping selection
        (contributes ``eps_q_total`` to every alpha -- Lemma sec 1.4).
    shape : per-step multipliers mu_t = mu * shape[t] (propagation schedule).
        Length must equal ``n_steps`` when provided.

    Returns
    -------
    mu : normalized noise scale.
    """
    global _DELTA_REF
    _DELTA_REF = float(target_delta)
    if n_steps == 1 and (shape is None):
        # Single release: Balle-Wang is tighter than RDP (sec 2.6).
        from privforge.domain.mechanisms import calibrate_gaussian_bw

        return calibrate_gaussian_bw(target_epsilon, target_delta)

    if shape is not None:
        shape = np.asarray(list(shape), dtype=np.float64)
        if shape.shape[0] != n_steps:
            raise ValueError("len(shape) must equal n_steps")

    def f(mu: float) -> float:
        return _eps_total(mu, q, orders, n_steps, eps_q_total, shape) - target_epsilon

    # epsilon(mu) is strictly decreasing in mu.
    lo = 1e-9
    hi = 1.0
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e12:
            break
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-14))
