"""Baseline DP-SGD (SPEC.md baseline B4).

@adjacency: add-remove-one

Poisson subsampling + per-example gradient clipping + Gaussian noise, with the
RDP accountant doing the *amplified* composition (pitfall #6: a baseline without
subsampling amplification is a straw man). The noise scale ``mu`` is calibrated
upstream (pipeline/flagship) so that the composed RDP total hits the declared
budget. The clip norm ``C`` is fixed here; the adaptive version lives in
``adaclip_budget.py``.

Critical invariant (INV-12 / I9): the released noise std is the *naked* ``mu_t``
(unit sensitivity). ``C`` only rescales the *update direction* after the noise is
drawn, so it never enters the privacy ledger.

Author: 晨星
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from privforge.core.types import Budget
from privforge.domain.accountant import RDPAccountant
from privforge.domain.budget import clip_epsilon_total
from privforge.domain.clipping import DEFAULT_GRID, quantile_clip_threshold
from privforge.domain.rdp import calibrate_mu
from privforge.training.base import DPLogisticModel
from privforge.training.logistic import per_example_grad, rescale_R


@dataclass
class DPSGDResult:
    theta: np.ndarray
    accountant: RDPAccountant
    eps_q_per_step: float
    mu_base: float
    shape: np.ndarray | None
    clip_mode: str
    c_trajectory: list[float]


def train_dpsgd(
    Xs: np.ndarray,
    y: np.ndarray,
    *,
    q: float,
    T: int,
    eps_q_per_step: float,
    mu: float,
    shape: np.ndarray | None = None,
    fixed_C: float | None = None,
    clip_mode: str = "fixed",
    eta: float = 0.1,
    lam: float = 1e-3,
    rng: np.random.Generator | None = None,
    grid: np.ndarray = DEFAULT_GRID,
    p: float = 0.7,
    seed_tag: object = "dpsgd",
    unsafe_mu_times_C: bool = False,
) -> DPSGDResult:
    """Run DP-SGD.

    Parameters
    ----------
    Xs : re-scaled features, rows ||x_i|| <= 1.
    y : labels in {-1, +1}.
    q : Poisson sampling rate.
    T : number of steps.
    eps_q_per_step : pure-DP budget per step for clip selection (0 for fixed C).
    mu : base normalized noise scale.
    shape : per-step multipliers mu_t = mu*shape[t] (propagation schedule).
    fixed_C : clip norm when clip_mode='fixed'.
    clip_mode : 'fixed' or 'quantile'.

    Returns
    -------
    DPSGDResult with the trained coefficient (in the re-scaled space) and ledger.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = Xs.shape
    if rng is None:
        rng = np.random.default_rng(0)
    L = q * n  # fixed divisor (sec 3.3): never divide by random |B_t|

    acc = RDPAccountant()
    theta = np.zeros(d, dtype=np.float64)
    c_traj: list[float] = []
    int(rng.integers(0, 1 << 62))

    for t in range(T):
        mu_t = mu if shape is None else mu * float(shape[t])
        # Poisson subsample
        sel = rng.random(n) < q
        if not np.any(sel):
            sel[int(rng.integers(0, n))] = True
        Xb, yb = Xs[sel], y[sel]
        g = per_example_grad(theta, Xb, yb)  # (|B|, d)
        r = np.linalg.norm(g, axis=1)  # (|B|,)
        if clip_mode == "quantile":
            C_t, _ = quantile_clip_threshold(r, grid=grid, p=p, epsilon_q=eps_q_per_step, rng=rng)
            acc.step_pure(eps_q_per_step)
        else:
            C_t = float(fixed_C) if fixed_C is not None else 1.0
        h = g / np.maximum(C_t, r)[:, None]  # ||h_i|| <= 1, independent of C_t
        if unsafe_mu_times_C:
            # NEGATIVE CONTROL (ablation A6): couples the noise scale to C_t.
            # This violates the AdaClip-Budget decoupling theorem and is caught
            # by INV-13 / I9. Never used by any shipped configuration.
            noise = rng.normal(0.0, mu_t * C_t, size=d)
            ghat = (h.sum(axis=0) + noise) / L
            theta = theta - eta * ghat
            acc.step(q, mu_t * C_t)
        else:
            noise = rng.normal(0.0, mu_t, size=d)  # naked mu_t, NOT mu_t * C_t
            ghat = (h.sum(axis=0) + noise) / L
            # Unit-sensitivity update: h_i already carries the per-example clip,
            # so we step by the averaged unit-sensitivity gradient.  C_t never
            # enters the noise scale (ledger stays decoupled, INV-13 / I9).
            theta = theta - eta * ghat
            acc.step(q, mu_t)
        if lam > 0.0:
            theta = theta / (1.0 + eta * lam)  # optional L2 regularisation
        c_traj.append(float(C_t))
    return DPSGDResult(
        theta=theta,
        accountant=acc,
        eps_q_per_step=eps_q_per_step,
        mu_base=mu,
        shape=shape,
        clip_mode=clip_mode,
        c_trajectory=c_traj,
    )


class DPSGDBaseline(DPLogisticModel):
    """Baseline B4: fixed-C, uniform-mu DP-SGD with amplified RDP (no AdaClip)."""

    def __init__(
        self,
        T: int = 200,
        q: float = 0.1,
        eta: float = 0.1,
        lam: float = 1e-3,
        fixed_C: float | None = None,
        clip_mode: str = "fixed",
    ) -> None:
        super().__init__()
        self.T = T
        self.q = q
        self.eta = eta
        self.lam = lam
        self.fixed_C = fixed_C
        self.clip_mode = clip_mode
        self.last_result = None

    def _fixed_C(self, dataset) -> float:
        if self.fixed_C is not None:
            return float(self.fixed_C)
        # "C = sqrt(d)" in the original space -> sqrt(d)/R in the re-scaled space.
        return math.sqrt(float(dataset.X.shape[1])) / float(dataset.bounds.x_norm_max)

    def fit(self, dataset, budget: Budget, rng: np.random.Generator) -> DPSGDBaseline:
        R = float(dataset.bounds.x_norm_max)
        Xs = rescale_R(dataset.X, R)
        y = np.asarray(dataset.y, dtype=np.float64)
        eps_q_total = clip_epsilon_total(budget.epsilon) if self.clip_mode == "quantile" else 0.0
        eps_q_per_step = eps_q_total / self.T
        mu = calibrate_mu(
            target_epsilon=budget.epsilon,
            target_delta=budget.delta,
            q=self.q,
            n_steps=self.T,
            eps_q_total=eps_q_total,
            shape=None,
        )
        res = train_dpsgd(
            Xs,
            y,
            q=self.q,
            T=self.T,
            eps_q_per_step=eps_q_per_step,
            mu=mu,
            fixed_C=self._fixed_C(dataset),
            clip_mode=self.clip_mode,
            eta=self.eta,
            lam=self.lam,
            rng=rng,
        )
        self.R = R
        self.w = res.theta / R
        self.last_result = res
        self.sigma = float(mu)
        self.eps_spent = float(res.accountant.epsilon(budget.delta))
        return self
