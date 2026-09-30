"""AdaClip-Budget: PrivForge's flagship algorithm (DP_MATH_SPEC.md section 4).

Three components, all realised with privacy ledger decoupled from ``C_t``:

  1. DP quantile-adaptive clipping -- Report-Noisy-Max selects ``C_t`` from a
     public grid; charged via ``step_pure(eps_q)`` (pure epsilon-DP, Lemma 1.4).
  2. Unit-sensitivity normalisation -- ``h_i = g_i / max(C_t, ||g_i||)`` gives
     ``||h_i|| <= 1`` *independent of* ``C_t``; the released noise std is the
     naked ``mu_t``. This is what keeps the ledger (INV-13 / I10) free of ``C_t``.
  3. Propagation-weighted budget reallocation ("privacy annealing") -- the per-step
     scale ``mu_t = mu0 * rho^{-(T-t)/2}`` follows the noise-propagation weight
     ``w_t = rho^{T-t}``. Degenerates to uniform ``mu`` (standard DP-SGD) when
     ``rho = 1`` (INV / degeneration test).

The class is the *algorithm kernel*; the pipeline (`pipeline/flagship.py`) wraps it
with the S1/S2/S3 safeguards. ``AdaClip-Budget`` and the ``AQUA-DP`` pipeline name
are kept distinct per architecture.md section 8.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from privforge.core.types import Budget
from privforge.domain.budget import clip_epsilon_total
from privforge.domain.clipping import DEFAULT_GRID
from privforge.domain.rdp import calibrate_mu
from privforge.training.base import DPLogisticModel
from privforge.training.dp_sgd import train_dpsgd
from privforge.training.logistic import rescale_R


@dataclass
class AdaClipConfig:
    T: int = 200
    q: float = 0.1
    eta: float = 0.1
    lam: float = 1e-3
    rho: float = 0.99
    p: float = 0.7
    clip_mode: str = "quantile"  # 'quantile' | 'fixed'
    mu_schedule: str = "propagation"  # 'propagation' | 'uniform'
    fixed_C: float | None = None
    unsafe_mu_times_C: bool = False  # ablation A6 only


class AdaClipBudget(DPLogisticModel):
    """Flagship: DP quantile clip + unit-sensitivity norm + propagation realloc."""

    def __init__(self, config: AdaClipConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdaClipConfig()
        self.last_result = None

    def _shape(self, T: int) -> np.ndarray | None:
        cfg = self.config
        if cfg.mu_schedule == "propagation" and 0.0 < cfg.rho < 1.0:
            # Privacy annealing: a step's noise propagates through the remaining
            # (T-t) steps, so later steps (less propagation) may carry MORE noise
            # while early steps (more propagation, should converge first) carry
            # LESS.  w_t = rho**(T-t) is the propagation weight -> mu_t = mu0 * w_t**0.5.
            # This yields small mu early (fast convergence) and mu0 late.
            t = np.arange(T)
            return np.array([cfg.rho ** ((T - t_i) / 2.0) for t_i in t], dtype=np.float64)
        return None

    def fit(self, dataset, budget: Budget, rng: np.random.Generator) -> AdaClipBudget:
        cfg = self.config
        R = float(dataset.bounds.x_norm_max)
        Xs = rescale_R(dataset.X, R)
        y = np.asarray(dataset.y, dtype=np.float64)

        eps_q_total = clip_epsilon_total(budget.epsilon)
        eps_q_per_step = eps_q_total / cfg.T
        shape = self._shape(cfg.T)
        mu0 = calibrate_mu(
            target_epsilon=budget.epsilon,
            target_delta=budget.delta,
            q=cfg.q,
            n_steps=cfg.T,
            eps_q_total=eps_q_total,
            shape=shape,
        )
        res = train_dpsgd(
            Xs,
            y,
            q=cfg.q,
            T=cfg.T,
            eps_q_per_step=eps_q_per_step,
            mu=mu0,
            shape=shape,
            fixed_C=cfg.fixed_C,
            clip_mode=cfg.clip_mode,
            eta=cfg.eta,
            lam=cfg.lam,
            rng=rng,
            grid=DEFAULT_GRID,
            p=cfg.p,
            unsafe_mu_times_C=cfg.unsafe_mu_times_C,
        )
        self.R = R
        self.w = res.theta / R  # back-map to original feature space
        self.last_result = res
        self.sigma = float(mu0 * (1.0 if shape is None else float(np.mean(shape))))
        self.eps_spent = float(res.accountant.epsilon(budget.delta))
        return self
