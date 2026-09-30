"""HPO search space + default configuration (SPEC.md pitfall #18 safe).

Only architecture-level hyperparameters are exposed.  `C` is intentionally NOT
tunable: it is either the public `sqrt(d)/R` default or quantile-adaptive (which
is itself a DP mechanism), so it never consumes private statistics silently.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HpoConfig:
    T: int = 200
    q: float = 0.1
    eta: float = 0.1
    lam: float = 1e-3
    rho: float = 0.99
    p: float = 0.7
    clip_mode: str = "quantile"  # 'quantile' | 'fixed'
    mu_schedule: str = "propagation"  # 'propagation' | 'uniform'

    def as_adaclip(self):
        from privforge.training.adaclip_budget import AdaClipConfig

        return AdaClipConfig(
            T=self.T,
            q=self.q,
            eta=self.eta,
            lam=self.lam,
            rho=self.rho,
            p=self.p,
            clip_mode=self.clip_mode,
            mu_schedule=self.mu_schedule,
        )


def default_space():
    """Return the hyperparameter bounds as a plain dict (optuna-agnostic)."""
    return {
        "T": (80, 320),
        "q": (0.05, 0.2),
        "eta": (0.05, 0.5),
        "lam": (1e-4, 1e-2),
        "rho": (0.90, 0.999),
        "p": (0.5, 0.9),
        "clip_mode": ("quantile", "fixed"),
        "mu_schedule": ("propagation", "uniform"),
    }


def default_config() -> HpoConfig:
    return HpoConfig()
