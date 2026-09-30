"""Hyper-parameter optimisation layer (SPEC.md pitfalls #5 / #18).

HPO tunes *architecture* choices only (T, eta, lam, rho, p, clip_mode,
mu_schedule).  It never tunes data-dependent quantities (C, bounds, class
priors) on private data without accounting -- that would leak (pitfall #18).
The validation fold uses only the public bounds, never private statistics.

Author: 晨星
"""

from __future__ import annotations

from privforge.hpo import space, transfer, tuner

__all__ = ["space", "transfer", "tuner"]
