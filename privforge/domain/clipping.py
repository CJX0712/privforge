"""DP quantile-adaptive clipping (DP_MATH_SPEC.md section 4.2).

@adjacency: add-remove-one

The clip norm ``C_t`` is selected from a public candidate grid ``G`` using
**Report-Noisy-Max** (the Exponential mechanism). The score of each candidate
``G[k]`` is ``s_k = -|N_k - p*|B||`` where ``N_k`` is the number of batch
gradients whose norm does not exceed ``G[k]``. Adding or removing one sample
changes a single count by at most 1, so ``Delta s = 1`` and the mechanism is
pure epsilon_q-DP (Lemma sec 1.4). It is charged via ``accountant.step_pure``.

Crucially, the *value* of ``C_t`` never enters the privacy ledger -- only
``epsilon_q`` does (INV-13 / I10).

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from privforge.domain.mechanisms import report_noisy_max

DEFAULT_GRID: np.ndarray = np.logspace(-3.0, 2.0, 32)  # [1e-3, 1e2], K=32
DEFAULT_P: float = 0.7


def quantile_clip_threshold(
    grad_norms: Sequence[float] | np.ndarray,
    grid: np.ndarray = DEFAULT_GRID,
    p: float = DEFAULT_P,
    epsilon_q: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """Select a DP quantile-based clip norm ``C_t``.

    Parameters
    ----------
    grad_norms : per-example gradient L2 norms of the current batch.
    grid : candidate clip norms (public, data independent).
    p : target quantile (default 0.7).
    epsilon_q : pure-DP budget for this selection (0 disables privatisation).

    Returns
    -------
    C_t : chosen clip norm.
    idx : index into ``grid``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    r = np.asarray(list(grad_norms), dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    b = r.shape[0]
    counts = np.searchsorted(np.sort(r), grid, side="right")  # N_k
    scores = -np.abs(counts - p * b)  # Delta s == 1
    if epsilon_q <= 0.0:
        idx = int(np.argmax(scores))
    else:
        idx = report_noisy_max(scores.tolist(), epsilon_q, rng)
    return float(grid[idx]), idx
