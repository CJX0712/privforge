"""Noise-aware ensemble (SPEC.md pitfalls #5 / E401).

An ensemble is only allowed when its members are (a) post-processing views of a
*single* DP release (e.g. the NAP lambda-path in AdaClip-Budget) or (b) trained on
disjoint shards. Splitting one epsilon across M independently-trained models wastes
M times the noise and is forbidden (E401).

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from privforge.core.errors import CompositionError
from privforge.core.types import DPFitResult


def combine_views(result: DPFitResult) -> np.ndarray:
    """Combine NAP views of one release via their (sum-to-1) weights.

    Never re-spends privacy: all views are post-processing of a single DP output.
    """
    if not result.views:
        return np.asarray(result.w, dtype=np.float64)
    views = [np.asarray(v, dtype=np.float64) for v in result.views]
    weights = np.asarray(result.weights, dtype=np.float64)
    return sum(w * v for w, v in zip(weights, views, strict=False))


def assert_no_epsilon_split(members: Sequence[DPFitResult]) -> None:
    """Raise E401 if members carry independently-spent privacy budgets.

    Two members are compatible only if they share the same budget ledger (one
    release, post-processing) or were fit on disjoint shards. We enforce the
    post-processing case: all members must reference the same budget object.
    """
    ledgers = {id(m.budget_spent) for m in members if m.budget_spent is not None}
    if len(ledgers) > 1:
        raise CompositionError(
            "ensemble splits epsilon across independent DP releases",
            n_ledgers=len(ledgers),
        )


def noise_aware_ensemble(members: Sequence[DPFitResult]) -> np.ndarray:
    """Weighted combine of one-release views (INV-2 safe: monotonic in weights)."""
    assert_no_epsilon_split(members)
    vecs = [combine_views(m) for m in members]
    weights = np.asarray([m.weights.sum() for m in members], dtype=np.float64)
    weights = weights / weights.sum()
    out = np.zeros_like(vecs[0])
    for w, v in zip(weights, vecs, strict=False):
        out = out + w * v
    return out
