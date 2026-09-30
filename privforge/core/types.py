"""Frozen dataclass contracts (architecture.md section 4.1).

All value objects are immutable and slot based. Validation happens in
`__post_init__` so an illegal object can never exist.
Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from privforge.core.constants import (
    BUDGET_FRACTIONS,
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    EPS_TOL,
)
from privforge.core.errors import BoundsError, BudgetConfigError, InputError, SplitError

Array = np.ndarray


def _check_finite(name: str, value: float) -> float:
    if value is None or not np.isfinite(float(value)):
        raise InputError(f"{name} must be a finite number", name=name, value=value)
    return float(value)


@dataclass(frozen=True, slots=True)
class Bounds:
    """Public, data independent bounds. Both fields are mandatory."""

    x_norm_max: float  # public bound: ||x|| <= x_norm_max
    y_abs_max: float  # public bound: |y| <= y_abs_max

    def __post_init__(self) -> None:
        if self.x_norm_max is None or self.y_abs_max is None:
            raise BoundsError("Bounds fields are mandatory and must not be None")
        if not np.isfinite(self.x_norm_max) or self.x_norm_max <= 0.0:
            raise BoundsError("x_norm_max must be finite and positive", x_norm_max=self.x_norm_max)
        if not np.isfinite(self.y_abs_max) or self.y_abs_max <= 0.0:
            raise BoundsError("y_abs_max must be finite and positive", y_abs_max=self.y_abs_max)

    def as_dict(self) -> dict[str, float]:
        """Serialisable view."""
        return {"x_norm_max": float(self.x_norm_max), "y_abs_max": float(self.y_abs_max)}


@dataclass(frozen=True, slots=True)
class Budget:
    """Privacy budget with its three private shares."""

    epsilon: float
    delta: float
    eps_clip: float
    eps_train: float
    eps_select: float

    @classmethod
    def from_fractions(
        cls,
        epsilon: float = DEFAULT_EPSILON,
        delta: float = DEFAULT_DELTA,
        fractions: tuple[float, float, float] = BUDGET_FRACTIONS,
    ) -> Budget:
        """Build a budget from (frac_clip, frac_train, frac_select)."""
        frac_clip, frac_select = float(fractions[0]), float(fractions[2])
        return cls(
            epsilon=float(epsilon),
            delta=float(delta),
            eps_clip=epsilon * frac_clip,
            eps_train=epsilon * (1.0 - frac_clip - frac_select),
            eps_select=epsilon * frac_select,
        )

    def __post_init__(self) -> None:
        _check_finite("epsilon", self.epsilon)
        _check_finite("delta", self.delta)
        if self.epsilon <= 0.0:
            raise BudgetConfigError("epsilon must be > 0", epsilon=self.epsilon)
        if not 0.0 < self.delta < 1.0:
            raise BudgetConfigError("delta must satisfy 0 < delta < 1", delta=self.delta)
        for name in ("eps_clip", "eps_train", "eps_select"):
            value = _check_finite(name, getattr(self, name))
            if value < 0.0:
                raise BudgetConfigError(f"{name} must be >= 0", **{name: value})
        if self.allocated > self.epsilon + EPS_TOL:
            raise BudgetConfigError(
                "eps_clip + eps_train + eps_select must not exceed epsilon",
                allocated=self.allocated,
                epsilon=self.epsilon,
            )

    @property
    def allocated(self) -> float:
        """Sum of the three private shares."""
        return float(self.eps_clip + self.eps_train + self.eps_select)

    @property
    def remaining(self) -> float:
        """Unallocated epsilon. Never negative by construction."""
        return float(max(0.0, self.epsilon - self.allocated))

    def as_dict(self) -> dict[str, float]:
        """Serialisable view."""
        return {
            "epsilon": float(self.epsilon),
            "delta": float(self.delta),
            "eps_clip": float(self.eps_clip),
            "eps_train": float(self.eps_train),
            "eps_select": float(self.eps_select),
            "allocated": self.allocated,
            "remaining": self.remaining,
        }


@dataclass(frozen=True, slots=True)
class Dataset:
    """A binary classification dataset with mandatory public bounds."""

    name: str
    X: Array
    y: Array
    bounds: Bounds
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bounds is None:
            raise BoundsError("Dataset.bounds is mandatory (missing bounds leaks privacy)")
        if not isinstance(self.bounds, Bounds):
            raise InputError("Dataset.bounds must be a Bounds instance")
        X = np.asarray(self.X, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if X.ndim != 2:
            raise InputError("X must be 2-D (n, d)", shape=str(X.shape))
        if y.ndim != 1:
            raise InputError("y must be 1-D (n,)", shape=str(y.shape))
        if X.shape[0] != y.shape[0]:
            raise InputError(
                "X and y must have the same number of rows",
                n_x=X.shape[0],
                n_y=y.shape[0],
            )
        if X.shape[0] == 0:
            raise SplitError("Dataset must not be empty")
        uniq = np.unique(y)
        if uniq.size < 2:
            raise SplitError("Dataset must contain both classes", classes=str(uniq.tolist()))
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)

    @property
    def n_samples(self) -> int:
        """Number of rows."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of columns."""
        return int(self.X.shape[1])

    @property
    def is_public_proxy(self) -> bool:
        """True when this dataset is the HPO-only public proxy."""
        return bool(self.meta.get("public_proxy", False))


@dataclass(frozen=True, slots=True)
class DPFitResult:
    """Result of a differentially private fit."""

    w: Array
    views: list[Array] = field(default_factory=list)
    weights: Array = field(default_factory=lambda: np.asarray([1.0], dtype=np.float64))
    C_hat: float = 1.0
    sigma: float = 0.0
    budget_spent: Budget | None = None
    backend: str = "numpy"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        w = np.asarray(self.w, dtype=np.float64)
        if w.ndim != 1:
            raise InputError("w must be 1-D (d,)")
        object.__setattr__(self, "w", w)
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise InputError("weights must be a non empty 1-D array")
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise InputError("weights must be finite and non negative")
        total = float(weights.sum())
        if abs(total - 1.0) > 1e-6:
            raise InputError("weights must sum to 1.0", total=total)
        expected = len(self.views) if self.views else 1
        if weights.size != expected:
            raise InputError(
                "weights length must match the number of views",
                n_weights=int(weights.size),
                n_views=expected,
            )
        object.__setattr__(self, "weights", weights)

    @property
    def n_views(self) -> int:
        """Number of released views (1 when NAP is disabled)."""
        return len(self.views) if self.views else 1


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One (dataset, method, epsilon, seed) evaluation record."""

    dataset: str
    method: str
    epsilon: float
    seed: int
    accuracy: float
    auc: float
    macro_f1: float
    eps_spent: float
    utility_gap: float
    ugc: float
    uac: float
    eps_min_at_target: float | None = None

    def __post_init__(self) -> None:
        for name in ("accuracy", "auc", "macro_f1", "ugc", "uac"):
            _check_finite(name, getattr(self, name))
        _check_finite("eps_spent", self.eps_spent)
        _check_finite("utility_gap", self.utility_gap)
        if self.eps_spent < 0.0:
            raise BudgetConfigError("eps_spent must be >= 0", eps_spent=self.eps_spent)

    @property
    def utility(self) -> float:
        """Headline utility used by the DoD gates (accuracy)."""
        return float(self.accuracy)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return {
            "dataset": self.dataset,
            "method": self.method,
            "epsilon": float(self.epsilon),
            "seed": int(self.seed),
            "accuracy": float(self.accuracy),
            "auc": float(self.auc),
            "macro_f1": float(self.macro_f1),
            "eps_spent": float(self.eps_spent),
            "utility_gap": float(self.utility_gap),
            "ugc": float(self.ugc),
            "uac": float(self.uac),
            "eps_min_at_target": (
                None if self.eps_min_at_target is None else float(self.eps_min_at_target)
            ),
        }


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """A single point of the utility@epsilon sweep."""

    epsilon: float
    method: str
    dataset: str
    seed: int
    metrics: EvalResult

    def __post_init__(self) -> None:
        _check_finite("epsilon", self.epsilon)
        if abs(self.metrics.epsilon - self.epsilon) > EPS_TOL:
            raise InputError(
                "SweepPoint.epsilon must match metrics.epsilon",
                epsilon=self.epsilon,
                metrics_epsilon=self.metrics.epsilon,
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return {
            "epsilon": float(self.epsilon),
            "method": self.method,
            "dataset": self.dataset,
            "seed": int(self.seed),
            "metrics": self.metrics.as_dict(),
        }
