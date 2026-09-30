"""Preprocessing that only ever uses PUBLIC bounds.

Pitfall #4 of SPEC.md: standardising with the private mean/variance leaks
information that is not accounted for. Every transform here is either a pure
function of the public `Bounds` or an explicitly DP-released quantity.
Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.core.errors import BoundsError, InputError
from privforge.core.types import Bounds, Dataset


def clip_rows(X: np.ndarray, radius: float) -> np.ndarray:
    """Project each row onto the L2 ball of radius `radius`."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise InputError("X must be 2-D (n, d)", shape=str(X.shape))
    if not np.isfinite(radius) or radius <= 0.0:
        raise InputError("radius must be finite and positive", radius=radius)
    norms = np.linalg.norm(X, axis=1)
    scale = np.ones_like(norms)
    over = norms > radius
    scale[over] = radius / norms[over]
    return X * scale[:, None]


def clip_dataset(dataset: Dataset, radius: float) -> Dataset:
    """Return a copy of `dataset` whose rows are clipped to `radius`."""
    if dataset.bounds is None:
        raise BoundsError("Dataset.bounds is mandatory")
    meta = dict(dataset.meta)
    meta["clip_radius"] = float(radius)
    from privforge.core.types import Dataset as _Dataset

    return _Dataset(
        name=dataset.name,
        X=clip_rows(dataset.X, radius),
        y=dataset.y,
        bounds=dataset.bounds,
        meta=meta,
    )


class PublicBoundsScaler:
    """Feature scaler that uses the public bound as its only statistic.

    `fit` deliberately ignores the data: it exists only to satisfy the
    sklearn-style call convention and to validate the bounds. Any attempt to
    construct it without bounds is a hard E202 failure.
    """

    def __init__(self, bounds: Bounds | None = None) -> None:
        if bounds is None:
            raise BoundsError("PublicBoundsScaler requires public bounds (E202)")
        if not isinstance(bounds, Bounds):
            raise InputError("bounds must be a Bounds instance")
        self.bounds = bounds
        self.scale_: float = float(bounds.x_norm_max)
        self.label_scale_: float = float(bounds.y_abs_max)
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PublicBoundsScaler:
        """No-op fit: no private statistic is ever computed."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise InputError("X must be 2-D (n, d)", shape=str(X.shape))
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Divide by the public row-norm bound. Deterministic, data blind."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise InputError("X must be 2-D (n, d)", shape=str(X.shape))
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise InputError(
                "X has a different number of features than fit",
                expected=self.n_features_in_,
                got=int(X.shape[1]),
            )
        return X / self.scale_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """fit then transform."""
        return self.fit(X, y).transform(X)

    def transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Scale labels by the public |y| bound."""
        return np.asarray(y, dtype=np.float64) / self.label_scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Multiply back by the public bound."""
        return np.asarray(X, dtype=np.float64) * self.scale_


def scale_dataset(dataset: Dataset) -> Dataset:
    """Return a copy of `dataset` standardised by its public bounds."""
    if dataset.bounds is None:
        raise BoundsError("Dataset.bounds is mandatory")
    scaler = PublicBoundsScaler(dataset.bounds).fit(dataset.X, dataset.y)
    meta = dict(dataset.meta)
    meta["scaled_by"] = "public_bounds"
    from privforge.core.types import Dataset as _Dataset

    return _Dataset(
        name=dataset.name,
        X=scaler.transform(dataset.X),
        y=dataset.y,
        bounds=dataset.bounds,
        meta=meta,
    )


def assert_within_bounds(dataset: Dataset, tol: float = 1e-9) -> None:
    """Assert every row satisfies the declared public bound (invariant I3)."""
    if dataset.bounds is None:
        raise BoundsError("Dataset.bounds is mandatory")
    norms = np.linalg.norm(dataset.X, axis=1)
    if float(norms.max()) > float(dataset.bounds.x_norm_max) + tol:
        raise BoundsError(
            "a row exceeds the declared public bound",
            max_norm=float(norms.max()),
            bound=float(dataset.bounds.x_norm_max),
        )
