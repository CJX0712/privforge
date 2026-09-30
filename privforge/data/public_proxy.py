"""Public proxy dataset for zero-privacy hyper-parameter search.

The proxy is generated from public knowledge only: the dimensionality of the
target task and a fixed proxy seed. It is physically isolated from the private
data (different seed namespace, different generator, different object), so a
bug that leaks the proxy cannot leak the private split.
Author: 晨星
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from privforge.core.constants import row_norm_bound
from privforge.core.errors import DataError, InputError
from privforge.core.rng import make_rng
from privforge.core.types import Bounds, Dataset
from privforge.data.synthetic import DATASET_IDS, public_dim

PROXY_SEED: int = 777001
PROXY_SAMPLES: int = 600
PROXY_SEP: float = 1.0


def build_proxy(
    dim: int,
    n_samples: int = PROXY_SAMPLES,
    seed: int = PROXY_SEED,
    class_sep: float = PROXY_SEP,
) -> Dataset:
    """Build a public proxy of the given dimensionality."""
    if dim <= 0:
        raise InputError("dim must be positive", dim=dim)
    rng = make_rng(seed, "public_proxy", dim, n_samples, class_sep)
    X, y01 = make_classification(
        n_samples=n_samples,
        n_features=dim,
        n_informative=max(1, dim // 2),
        n_redundant=0,
        n_classes=2,
        class_sep=class_sep,
        flip_y=0.05,
        random_state=int(rng.integers(0, 1 << 31)),
    )
    y = np.where(y01 > 0, 1.0, -1.0)
    bounds = Bounds(x_norm_max=row_norm_bound(dim), y_abs_max=1.0)
    meta = {
        "public_proxy": True,
        "source": "make_classification",
        "proxy_seed": int(seed),
        "n_informative": max(1, dim // 2),
        "class_sep": class_sep,
    }
    return Dataset(name="public_proxy", X=X, y=y, bounds=bounds, meta=meta)


def proxy_for(dataset_name: str, **kwargs: object) -> Dataset:
    """Build the proxy that matches a dataset's *public* dimensionality."""
    if dataset_name not in DATASET_IDS:
        raise DataError(f"unknown dataset: {dataset_name}", known=str(list(DATASET_IDS)))
    return build_proxy(public_dim(dataset_name), **kwargs)  # type: ignore[arg-type]


def assert_public(dataset: Dataset) -> None:
    """Guard: raise if `dataset` is not marked as the public proxy."""
    if not dataset.is_public_proxy:
        raise DataError(
            "this dataset is not the public proxy; HPO must not touch private data",
            name=dataset.name,
        )


def estimate_w_prior(dataset: Dataset) -> float:
    """Estimate the ||w|| prior used by QAC. Zero privacy cost (public data)."""
    assert_public(dataset)
    clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=1)
    clf.fit(dataset.X, dataset.y)
    w = np.asarray(clf.coef_, dtype=np.float64).ravel()
    return float(np.linalg.norm(w))
