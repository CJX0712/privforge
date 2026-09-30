"""Difficulty ladder D1..D6 (SPEC.md section 3).

Every dataset carries a **public** `Bounds` object: the row-norm bound is a
published function of the dimensionality only (`R = 4 * sqrt(d)`), never an
estimate taken from the data. Rows are clipped to that public bound at
generation time so the declared bound is always true.
Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits, make_classification

from privforge.core.constants import (
    DATASET_SEED,
    DIGITS_PIXEL_MAX,
    ROW_NORM_SAFETY,
    row_norm_bound,
)
from privforge.core.errors import DataError
from privforge.core.rng import make_rng
from privforge.core.types import Bounds, Dataset

POSITIVE_DIGITS: tuple[int, ...] = (0, 1, 8)
DIGITS_SUBSAMPLE: int = 1200
PARETO_ALPHA: float = 2.0


@dataclass(frozen=True, slots=True)
class _Spec:
    """Generation recipe. Purely public information."""

    name: str
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int
    class_sep: float
    flip_y: float
    weights: tuple[float, float] | None = None
    heavy_tail: bool = False
    source: str = "make_classification"
    note: str = ""


SPECS: dict[str, _Spec] = {
    "easy_separable": _Spec(
        name="easy_separable",
        n_samples=1200,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        class_sep=1.5,
        flip_y=0.01,
        note="D1: almost lossless under DP; tiny oracle gap stresses DoD-2",
    ),
    "medium": _Spec(
        name="medium",
        n_samples=1200,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        class_sep=1.0,
        flip_y=0.05,
        note="D2: main battleground",
    ),
    "hard_lowsep": _Spec(
        name="hard_lowsep",
        n_samples=800,
        n_features=40,
        n_informative=8,
        n_redundant=4,
        class_sep=0.6,
        flip_y=0.10,
        note="D3: high dimension, low SNR; QAC failure zone, S1 must fire",
    ),
    "heavy_tail": _Spec(
        name="heavy_tail",
        n_samples=1200,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        class_sep=1.0,
        flip_y=0.05,
        heavy_tail=True,
        note="D4: D2 rows rescaled by a Pareto(2) factor; QAC home turf",
    ),
    "imbalanced": _Spec(
        name="imbalanced",
        n_samples=1200,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        class_sep=1.0,
        flip_y=0.05,
        weights=(0.9, 0.1),
        note="D5: 9:1 class ratio",
    ),
    "digits_subsample": _Spec(
        name="digits_subsample",
        n_samples=DIGITS_SUBSAMPLE,
        n_features=64,
        n_informative=0,
        n_redundant=0,
        class_sep=0.0,
        flip_y=0.0,
        source="load_digits",
        note="D6: real pixels, positive class = digits {0, 1, 8}",
    ),
}

DATASET_IDS: tuple[str, ...] = tuple(SPECS)


def _clip_rows(X: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1)
    scale = np.ones_like(norms)
    over = norms > radius
    scale[over] = radius / norms[over]
    return X * scale[:, None]


def _public_bounds(spec: _Spec) -> Bounds:
    if spec.source == "load_digits":
        # A pixel row lives in [0, 1]^64, so ||x|| <= sqrt(64) exactly.
        return Bounds(x_norm_max=row_norm_bound(spec.n_features, safety=1.0), y_abs_max=1.0)
    return Bounds(x_norm_max=row_norm_bound(spec.n_features), y_abs_max=1.0)


def _generate_make_classification(spec: _Spec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = make_rng(seed, "dataset", spec.name)
    X, y01 = make_classification(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        n_informative=spec.n_informative,
        n_redundant=spec.n_redundant,
        n_classes=2,
        class_sep=spec.class_sep,
        flip_y=spec.flip_y,
        weights=list(spec.weights) if spec.weights is not None else None,
        random_state=int(rng.integers(0, 1 << 31)),
    )
    y = np.where(y01 > 0, 1.0, -1.0)
    if spec.heavy_tail:
        nrm_before = np.linalg.norm(X, axis=1)
        p = 1.0 + rng.pareto(PARETO_ALPHA, size=X.shape[0])
        X = X * p[:, None]
        nrm_after = np.linalg.norm(X, axis=1)
        # Renormalise on the median so the bulk keeps its scale and only the
        # tail grows. This is what makes a fixed C destructive.
        X = X * (float(np.median(nrm_before)) / float(np.median(nrm_after)))
    return X, y


def _generate_digits(spec: _Spec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = make_rng(seed, "dataset", spec.name)
    digits = load_digits()
    idx = rng.choice(digits.data.shape[0], size=spec.n_samples, replace=False)
    X = np.asarray(digits.data[idx], dtype=np.float64) / DIGITS_PIXEL_MAX
    labels = np.asarray(digits.target[idx], dtype=np.int64)
    y = np.where(np.isin(labels, POSITIVE_DIGITS), 1.0, -1.0)
    return X, y


def generate_dataset(name: str, seed: int | None = None) -> Dataset:
    """Generate one of D1..D6. The generation seed is fixed by protocol."""
    spec = SPECS.get(name)
    if spec is None:
        raise DataError(f"unknown dataset: {name}", known=str(list(DATASET_IDS)))
    use_seed = DATASET_SEED if seed is None else int(seed)
    if spec.source == "load_digits":
        X, y = _generate_digits(spec, use_seed)
    else:
        X, y = _generate_make_classification(spec, use_seed)

    bounds = _public_bounds(spec)
    raw_norms = np.linalg.norm(X, axis=1)
    X = _clip_rows(np.asarray(X, dtype=np.float64), bounds.x_norm_max)
    clipped = float(np.mean(raw_norms > bounds.x_norm_max))
    meta = {
        "spec": spec.name,
        "seed": use_seed,
        "source": spec.source,
        "n_informative": spec.n_informative,
        "n_redundant": spec.n_redundant,
        "class_sep": spec.class_sep,
        "flip_y": spec.flip_y,
        "heavy_tail": spec.heavy_tail,
        "row_norm_safety": ROW_NORM_SAFETY if spec.source != "load_digits" else 1.0,
        "frac_clipped_at_public_bound": clipped,
        "positive_rate": float(np.mean(y > 0)),
        "note": spec.note,
    }
    return Dataset(name=spec.name, X=X, y=y, bounds=bounds, meta=meta)


def generate_all(seed: int | None = None) -> dict[str, Dataset]:
    """Generate the whole D1..D6 ladder with the same fixed seed."""
    return {name: generate_dataset(name, seed) for name in DATASET_IDS}


def public_dim(name: str) -> int:
    """Public dimensionality of a dataset spec (used to build the proxy)."""
    spec = SPECS.get(name)
    if spec is None:
        raise DataError(f"unknown dataset: {name}", known=str(list(DATASET_IDS)))
    return int(spec.n_features)
