"""Dataset construction and the single canonical train/test split.

`train_test_split` is the only place in the codebase allowed to cut a
dataset, so every method in a benchmark sees byte identical folds.
Author: 晨星
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from privforge.core.constants import TEST_SIZE, row_norm_bound
from privforge.core.errors import BoundsError, DataError, InputError, SplitError
from privforge.core.rng import make_rng, shuffle_indices
from privforge.core.types import Bounds, Dataset


def from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    name: str = "custom",
    bounds: Bounds | None = None,
    **meta: object,
) -> Dataset:
    """Wrap raw arrays into a `Dataset`, deriving public bounds if needed."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise InputError("X must be (n, d) and y must be (n,)", x=str(X.shape), y=str(y.shape))
    if bounds is None:
        # The only legal default is the published dimension-only prior.
        bounds = Bounds(x_norm_max=row_norm_bound(X.shape[1]), y_abs_max=float(np.max(np.abs(y))))
    info = dict(meta)
    info.setdefault("source", "from_arrays")
    return Dataset(name=name, X=X, y=y, bounds=bounds, meta=info)


def load_csv(
    path: str | Path,
    label_col: int | str = -1,
    name: str | None = None,
    bounds: Bounds | None = None,
    has_header: bool = True,
) -> Dataset:
    """Load a CSV into a `Dataset`. Encoding is always explicit (Windows)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DataError(f"csv not found: {file_path}")
    with file_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        raise DataError(f"csv is empty: {file_path}")
    header = rows[0] if has_header else None
    body = rows[1:] if has_header else rows
    if not body:
        raise DataError(f"csv has no data rows: {file_path}")

    if isinstance(label_col, str):
        if header is None:
            raise DataError("a string label_col requires a header row")
        if label_col not in header:
            raise DataError(f"label column not found: {label_col}", header=str(header))
        col_idx = header.index(label_col)
        feature_names = [h for i, h in enumerate(header) if i != col_idx]
    else:
        n_cols = len(body[0])
        col_idx = label_col if label_col >= 0 else n_cols + label_col
        if not 0 <= col_idx < n_cols:
            raise DataError(f"label_col out of range: {label_col}")
        feature_names = [f"f{i}" for i in range(n_cols) if i != col_idx]

    try:
        table = np.asarray([[float(v) for v in row] for row in body], dtype=np.float64)
    except ValueError as exc:
        raise DataError(f"csv contains non numeric cells: {exc}") from exc

    y_raw = table[:, col_idx]
    X = np.delete(table, col_idx, axis=1)
    uniq = np.unique(y_raw)
    if uniq.size != 2:
        raise DataError("csv label column must have exactly 2 classes", classes=str(uniq.tolist()))
    y = np.where(y_raw == uniq[1], 1.0, -1.0)
    return from_arrays(
        X,
        y,
        name=name or file_path.stem,
        bounds=bounds,
        source="csv",
        feature_names=feature_names,
    )


def split_indices(n: int, seed: int, test_size: float = TEST_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 70/30 index split."""
    if n <= 1:
        raise SplitError("cannot split fewer than 2 rows", n=n)
    if not 0.0 < test_size < 1.0:
        raise SplitError("test_size must satisfy 0 < test_size < 1", test_size=test_size)
    n_test = round(n * test_size)
    n_test = min(max(n_test, 1), n - 1)
    perm = shuffle_indices(make_rng(seed, "split"), n)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def train_test_split(
    dataset: Dataset, seed: int, test_size: float = TEST_SIZE
) -> tuple[Dataset, Dataset]:
    """Split into (train, test), reusing the parent's public bounds."""
    if dataset.bounds is None:
        raise BoundsError("Dataset.bounds is mandatory (missing bounds leaks privacy)")
    train_idx, test_idx = split_indices(dataset.n_samples, seed, test_size)
    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
    for fold_name, yy in (("train", y_train), ("test", y_test)):
        if yy.size == 0:
            raise SplitError(f"{fold_name} fold is empty", fold=fold_name)
        if np.unique(yy).size < 2:
            raise SplitError(f"{fold_name} fold misses a class", fold=fold_name)
    base_meta = dict(dataset.meta)
    train_meta = {**base_meta, "split_seed": int(seed), "fold": "train", "n": int(y_train.size)}
    test_meta = {**base_meta, "split_seed": int(seed), "fold": "test", "n": int(y_test.size)}
    train = Dataset(name=dataset.name, X=X_train, y=y_train, bounds=dataset.bounds, meta=train_meta)
    test = Dataset(name=dataset.name, X=X_test, y=y_test, bounds=dataset.bounds, meta=test_meta)
    return train, test
