"""preprocess/scaler.py — 标准化（防数据泄漏：scaler 只在 train 上 fit）。"""

from __future__ import annotations

import numpy as np

from core.errors import DataError


class Standardizer:
    """在 train 上估计均值/方差，对 train 与 test 施加同一变换。"""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise DataError("fit 需要二维且至少 2 个样本")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_ = np.where(self.scale_ > self.eps, self.scale_, 1.0)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise DataError("Standardizer 未 fit")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def available(self) -> bool:
        return True
