"""Shared base for DP logistic trainers (keeps each algorithm file focused).

Provides feature re-scaling (by the public bound R), coefficient back-mapping, and
the common predict/score metrics. Depends on core + domain + training.logistic
only.

Author: 晨星
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from privforge.training.logistic import sigmoid


class DPLogisticModel(ABC):
    """A fitted differentially private linear classifier (labels {-1, +1})."""

    def __init__(self) -> None:
        self.w: np.ndarray | None = None
        self.R: float = 1.0

    @property
    def fitted(self) -> bool:
        return self.w is not None

    def _scores(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            from privforge.core.errors import NotFittedError

            raise NotFittedError("model is not fitted")
        return np.asarray(X, dtype=np.float64) @ self.w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability for each row."""
        s = self._scores(X)
        pos = sigmoid(s)
        out = np.empty((s.shape[0], 2), dtype=np.float64)
        out[:, 0] = 1.0 - pos
        out[:, 1] = pos
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self._scores(X) >= 0.0, 1.0, -1.0)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        return float(np.mean(self.predict(X) == np.sign(y)))

    def auc(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        y01 = (y > 0).astype(int)
        return float(roc_auc_score(y01, self._scores(X)))

    def macro_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        return float(f1_score(np.sign(y), pred, average="macro", zero_division=0))

    @abstractmethod
    def fit(self, dataset, budget, rng: np.random.Generator) -> DPLogisticModel:
        """Train under `budget`; store `self.w` in the original feature space."""
        ...
