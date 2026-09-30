"""Runtime protocols (architecture.md section 4.3).

Cross module code is written against these protocols; concrete implementations
are injected at runtime (rule R10), which keeps the unit tests free of heavy
dependencies.
Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from privforge.core.types import Budget, Dataset, DPFitResult


@runtime_checkable
class Mechanism(Protocol):
    """A single (epsilon, delta) release primitive."""

    def release(self, value: np.ndarray, accountant: Accountant) -> np.ndarray:
        """Release a privatised version of `value` and charge `accountant`."""
        ...


@runtime_checkable
class DPModel(Protocol):
    """A differentially private estimator."""

    def fit(self, data: Dataset, budget: Budget, rng: np.random.Generator) -> DPFitResult:
        """Fit under `budget` using `rng` as the sole randomness source."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive class probability for each row of `X`."""
        ...


@runtime_checkable
class Accountant(Protocol):
    """Privacy budget ledger."""

    def spend(self, epsilon: float, delta: float) -> None:
        """Charge (epsilon, delta); must raise E400 when the budget is blown."""
        ...

    def spent(self) -> Budget:
        """The budget actually spent so far."""
        ...

    def remaining(self) -> float:
        """Unspent epsilon."""
        ...


class Backend(Protocol):
    """A numeric backend (Tier-0 diffprivlib / Tier-1 numpy).

    Not runtime_checkable on purpose: it exposes the data attribute `name`,
    and `isinstance` against a protocol with non method members raises
    TypeError. Structural checks use `hasattr` instead.
    """

    name: str

    def available(self) -> bool:
        """True when the backend can be used in this environment."""
        ...
