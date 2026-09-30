"""Tier-1 numpy numeric backend (architecture.md section 4.3).

A :class:`~privforge.core.interfaces.Backend` implementation backed by a numpy
``Generator``. All randomness flows through ``privforge.core.rng.make_rng`` so
streams stay reproducible (rule R5). The privacy math here is just sampling
from the already calibrated scales -- no ``mu * C_t`` coupling is introduced.
Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.core.rng import make_rng


class NumpyBackend:
    """Numeric backend implemented with numpy's ``Generator``.

    Conforms to the :class:`~privforge.core.interfaces.Backend` protocol.
    """

    name = "numpy"

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else make_rng(0)

    def available(self) -> bool:
        """The numpy backend is always available."""
        return True

    def gaussian(
        self, mu: float, size: int | tuple[int, ...], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw N(0, mu^2 I) samples of shape ``size``."""
        g = rng if rng is not None else self.rng
        return g.normal(0.0, float(mu), size=size)

    def laplace(
        self, scale: float, size: int | tuple[int, ...], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw Laplace(0, scale) samples of shape ``size``."""
        g = rng if rng is not None else self.rng
        return g.laplace(0.0, float(scale), size=size)

    def report_noisy_max(
        self, scores: object, epsilon: float, rng: np.random.Generator | None = None
    ) -> int:
        """Report-Noisy-Max: add Laplace(1/epsilon) noise, return argmax index."""
        g = rng if rng is not None else self.rng
        s = np.asarray(list(scores), dtype=np.float64)
        noise = g.laplace(0.0, 1.0 / float(epsilon), size=s.shape)
        return int(np.argmax(s + noise))

    def clip(self, grads: np.ndarray, C: float) -> np.ndarray:
        """Per-row clip of ``grads`` so each row has norm at most ``C``."""
        arr = np.asarray(grads, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe = np.maximum(norms, 1e-12)
        scales = np.minimum(1.0, float(C) / safe)
        return arr * scales
