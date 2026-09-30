"""Tier-0 diffprivlib numeric backend (architecture.md section 4.3).

A :class:`~privforge.core.interfaces.Backend` implementation that delegates the
noise draws to ``diffprivlib`` mechanisms where the parameterisation matches
the required scale, and falls back to the numpy backend when ``diffprivlib`` is
unavailable or any construction fails. The module imports cleanly even when
``diffprivlib`` is absent -- the heavy import is deferred into the methods.

No privacy math is re-derived here; the Gaussian scale is recovered from the
golden ``calibrate_gaussian_bw`` calibration so the drawn noise has the
requested std ``mu`` (sensitivity 1.0).
Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.backends.detect import DIFFPRIVLIB_AVAILABLE, detect_backend
from privforge.core.rng import make_rng


class DiffprivlibBackend:
    """Numeric backend implemented with ``diffprivlib`` mechanisms.

    Conforms to the :class:`~privforge.core.interfaces.Backend` protocol. When a
    method cannot use ``diffprivlib`` it sets ``fallback_to = "numpy"`` and uses
    the numpy ``Generator`` for that draw.
    """

    name = "diffprivlib"

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else make_rng(0)
        self.fallback_to: str | None = None

    def available(self) -> bool:
        """True when ``diffprivlib`` is importable in this environment."""
        return DIFFPRIVLIB_AVAILABLE

    @staticmethod
    def _invert_mu(mu: float, delta: float) -> float:
        """Find epsilon with ``calibrate_gaussian_bw(epsilon, delta) == mu``.

        ``calibrate_gaussian_bw`` is strictly decreasing in epsilon, so a
        bisection on a fixed ``delta`` recovers the matching epsilon.
        """
        from privforge.domain.mechanisms import calibrate_gaussian_bw

        lo, hi = 1e-4, 1e3
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            try:
                val = calibrate_gaussian_bw(mid, delta)
            except Exception:
                val = float("inf")
            if val > float(mu):
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def gaussian(
        self, mu: float, size: int | tuple[int, ...], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw N(0, mu^2 I) samples, using diffprivlib's Gaussian when possible."""
        try:
            from diffprivlib.mechanisms import GaussianAnalytic

            delta = 1e-5
            eps = self._invert_mu(float(mu), delta)
            mech = GaussianAnalytic(epsilon=eps, delta=delta, sensitivity=1.0)
            out = np.empty(size, dtype=np.float64)
            flat = out.ravel()
            for i in range(flat.size):
                flat[i] = float(mech.randomise(0.0))
            return out
        except Exception:
            self.fallback_to = "numpy"
            g = rng if rng is not None else self.rng
            return g.normal(0.0, float(mu), size=size)

    def laplace(
        self, scale: float, size: int | tuple[int, ...], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw Laplace(0, scale) samples via diffprivlib's Laplace mechanism."""
        try:
            from diffprivlib.mechanisms import Laplace

            eps = 1.0 / float(scale) if float(scale) > 0.0 else 1.0
            mech = Laplace(epsilon=eps, sensitivity=1.0)
            out = np.empty(size, dtype=np.float64)
            flat = out.ravel()
            for i in range(flat.size):
                flat[i] = float(mech.randomise(0.0))
            return out
        except Exception:
            self.fallback_to = "numpy"
            g = rng if rng is not None else self.rng
            return g.laplace(0.0, float(scale), size=size)

    def report_noisy_max(
        self, scores: object, epsilon: float, rng: np.random.Generator | None = None
    ) -> int:
        """Add diffprivlib Laplace(1/epsilon) noise to scores and return argmax."""
        try:
            from diffprivlib.mechanisms import Laplace

            s = np.asarray(list(scores), dtype=np.float64)
            mech = Laplace(epsilon=float(epsilon), sensitivity=1.0)
            noise = np.array([float(mech.randomise(0.0)) for _ in range(s.size)], dtype=np.float64)
            return int(np.argmax(s + noise))
        except Exception:
            self.fallback_to = "numpy"
            g = rng if rng is not None else self.rng
            s = np.asarray(list(scores), dtype=np.float64)
            noise = g.laplace(0.0, 1.0 / float(epsilon), size=s.shape)
            return int(np.argmax(s + noise))


def get_backend(name: str | None = None):
    """Return the preferred backend.

    Returns :class:`DiffprivlibBackend` when ``detect_backend()`` is
    ``"diffprivlib"`` and ``name`` is ``None`` or ``"diffprivlib"`` (and
    diffprivlib is importable); otherwise returns :class:`NumpyBackend`.
    """
    from privforge.backends.numpy_backend import NumpyBackend

    chosen = name if name is not None else detect_backend()
    if chosen == "diffprivlib" and DIFFPRIVLIB_AVAILABLE:
        return DiffprivlibBackend()
    return NumpyBackend()
