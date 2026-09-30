"""DP mechanisms (DP_MATH_SPEC.md section 1).

@adjacency: add-remove-one

Mechanisms implemented here:
  * Laplace           -- pure epsilon-DP (sec 1.2)
  * Gaussian (BW)     -- (epsilon, delta)-DP via Balle-Wang analytic calibration
                         (sec 1.3). The classic formula is FORBIDDEN (INV-12/I12).
  * Exponential / Report-Noisy-Max -- pure epsilon-DP discrete selection (sec 1.4)
  * SVT               -- Sparse Vector Technique for threshold release

All releases operate in *normalized* space: callers pass the sensitivity and
the mechanism returns the privatised value. Noise is a function of ``mu``
(normalized scale) only -- never ``mu * C_t``.

Author: 晨星
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def calibrate_gaussian_bw(epsilon: float, delta: float) -> float:
    """Balle-Wang analytic Gaussian calibration (sec 1.3).

    Returns the normalized scale ``mu`` (sigma_abs = mu * sensitivity) such that
    the Gaussian mechanism satisfies (epsilon, delta)-DP. Solves the
    characteristic equation by bisection on a strictly decreasing function.

    This is the golden-reference calibrated value; it matches
    ``diffprivlib.mechanisms.GaussianAnalytic._scale`` to 1e-13 (INV-9).
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must satisfy 0 < delta < 1")

    def g(mu: float) -> float:
        return (
            ndtr(1.0 / (2.0 * mu) - epsilon * mu)
            - math.exp(epsilon) * ndtr(-1.0 / (2.0 * mu) - epsilon * mu)
            - delta
        )

    hi = 1.0
    while g(hi) > 0.0:
        hi *= 2.0
    return float(brentq(g, 1e-9, hi, xtol=1e-14, rtol=8.9e-16))


def gaussian_sigma(mu: float, sensitivity: float) -> float:
    """Absolute noise std = mu * sensitivity (the ONLY place sensitivity enters)."""
    return float(mu) * float(sensitivity)


class Laplace:
    """Laplace mechanism, pure epsilon-DP (sec 1.2)."""

    def __init__(self, sensitivity: float, epsilon: float, rng: np.random.Generator) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if sensitivity < 0.0:
            raise ValueError("sensitivity must be non negative")
        self.sensitivity = float(sensitivity)
        self.epsilon = float(epsilon)
        self.rng = rng

    @property
    def scale(self) -> float:
        """b = Delta1 / epsilon."""
        return self.sensitivity / self.epsilon

    def release(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        return value + self.rng.laplace(0.0, self.scale, size=value.shape)

    def density_ratio_bound(self, value: np.ndarray) -> float:
        """Worst-case log density ratio max_x log p_D(x)/p_D'(x) == epsilon (INV-8)."""
        return self.epsilon


class GaussianBW:
    """Gaussian mechanism with Balle-Wang analytic calibration (sec 1.3)."""

    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
        rng: np.random.Generator,
    ) -> None:
        if sensitivity < 0.0:
            raise ValueError("sensitivity must be non negative")
        self.sensitivity = float(sensitivity)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.rng = rng
        self.mu = calibrate_gaussian_bw(epsilon, delta)

    @property
    def sigma(self) -> float:
        """Absolute std = mu * sensitivity."""
        return gaussian_sigma(self.mu, self.sensitivity)

    def release(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        return value + self.rng.normal(0.0, self.sigma, size=value.shape)


def report_noisy_max(
    scores: Sequence[float],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Report-Noisy-Max / Exponential selection (sec 1.4).

    Given per-candidate scores ``s_k`` with global sensitivity ``Delta s = 1``,
    returns an index drawn with probability proportional to
    ``exp(epsilon * s_k / 2)`` -- pure epsilon-DP. Subtraction of the max score
    before exp prevents overflow (pitfall #16); all math is float64 (pitfall #14).
    """
    s = np.asarray(list(scores), dtype=np.float64)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    s = s - s.max()
    w = np.exp(epsilon * s / 2.0)
    p = w / w.sum()
    candidates = list(range(s.shape[0]))
    return int(rng.choice(candidates, p=p))


class ReportNoisyMax:
    """Stateful wrapper around :func:`report_noisy_max`."""

    def __init__(self, epsilon: float, rng: np.random.Generator) -> None:
        self.epsilon = float(epsilon)
        self.rng = rng

    def select(self, scores: Sequence[float]) -> int:
        return report_noisy_max(scores, self.epsilon, self.rng)


def sparse_vector_threshold(
    queries: Sequence[np.ndarray],
    noisy_answers: Sequence[float],
    threshold: float,
    epsilon: float,
    rng: np.random.Generator,
) -> list[bool]:
    """Sparse Vector Technique (SVT): release which answers beat a DP threshold.

    ``noisy_answers`` are assumed already privatised (caller adds Laplace noise
    of scale 2*Delta/epsilon); the threshold itself is released with Laplace
    noise of scale Delta/epsilon. Returns a boolean flag per query.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    delta = 1.0  # unit sensitivity of each answer
    noisy_threshold = threshold + rng.laplace(0.0, delta / epsilon)
    flags: list[bool] = []
    for ans in noisy_answers:
        flags.append(float(ans) >= float(noisy_threshold))
    return flags
