"""Public constants and closed form mechanism formulas.

Everything here is pure mathematics / policy. No I/O, no upper layer
imports (rule R1). Author: 晨星

The formulas in this module are the *analytic* (epsilon, delta) Gaussian
and Laplace calibrations. `domain/sensitivity.py` computes the sensitivity
of each loss and feeds it into these helpers, so the calibration lives in
exactly one place.
"""

from __future__ import annotations

import math

# --- protocol policy -------------------------------------------------------
DEFAULT_EPSILON: float = 1.0
DEFAULT_DELTA: float = 1e-5
N_SEEDS: int = 5
TEST_SIZE: float = 0.3

# --- utility@epsilon curve (SPEC.md section 6) -----------------------------
EPSILON_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
TARGET_FRAC: float = 0.9  # eps_min_at_target solves U(eps) >= 0.9 * U_nodp

# --- budget policy ---------------------------------------------------------
BUDGET_FRACTIONS: tuple[float, ...] = (0.05, 0.92, 0.03)  # clip / train / select
EPS_TOL: float = 1e-9  # invariant I1 tolerance
AUDIT_TOL: float = 0.05  # invariant / safeguard S3 tolerance

# --- UGC guard rail (SPEC.md section 1.1) ----------------------------------
UGC_MIN_DENOM: float = 0.005

# --- dataset protocol (SPEC.md section 5) ----------------------------------
DATASET_SEED: int = 20250930
SPLIT_SEED_BASE: int = 4001
NOISE_SEED_BASE: int = 9001

# --- public row-norm prior -------------------------------------------------
# The public bound R is derived from the dimensionality only, never from the
# data. R = ROW_NORM_SAFETY * sqrt(d) is a published, data independent prior.
ROW_NORM_SAFETY: float = 4.0
DIGITS_PIXEL_MAX: float = 16.0

# --- seed derivation -------------------------------------------------------
SEED_MASK: int = (1 << 63) - 1


def row_norm_bound(dim: int, safety: float = ROW_NORM_SAFETY) -> float:
    """Public L2 bound on a feature row: R(d) = safety * sqrt(d)."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    return safety * math.sqrt(float(dim))


def k_gm(delta: float) -> float:
    """Gaussian mechanism constant K_GM = sqrt(2 * ln(1.25 / delta))."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must satisfy 0 < delta < 1")
    return math.sqrt(2.0 * math.log(1.25 / delta))


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Analytic Gaussian sigma for an L2 sensitivity under (eps, delta)-DP."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if sensitivity < 0.0:
        raise ValueError("sensitivity must be non negative")
    return sensitivity * k_gm(delta) / epsilon


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Laplace scale b for an L1 sensitivity under epsilon-DP."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if sensitivity < 0.0:
        raise ValueError("sensitivity must be non negative")
    return sensitivity / epsilon
