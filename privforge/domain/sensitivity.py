"""Sensitivity bounds for the losses used (DP_MATH_SPEC.md section 3 / 4).

@adjacency: add-remove-one

The features are always row-clipped to the public bound ``R = bounds.x_norm_max``
and (in the trainer) re-scaled by ``R`` so the working norm is <= 1. For a
logistic loss with labels in {-1, +1} the per-example gradient satisfies

    || grad_i ||_2 = |y_i| * ||x_i|| * sigma(-y_i x_i^T theta) <= R,

so the effective L2 sensitivity of one clipped gradient is 1 in the re-scaled
space (and ``R`` before re-scaling). All bounds live in this one module so they
are never scattered or silently changed.

Author: 晨星
"""

from __future__ import annotations

import math

from privforge.core.types import Bounds


def grad_l2_bound(bounds: Bounds) -> float:
    """Per-example gradient L2 bound for logistic loss, pre re-scaling = ``R``."""
    return float(bounds.x_norm_max)


def rescaled_grad_l2_bound() -> float:
    """After re-scaling features by ``R`` the per-example gradient norm <= 1."""
    return 1.0


def default_clip_norm(dim: int) -> float:
    """Baseline fixed clip norm used by B2/B3/B4 (SPEC.md section 4): C = sqrt(d)."""
    return math.sqrt(float(dim))


def output_perturbation_sensitivity(R: float, lam: float, n: int) -> float:
    """L2 sensitivity of the ERM solution under add/remove-one (sec 3.1).

    Delta_2 = 2 L / (lambda n), with L = R the per-example gradient bound.
    """
    return 2.0 * R / (lam * n)


def output_perturbation_l1_sensitivity(R: float, lam: float, n: int, dim: int) -> float:
    """L1 sensitivity used by the Laplace mechanism: sqrt(d) * Delta_2."""
    return math.sqrt(float(dim)) * output_perturbation_sensitivity(R, lam, n)


def objective_perturbation_c(R: float, lam: float, epsilon: float, n: int) -> float:
    """K-norm mechanism constant c = n*lambda*epsilon / (2 L) (sec 3.2)."""
    return n * lam * epsilon / (2.0 * R)
