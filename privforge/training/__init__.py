"""Training algorithms (architecture.md section 3 / DP_MATH_SPEC.md section 3-4).

This layer depends on `core` and `domain` only (rule R3). It never imports
`pipeline` or `cli`. All trainers share the logistic loss + Balle-Wang / RDP
calibration primitives from `domain`.

Author: 晨星
"""

from __future__ import annotations

from privforge.training import (
    adaclip_budget,
    dp_sgd,
    ensemble,
    logistic,
    objective_perturbation,
    output_perturbation,
    safeguards,
)
from privforge.training.adaclip_budget import AdaClipBudget, AdaClipConfig
from privforge.training.dp_sgd import DPSGDBaseline
from privforge.training.objective_perturbation import ObjectivePerturbation
from privforge.training.output_perturbation import OutputPerturbation

__all__ = [
    "AdaClipBudget",
    "AdaClipConfig",
    "DPSGDBaseline",
    "ObjectivePerturbation",
    "OutputPerturbation",
    "adaclip_budget",
    "dp_sgd",
    "ensemble",
    "logistic",
    "objective_perturbation",
    "output_perturbation",
    "safeguards",
]
