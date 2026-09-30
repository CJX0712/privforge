"""Pipeline layer (SPEC.md section 2.3 / architecture.md section 3).

`AQUA-DP` is the pipeline *name*; `AdaClip-Budget` is the algorithm kernel. The
pipeline wraps the kernel with the three safeguards (S1 non-inferiority, S2 budget,
S3 audit) and emits `EvalResult`s for the DoD protocol.

Depends on `core` + `domain` + `training` + `eval` (rule R5). Never imports `cli`.

Author: 晨星
"""

from __future__ import annotations

from privforge.pipeline import flagship, pipeline

__all__ = ["flagship", "pipeline"]
