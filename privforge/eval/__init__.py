"""Evaluation layer (SPEC.md section 6 / architecture.md section 3).

Turns a fitted model into an `EvalResult`, runs the DoD gate checks, and builds
the benchmark report (with the required `backend_fallback` field). Depends on
`core` + `training` + `domain` only (rule R4). Never imports `pipeline` or `cli`.

Author: 晨星
"""

from __future__ import annotations

from privforge.eval import curves, metrics, protocol, report

__all__ = ["curves", "metrics", "protocol", "report"]
