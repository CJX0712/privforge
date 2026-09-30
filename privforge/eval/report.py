"""Benchmark report + JSON serialisation (SPEC.md section 6.5).

The report carries the per-seed raw `EvalResult`s and the required
`backend_fallback` field (set when a safeguard fired and a usable model was
returned instead).  DoD gates are computed on demand via `protocol.evaluate_dod`.

Author: 晨星
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from privforge.core.types import EvalResult
from privforge.eval.protocol import DodReport, evaluate_dod


@dataclass
class BenchmarkReport:
    """One evaluation run, ready to be serialised to `benchmark.json`."""

    results: list[EvalResult] = field(default_factory=list)
    backend_fallback: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_fallback": self.backend_fallback,
            "note": self.note,
            "n_results": len(self.results),
            "results": [r.as_dict() for r in self.results],
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> BenchmarkReport:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        results = [EvalResult(**r) for r in data.get("results", [])]
        return cls(
            results=results,
            backend_fallback=data.get("backend_fallback"),
            note=data.get("note", ""),
        )

    def dod(self) -> DodReport:
        """Run the DoD gate checks against the stored results."""
        return evaluate_dod(self.results)
