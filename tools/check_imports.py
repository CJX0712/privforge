"""Import every ``privforge`` submodule and report success / failure.

Used by CI and local checks to catch import-time breakage (missing dependency,
syntax error, circular import). ``check()`` returns a mapping of submodule name
to a boolean (True == imported cleanly). ``main()`` prints a table and exits
non-zero only when a *required* (already-existing) module fails to import; the
not-yet-authored ``eval`` / ``hpo`` / ``pipeline`` packages are listed but do
not fail the gate.
Author: 晨星
"""

from __future__ import annotations

import importlib
import os
import sys

# Ensure the repository root (parent of this file's directory) is importable
# even when the script is launched directly as ``python tools/check_imports.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


SUBMODULES: list[str] = [
    "privforge",
    "privforge.core",
    "privforge.core.constants",
    "privforge.core.errors",
    "privforge.core.interfaces",
    "privforge.core.rng",
    "privforge.core.types",
    "privforge.data",
    "privforge.data.loader",
    "privforge.data.preprocessing",
    "privforge.data.public_proxy",
    "privforge.data.synthetic",
    "privforge.domain",
    "privforge.domain.accountant",
    "privforge.domain.audit",
    "privforge.domain.budget",
    "privforge.domain.clipping",
    "privforge.domain.mechanisms",
    "privforge.domain.rdp",
    "privforge.domain.sensitivity",
    "privforge.training",
    "privforge.training.adaclip_budget",
    "privforge.training.base",
    "privforge.training.dp_sgd",
    "privforge.training.ensemble",
    "privforge.training.logistic",
    "privforge.training.objective_perturbation",
    "privforge.training.output_perturbation",
    "privforge.training.safeguards",
    "privforge.backends",
    "privforge.backends.detect",
    "privforge.backends.numpy_backend",
    "privforge.backends.diffprivlib_backend",
    # Not yet authored; listed for future coverage.
    "privforge.eval",
    "privforge.hpo",
    "privforge.pipeline",
]

# Modules that must import cleanly today.
REQUIRED: set = {
    m for m in SUBMODULES if m not in ("privforge.eval", "privforge.hpo", "privforge.pipeline")
}


def check() -> dict[str, bool]:
    """Import each submodule and return name -> success mapping."""
    results: dict[str, bool] = {}
    for name in SUBMODULES:
        try:
            importlib.import_module(name)
            results[name] = True
        except Exception:
            results[name] = False
    return results


def main() -> int:
    results = check()
    width = max(len(n) for n in SUBMODULES)
    failed: list[str] = []
    for name in SUBMODULES:
        ok = results.get(name, False)
        if ok:
            print(f"{name.ljust(width)}  OK")
        else:
            missing = name in ("privforge.eval", "privforge.hpo", "privforge.pipeline")
            print(f"{name.ljust(width)}  {'MISSING' if missing else 'FAIL'}")
            if not missing:
                failed.append(name)
    if failed:
        print(f"\nFAIL: {len(failed)} required module(s) failed to import: {failed}")
        return 1
    print("\nAll required modules imported cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
