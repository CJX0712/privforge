"""Backend auto detection (architecture.md section 4.3).

Detects whether the ``diffprivlib`` Tier-0 backend is importable in this
environment and exposes the preferred backend name. The module is safe to
import even when ``diffprivlib`` is missing.
Author: 晨星
"""

from __future__ import annotations

try:
    import diffprivlib  # noqa: F401

    DIFFPRIVLIB_AVAILABLE = True
except Exception:
    DIFFPRIVLIB_AVAILABLE = False


def detect_backend() -> str:
    """Return the preferred backend name for this environment.

    Returns ``"diffprivlib"`` when the Tier-0 backend is importable, otherwise
    the Tier-1 ``"numpy"`` backend.
    """
    return "diffprivlib" if DIFFPRIVLIB_AVAILABLE else "numpy"
