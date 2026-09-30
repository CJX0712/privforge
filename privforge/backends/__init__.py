"""Numeric backends for PrivForge (architecture.md section 4.3).

Re-exports the backend detection helpers and the concrete backend classes.
Author: 晨星
"""

from __future__ import annotations

from privforge.backends.detect import DIFFPRIVLIB_AVAILABLE, detect_backend
from privforge.backends.diffprivlib_backend import DiffprivlibBackend, get_backend
from privforge.backends.numpy_backend import NumpyBackend

__all__ = [
    "DIFFPRIVLIB_AVAILABLE",
    "DiffprivlibBackend",
    "NumpyBackend",
    "detect_backend",
    "get_backend",
]
