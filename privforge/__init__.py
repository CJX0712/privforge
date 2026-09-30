"""PrivForge - a differentially private machine learning system.

Author: 晨星
License: MIT

The top level package stays deliberately thin: it exposes the package
metadata only. Sub packages (`core`, `data`, `backends`, `eval`, ...) are
imported explicitly by callers so that the dependency direction rules
R1~R10 stay observable by `tools/check_imports.py`.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"
__license__ = "MIT"

__all__ = ["__author__", "__license__", "__version__"]
