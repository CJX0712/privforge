"""PrivForge data layer: generation, loading, public proxy, preprocessing.

Depends on `core` only (rule R5). No DP mechanism may live here.
Author: 晨星
"""

from __future__ import annotations

from privforge.data import loader, preprocessing, public_proxy, synthetic

__all__ = ["loader", "preprocessing", "public_proxy", "synthetic"]
