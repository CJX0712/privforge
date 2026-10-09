"""training/nonprivate.py — 非私有 mini-batch SGD（效用上界基线）。"""

from __future__ import annotations

from training.base import BaseTrainer


class NonPrivateTrainer(BaseTrainer):
    method_name = "NonPrivate"

    def _is_private(self) -> bool:
        return False
