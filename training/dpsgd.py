"""training/dpsgd.py — 固定裁剪界 DP-SGD（公认 SOTA 基线，Abadi et al. 2016）。"""

from __future__ import annotations

from training.base import BaseTrainer


class FixedClipDPSGDTrainer(BaseTrainer):
    method_name = "FixedClip-DP-SGD"

    def _is_private(self) -> bool:
        return True
