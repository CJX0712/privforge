"""training/dpfuse.py — DPFuse：自适应裁剪 DP-SGD（旗舰）。

自适应裁剪（Andrew et al. 2021, "Differentially Private Learning with Adaptive
Clipping"）：每轮依据被裁剪样本占比 f 调整裁剪界 C，使其逼近目标裁剪比例
``target_fraction``（默认 0.5）：
    C <- C · exp( clip_lr · (f − target_fraction) )
裁剪界只影响效用（偏差/方差权衡），不改变隐私开销（RDP 仅由 σ 决定），因此
DPFuse 与 FixedClip 在**相同 (ε,δ)** 下比较，预期获得更优效用。
"""

from __future__ import annotations

import numpy as np

from core.config import DEFAULT_CONFIG
from training.base import BaseTrainer


class DPFuseTrainer(BaseTrainer):
    method_name = "DPFuse"

    def _is_private(self) -> bool:
        return True

    def _maybe_adapt(self, clip_norm: float, flags: np.ndarray) -> float:
        f = float(np.mean(flags.astype(np.float64)))
        target = self.cfg.target_fraction or DEFAULT_CONFIG.target_clip_fraction
        new_c = clip_norm * np.exp(self.cfg.clip_lr * (f - target))
        # 合理界：避免数值爆炸或坍缩；相对初值 ±10×。
        lo = self.cfg.clip_norm * 0.1
        hi = self.cfg.clip_norm * 10.0
        return float(min(max(new_c, lo), hi))
