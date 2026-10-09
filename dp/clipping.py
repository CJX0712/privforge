"""dp/clipping.py — 逐样本梯度裁剪（DP-SGD 的敏感度界）。

裁剪算子对非负标量齐次：clip(k·g, C) = k·clip(g, C)（k≥0），且
||clip(g_i, C)||₂ ≤ C。这是两条可验证不变量，也是敏感度上界保证隐私的关键。
"""

from __future__ import annotations

import numpy as np

from core.errors import PrivacyError


def clip_per_sample(
    grads: np.ndarray, clip_norm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对逐样本梯度做 L2 裁剪。

    参数
    ----
    grads : (B, P) 每个样本一条展平梯度。
    clip_norm : C>0。

    返回
    ----
    clipped   : (B, P) 裁剪后梯度，每行范数 ≤ C。
    norms     : (B,)   裁剪前范数。
    flags     : (B,)   bool，该样本是否被裁剪（norm > C）。
    """
    if clip_norm <= 0:
        raise PrivacyError("clip_norm 必须为正")
    grads = np.asarray(grads, dtype=np.float64)
    norms = np.linalg.norm(grads, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    factors = np.minimum(1.0, clip_norm / safe)
    clipped = grads * factors[:, None]
    flags = norms > clip_norm
    return clipped, norms, flags


def max_clipped_norm(clipped: np.ndarray) -> float:
    """裁剪后梯度的最大 L2 范数（应 ≤ clip_norm）。"""
    if clipped.size == 0:
        return 0.0
    return float(np.linalg.norm(clipped, axis=1).max())


def clip_fraction(flags: np.ndarray) -> float:
    """被裁剪样本占比。"""
    if flags.size == 0:
        return 0.0
    return float(np.mean(flags.astype(np.float64)))
