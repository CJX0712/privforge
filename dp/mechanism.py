"""dp/mechanism.py — 高斯机制与敏感度。

DP-SGD 在「逐样本裁剪梯度之和」上注入高斯噪声。设裁剪界为 C、噪声倍率为 σ，
注入噪声为 N(0, (C·σ)² I)（作用于 sum，再除以 batch size 得均值上的噪声
N(0, (C·σ/B)² I)）。敏感度（对均值）为 C，故 RDP 仅由 σ 决定（见 accountant）。
"""

from __future__ import annotations

import numpy as np

from core.errors import PrivacyError


def gaussian_noise(rng: np.random.Generator, shape: tuple[int, ...], std: float) -> np.ndarray:
    """采样 N(0, std² I)。"""
    if std < 0:
        raise PrivacyError("std 必须非负")
    if std == 0.0:
        return np.zeros(shape, dtype=np.float64)
    return rng.standard_normal(shape) * std


def noise_scale_for_sum(clip_norm: float, sigma: float) -> float:
    """作用于「裁剪梯度之和」的噪声标准差 = C·σ。"""
    return clip_norm * sigma


def sensitivity_of_averaged_clipped_gradient(clip_norm: float) -> float:
    """均值（除以 B）后裁剪梯度的 L2 敏感度上界 = C。"""
    return clip_norm
