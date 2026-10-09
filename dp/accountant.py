"""dp/accountant.py — Rényi Differential Privacy (RDP) 矩会计。

参考：
  - Mironov (2017), "Renyi Differential Privacy", IEEE S&P.
  - Abadi et al. (2016), "Deep Learning with Differential Privacy" (DP-SGD).

高斯机制（敏感度 1、噪声倍率 sigma）的 RDP 为
    rdp(alpha) = alpha / (2 sigma^2)
与裁剪界 C 无关——这是 DP-SGD 的核心性质：隐私开销只取决于 (sigma, T)，
裁剪界 C 只影响效用（偏差/方差权衡）。本文件实现严格的矩会计与 RDP→(eps,delta)
转换，并暴露可验证不变量。
"""

from __future__ import annotations

import math

import numpy as np

from core.errors import PrivacyError
from core.types import PrivacyBudget

# 用于 RDP→DP 转换的 alpha 网格（覆盖常见取值，对数+线性混合）。
_ALPHA_GRID = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 32.0, 64.0]


def gaussian_rdp(alpha: float, sigma: float) -> float:
    """高斯机制（敏感度 1、噪声倍率 sigma）的 α 阶 RDP。

    标准结果：rdp_α = α / (2 σ²)。与裁剪界无关。
    """
    if sigma <= 0:
        raise PrivacyError("sigma 必须为正")
    return alpha / (2.0 * sigma * sigma)


class RDPAccountant:
    """按步累加重力 RDP，并转换到 (epsilon, delta)。

    组成定理：T 步独立高斯机制的 RDP 线性相加，
        rdp_total(α) = Σ_t rdp_t(α) = T · α / (2 σ²)
    （每步 sigma 可不同；自适应裁剪改变 C 但不改变 sigma，故每步 RDP 不变。）
    """

    def __init__(self) -> None:
        self._steps: list[float] = []  # 每步记录 sigma
        self._total_rdp: dict[float, float] = {a: 0.0 for a in _ALPHA_GRID}

    def step(self, sigma: float) -> None:
        """记录一步（噪声倍率 sigma）。"""
        if sigma <= 0:
            raise PrivacyError("sigma 必须为正")
        self._steps.append(sigma)
        for a in _ALPHA_GRID:
            self._total_rdp[a] += gaussian_rdp(a, sigma)

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    def rdp(self, alpha: float) -> float:
        """累计 RDP（任意 alpha，按网格外推：线性于 alpha）。"""
        if alpha in self._total_rdp:
            return self._total_rdp[alpha]
        # 超出网格：rdp 对 alpha 线性（高斯机制），按斜率外推。
        lo, hi = 1.5, 64.0
        slope = (self._total_rdp[hi] - self._total_rdp[lo]) / (hi - lo)
        return self._total_rdp[lo] + slope * (alpha - lo)

    def get_epsilon(self, delta: float) -> float:
        """RDP→DP 转换：ε = min_α [ rdp(α) + log(1/δ)/(α−1) ]。"""
        if not (0.0 < delta < 1.0):
            raise PrivacyError("delta 必须落在 (0,1)")
        best = float("inf")
        for a in _ALPHA_GRID:
            eps = self._total_rdp[a] + math.log(1.0 / delta) / (a - 1.0)
            if eps < best:
                best = eps
        return best

    def get_budget(self, delta: float) -> PrivacyBudget:
        return PrivacyBudget(
            epsilon=self.get_epsilon(delta),
            delta=delta,
            sigma=float(np.mean(self._steps)) if self._steps else 0.0,
            steps=self.n_steps,
        )

    def available(self) -> bool:
        return True
