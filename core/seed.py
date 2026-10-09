"""core/seed.py — 全局确定性 RNG 管理。

唯一入口 ``set_all(seed)``：一次性设齐 numpy / random / torch(可选) / 本系统
私有 ``Generator``。保证同 seed 两次运行产生逐位一致的结果（elapsed 除外）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RngBundle:
    """一组可复现的随机源，集中持有以避免散落的全局状态。"""

    numpy: np.random.Generator
    python: random.Random

    @classmethod
    def from_seed(cls, seed: int) -> "RngBundle":
        return cls(
            numpy=np.random.default_rng(seed),
            python=random.Random(seed),
        )


def set_all(seed: int) -> RngBundle:
    """设齐所有随机源并返回 bundle。全局副作用用于兼容直接 ``np.random`` 调用。

    本系统一律优先使用返回的 ``bundle.numpy``（``default_rng`` 而非 legacy
    ``np.random``），此处仍同步重置 legacy 全局状态，避免任何隐式调用破坏确定性。
    """
    np.random.seed(seed)
    random.seed(seed)
    try:  # torch 可选；本系统纯 numpy，无 torch 依赖
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass
    return RngBundle.from_seed(seed)


_GLOBAL: RngBundle | None = None


def get_default() -> RngBundle:
    """返回（惰性地以 0 初始化）全局 bundle。"""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = RngBundle.from_seed(0)
    return _GLOBAL
