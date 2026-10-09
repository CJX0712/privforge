"""core/types.py — 共享数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PrivacyBudget:
    """一次训练累计的隐私开销。

    ``epsilon`` 由 RDP 矩会计转换得到；``delta`` 为目标失败概率；
    ``sigma`` 为噪声倍率；``steps`` 为组成步数。
    """

    epsilon: float
    delta: float
    sigma: float
    steps: int

    def as_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sigma": self.sigma,
            "steps": self.steps,
        }


@dataclass(frozen=True)
class TrainResult:
    """单次训练结果。"""

    method: str
    test_accuracy: float
    test_loss: float
    privacy: Optional[PrivacyBudget]
    final_clip_norm: float
    n_params: int
    elapsed_sec: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "test_accuracy": self.test_accuracy,
            "test_loss": self.test_loss,
            "privacy": None if self.privacy is None else self.privacy.as_dict(),
            "final_clip_norm": self.final_clip_norm,
            "n_params": self.n_params,
            "elapsed_sec": self.elapsed_sec,
        }


@dataclass
class BenchmarkReport:
    """多方法、多 seed 的基准报告。"""

    per_seed: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    ablation: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "per_seed": self.per_seed,
            "summary": self.summary,
            "failures": self.failures,
            "ablation": self.ablation,
        }


@dataclass(frozen=True)
class Dataset:
    """简单数据集容器（已切分为 train/val/test）。"""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int

    def n_train(self) -> int:
        return int(self.X_train.shape[0])
