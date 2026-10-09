"""core/interfaces.py — 接口契约（Protocol）。

调用关系单向无环：``cli -> pipeline -> {data, preprocess, dp, models, training,
eval} -> core``。模块间只经以下抽象契约通信，便于替换后端与离线兜底。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from core.types import Dataset, PrivacyBudget, TrainResult


@runtime_checkable
class DataSource(Protocol):
    """生成/载入数据集。"""

    def get(self) -> Dataset: ...


@runtime_checkable
class PrivacyAccountant(Protocol):
    """隐私会计：按步累加 RDP，并转换到 (epsilon, delta)。"""

    def step(self, sigma: float) -> None:
        """记录一步（噪声倍率 sigma）的隐私消耗。"""

    def get_budget(self, delta: float) -> PrivacyBudget:
        """返回累计隐私预算。"""

    def available(self) -> bool:
        """后端是否可用（离线兜底用）。"""


@runtime_checkable
class Trainer(Protocol):
    """可替换的训练器：非私有 / 固定裁剪 / 自适应裁剪。"""

    def fit(
        self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> TrainResult: ...


@dataclass
class ModelProtocolCarrier:
    """占位：满足 Trainer 的对象可由任意后端提供。"""

    backend: str = "numpy"
