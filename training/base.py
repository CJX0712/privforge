"""training/base.py — 共享训练循环。

三种训练器共用同一 MLP 与前向/反向；差异仅在「是否注入噪声」与「裁剪界是否自适应」。
隐私开销只由 (sigma, steps) 决定（见 dp.accountant），与裁剪界无关。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from core.seed import set_all
from core.types import PrivacyBudget, TrainResult
from dp.accountant import RDPAccountant
from dp.clipping import clip_per_sample
from dp.mechanism import gaussian_noise
from models.mlp import MLP


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.5
    sigma: float = 1.1
    clip_norm: float = 1.0
    adaptive: bool = False
    target_fraction: float = 0.5
    clip_lr: float = 0.2
    seed: int = 0
    hidden1: int = 32
    hidden2: int = 16


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator):
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


class BaseTrainer:
    """DP-SGD 家族的统一训练器；子类覆写 ``_noise_std`` / ``_maybe_adapt``。"""

    method_name = "base"

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    # 子类钩子
    def _is_private(self) -> bool:
        return True

    def _noise_std(self, clip_norm: float, batch_size: int) -> float:
        """作用于「平均裁剪梯度」的噪声标准差 = σ·C/B。"""
        return self.cfg.sigma * clip_norm / float(batch_size)

    def _maybe_adapt(self, clip_norm: float, flags: np.ndarray) -> float:
        return clip_norm

    def fit(self, X, y, X_test, y_test) -> TrainResult:
        t0 = time.perf_counter()
        rng_bundle = set_all(self.cfg.seed)
        rng = rng_bundle.numpy
        model = MLP(
            X.shape[1],
            int(np.unique(y).size),
            hidden1=self.cfg.hidden1,
            hidden2=self.cfg.hidden2,
            rng=rng_bundle,
        )
        accountant = RDPAccountant()
        clip_norm = float(self.cfg.clip_norm)
        n = X.shape[0]

        private = self._is_private()
        for epoch in range(self.cfg.epochs):
            epoch_flags = []
            for idx in _batch_indices(n, self.cfg.batch_size, rng):
                Xb, yb = X[idx], y[idx]
                B = int(Xb.shape[0])
                grads, _ = model.per_sample_gradients(Xb, yb)
                clipped, _, flags = clip_per_sample(grads, clip_norm)
                epoch_flags.append(flags)
                g_avg = clipped.mean(axis=0)
                if private and self.cfg.sigma > 0:
                    accountant.step(self.cfg.sigma)
                    noise = gaussian_noise(rng, g_avg.shape, self._noise_std(clip_norm, B))
                    g_avg = g_avg + noise
                vec = model.pack() - self.cfg.learning_rate * g_avg
                model.unpack(vec)
            if self.cfg.adaptive:
                all_flags = np.concatenate(epoch_flags)
                clip_norm = self._maybe_adapt(clip_norm, all_flags)

        privacy: PrivacyBudget | None = None
        if private and self.cfg.sigma > 0:
            privacy = accountant.get_budget(delta=1e-5)

        elapsed = time.perf_counter() - t0
        return TrainResult(
            method=self.method_name,
            test_accuracy=model.accuracy(X_test, y_test),
            test_loss=model.loss(X_test, y_test),
            privacy=privacy,
            final_clip_norm=float(clip_norm),
            n_params=model.n_params(),
            elapsed_sec=elapsed,
        )
