"""data/synthetic.py — 确定性合成分类数据（可复现 DGP + 难度旋钮）。

DGP：两类各半，特征均值沿单位方向偏移 ±separation/√d，各维独立 N(0,1) 噪声。
``separation`` 控制可分性：太小→全崩、太大→全满分，都无区分度；需落入甜点区
（NonPrivate 明显优于随机、固定裁剪明显低于 NonPrivate 上界）。
"""

from __future__ import annotations

import numpy as np

from core.seed import RngBundle
from core.types import Dataset


def make_classification(
    seed: int = 0,
    n_samples: int = 4000,
    n_features: int = 20,
    separation: float = 2.2,
    test_fraction: float = 0.3,
    outlier_frac: float = 0.0,
    outlier_scale: float = 8.0,
) -> Dataset:
    """生成二分类合成数据集，按 seed 确定性切分 train/test。

    ``outlier_frac>0`` 时按比例注入重尾离群样本（特征尺度放大 ``outlier_scale``
    倍），使逐样本梯度呈重尾分布——固定裁剪界明显次优、自适应裁剪显著占优的典型场景。
    """
    rng = RngBundle.from_seed(seed).numpy
    d = int(n_features)
    n = int(n_samples)
    half = n // 2
    direction = np.ones(d) / np.sqrt(d)
    X = np.empty((n, d), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    # 类 0
    X[:half] = -separation * direction + rng.standard_normal((half, d))
    y[:half] = 0
    # 类 1
    X[half:] = separation * direction + rng.standard_normal((n - half, d))
    y[half:] = 1
    # 重尾离群样本
    if outlier_frac > 0.0:
        n_out = int(round(n * outlier_frac))
        oi = rng.permutation(n)[:n_out]
        signs = rng.choice([-1.0, 1.0], size=n_out)
        X[oi] = signs[:, None] * outlier_scale * direction + rng.standard_normal((n_out, d))
    # 洗牌（确定性）
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    n_test = int(round(n * test_fraction))
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]
    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_features=d,
        n_classes=2,
    )
