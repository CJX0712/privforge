"""data/loader.py — 数据源接口实现（合成数据 + 防泄漏约定）。"""

from __future__ import annotations

from core.interfaces import DataSource
from core.types import Dataset
from data.synthetic import make_classification


class SyntheticDataSource(DataSource):
    """确定性合成数据源。"""

    def __init__(
        self,
        seed: int = 0,
        n_samples: int = 4000,
        n_features: int = 20,
        separation: float = 2.2,
        test_fraction: float = 0.3,
        outlier_frac: float = 0.0,
        outlier_scale: float = 8.0,
    ) -> None:
        self.seed = seed
        self.n_samples = n_samples
        self.n_features = n_features
        self.separation = separation
        self.test_fraction = test_fraction
        self.outlier_frac = outlier_frac
        self.outlier_scale = outlier_scale

    def get(self) -> Dataset:
        return make_classification(
            seed=self.seed,
            n_samples=self.n_samples,
            n_features=self.n_features,
            separation=self.separation,
            test_fraction=self.test_fraction,
            outlier_frac=self.outlier_frac,
            outlier_scale=self.outlier_scale,
        )
