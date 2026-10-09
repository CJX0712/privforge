"""eval/metrics.py — 评估指标与汇总。"""

from __future__ import annotations

import numpy as np

from core.types import TrainResult


def accuracy_from_result(r: TrainResult) -> float:
    return r.test_accuracy


def excess_loss(r: TrainResult, ceiling: float) -> float:
    """相对非私有上界的效用损失（越小越好）。

    ceiling：非私有模型精度（上界）。返回 (ceiling - r.accuracy)。
    """
    return float(ceiling - r.test_accuracy)


def summarize(results: list[TrainResult]) -> dict:
    """对一组同方法多 seed 结果做 mean±std。"""
    acc = np.array([r.test_accuracy for r in results], dtype=np.float64)
    los = np.array([r.test_loss for r in results], dtype=np.float64)
    return {
        "method": results[0].method,
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=0)),
        "loss_mean": float(los.mean()),
        "loss_std": float(los.std(ddof=0)),
        "n_seeds": len(results),
        "privacy": None if results[0].privacy is None else results[0].privacy.as_dict(),
    }


def significance_mean_diff(a: list[TrainResult], b: list[TrainResult]) -> tuple[float, bool]:
    """均值差与显著性判据：|mean(a) − mean(b)| > ½(σ_a + σ_b)。"""
    am = np.mean([r.test_accuracy for r in a])
    bm = np.mean([r.test_accuracy for r in b])
    sa = np.std([r.test_accuracy for r in a], ddof=0)
    sb = np.std([r.test_accuracy for r in b], ddof=0)
    diff = float(am - bm)
    threshold = 0.5 * (float(sa) + float(sb))
    return diff, bool(abs(diff) > threshold)
