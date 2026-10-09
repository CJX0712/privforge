"""tests/test_pipeline.py — 流水线：确定性、隐私等价、效用排序、失败案例。"""

from __future__ import annotations

from pipeline.pipeline import PrivForgePipeline
from training.base import TrainConfig


def _pipe():
    cfg = TrainConfig(
        epochs=12, batch_size=256, learning_rate=0.5, sigma=1.1, clip_norm=1.0, seed=0
    )
    return PrivForgePipeline(
        methods=("NonPrivate", "FixedClip-DP-SGD", "DPFuse"),
        seeds=(0, 1, 2),
        cfg=cfg,
        n_samples=2400,
        n_features=16,
        separation=2.0,
    )


def test_determinism():
    r1 = _pipe().run()
    r2 = _pipe().run()
    for m in ("NonPrivate", "FixedClip-DP-SGD", "DPFuse"):
        a = r1.summary[m]["accuracy_mean"]
        b = r2.summary[m]["accuracy_mean"]
        assert abs(a - b) < 1e-12, f"{m} 确定性失败: {a} vs {b}"


def test_privacy_independent_of_clip_adaptation():
    # FixedClip 与 DPFuse 同 σ、同步数 → 相同 (ε,δ)
    rep = _pipe().run()
    e_fixed = rep.summary["FixedClip-DP-SGD"]["privacy"]["epsilon"]
    e_fuse = rep.summary["DPFuse"]["privacy"]["epsilon"]
    assert abs(e_fixed - e_fuse) < 1e-9


def test_utility_ordering():
    rep = _pipe().run()
    np_acc = rep.summary["NonPrivate"]["accuracy_mean"]
    fx_acc = rep.summary["FixedClip-DP-SGD"]["accuracy_mean"]
    dp_acc = rep.summary["DPFuse"]["accuracy_mean"]
    # 非私有是效用上界
    assert np_acc >= dp_acc - 0.03
    # 隐私方法应明显优于随机（0.5）
    assert dp_acc > 0.6
    # 自适应裁剪不应劣于固定裁剪（相同隐私）
    assert dp_acc >= fx_acc - 0.03


def test_failures_and_ablation_present():
    rep = _pipe().run()
    assert len(rep.failures) >= 3
    assert "comparison" in rep.ablation
