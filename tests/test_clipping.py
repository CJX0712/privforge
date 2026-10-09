"""tests/test_clipping.py — 逐样本裁剪不变量。"""

from __future__ import annotations

import numpy as np

from core.seed import RngBundle
from dp.clipping import clip_fraction, clip_per_sample, max_clipped_norm


def _rand_grads(rng, B=16, P=20):
    return rng.standard_normal((B, P))


def test_clipped_norm_below_clip():
    rng = RngBundle.from_seed(1).numpy
    g = _rand_grads(rng)
    C = 0.7
    clipped, _, _ = clip_per_sample(g, C)
    assert max_clipped_norm(clipped) <= C + 1e-9


def test_scale_equivariance():
    # 正确尺度等价：clip(g, C) = C · clip(g/C, 1)
    rng = RngBundle.from_seed(2).numpy
    g = _rand_grads(rng)
    for C in (0.3, 1.0, 2.0):
        cg, _, _ = clip_per_sample(g, C)
        cg_unit, _, _ = clip_per_sample(g / C, 1.0)
        assert np.allclose(cg, C * cg_unit, atol=1e-9)


def test_unclipped_identity():
    # 范数 ≤ C 的样本不被裁剪（裁剪 = 原梯度）
    rng = RngBundle.from_seed(4).numpy
    g = rng.standard_normal((16, 20)) * 0.01  # 范数很小
    cg, norms, flags = clip_per_sample(g, 1.0)
    assert not flags.any()
    assert np.allclose(cg, g, atol=1e-12)


def test_clip_fraction():
    rng = RngBundle.from_seed(3).numpy
    g = _rand_grads(rng)
    C = 0.5
    _, _, flags = clip_per_sample(g, C)
    assert abs(clip_fraction(flags) - float(np.mean(flags))) < 1e-12


def test_zero_grad_safe():
    g = np.zeros((4, 5))
    clipped, norms, flags = clip_per_sample(g, 1.0)
    assert np.allclose(clipped, 0.0)
    assert np.all(norms == 0.0)
    assert not flags.any()
