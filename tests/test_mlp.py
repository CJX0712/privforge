"""tests/test_mlp.py — MLP 梯度检验与参数往返。"""

from __future__ import annotations

import numpy as np

from core.seed import RngBundle
from models.mlp import MLP


def _tiny(seed=0):
    rng = RngBundle.from_seed(seed)
    X = rng.numpy.standard_normal((8, 6))
    y = rng.numpy.integers(0, 2, size=8)
    return X, y


def test_numgrad_check():
    X, y = _tiny(5)
    model = MLP(6, 2, hidden1=8, hidden2=4, rng=RngBundle.from_seed(7))
    max_rel = model.numgrad_check(X, y, eps=1e-6)
    assert max_rel < 1e-6, f"梯度检验 maxRelErr={max_rel}"


def test_pack_unpack_roundtrip():
    model = MLP(6, 2, hidden1=8, hidden2=4, rng=RngBundle.from_seed(9))
    vec = model.pack()
    model.W1 += 1.0  # 破坏
    model.unpack(vec)
    assert np.allclose(model.pack(), vec)


def test_predict_shape_and_accuracy_bounds():
    X, y = _tiny(11)
    model = MLP(6, 2, hidden1=8, hidden2=4, rng=RngBundle.from_seed(13))
    p = model.predict_proba(X)
    assert p.shape == (8, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    pred = model.predict(X)
    assert pred.shape == (8,)
    assert set(np.unique(pred)).issubset({0, 1})
