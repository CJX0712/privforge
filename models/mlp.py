"""models/mlp.py — 纯 NumPy 多层感知机（手写前向/反向，支持逐样本梯度）。

设计要点
--------
- 两层隐藏层（ReLU）+ 输出 logits + 交叉熵。容量小、CPU 友好、确定性可复现。
- ``per_sample_gradients`` 返回 (B, P) 的逐样本梯度展平矩阵，是 DP-SGD
  逐样本裁剪的直接输入。
- 提供 ``numgrad_check``（中心差分）用于梯度检验，maxRelErr 应 ~1e-7。
"""

from __future__ import annotations

import numpy as np

from core.seed import RngBundle


class MLP:
    """小型分类 MLP。"""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden1: int = 32,
        hidden2: int = 16,
        rng: RngBundle | None = None,
        scale: float = 0.5,
    ) -> None:
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.scale = float(scale)
        rng = rng or RngBundle.from_seed(0)
        self._init_params(rng.numpy)

    # ---- 参数管理 -------------------------------------------------------
    def _init_params(self, rng: np.random.Generator) -> None:
        d, h1, h2, k = self.n_features, self.hidden1, self.hidden2, self.n_classes
        s = self.scale
        self.W1 = rng.standard_normal((h1, d)) * s / np.sqrt(d)
        self.b1 = np.zeros(h1)
        self.W2 = rng.standard_normal((h2, h1)) * s / np.sqrt(h1)
        self.b2 = np.zeros(h2)
        self.W3 = rng.standard_normal((k, h2)) * s / np.sqrt(h2)
        self.b3 = np.zeros(k)

    def _shapes(self):
        return [
            self.W1.shape,
            self.b1.shape,
            self.W2.shape,
            self.b2.shape,
            self.W3.shape,
            self.b3.shape,
        ]

    def n_params(self) -> int:
        return sum(int(np.prod(s)) for s in self._shapes())

    def pack(self) -> np.ndarray:
        return np.concatenate(
            [
                self.W1.ravel(),
                self.b1.ravel(),
                self.W2.ravel(),
                self.b2.ravel(),
                self.W3.ravel(),
                self.b3.ravel(),
            ]
        )

    def unpack(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float64)
        idx = 0
        out = {}
        for name, shp in zip(["W1", "b1", "W2", "b2", "W3", "b3"], self._shapes()):
            n = int(np.prod(shp))
            out[name] = vec[idx : idx + n].reshape(shp)
            idx += n
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = (
            out["W1"],
            out["b1"],
            out["W2"],
            out["b2"],
            out["W3"],
            out["b3"],
        )

    # ---- 前向 -----------------------------------------------------------
    def forward(self, X: np.ndarray):
        """返回 (logits, cache)。cache 供反向复用。"""
        X = np.asarray(X, dtype=np.float64)
        z1 = X @ self.W1.T + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2.T + self.b2
        a2 = np.maximum(0.0, z2)
        z3 = a2 @ self.W3.T + self.b3
        return z3, (X, z1, a1, z2, a2)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        m = np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits - m)
        return e / np.sum(e, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z3, _ = self.forward(X)
        return self.softmax(z3)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        z3, _ = self.forward(X)
        p = self.softmax(z3)
        # 数值稳定交叉熵
        logp = np.log(np.clip(p, 1e-12, 1.0))
        return float(-np.mean(logp[np.arange(len(y)), y]))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ---- 逐样本梯度 -----------------------------------------------------
    def per_sample_gradients(self, X: np.ndarray, y: np.ndarray):
        """返回 (grads_BP, cache)。grads_BP[i] 为样本 i 的展平梯度。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        B = X.shape[0]
        z3, (Xc, z1, a1, z2, a2) = self.forward(X)
        p = self.softmax(z3)
        # dZ3 = p - onehot(y)  -> (B, K)
        dZ3 = p.copy()
        dZ3[np.arange(B), y] -= 1.0
        # 输出层
        dW3 = dZ3[:, :, None] * a2[:, None, :]  # (B, K, h2)
        db3 = dZ3
        # 隐藏层 2
        dA2 = dZ3 @ self.W3  # (B, h2)
        dZ2 = dA2 * (z2 > 0).astype(np.float64)
        dW2 = dZ2[:, :, None] * a1[:, None, :]  # (B, h2, h1)
        db2 = dZ2
        # 隐藏层 1
        dA1 = dZ2 @ self.W2
        dZ1 = dA1 * (z1 > 0).astype(np.float64)
        dW1 = dZ1[:, :, None] * Xc[:, None, :]  # (B, h1, d)
        db1 = dZ1
        grads = np.concatenate(
            [
                dW1.reshape(B, -1),
                db1,
                dW2.reshape(B, -1),
                db2,
                dW3.reshape(B, -1),
                db3,
            ],
            axis=1,
        )
        return grads, (Xc, z1, a1, z2, a2)

    # ---- 梯度数值校验 ---------------------------------------------------
    def _per_sample_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        z3, _ = self.forward(X)
        p = self.softmax(z3)
        logp = np.log(np.clip(p, 1e-12, 1.0))
        return -logp[np.arange(len(y)), y]

    def numgrad_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """中心差分梯度检验（单样本口径），返回 maxRelErr（应 ~1e-7）。

        analytic[0] 是样本 0 的逐样本损失梯度；numeric 对样本 0 的逐样本损失
        做中心差分，二者口径一致。
        """
        analytic, _ = self.per_sample_gradients(X, y)
        an = analytic[0]
        num = np.zeros_like(an)
        vec = self.pack()
        x0 = X[0:1]
        y0 = y[0:1]

        def loss0(v):
            self.unpack(v)
            return self._per_sample_loss(x0, y0)[0]

        for j in range(len(vec)):
            vplus = vec.copy()
            vplus[j] += eps
            vminus = vec.copy()
            vminus[j] -= eps
            num[j] = (loss0(vplus) - loss0(vminus)) / (2.0 * eps)
        self.unpack(vec)
        denom = np.maximum(1e-8, np.abs(an) + np.abs(num))
        return float(np.max(np.abs(an - num) / denom))


def build_mlp(
    n_features: int,
    n_classes: int,
    hidden1: int = 32,
    hidden2: int = 16,
    rng: RngBundle | None = None,
) -> MLP:
    return MLP(n_features, n_classes, hidden1, hidden2, rng=rng)
