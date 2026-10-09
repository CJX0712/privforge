"""training/sklearn_ref.py — scikit-learn 非私有参考后端（Tier-0 复用，可选）。

复用世界顶级开源 scikit-learn 的 ``MLPClassifier`` 作为非私有效用参考上界。
不可用时（纯离线兜底）自动跳过，不影响系统运行。该方法仅作参考，不参与
「相同 (ε,δ) 下 DPFuse vs FixedClip」的核心隐私效用比较。
"""

from __future__ import annotations

import time

import numpy as np

from core.errors import ModelError
from core.types import TrainResult


def _available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def run_sklearn_reference(
    X_train,
    y_train,
    X_test,
    y_test,
    seed: int = 0,
    hidden1: int = 32,
    hidden2: int = 16,
) -> TrainResult:
    try:
        from sklearn.neural_network import MLPClassifier
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"scikit-learn 不可用: {e}")

    t0 = time.perf_counter()
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden1, hidden2),
        activation="relu",
        solver="adam",
        max_iter=200,
        learning_rate_init=0.001,
        random_state=seed,
        early_stopping=False,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = float(np.mean(pred == y_test))
    proba = clf.predict_proba(X_test)
    logp = np.log(np.clip(proba, 1e-12, 1.0))
    loss = float(-np.mean(logp[np.arange(len(y_test)), y_test]))
    elapsed = time.perf_counter() - t0
    return TrainResult(
        method="SklearnRef-NonPrivate",
        test_accuracy=acc,
        test_loss=loss,
        privacy=None,
        final_clip_norm=0.0,
        n_params=int(
            clf.n_outputs_ * (hidden2 * hidden1 + hidden1 * X_train.shape[1] + hidden1 + hidden2)
            + clf.n_outputs_
        ),
        elapsed_sec=elapsed,
    )
