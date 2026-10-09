"""hpo/tuner.py — 可选 HPO（Optuna）。不可用时自动降级（返回默认配置）。"""

from __future__ import annotations

from dataclasses import asdict

from core.errors import ConfigError
from training.base import TrainConfig


def _available() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def tune(cfg: TrainConfig, objective) -> TrainConfig:
    """用 Optuna 微调 clip_norm / learning_rate / sigma。

    离线兜底：Optuna 不可用时直接返回原 cfg（不阻塞系统）。
    """
    if not _available():
        return cfg
    try:
        import optuna

        def _obj(trial):
            c = TrainConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                learning_rate=trial.suggest_float("lr", 0.1, 1.5, log=True),
                sigma=trial.suggest_float("sigma", 0.5, 2.0),
                clip_norm=trial.suggest_float("clip_norm", 0.2, 3.0, log=True),
                seed=cfg.seed,
                hidden1=cfg.hidden1,
                hidden2=cfg.hidden2,
            )
            return objective(c)

        study = optuna.create_study(direction="maximize")
        study.optimize(_obj, n_trials=12, show_progress_bar=False)
        best = study.best_params
        out = TrainConfig(**{**asdict(cfg), **best})
        return out
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"HPO 失败且无法降级: {e}")
