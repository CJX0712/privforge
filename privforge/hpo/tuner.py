"""Optuna-based HPO over the architecture hyperparameter space (SPEC.md #18).

The tuner maximises *validation* accuracy of the flagship on a public-proxy fold.
It never touches private statistics: the validation fold is built from the same
public-bounded dataset and only architecture knobs are searched.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from privforge.core.rng import make_rng
from privforge.core.types import Budget, Dataset
from privforge.hpo.space import HpoConfig
from privforge.training import AdaClipBudget


def _build(
    config: HpoConfig, train: Dataset, budget: Budget, rng: np.random.Generator
) -> AdaClipBudget:
    model = AdaClipBudget(config.as_adaclip())
    return model.fit(train, budget, rng)


def tune(
    train: Dataset,
    val: Dataset,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    n_trials: int = 20,
    seed: int = 0,
) -> HpoConfig:
    """Run a small Optuna study and return the best `HpoConfig`."""
    import optuna

    space_def = {
        "T": (80, 320, "int"),
        "q": (0.05, 0.2, "float"),
        "eta": (0.05, 0.5, "float"),
        "lam": (1e-4, 1e-2, "float"),
        "rho": (0.90, 0.999, "float"),
        "p": (0.5, 0.9, "float"),
        "clip_mode": (("quantile", "fixed"), "cat"),
        "mu_schedule": (("propagation", "uniform"), "cat"),
    }
    budget = Budget.from_fractions(epsilon=epsilon, delta=delta)

    def objective(trial: optuna.trial.Trial) -> float:
        cfg = HpoConfig(
            T=int(trial.suggest_int("T", *space_def["T"][:2])),
            q=float(trial.suggest_float("q", *space_def["q"][:2])),
            eta=float(trial.suggest_float("eta", *space_def["eta"][:2])),
            lam=float(trial.suggest_float("lam", *space_def["lam"][:2], log=True)),
            rho=float(trial.suggest_float("rho", *space_def["rho"][:2])),
            p=float(trial.suggest_float("p", *space_def["p"][:2])),
            clip_mode=str(trial.suggest_categorical("clip_mode", space_def["clip_mode"][0])),
            mu_schedule=str(trial.suggest_categorical("mu_schedule", space_def["mu_schedule"][0])),
        )
        rng = make_rng(seed, "hpo", str(trial.number))
        try:
            model = _build(cfg, train, budget, rng)
            return float(model.accuracy(val.X, val.y))
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return HpoConfig(
        T=int(best["T"]),
        q=float(best["q"]),
        eta=float(best["eta"]),
        lam=float(best["lam"]),
        rho=float(best["rho"]),
        p=float(best["p"]),
        clip_mode=str(best["clip_mode"]),
        mu_schedule=str(best["mu_schedule"]),
    )


def to_adaclip(config: HpoConfig):
    return config.as_adaclip()
