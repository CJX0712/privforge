"""AQUA-DP pipeline with the S1 / S2 / S3 safeguards (SPEC.md section 2.3).

S1 -- non-inferiority: compare the flagship against a fixed-C baseline on the
     test fold and keep the better one (Exponential mechanism, budget eps_select).
S2 -- budget: the chosen accountant must not report more than the declared eps.
S3 -- audit: empirical MIA audit; if eps_emp exceeds epsilon + tol, fall back to
     Output Perturbation (a usable, pure-epsilon-DP model is always returned).

The pipeline ALWAYS returns a usable model; a safeguard firing is recorded in
`backend_fallback` rather than raising (E502 semantics).

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from privforge.core.errors import AuditViolation, BudgetExceeded
from privforge.core.rng import make_rng
from privforge.core.types import Budget, Dataset, EvalResult
from privforge.eval.metrics import evaluate_model
from privforge.training import (
    AdaClipBudget,
    AdaClipConfig,
    DPSGDBaseline,
    ObjectivePerturbation,
    OutputPerturbation,
)
from privforge.training.logistic import logistic_loss
from privforge.training.safeguards import audit_safeguard, budget_safeguard, non_inferiority_select

FLAGSHIP_NAME = "AQUA-DP"


@dataclass
class AquaConfig:
    epsilon: float = 1.0
    delta: float = 1e-5
    T: int = 120
    q: float = 0.1
    eta: float = 0.1
    lam: float = 1e-3
    rho: float = 0.99
    p: float = 0.7
    clip_mode: str = "quantile"  # 'quantile' | 'fixed'
    mu_schedule: str = "propagation"  # 'propagation' | 'uniform'
    fixed_C: float | None = None
    do_audit: bool = False
    audit_trials: int = 40
    audit_tol: float = 0.05


class AquaDP:
    """The AQUA-DP pipeline: AdaClip-Budget + S1/S2/S3 safeguards."""

    def __init__(self, config: AquaConfig | None = None) -> None:
        self.config = config or AquaConfig()

    # -- internal building blocks ------------------------------------------------
    def _flagship_model(self, cfg: AquaConfig) -> AdaClipBudget:
        ac = AdaClipConfig(
            T=cfg.T,
            q=cfg.q,
            eta=cfg.eta,
            lam=cfg.lam,
            rho=cfg.rho,
            p=cfg.p,
            clip_mode=cfg.clip_mode,
            mu_schedule=cfg.mu_schedule,
            fixed_C=cfg.fixed_C,
        )
        return AdaClipBudget(ac)

    def _baseline_model(self, cfg: AquaConfig, dataset: Dataset) -> DPSGDBaseline:
        R = float(dataset.bounds.x_norm_max)
        n_features = int(dataset.X.shape[1])
        fixed_C = float(np.sqrt(n_features)) / R
        return DPSGDBaseline(
            T=cfg.T, q=cfg.q, eta=cfg.eta, lam=cfg.lam, clip_mode="fixed", fixed_C=fixed_C
        )

    # -- single (dataset, seed, epsilon) evaluation ------------------------------
    def fit_and_evaluate(
        self,
        train: Dataset,
        test: Dataset,
        budget: Budget,
        rng: np.random.Generator,
        u_nodp: float,
        u_best_base: float,
        seed: int,
        dataset_name: str,
    ) -> tuple[EvalResult, str | None]:
        cfg = self.config
        fallback: str | None = None

        # 1) flagship + fixed-C baseline (S1 comparison pair)
        aqua = self._flagship_model(cfg).fit(train, budget, make_rng(seed, "aqua", dataset_name))
        base = self._baseline_model(cfg, train).fit(
            train, budget, make_rng(seed, "base", dataset_name)
        )
        aqua_acc = aqua.accuracy(test.X, test.y)
        base_acc = base.accuracy(test.X, test.y)

        # S1: keep the better of the two under eps_select-DP selection
        choice = non_inferiority_select(
            aqua_acc, base_acc, budget.eps_select, make_rng(seed, "s1", dataset_name)
        )
        if choice == 0:
            chosen = aqua
        else:
            chosen = base
            fallback = "S1:non-inferiority->B4"

        # S2: the chosen accountant must not overspend
        try:
            budget_safeguard(chosen.last_result.accountant, budget, tol=1e-6)
        except BudgetExceeded:
            chosen = base
            fallback = "S2:budget->B4"

        # S3: empirical privacy audit (optional, expensive)
        if cfg.do_audit and fallback is None:
            fit_fn = lambda ds, r, b=budget: self._flagship_model(cfg).fit(ds, b, r)
            loss_fn = lambda m, X, y: float(logistic_loss(m.w, X, y, lam=0.0))

            def _safe():
                try:
                    audit_safeguard(
                        fit_fn,
                        loss_fn,
                        train,
                        budget,
                        make_rng(seed, "s3", dataset_name),
                        n_trials=cfg.audit_trials,
                        tol=cfg.audit_tol,
                        fallback_model=base,
                    )
                    return None
                except AuditViolation:
                    return "S3:audit->OP"

            triggered = _safe()
            if triggered is not None:
                chosen = OutputPerturbation(lam=cfg.lam).fit(
                    train, budget, make_rng(seed, "op", dataset_name)
                )
                fallback = triggered

        result = evaluate_model(
            chosen,
            test,
            budget,
            dataset=dataset_name,
            method=FLAGSHIP_NAME,
            seed=seed,
            u_nodp=u_nodp,
            u_best_base=u_best_base,
        )
        return result, fallback

    # -- ablation runner (SPEC.md section 2.4, A0..A6) ---------------------------
    def run_ablation(
        self,
        name: str,
        train: Dataset,
        test: Dataset,
        budget: Budget,
        rng: np.random.Generator,
        u_nodp: float,
    ) -> EvalResult:
        cfg = self.config
        if name in ("A0", "A1", "A2", "A3", "A5"):
            mode = {
                "A0": "fixed",
                "A1": "quantile",
                "A2": "quantile",
                "A3": "quantile",
                "A5": "fixed",
            }[name]
            schedule = "uniform" if name in ("A0", "A1", "A5") else cfg.mu_schedule
            ac = AdaClipConfig(
                T=cfg.T,
                q=cfg.q,
                eta=cfg.eta,
                lam=cfg.lam,
                rho=cfg.rho,
                p=cfg.p,
                clip_mode=mode,
                mu_schedule=schedule,
                fixed_C=(
                    float(np.sqrt(train.X.shape[1]) / train.bounds.x_norm_max)
                    if name in ("A0", "A5")
                    else None
                ),
                unsafe_mu_times_C=(name == "A6"),
            )
            m = AdaClipBudget(ac).fit(train, budget, rng)
        elif name == "A6":
            ac = AdaClipConfig(
                T=cfg.T,
                q=cfg.q,
                clip_mode="quantile",
                mu_schedule="propagation",
                unsafe_mu_times_C=True,
            )
            m = AdaClipBudget(ac).fit(train, budget, rng)
        elif name == "B2":
            m = OutputPerturbation(lam=cfg.lam).fit(train, budget, rng)
        elif name == "B3":
            m = ObjectivePerturbation(lam=cfg.lam).fit(train, budget, rng)
        elif name == "B4":
            m = self._baseline_model(cfg, train).fit(train, budget, rng)
        else:
            raise ValueError(f"unknown ablation: {name}")
        # baselines define their own u_best_base trivially
        return evaluate_model(
            m,
            test,
            budget,
            dataset=train.name,
            method=name,
            seed=0,
            u_nodp=u_nodp,
            u_best_base=u_nodp,
        )
