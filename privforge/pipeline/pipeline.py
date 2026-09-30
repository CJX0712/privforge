"""Top-level benchmark driver (SPEC.md section 6).

`run_benchmark` generates the D1..D6 ladder (fixed seed), splits per seed, fits
the baselines (B2/B3/B4) and the AQUA-DP pipeline for each epsilon, and collects
everything into a `BenchmarkReport`.  `quick_benchmark` is the demo-sized variant.

Author: 晨星
"""

from __future__ import annotations

from privforge.core.rng import make_rng
from privforge.core.types import Budget
from privforge.data.loader import train_test_split
from privforge.data.synthetic import generate_dataset
from privforge.eval.metrics import evaluate_model, non_dp_ceiling
from privforge.eval.report import BenchmarkReport
from privforge.pipeline.flagship import AquaConfig, AquaDP
from privforge.training import DPSGDBaseline, ObjectivePerturbation, OutputPerturbation

# Baselines that compete for "best base" (A5 oracle is diagnostic only, excluded).
BASELINE_BUILDERS = [
    (lambda cfg: DPSGDBaseline(T=cfg.T, q=cfg.q, eta=cfg.eta, lam=cfg.lam), "B4"),
    (lambda cfg: OutputPerturbation(lam=cfg.lam), "B2"),
    (lambda cfg: ObjectivePerturbation(lam=cfg.lam), "B3"),
]


def run_benchmark(
    datasets=("medium",),
    epsilons=(1.0,),
    seeds=(1, 2, 3),
    config: AquaConfig | None = None,
) -> BenchmarkReport:
    """Run the full evaluation protocol and return a `BenchmarkReport`."""
    cfg = config or AquaConfig()
    results = []
    fallback_seen: list[str] = []

    for name in datasets:
        ds = generate_dataset(name)
        for seed in seeds:
            train, test = train_test_split(ds, seed=int(seed))
            u_nodp = non_dp_ceiling(
                train,
                test,
                T=cfg.T,
                q=cfg.q,
                eta=cfg.eta,
                lam=cfg.lam,
                clip_mode=cfg.clip_mode,
                p=cfg.p,
                rng=make_rng(int(seed), "ceil", name),
            )
            for eps in epsilons:
                budget = Budget.from_fractions(epsilon=float(eps), delta=cfg.delta)

                # --- baselines -------------------------------------------------
                fitted = []
                for builder, mname in BASELINE_BUILDERS:
                    m = builder(cfg).fit(train, budget, make_rng(int(seed), mname, name))
                    fitted.append((m, mname))
                u_best = max(m.accuracy(test.X, test.y) for m, _ in fitted)
                for m, mname in fitted:
                    results.append(
                        evaluate_model(
                            m,
                            test,
                            budget,
                            dataset=name,
                            method=mname,
                            seed=int(seed),
                            u_nodp=u_nodp,
                            u_best_base=u_best,
                        )
                    )

                # --- flagship pipeline (S1/S2/S3) -------------------------------
                aqua = AquaDP(cfg)
                res, fb = aqua.fit_and_evaluate(
                    train,
                    test,
                    budget,
                    make_rng(int(seed), "pipe", name),
                    u_nodp=u_nodp,
                    u_best_base=u_best,
                    seed=int(seed),
                    dataset_name=name,
                )
                results.append(res)
                if fb:
                    fallback_seen.append(fb)

    report = BenchmarkReport(
        results=results, backend_fallback=(";".join(sorted(set(fallback_seen))) or None)
    )
    return report


def quick_benchmark() -> BenchmarkReport:
    """Demo-sized benchmark (single dataset, single seed, one epsilon)."""
    return run_benchmark(
        datasets=("medium",), epsilons=(1.0,), seeds=(1,), config=AquaConfig(T=80, do_audit=False)
    )
