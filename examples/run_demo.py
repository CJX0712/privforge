"""PrivForge end-to-end demo (SPEC.md section 5 / SOP section 5).

Generates the synthetic D1..D6 ladder, trains the AQUA-DP pipeline against the
three baselines (B2 Output Perturbation, B3 Objective Perturbation, B4 DP-SGD),
prints a fixed-width utility table and writes ``benchmark.json`` (per-seed raw
records + the required ``backend_fallback`` field).

Default config (``quick``) completes in well under 60s so it is safe to ship as
the one-click reproducibility demo.  Use ``python cli.py benchmark`` for a fuller
sweep used by the DoD gates in the acceptance report.

Author: 晨星
"""

from __future__ import annotations

import time
from pathlib import Path

# `privforge` is importable because the repo root is on sys.path when this script
# is launched directly (SOP: python examples/run_demo.py).
from privforge import __author__, __version__
from privforge.pipeline.flagship import AquaConfig
from privforge.pipeline.pipeline import run_benchmark

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    t0 = time.time()
    # Demo-sized: one dataset, one seed, one epsilon -- comfortably <= 60s.
    datasets = ("medium",)
    seeds = (1,)
    epsilons = (1.0,)
    cfg = AquaConfig(T=80, do_audit=False)

    print(f"PrivForge v{__version__}  author={__author__}")
    print(f"demo: datasets={datasets} seeds={seeds} epsilons={epsilons} T={cfg.T}")
    print("generating data + training AQUA-DP and baselines ...", flush=True)

    rep = run_benchmark(datasets=datasets, epsilons=epsilons, seeds=seeds, config=cfg)

    print()
    print(
        f"{'dataset':<12} {'method':<9} {'eps':>6} {'seed':>5} "
        f"{'acc':>8} {'ugc':>8} {'uac':>7} {'spent':>8}"
    )
    for r in rep.results:
        print(
            f"{r.dataset:<12} {r.method:<9} {r.epsilon:>6.3f} {r.seed:>5} "
            f"{r.accuracy:>8.4f} {r.ugc:>+8.3f} {r.uac:>7.3f} {r.eps_spent:>8.4f}"
        )

    dod = rep.dod()
    d = dod.as_dict()
    print("\nDoD gates:")
    for k, v in d.items():
        if k == "notes":
            continue
        mark = "✅" if (isinstance(v, bool) and v) else ("⚠️ " if isinstance(v, bool) else "")
        print(f"  {mark}{k}: {v}")
    for n in d["notes"]:
        print(f"  - {n}")

    out = ROOT / "benchmark.json"
    rep.to_json(str(out))
    print(f"\nwrote {out} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
