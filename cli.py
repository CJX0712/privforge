#!/usr/bin/env python
"""PrivForge CLI (SPEC.md section 7).

Subcommands
-----------
  demo        run the end-to-end demo (quick_benchmark) and write benchmark.json
  benchmark  run a custom benchmark (--datasets/--epsilons/--seeds) -> json
  dod        evaluate the DoD gates against a benchmark.json
  verify     run the pytest suite

All commands are reproducible: every stream is derived from the documented
seed protocol (core/rng.py), so re-running yields byte-identical results.

Usage
-----
  python cli.py demo
  python cli.py benchmark --datasets medium heavy_tail --seeds 1 2 3 --epsilons 0.5 1.0 2.0
  python cli.py dod benchmark.json
  python cli.py verify

Author: 晨星
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from privforge import __author__, __version__
from privforge.eval.report import BenchmarkReport
from privforge.pipeline.flagship import AquaConfig
from privforge.pipeline.pipeline import quick_benchmark, run_benchmark

ROOT = Path(__file__).resolve().parent


def _fmt_row(r) -> str:
    return (
        f"{r.dataset:<12} {r.method:<9} {r.epsilon:>6.3f} {r.seed:>5} "
        f"{r.accuracy:>8.4f} {r.ugc:>+8.3f} {r.uac:>7.3f} {r.eps_spent:>8.4f}"
    )


def _emit(rep: BenchmarkReport, out: Path) -> None:
    # Persist the raw results BEFORE any DoD evaluation: a gate crash must
    # never cost the benchmark compute (it happened once -- 35 min lost).
    rep.to_json(str(out))
    dod = rep.dod()
    print(f"PrivForge v{__version__}  author={__author__}")
    print(
        f"{'dataset':<12} {'method':<9} {'eps':>6} {'seed':>5} "
        f"{'acc':>8} {'ugc':>8} {'uac':>7} {'spent':>8}"
    )
    for r in rep.results:
        print(_fmt_row(r))
    d = dod.as_dict()
    print("\nDoD gates:")
    for k, v in d.items():
        if k == "notes":
            continue
        mark = "✅" if (isinstance(v, bool) and v) else ("⚠️ " if isinstance(v, bool) else "")
        print(f"  {mark}{k}: {v}")
    for n in d["notes"]:
        print(f"  - {n}")
    print(f"\nwrote {out}")


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = AquaConfig(T=args.T, do_audit=False)
    rep = (
        quick_benchmark()
        if args.quick
        else run_benchmark(
            datasets=tuple(args.datasets),
            epsilons=tuple(args.epsilons),
            seeds=tuple(args.seeds),
            config=cfg,
        )
    )
    _emit(rep, ROOT / args.out)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = AquaConfig(T=args.T, do_audit=args.audit, audit_trials=args.audit_trials)
    rep = run_benchmark(
        datasets=tuple(args.datasets),
        epsilons=tuple(args.epsilons),
        seeds=tuple(args.seeds),
        config=cfg,
    )
    _emit(rep, ROOT / args.out)
    return 0


def cmd_dod(args: argparse.Namespace) -> int:
    rep = BenchmarkReport.from_json(str(args.json))
    dod = rep.dod()
    d = dod.as_dict()
    print(f"DoD evaluation of {args.json} ({rep.n_results} results)")
    for k, v in d.items():
        if k == "notes":
            continue
        mark = "✅" if (isinstance(v, bool) and v) else ("⚠️ " if isinstance(v, bool) else "")
        print(f"  {mark}{k}: {v}")
    for n in d["notes"]:
        print(f"  - {n}")
    return 0 if d["all_passed"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "-W", "ignore::UserWarning", "tests"]
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="privforge", description="PrivForge DP-ML toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the end-to-end demo")
    d.add_argument("--quick", action="store_true", help="quick single-dataset demo")
    d.add_argument("--T", type=int, default=80)
    d.add_argument("--datasets", nargs="+", default=["medium"])
    d.add_argument("--seeds", nargs="+", type=int, default=[1])
    d.add_argument("--epsilons", nargs="+", type=float, default=[1.0])
    d.add_argument("--out", default="benchmark.json")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("benchmark", help="run a custom benchmark")
    b.add_argument("--datasets", nargs="+", default=["medium", "heavy_tail", "hard_lowsep"])
    b.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    b.add_argument("--epsilons", nargs="+", type=float, default=[1.0])
    b.add_argument("--T", type=int, default=100)
    b.add_argument("--audit", action="store_true")
    b.add_argument("--audit-trials", dest="audit_trials", type=int, default=40)
    b.add_argument("--out", default="benchmark.json")
    b.set_defaults(func=cmd_benchmark)

    g = sub.add_parser("dod", help="evaluate DoD gates on a benchmark.json")
    g.add_argument("json")
    g.set_defaults(func=cmd_dod)

    v = sub.add_parser("verify", help="run the pytest suite")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
