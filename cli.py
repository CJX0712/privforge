"""cli.py — PrivForge 命令行入口（argparse）。

子命令：
  benchmark  运行多方法×多 seed 基准，打印表格并落盘 benchmark.json
  train      单方法训练，打印结果
  selftest   快速跑通一个最小训练，验证环境
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 允许 `python cli.py` 直接运行：把仓库根加入路径。
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.pipeline import PrivForgePipeline  # noqa: E402
from training.base import TrainConfig  # noqa: E402


def _print_summary(report) -> None:
    print("\n=== 汇总（mean±std，多 seed）===")
    print(f"{'方法':<18}{'acc_mean':>10}{'acc_std':>9}{'excess':>9}  privacy(ε,δ)")
    for m, s in report.summary.items():
        if m.startswith("_"):
            continue
        priv = s.get("privacy")
        if priv:
            priv_s = f"({priv['epsilon']:.2f},{priv['delta']:.0e})"
        else:
            priv_s = "non-private"
        print(
            f"{m:<18}{s['accuracy_mean']:>10.4f}{s['accuracy_std']:>9.4f}"
            f"{s.get('excess_loss_vs_nonprivate', 0):>9.4f}  {priv_s}"
        )
    ab = report.ablation
    if ab and "accuracy_diff_mean_pp" in ab:
        print("\n=== 消融：DPFuse vs FixedClip（相同 ε,δ）===")
        print(f"  Δacc(mean) = {ab['accuracy_diff_mean_pp']:+.4f} pp | 显著={ab['significant']}")


def cmd_benchmark(args) -> int:
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sigma=args.sigma,
        clip_norm=args.clip,
        seed=0,
        hidden1=args.h1,
        hidden2=args.h2,
    )
    pipe = PrivForgePipeline(
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        cfg=cfg,
        delta=args.delta,
        n_samples=args.n_samples,
        n_features=args.n_features,
        separation=args.separation,
        include_reference=True,
        outlier_frac=args.outlier_frac,
        outlier_scale=args.outlier_scale,
    )
    report = pipe.run()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, ensure_ascii=False, indent=2)
    _print_summary(report)
    print(f"\nbenchmark.json -> {out_path}")
    return 0


def cmd_train(args) -> int:
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sigma=args.sigma,
        clip_norm=args.clip,
        seed=args.seed,
        hidden1=args.h1,
        hidden2=args.h2,
    )
    pipe = PrivForgePipeline(
        methods=(args.method,),
        seeds=(args.seed,),
        cfg=cfg,
        n_samples=args.n_samples,
        n_features=args.n_features,
        separation=args.separation,
        outlier_frac=args.outlier_frac,
        outlier_scale=args.outlier_scale,
    )
    r = pipe.run_single(args.method, args.seed)
    print(json.dumps(r.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_selftest(args) -> int:
    cfg = TrainConfig(epochs=2, batch_size=256, sigma=1.0, clip_norm=1.0, seed=0)
    pipe = PrivForgePipeline(methods=("DPFuse",), seeds=(0,), cfg=cfg, n_samples=800)
    r = pipe.run_single("DPFuse", 0)
    print(
        f"selftest OK: DPFuse acc={r.test_accuracy:.4f} "
        f"ε={r.privacy.epsilon if r.privacy else 'NA'}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="privforge", description="差分隐私机器学习系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="运行基准")
    b.add_argument("--methods", nargs="+", default=["NonPrivate", "FixedClip-DP-SGD", "DPFuse"])
    b.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    b.add_argument("--epochs", type=int, default=30)
    b.add_argument("--batch-size", type=int, default=256)
    b.add_argument("--lr", type=float, default=0.5)
    b.add_argument("--sigma", type=float, default=1.5)
    b.add_argument("--clip", type=float, default=1.0)
    b.add_argument("--delta", type=float, default=1e-5)
    b.add_argument("--h1", type=int, default=32)
    b.add_argument("--h2", type=int, default=16)
    b.add_argument("--n-samples", type=int, default=4000)
    b.add_argument("--n-features", type=int, default=20)
    b.add_argument("--separation", type=float, default=1.6)
    b.add_argument("--outlier-frac", type=float, default=0.1)
    b.add_argument("--outlier-scale", type=float, default=8.0)
    b.add_argument("--out-dir", default="examples")
    b.set_defaults(func=cmd_benchmark)

    t = sub.add_parser("train", help="单方法训练")
    t.add_argument("--method", default="DPFuse")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=0.5)
    t.add_argument("--sigma", type=float, default=1.5)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--h1", type=int, default=32)
    t.add_argument("--h2", type=int, default=16)
    t.add_argument("--n-samples", type=int, default=4000)
    t.add_argument("--n-features", type=int, default=20)
    t.add_argument("--separation", type=float, default=1.6)
    t.add_argument("--outlier-frac", type=float, default=0.1)
    t.add_argument("--outlier-scale", type=float, default=8.0)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("selftest", help="环境自检")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
