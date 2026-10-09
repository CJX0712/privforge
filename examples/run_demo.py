"""examples/run_demo.py — 端到端演示：基准 + 确定性二次校验 + 落盘。

运行：``python -m examples.run_demo``（仓库根在 sys.path）。
产物：examples/benchmark.json（核心指标来自真实运行）。
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.pipeline import PrivForgePipeline  # noqa: E402
from training.base import TrainConfig  # noqa: E402

SEEDS = (0, 1, 2)
METHODS = ("NonPrivate", "FixedClip-DP-SGD", "DPFuse")


def _build() -> PrivForgePipeline:
    cfg = TrainConfig(
        epochs=30, batch_size=256, learning_rate=0.5, sigma=1.5, clip_norm=1.0, seed=0
    )
    return PrivForgePipeline(
        methods=METHODS,
        seeds=SEEDS,
        cfg=cfg,
        delta=1e-5,
        n_samples=4000,
        n_features=20,
        separation=1.6,
        include_reference=True,
        outlier_frac=0.1,
        outlier_scale=8.0,
    )


def _core(report) -> dict:
    # 仅取可复现核心指标（排除 elapsed_sec）
    core = {}
    for m, s in report.summary.items():
        if m.startswith("_"):
            continue
        core[m] = {
            "accuracy_mean": s["accuracy_mean"],
            "accuracy_std": s["accuracy_std"],
            "loss_mean": s["loss_mean"],
            "excess_loss_vs_nonprivate": s.get("excess_loss_vs_nonprivate"),
            "privacy": s.get("privacy"),
        }
    return core


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    print(">> 第一次运行 ...")
    r1 = _build().run()
    print(">> 第二次运行（确定性校验）...")
    r2 = _build().run()

    c1, c2 = _core(r1), _core(r2)
    max_diff = 0.0
    for m in c1:
        max_diff = max(max_diff, abs(c1[m]["accuracy_mean"] - c2[m]["accuracy_mean"]))
    deterministic = max_diff < 1e-12
    print(
        f">> 确定性校验: max|Δacc_mean| = {max_diff:.2e} -> {'PASS' if deterministic else 'FAIL'}"
    )

    out = {
        "system": "PrivForge",
        "flagship": "DPFuse",
        "deterministic": deterministic,
        "summary": r1.summary,
        "per_seed": r1.per_seed,
        "failures": r1.failures,
        "ablation": r1.ablation,
    }
    out_dir = os.path.join(_ROOT, "examples")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== 汇总（mean±std）===")
    for m, s in r1.summary.items():
        if m.startswith("_"):
            continue
        priv = s.get("privacy")
        ps = f"(ε={priv['epsilon']:.2f},δ={priv['delta']:.0e})" if priv else "non-private"
        print(
            f"  {m:<18} acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} "
            f"excess={s.get('excess_loss_vs_nonprivate', 0):.4f}  {ps}"
        )
    ab = r1.ablation
    print("\n=== 消融（相同 ε,δ）===")
    print(
        f"  DPFuse − FixedClip = {ab['accuracy_diff_mean_pp']:+.4f} pp | 显著={ab['significant']}"
    )
    print(f"\nbenchmark.json -> {path}")
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
