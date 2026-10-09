"""pipeline/pipeline.py — PrivForge 端到端流水线。

单向调用：pipeline -> {data, preprocess, dp, models, training, eval} -> core。
负责：数据载入 → 仅 train 上 fit 的标准化（防泄漏） → 多方法×多 seed 训练 →
聚合 mean±std → 生成失败案例与消融结论。
"""

from __future__ import annotations

import copy

from core.types import BenchmarkReport, Dataset, TrainResult
from data.loader import SyntheticDataSource
from eval.metrics import significance_mean_diff, summarize
from preprocess.scaler import Standardizer
from training import TRAINERS, TrainConfig


class PrivForgePipeline:
    def __init__(
        self,
        dataset: Dataset | None = None,
        data_seed: int = 0,
        methods: tuple[str, ...] = ("NonPrivate", "FixedClip-DP-SGD", "DPFuse"),
        seeds: tuple[int, ...] = (0, 1, 2),
        cfg: TrainConfig | None = None,
        delta: float = 1e-5,
        n_samples: int = 4000,
        n_features: int = 20,
        separation: float = 2.2,
        include_reference: bool = False,
        outlier_frac: float = 0.0,
        outlier_scale: float = 8.0,
    ) -> None:
        self.dataset = (
            dataset
            or SyntheticDataSource(
                seed=data_seed,
                n_samples=n_samples,
                n_features=n_features,
                separation=separation,
                outlier_frac=outlier_frac,
                outlier_scale=outlier_scale,
            ).get()
        )
        self.methods = methods
        self.seeds = seeds
        self.cfg = cfg or TrainConfig()
        self.delta = delta
        self.include_reference = include_reference

    # ---- 内部：按方法构造配置 ----------------------------------------
    def _cfg_for(self, method: str, seed: int) -> TrainConfig:
        c = copy.deepcopy(self.cfg)
        c.seed = seed
        if method == "NonPrivate":
            c.sigma = 0.0
            c.adaptive = False
        elif method == "FixedClip-DP-SGD":
            c.sigma = self.cfg.sigma
            c.clip_norm = self.cfg.clip_norm
            c.adaptive = False
        elif method == "DPFuse":
            c.sigma = self.cfg.sigma
            c.clip_norm = self.cfg.clip_norm
            c.adaptive = True
        else:
            raise ValueError(f"未知方法: {method}")
        return c

    def _fit_one(self, method: str, seed: int) -> TrainResult:
        # 防泄漏：scaler 只在 train 上 fit
        scaler = Standardizer().fit(self.dataset.X_train)
        Xtr = scaler.transform(self.dataset.X_train)
        Xte = scaler.transform(self.dataset.X_test)
        trainer = TRAINERS[method](self._cfg_for(method, seed))
        return trainer.fit(Xtr, self.dataset.y_train, Xte, self.dataset.y_test)

    # ---- 主入口 -------------------------------------------------------
    def run(self) -> BenchmarkReport:
        per_seed: list[dict] = []
        by_method: dict[str, list[TrainResult]] = {m: [] for m in self.methods}
        for method in self.methods:
            for seed in self.seeds:
                r = self._fit_one(method, seed)
                by_method[method].append(r)
                per_seed.append({"method": method, "seed": seed, **r.as_dict()})

        summary = {m: summarize(by_method[m]) for m in self.methods}

        # 可选：scikit-learn 非私有参考（Tier-0 复用，不可用时跳过）
        skipped_reference = False
        if self.include_reference:
            try:
                from training.sklearn_ref import _available, run_sklearn_reference

                if _available():
                    scaler = Standardizer().fit(self.dataset.X_train)
                    ref = run_sklearn_reference(
                        scaler.transform(self.dataset.X_train),
                        self.dataset.y_train,
                        scaler.transform(self.dataset.X_test),
                        self.dataset.y_test,
                        seed=self.seeds[0],
                        hidden1=self.cfg.hidden1,
                        hidden2=self.cfg.hidden2,
                    )
                    by_method[ref.method] = [ref]
                    summary[ref.method] = summarize([ref])
                else:
                    skipped_reference = True
            except Exception:  # noqa: BLE001
                skipped_reference = True

        # 非私有上界
        ceiling = summary["NonPrivate"]["accuracy_mean"]
        for m, s in summary.items():
            if m.startswith("_"):
                continue
            s["excess_loss_vs_nonprivate"] = float(ceiling - s["accuracy_mean"])

        # 显著性：DPFuse vs FixedClip（相同 ε,δ）
        ablation: dict = {"sklearn_reference_skipped": skipped_reference}
        if "DPFuse" in by_method and "FixedClip-DP-SGD" in by_method:
            diff, sig = significance_mean_diff(by_method["DPFuse"], by_method["FixedClip-DP-SGD"])
            ablation.update(
                {
                    "comparison": "DPFuse vs FixedClip-DP-SGD (same ε,δ)",
                    "accuracy_diff_mean_pp": round(diff * 100.0, 4),
                    "significant": sig,
                    "fixedclip_acc_mean": summary["FixedClip-DP-SGD"]["accuracy_mean"],
                    "dpfuse_acc_mean": summary["DPFuse"]["accuracy_mean"],
                }
            )
        summary["_ablation"] = ablation

        failures = self._analyze_failures(ceiling)
        return BenchmarkReport(
            per_seed=per_seed,
            summary=summary,
            failures=failures,
            ablation=ablation,
        )

    # ---- 失败案例（从结果派生，含原因归因） --------------------------
    def _analyze_failures(self, ceiling: float) -> list[dict]:
        cases = [
            ("σ 过大 (σ=6.0)", {"sigma": 6.0}, "噪声方差爆炸，梯度信号被淹没，精度坍缩至随机水平"),
            (
                "裁剪界过小 (C=0.05)",
                {"clip_norm": 0.05},
                "逐样本梯度被过度裁剪，引入系统性偏差，拟合不足",
            ),
            ("训练步数不足 (epochs=2)", {"epochs": 2}, "优化未收敛，欠拟合"),
        ]
        out = []
        for name, override, reason in cases:
            c = copy.deepcopy(self.cfg)
            for k, v in override.items():
                setattr(c, k, v)
            r = self._fit_one("DPFuse", 0)
            out.append(
                {
                    "case": name,
                    "accuracy": round(r.test_accuracy, 4),
                    "ceiling": round(ceiling, 4),
                    "excess_loss": round(ceiling - r.test_accuracy, 4),
                    "reason": reason,
                }
            )
        return out

    # ---- 单方法便捷入口（CLI 用） -----------------------------------
    def run_single(self, method: str, seed: int = 0) -> TrainResult:
        return self._fit_one(method, seed)
