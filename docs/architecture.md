# 架构（Architecture）

PrivForge 采用单向无环调用骨架：`cli → pipeline → {data, preprocess, dp, models, training, eval} → core`。

```
privforge/
├── core/        类型 / 错误码(E100~E500) / 配置(ENV_* 覆盖+schema 校验)
│                接口契约(Protocol) / 全局确定性种子(set_all)
├── data/        确定性合成分类 DGP（难度旋钮 separation）+ 数据源接口
├── preprocess/  标准化（scaler 只在 train 上 fit，防泄漏）
├── dp/          隐私引擎：RDP 矩会计 / 逐样本裁剪 / 高斯机制
├── models/      纯 NumPy MLP（手写前向/反向，支持逐样本梯度）
├── training/    非私有 / 固定裁剪 DP-SGD / 自适应裁剪 DPFuse + sklearn 参考
├── hpo/         可选超参优化（Optuna；不可用时自动降级）
├── eval/        评估指标（summary / excess loss / 显著性）
├── pipeline/    PrivForgePipeline.run() 编排 + benchmark() + 失败案例 + 消融
├── cli.py       argparse 入口（benchmark / train / selftest）
└── examples/run_demo.py  端到端演示（落盘 benchmark.json + 确定性二次校验）
```

## 调用关系
- 每个模块只经 `core.interfaces` 的 Protocol 通信，可独立替换/验证。
- 训练器统一在 `training.base.BaseTrainer`，三种后端仅差「是否注入噪声」「裁剪界是否自适应」。
- 隐私开销只由 `(σ, steps)` 决定（见 `dp/accountant.py`），与裁剪界无关——这是 DPFuse 能在**相同 (ε,δ)** 下比较效用的理论基石。

## 确定性
唯一入口 `core.seed.set_all(seed)` 设齐 numpy / random / torch(可选)；每个训练 fit 内部重置 RNG，故同 seed 两次运行核心指标逐位一致（elapsed 除外）。
