# Changelog

## v0.1.0 (2026-10-09)
- 首发：PrivForge · 差分隐私机器学习系统
- 旗舰 `DPFuse`：DP-SGD + Rényi DP 矩会计 + 自适应梯度裁剪
- 基线：`NonPrivate`（效用上界）、`FixedClip-DP-SGD`（公认 SOTA 算法）
- 纯 NumPy 离线引擎（零下载、确定性可复现、CPU-only）
- Tier-0 复用：scikit-learn（非私有参考）、Optuna（可选 HPO）、SciPy（数学内核）
- ≥10 条可验证数学不变量门禁（RDP 可加性 / RDP→DP 转换 / 裁剪齐次性 / 灵敏度上界 / 隐私单调 / 确定性）
- 作者：晨星 (CJX0712)
