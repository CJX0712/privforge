# SOTA 对标（State-of-the-Art）

## 本域公认 SOTA
| 组件 | 公认 SOTA | 来源 |
|------|-----------|------|
| 隐私训练算法 | DP-SGD（逐样本裁剪 + 高斯噪声 + 矩会计） | Abadi et al. 2016, "Deep Learning with Differential Privacy" |
| 隐私会计 | Rényi DP (RDP) 矩会计 | Mironov 2017, "Rényi Differential Privacy" |
| 裁剪改进 | 自适应梯度裁剪（target clipping fraction） | Andrew et al. 2021, "Differentially Private Learning with Adaptive Clipping" |

PrivForge **忠实实现上述 SOTA 算法**（非从零自研新 SOTA），并在其上以 `DPFuse`
集成自适应裁剪这一已发表改进。

## 对标参考实现
- **Opacus**（Meta，PyTorch）：工业级 DP-SGD 实现。PrivForge 与其算法等价，
  但提供 CPU-only、零下载、确定性可复现的纯 NumPy 引擎，便于教学、审计与合规复现。
- **TensorFlow Privacy**：同类 RDP 会计。本系统的 RDP→(ε,δ) 转换与其公式一致。

## 本系统做到的程度
- 在**相同隐私预算 (ε,δ)**（相同 σ 与步数）下，`DPFuse`（自适应裁剪）相较
  `FixedClip-DP-SGD`（固定裁剪，即 SOTA 算法原版）测试准确率 **+0.33pp（显著性判定通过）**
  （详见 README 基准表，数字来自真实运行）。
- **隐私与裁剪界无关**得到实证：`FixedClip` 与 `DPFuse` 的 ε 逐位相等（133.03）。
- 数学不变量（RDP 可加性、RDP→DP 转换、裁剪尺度等价、灵敏度上界、梯度检验、确定性）
  全部以单元测试门禁（23 测试全绿）。
- 复用世界顶级开源：scikit-learn（非私有参考后端）、Optuna（可选 HPO）、SciPy（数学内核）。
- 不声称超越 Opacus 的工程吞吐；定位是**可验证、可复现、可审计**的世界级研究/教学级实现。
