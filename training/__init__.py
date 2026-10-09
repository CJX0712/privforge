"""training — 可替换训练后端：非私有 / 固定裁剪 DP-SGD / 自适应裁剪 DPFuse。"""

from training.base import BaseTrainer, TrainConfig
from training.dpfuse import DPFuseTrainer
from training.dpsgd import FixedClipDPSGDTrainer
from training.nonprivate import NonPrivateTrainer

__all__ = [
    "BaseTrainer",
    "TrainConfig",
    "DPFuseTrainer",
    "FixedClipDPSGDTrainer",
    "NonPrivateTrainer",
]

# 方法注册表（pipeline 按名取用，离线兜底统一入口）。
TRAINERS = {
    "NonPrivate": NonPrivateTrainer,
    "FixedClip-DP-SGD": FixedClipDPSGDTrainer,
    "DPFuse": DPFuseTrainer,
}
