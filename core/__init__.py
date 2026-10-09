"""core — PrivForge 内核骨架：类型 / 错误 / 配置 / 接口 / 确定性种子。"""

from core.config import DEFAULT_CONFIG, SystemConfig, load_config
from core.errors import (
    ConfigError,
    DataError,
    ModelError,
    PipelineError,
    PrivacyError,
    PrivForgeError,
)
from core.seed import RngBundle, get_default, set_all
from core.types import (
    BenchmarkReport,
    Dataset,
    PrivacyBudget,
    TrainResult,
)

__all__ = [
    "DEFAULT_CONFIG",
    "SystemConfig",
    "load_config",
    "ConfigError",
    "DataError",
    "ModelError",
    "PipelineError",
    "PrivForgeError",
    "PrivacyError",
    "RngBundle",
    "get_default",
    "set_all",
    "BenchmarkReport",
    "Dataset",
    "PrivacyBudget",
    "TrainResult",
]
