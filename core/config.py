"""core/config.py — 配置（环境变量覆盖 + schema 校验）。

所有配置项以 ``PRIVFORGE_*`` 环境变量为最高优先级，落地默认值后做类型/范围校验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

from core.errors import ConfigError


def _env(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if raw is None:
        return default
    # 类型推断：bool / int / float / str
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


@dataclass
class SystemConfig:
    """全局系统配置，可由环境变量覆盖。"""

    default_seed: int = 0
    clip_norm_default: float = _env("PRIVFORGE_CLIP_NORM", 1.0)
    noise_multiplier_default: float = _env("PRIVFORGE_SIGMA", 1.1)
    target_clip_fraction: float = _env("PRIVFORGE_TARGET_FRAC", 0.5)
    clip_lr: float = _env("PRIVFORGE_CLIP_LR", 0.2)
    delta_default: float = _env("PRIVFORGE_DELTA", 1e-5)

    def validate(self) -> None:
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (int, float)) and f.name != "default_seed":
                if val <= 0:
                    raise ConfigError(f"{f.name}={val} 必须为正（default_seed 除外）")
        if not (0.0 < self.target_clip_fraction < 1.0):
            raise ConfigError("target_clip_fraction 必须落在 (0,1)")
        if not (0.0 < self.delta_default < 1.0):
            raise ConfigError("delta_default 必须落在 (0,1)")


DEFAULT_CONFIG = SystemConfig()


def load_config() -> SystemConfig:
    cfg = SystemConfig()
    cfg.validate()
    return cfg
