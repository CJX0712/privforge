"""core/errors.py — 错误码体系（E100~E500）。

每个模块的错误都映射到稳定错误码，便于文档化与测试断言。
"""

from __future__ import annotations


class PrivForgeError(Exception):
    """所有 PrivForge 异常的基类。"""

    code = "E000"
    message = "未分类错误"


class ConfigError(PrivForgeError):
    code = "E100"
    message = "配置错误"


class DataError(PrivForgeError):
    code = "E200"
    message = "数据错误"


class PrivacyError(PrivForgeError):
    code = "E300"
    message = "隐私预算/会计错误"


class ModelError(PrivForgeError):
    code = "E400"
    message = "模型/训练错误"


class PipelineError(PrivForgeError):
    code = "E500"
    message = "流水线错误"


def error_for_code(code: str) -> type[PrivForgeError]:
    mapping = {
        cls.code: cls for cls in (ConfigError, DataError, PrivacyError, ModelError, PipelineError)
    }
    return mapping.get(code, PrivForgeError)
