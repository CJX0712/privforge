"""PrivForge error hierarchy (architecture.md section 4.2).

Every exception carries a stable `code` attribute (E100 .. E502) so that the
report layer and CI can key on codes instead of on message text.
Author: 晨星
"""

from __future__ import annotations

from typing import Any


class PrivForgeError(Exception):
    """Root of the PrivForge error hierarchy."""

    code: str = "E000"
    default_message: str = "privforge error"

    def __init__(self, message: str = "", **context: Any) -> None:
        self.message = message or self.default_message
        self.context = dict(context)
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            extra = ", ".join(f"{k}={v}" for k, v in sorted(self.context.items()))
            return f"[{self.code}] {self.message} ({extra})"
        return f"[{self.code}] {self.message}"

    def as_dict(self) -> dict[str, Any]:
        """Serialisable view used by `eval/report.py`."""
        return {"code": self.code, "message": self.message, "context": self.context}


class ConfigError(PrivForgeError):
    """E100 - illegal configuration."""

    code = "E100"
    default_message = "illegal configuration"


class InputError(PrivForgeError):
    """E101 - illegal argument shape or type."""

    code = "E101"
    default_message = "illegal input"


class BudgetConfigError(PrivForgeError):
    """E102 - illegal epsilon/delta, or the three shares exceed epsilon."""

    code = "E102"
    default_message = "illegal privacy budget"


class DataError(PrivForgeError):
    """E200 - data cannot be parsed."""

    code = "E200"
    default_message = "data error"


class SplitError(PrivForgeError):
    """E201 - a split fold is empty or misses a class."""

    code = "E201"
    default_message = "illegal train/test split"


class BoundsError(PrivForgeError):
    """E202 - public bounds are missing. Hard fail, never degrade silently."""

    code = "E202"
    default_message = "public bounds are required"


class MechanismError(PrivForgeError):
    """E300 - illegal mechanism parameters."""

    code = "E300"
    default_message = "mechanism error"


class BackendUnavailable(PrivForgeError):
    """E301 - diffprivlib is unavailable; triggers a Tier-1 fallback."""

    code = "E301"
    default_message = "backend unavailable"


class NotFittedError(PrivForgeError):
    """E302 - predict before fit."""

    code = "E302"
    default_message = "model is not fitted"


class PrivacyLeakError(PrivForgeError):
    """E303 - a PrivacyLeakWarning was caught. Hard fail."""

    code = "E303"
    default_message = "privacy leak detected"


class BudgetExceeded(PrivForgeError):
    """E400 - the accountant spent more than the declared epsilon."""

    code = "E400"
    default_message = "privacy budget exceeded"


class CompositionError(PrivForgeError):
    """E401 - composition mismatch, or an ensemble split epsilon."""

    code = "E401"
    default_message = "illegal composition"


class AuditViolation(PrivForgeError):
    """E402 - empirical audit epsilon exceeded the declared epsilon."""

    code = "E402"
    default_message = "empirical privacy audit failed"


class EvalError(PrivForgeError):
    """E500 - evaluation protocol violation."""

    code = "E500"
    default_message = "evaluation error"


class PipelineError(PrivForgeError):
    """E501 - pipeline assembly failed."""

    code = "E501"
    default_message = "pipeline error"


class SafeguardError(PrivForgeError):
    """E502 - a safeguard fired and fell back; a usable model is returned."""

    code = "E502"
    default_message = "safeguard triggered a fallback"


ERROR_CODES: dict[str, type[PrivForgeError]] = {
    cls.code: cls
    for cls in (
        ConfigError,
        InputError,
        BudgetConfigError,
        DataError,
        SplitError,
        BoundsError,
        MechanismError,
        BackendUnavailable,
        NotFittedError,
        PrivacyLeakError,
        BudgetExceeded,
        CompositionError,
        AuditViolation,
        EvalError,
        PipelineError,
        SafeguardError,
    )
}

ALL_CODES: tuple[str, ...] = tuple(sorted(ERROR_CODES))


def raise_with_code(code: str, message: str = "", **context: Any) -> None:
    """Raise the exception registered for `code`."""
    cls = ERROR_CODES.get(code)
    if cls is None:
        raise KeyError(f"unknown privforge error code: {code}")
    raise cls(message, **context)
