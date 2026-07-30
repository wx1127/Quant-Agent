"""Shared error codes and typed application errors."""

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Stable machine-readable error codes."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    DATA_INVALID = "DATA_INVALID"
    RISK_REJECTED = "RISK_REJECTED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class QuantAgentError(Exception):
    """Base typed error that remains distinct from a valid empty result."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
