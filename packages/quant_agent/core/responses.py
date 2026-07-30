"""Standard tool response contracts."""

from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from quant_agent.core.errors import ErrorCode
from quant_agent.core.time import ensure_aware

T = TypeVar("T")


class Provenance(BaseModel):
    """Origin and version metadata for deterministic tool output."""

    model_config = ConfigDict(frozen=True)

    service: str
    version: str
    data_version: str | None = None


class ToolIssue(BaseModel):
    """Structured warning or error returned by a tool."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str


class ToolResponse[T](BaseModel):
    """Uniform response that keeps failures separate from empty data."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    request_id: str
    decision_id: str | None = None
    as_of: datetime
    data: T | None = None
    warnings: tuple[ToolIssue, ...] = ()
    errors: tuple[ToolIssue, ...] = ()
    provenance: Provenance

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: datetime) -> datetime:
        """Reject ambiguous naive timestamps at module boundaries."""

        return ensure_aware(value)

    def validate_consistency(self) -> "ToolResponse[T]":
        """Validate success/error consistency after model construction."""

        if self.ok and self.errors:
            raise ValueError("successful response cannot contain errors")
        if not self.ok and not self.errors:
            raise ValueError("failed response must contain at least one error")
        return self
