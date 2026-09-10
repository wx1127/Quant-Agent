"""Immutable contracts for the Agent input security boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_agent.agent.content_security import ContentDecision
from quant_agent.agent.runtime import AgentRunGoal, AgentRunState
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware

SECURITY_SCHEMA_VERSION: Final = "1"
SECURITY_POLICY_VERSION: Final = "agent-security-policy-v1"
_SHA256 = r"^[0-9a-f]{64}$"
_STRICT = ConfigDict(
    extra="forbid", frozen=True, revalidate_instances="always", strict=True, validate_default=True
)
Identifier = Annotated[str, Field(min_length=1, max_length=256)]


class InputDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    ACCEPTED_WITH_REDACTIONS = "ACCEPTED_WITH_REDACTIONS"
    REJECTED = "REJECTED"


class SecuritySource(StrEnum):
    USER = "USER"
    EXTERNAL_CONTENT = "EXTERNAL_CONTENT"
    SYSTEM = "SYSTEM"


class SecurityFindingCode(StrEnum):
    USER_MODE_OVERRIDE = "USER_MODE_OVERRIDE"
    USER_PRIVILEGE_ESCALATION = "USER_PRIVILEGE_ESCALATION"
    USER_INSTRUCTION_OVERRIDE = "USER_INSTRUCTION_OVERRIDE"
    USER_SECRET_REQUEST = "USER_SECRET_REQUEST"
    USER_SECRET_REDACTED = "USER_SECRET_REDACTED"
    EXTERNAL_CONTENT_QUARANTINED = "EXTERNAL_CONTENT_QUARANTINED"
    EXTERNAL_CONTENT_REDACTED = "EXTERNAL_CONTENT_REDACTED"
    EXECUTION_TOOLS_SUPPRESSED = "EXECUTION_TOOLS_SUPPRESSED"


class SecurityFinding(BaseModel):
    """Text-free security finding; source content is never copied into it."""

    model_config = _STRICT

    code: SecurityFindingCode
    source: SecuritySource
    content_id: Identifier


class ExternalContentInput(BaseModel):
    """Ephemeral caller input; raw text is excluded from serialization and repr."""

    model_config = _STRICT

    source_id: Identifier
    raw_payload_id: Annotated[int, Field(gt=0, le=1_000_000_000)]
    text: Annotated[str, Field(min_length=1, max_length=200_000)] = Field(
        repr=False, exclude=True
    )

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("source_id must be printable and trimmed")
        return value


class UntrustedContentBlock(BaseModel):
    """Data-only content that cannot carry tool authority."""

    model_config = _STRICT

    source_id: Identifier
    raw_payload_id: Annotated[int, Field(gt=0)]
    source_content_hash: Annotated[str, Field(pattern=_SHA256)]
    sanitizer_result_hash: Annotated[str, Field(pattern=_SHA256)]
    decision: ContentDecision
    safe_text: Annotated[str, Field(min_length=1, max_length=200_000)] | None = None
    instruction_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_safe_text(self) -> Self:
        if self.decision is ContentDecision.ACCEPTED and not self.safe_text:
            raise ValueError("accepted external content requires safe_text")
        if self.decision is not ContentDecision.ACCEPTED and self.safe_text is not None:
            raise ValueError("quarantined or rejected content cannot carry safe_text")
        return self


class AgentInputEnvelope(BaseModel):
    """Server-built model context with mode and tools outside user control."""

    model_config = _STRICT

    schema_version: Literal["1"] = SECURITY_SCHEMA_VERSION
    policy_version: Literal["agent-security-policy-v1"] = SECURITY_POLICY_VERSION
    run_id: Identifier
    decision_id: Identifier
    runtime_mode: RuntimeMode
    goal: AgentRunGoal
    run_state: AgentRunState
    user_request_hash: Annotated[str, Field(pattern=_SHA256)]
    user_intent: Annotated[str, Field(min_length=1, max_length=20_000)] | None = None
    user_disposition: InputDisposition
    external_content: tuple[UntrustedContentBlock, ...] = ()
    available_tool_names: tuple[Identifier, ...] = ()
    findings: tuple[SecurityFinding, ...] = ()
    prepared_at: datetime
    envelope_hash: Annotated[str, Field(pattern=_SHA256)]

    @field_validator("prepared_at")
    @classmethod
    def normalize_prepared_at(cls, value: datetime) -> datetime:
        return ensure_aware(value).astimezone(UTC)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if tuple(sorted(set(self.available_tool_names))) != self.available_tool_names:
            raise ValueError("available_tool_names must be unique and sorted")
        if self.user_disposition is InputDisposition.REJECTED and self.user_intent is not None:
            raise ValueError("rejected user input cannot carry user_intent")
        if self.user_disposition is not InputDisposition.REJECTED and self.user_intent is None:
            raise ValueError("accepted user input requires user_intent")
        expected = stable_security_hash(self._identity_payload())
        if self.envelope_hash != expected:
            raise ValueError("envelope_hash does not match the complete envelope")
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.model_dump(mode="python").items()
            if key != "envelope_hash"
        }


def stable_security_hash(value: object) -> str:
    """Hash JSON-compatible security identities deterministically."""

    def normalize(item: object) -> object:
        if isinstance(item, BaseModel):
            return normalize(item.model_dump(mode="python"))
        if isinstance(item, datetime):
            return ensure_aware(item).astimezone(UTC).isoformat(timespec="microseconds")
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        return item

    encoded = json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "SECURITY_POLICY_VERSION",
    "SECURITY_SCHEMA_VERSION",
    "AgentInputEnvelope",
    "ExternalContentInput",
    "InputDisposition",
    "SecurityFinding",
    "SecurityFindingCode",
    "SecuritySource",
    "UntrustedContentBlock",
    "stable_security_hash",
]
