"""Strict, content-addressed contracts for the explicit Agent runtime."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from typing import Annotated, Final, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from quant_agent.agent.tools.contracts import ToolEffect
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware

AGENT_RUNTIME_SCHEMA_VERSION: Final = "1"
AGENT_RUNTIME_VERSION: Final = "agent-runtime-v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
_TOOL_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", re.ASCII)
_MAX_JSON_BYTES: Final = 8 * 1024 * 1024
_STRICT_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    revalidate_instances="always",
    strict=True,
    validate_default=True,
)

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
ShortText = Annotated[str, Field(min_length=1, max_length=512)]
Sha256 = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
SequenceNumber = Annotated[int, Field(ge=0, le=10_000_000)]
Counter = Annotated[int, Field(ge=0, le=1_000_000)]
PositiveLimit = Annotated[int, Field(ge=1, le=1_000_000)]


def _exact_text(value: str, field_name: str, *, maximum: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    if not value or value != value.strip() or len(value) > maximum or not value.isprintable():
        raise ValueError(
            f"{field_name} must be non-empty, trimmed, printable, and at most {maximum} chars"
        )
    return value


def _identifier(value: str, field_name: str) -> str:
    return _exact_text(value, field_name, maximum=256)


def _sha256(value: str, field_name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _tool_name(value: str) -> str:
    if type(value) is not str or len(value) > 64 or _TOOL_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("tool_name must be a lower-ASCII snake-case identifier")
    return value


def _utc(value: datetime) -> datetime:
    return ensure_aware(value).astimezone(UTC)


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        return _utc(value).isoformat(timespec="microseconds")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    raise TypeError(f"unsupported canonical runtime value: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _validated_json(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} JSON must be an exact string")
    if not value or len(value.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} JSON is empty or exceeds the size limit")
    try:
        parsed: object = json.loads(value, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dump_model(model: BaseModel, adapter: TypeAdapter[object]) -> str:
    parsed = json.loads(adapter.dump_json(model, warnings="error"))
    return json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class AgentRunGoal(StrEnum):
    """Server-selected workflow profile; the model cannot expand it at runtime."""

    MARKET_RESEARCH = "MARKET_RESEARCH"
    BACKTEST_REPORT = "BACKTEST_REPORT"
    PORTFOLIO_REPORT = "PORTFOLIO_REPORT"
    ORDER_DRAFT = "ORDER_DRAFT"
    PAPER_EXECUTION = "PAPER_EXECUTION"
    LIVE_ASSISTED_EXECUTION = "LIVE_ASSISTED_EXECUTION"


class AgentRunState(StrEnum):
    """Persistent phases and distinct terminal outcomes for one Agent run."""

    RECEIVED = "RECEIVED"
    SNAPSHOT_READY = "SNAPSHOT_READY"
    DATA_VALIDATED = "DATA_VALIDATED"
    ANALYZED = "ANALYZED"
    PORTFOLIO_READY = "PORTFOLIO_READY"
    RISK_CHECKED = "RISK_CHECKED"
    DRAFT_READY = "DRAFT_READY"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    RECONCILING = "RECONCILING"

    REPORTED = "REPORTED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    DATA_INVALID = "DATA_INVALID"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    FAILED = "FAILED"
    INCIDENT = "INCIDENT"


TERMINAL_AGENT_RUN_STATES: Final[frozenset[AgentRunState]] = frozenset(
    {
        AgentRunState.REPORTED,
        AgentRunState.COMPLETED,
        AgentRunState.REJECTED,
        AgentRunState.DATA_INVALID,
        AgentRunState.EXPIRED,
        AgentRunState.CANCELLED,
        AgentRunState.TIMED_OUT,
        AgentRunState.LIMIT_EXCEEDED,
        AgentRunState.FAILED,
        AgentRunState.INCIDENT,
    }
)


class AgentRunEventKind(StrEnum):
    """Append-only runtime event types, including same-state observations."""

    RUN_CREATED = "RUN_CREATED"
    STATE_TRANSITION = "STATE_TRANSITION"
    TOOL_CALL_RESERVED = "TOOL_CALL_RESERVED"
    TOOL_CALL_SUCCEEDED = "TOOL_CALL_SUCCEEDED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    TOOL_CALL_REPLAYED = "TOOL_CALL_REPLAYED"
    TOOL_CALL_REJECTED = "TOOL_CALL_REJECTED"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    APPROVAL_RECORDED = "APPROVAL_RECORDED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    TIMEOUT_RECORDED = "TIMEOUT_RECORDED"
    LIMIT_REACHED = "LIMIT_REACHED"
    INCIDENT_RECORDED = "INCIDENT_RECORDED"


class AgentRunTrigger(StrEnum):
    """Trusted source responsible for a runtime event."""

    SYSTEM = "SYSTEM"
    TOOL = "TOOL"
    HUMAN = "HUMAN"
    TIMEOUT = "TIMEOUT"
    CANCELLATION = "CANCELLATION"
    KILL_SWITCH = "KILL_SWITCH"
    RECOVERY = "RECOVERY"


class AgentArtifactKind(StrEnum):
    """Typed identities retained from validated tool responses."""

    MARKET_SNAPSHOT = "MARKET_SNAPSHOT"
    DATA_QUALITY = "DATA_QUALITY"
    MARKET_REGIME = "MARKET_REGIME"
    MARKET_THEMES = "MARKET_THEMES"
    THEME_LEADERS = "THEME_LEADERS"
    STOCK_CANDIDATES = "STOCK_CANDIDATES"
    CANDIDATE_EXPLANATION = "CANDIDATE_EXPLANATION"
    BACKTEST_REPORT = "BACKTEST_REPORT"
    PORTFOLIO_SNAPSHOT = "PORTFOLIO_SNAPSHOT"
    TARGET_PORTFOLIO = "TARGET_PORTFOLIO"
    RISK_CHECK = "RISK_CHECK"
    ORDER_DRAFT = "ORDER_DRAFT"
    EXECUTION_RECEIPT = "EXECUTION_RECEIPT"
    RECONCILIATION = "RECONCILIATION"
    DECISION_REPORT = "DECISION_REPORT"


class AgentRunLimits(BaseModel):
    """Server-owned resource and deadline limits for one workflow."""

    model_config = _STRICT_MODEL_CONFIG

    max_tool_calls: Annotated[int, Field(ge=1, le=1_000)] = 32
    max_invalid_tool_calls: Annotated[int, Field(ge=1, le=100)] = 3
    max_reconciliation_attempts: Annotated[int, Field(ge=1, le=100)] = 5
    timeout_seconds: Annotated[int, Field(ge=1, le=604_800)] = 300
    approval_timeout_seconds: Annotated[int, Field(ge=1, le=86_400)] = 900

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.max_invalid_tool_calls > self.max_tool_calls:
            raise ValueError("max_invalid_tool_calls cannot exceed max_tool_calls")
        return self


class AgentArtifactRef(BaseModel):
    """Immutable domain and response hashes pinned to a completed runtime step."""

    model_config = _STRICT_MODEL_CONFIG

    kind: AgentArtifactKind
    artifact_id: Identifier
    content_hash: Sha256
    tool_name: Annotated[str, Field(min_length=1, max_length=64)]
    response_hash: Sha256
    truncated: bool = False
    expires_at: datetime | None = None

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return _identifier(value, "artifact_id")

    @field_validator("content_hash", "response_hash")
    @classmethod
    def validate_hashes(cls, value: str, info: object) -> str:
        return _sha256(value, getattr(info, "field_name", "hash"))

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _tool_name(value)

    @field_validator("expires_at")
    @classmethod
    def normalize_expiry(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def validate_expiry_scope(self) -> Self:
        if self.expires_at is not None and self.kind is not AgentArtifactKind.ORDER_DRAFT:
            raise ValueError("only an ORDER_DRAFT artifact may carry expires_at")
        return self


class AgentPendingToolCall(BaseModel):
    """One atomically reserved tool call whose outcome is not yet recorded."""

    model_config = _STRICT_MODEL_CONFIG

    request_id: Identifier
    tool_name: Annotated[str, Field(min_length=1, max_length=64)]
    argument_hash: Sha256
    idempotency_key_hash: Sha256 | None = None
    batch_hash: Sha256 | None = None
    started_at: datetime
    effect: ToolEffect

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _identifier(value, "request_id")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _tool_name(value)

    @field_validator("argument_hash", "idempotency_key_hash", "batch_hash")
    @classmethod
    def validate_hashes(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _sha256(value, getattr(info, "field_name", "hash"))

    @field_validator("started_at")
    @classmethod
    def normalize_started_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_write_identity(self) -> Self:
        if (
            self.effect
            in {
                ToolEffect.ARTIFACT_WRITE,
                ToolEffect.PAPER_EXECUTION_WRITE,
                ToolEffect.LIVE_EXTERNAL_WRITE,
            }
            and self.idempotency_key_hash is None
        ):
            raise ValueError("a write call reservation requires an idempotency-key hash")
        if (
            self.effect
            in {
                ToolEffect.PAPER_EXECUTION_WRITE,
                ToolEffect.LIVE_EXTERNAL_WRITE,
            }
            and self.batch_hash is None
        ):
            raise ValueError("an execution call reservation requires a batch hash")
        return self


class AgentExternalApproval(BaseModel):
    """Trusted control-plane evidence; approval secrets never enter this contract."""

    model_config = _STRICT_MODEL_CONFIG

    approval_id: Identifier
    approval_hash: Sha256
    decision_id: Identifier
    batch_hash: Sha256
    approved_at: datetime
    expires_at: datetime

    @field_validator("approval_id", "decision_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("approval_hash", "batch_hash")
    @classmethod
    def validate_hashes(cls, value: str, info: object) -> str:
        return _sha256(value, getattr(info, "field_name", "hash"))

    @field_validator("approved_at", "expires_at")
    @classmethod
    def normalize_times(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.expires_at <= self.approved_at:
            raise ValueError("external approval must expire after it was approved")
        return self


_TOOL_EVENT_KINDS: Final[frozenset[AgentRunEventKind]] = frozenset(
    {
        AgentRunEventKind.TOOL_CALL_RESERVED,
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
    }
)

_TOOL_OUTCOME_KINDS: Final[frozenset[AgentRunEventKind]] = frozenset(
    _TOOL_EVENT_KINDS - {AgentRunEventKind.TOOL_CALL_RESERVED}
)

_CONTROL_HASH_EVENT_KINDS: Final[frozenset[AgentRunEventKind]] = frozenset(
    {
        AgentRunEventKind.APPROVAL_RECORDED,
        AgentRunEventKind.APPROVAL_REJECTED,
    }
)


class AgentRunEvent(BaseModel):
    """One immutable, hash-chained runtime observation or state transition."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["1"] = AGENT_RUNTIME_SCHEMA_VERSION
    runtime_version: ShortText = AGENT_RUNTIME_VERSION
    run_id: Identifier
    decision_id: Identifier
    decision_snapshot_hash: Sha256
    runtime_mode: RuntimeMode
    goal: AgentRunGoal
    sequence: SequenceNumber
    kind: AgentRunEventKind
    trigger: AgentRunTrigger
    state_before: AgentRunState
    state_after: AgentRunState
    reason: ShortText
    occurred_at: datetime
    request_id: Identifier | None = None
    tool_name: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    tool_effect: ToolEffect | None = None
    argument_hash: Sha256 | None = None
    response_hash: Sha256 | None = None
    idempotency_key_hash: Sha256 | None = None
    batch_hash: Sha256 | None = None
    control_hash: Sha256 | None = None
    external_approval: AgentExternalApproval | None = None
    run_limits: AgentRunLimits | None = None
    artifact: AgentArtifactRef | None = None
    warning_codes: Annotated[tuple[ShortText, ...], Field(max_length=128)] = ()
    error_codes: Annotated[tuple[ShortText, ...], Field(max_length=128)] = ()
    previous_event_hash: Sha256 | None = None
    event_hash: Sha256

    @field_validator("runtime_version", "reason")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _exact_text(value, getattr(info, "field_name", "text"), maximum=512)

    @field_validator("run_id", "decision_id", "request_id")
    @classmethod
    def validate_identifiers(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator(
        "decision_snapshot_hash",
        "argument_hash",
        "response_hash",
        "idempotency_key_hash",
        "batch_hash",
        "control_hash",
        "previous_event_hash",
        "event_hash",
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _sha256(value, getattr(info, "field_name", "hash"))

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str | None) -> str | None:
        return None if value is None else _tool_name(value)

    @field_validator("warning_codes", "error_codes")
    @classmethod
    def validate_codes(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        if type(value) is not tuple:
            raise TypeError(f"{getattr(info, 'field_name', 'codes')} must be an exact tuple")
        result = tuple(
            _exact_text(item, getattr(info, "field_name", "code"), maximum=512) for item in value
        )
        if len(set(result)) != len(result):
            raise ValueError(f"{getattr(info, 'field_name', 'codes')} must be unique")
        return result

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if (self.sequence == 0) != (self.previous_event_hash is None):
            raise ValueError("only sequence zero may omit previous_event_hash")
        if (self.sequence == 0) != (self.kind is AgentRunEventKind.RUN_CREATED):
            raise ValueError("only sequence zero may be RUN_CREATED")
        if self.kind is AgentRunEventKind.RUN_CREATED and (
            self.trigger is not AgentRunTrigger.SYSTEM
            or self.state_before is not AgentRunState.RECEIVED
            or self.state_after is not AgentRunState.RECEIVED
        ):
            raise ValueError("RUN_CREATED must be a SYSTEM event in RECEIVED state")
        if (
            self.kind is AgentRunEventKind.STATE_TRANSITION
            and self.state_before is self.state_after
        ):
            raise ValueError("STATE_TRANSITION cannot be a self-loop")
        tool_fields = (
            self.tool_name,
            self.tool_effect,
            self.argument_hash,
            self.response_hash,
            self.idempotency_key_hash,
            self.artifact,
        )
        if self.kind in _TOOL_EVENT_KINDS:
            if (
                self.request_id is None
                or self.tool_name is None
                or self.tool_effect is None
                or self.argument_hash is None
            ):
                raise ValueError("tool events require request, tool, effect, and argument identity")
        elif self.request_id is not None or any(item is not None for item in tool_fields):
            raise ValueError("non-tool events cannot carry tool-call fields")
        if self.batch_hash is not None and self.kind not in {
            *_TOOL_EVENT_KINDS,
            *_CONTROL_HASH_EVENT_KINDS,
        }:
            raise ValueError("only tool or approval-control events may carry a batch hash")
        if (self.warning_codes or self.error_codes) and self.kind not in _TOOL_OUTCOME_KINDS:
            raise ValueError("only tool outcomes may carry warning or error codes")
        if (self.control_hash is not None) != (self.kind in _CONTROL_HASH_EVENT_KINDS):
            raise ValueError("approval control events require exactly one trusted control hash")
        if self.kind is AgentRunEventKind.APPROVAL_RECORDED:
            approval = self.external_approval
            if type(approval) is not AgentExternalApproval:
                raise ValueError("recorded approval requires complete trusted approval evidence")
            if (
                approval.approval_hash != self.control_hash
                or approval.decision_id != self.decision_id
                or approval.batch_hash != self.batch_hash
                or approval.approved_at > self.occurred_at
                or approval.expires_at <= self.occurred_at
            ):
                raise ValueError("recorded approval evidence does not bind the event")
        elif self.external_approval is not None:
            raise ValueError("only APPROVAL_RECORDED may carry external approval evidence")
        if (self.run_limits is not None) != (self.kind is AgentRunEventKind.RUN_CREATED):
            raise ValueError("only RUN_CREATED may carry the immutable run limits")
        if self.kind is AgentRunEventKind.TOOL_CALL_RESERVED:
            if self.response_hash is not None or self.artifact is not None:
                raise ValueError("a reserved call cannot already carry a response or artifact")
            if self.warning_codes or self.error_codes:
                raise ValueError("a reserved call cannot already carry outcome codes")
        if self.kind in {
            AgentRunEventKind.TOOL_CALL_SUCCEEDED,
            AgentRunEventKind.TOOL_CALL_REPLAYED,
        } and (self.response_hash is None or self.error_codes):
            raise ValueError("successful or replayed calls require a response and no errors")
        if (
            self.kind
            in {
                AgentRunEventKind.TOOL_CALL_FAILED,
                AgentRunEventKind.TOOL_CALL_REJECTED,
            }
            and not self.error_codes
        ):
            raise ValueError("failed or rejected calls require at least one error code")
        if self.artifact is not None:
            if self.kind not in {
                AgentRunEventKind.TOOL_CALL_SUCCEEDED,
                AgentRunEventKind.TOOL_CALL_REPLAYED,
            }:
                raise ValueError("only successful or replayed calls may carry an artifact")
            if self.artifact.tool_name != self.tool_name:
                raise ValueError("event artifact must belong to the event tool")
            if self.artifact.response_hash != self.response_hash:
                raise ValueError("event artifact must bind the exact response hash")
        expected = _stable_hash(self._content_payload())
        if self.event_hash != expected:
            raise ValueError("event_hash does not match complete event content")
        return self

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        decision_id: str,
        decision_snapshot_hash: str,
        runtime_mode: RuntimeMode,
        goal: AgentRunGoal,
        sequence: int,
        kind: AgentRunEventKind,
        trigger: AgentRunTrigger,
        state_before: AgentRunState,
        state_after: AgentRunState,
        reason: str,
        occurred_at: datetime,
        request_id: str | None = None,
        tool_name: str | None = None,
        tool_effect: ToolEffect | None = None,
        argument_hash: str | None = None,
        response_hash: str | None = None,
        idempotency_key_hash: str | None = None,
        batch_hash: str | None = None,
        control_hash: str | None = None,
        external_approval: AgentExternalApproval | None = None,
        run_limits: AgentRunLimits | None = None,
        artifact: AgentArtifactRef | None = None,
        warning_codes: tuple[str, ...] = (),
        error_codes: tuple[str, ...] = (),
        previous_event_hash: str | None = None,
        runtime_version: str = AGENT_RUNTIME_VERSION,
    ) -> AgentRunEvent:
        values: dict[str, object] = {
            "schema_version": AGENT_RUNTIME_SCHEMA_VERSION,
            "runtime_version": runtime_version,
            "run_id": run_id,
            "decision_id": decision_id,
            "decision_snapshot_hash": decision_snapshot_hash,
            "runtime_mode": runtime_mode,
            "goal": goal,
            "sequence": sequence,
            "kind": kind,
            "trigger": trigger,
            "state_before": state_before,
            "state_after": state_after,
            "reason": reason,
            "occurred_at": occurred_at,
            "request_id": request_id,
            "tool_name": tool_name,
            "tool_effect": tool_effect,
            "argument_hash": argument_hash,
            "response_hash": response_hash,
            "idempotency_key_hash": idempotency_key_hash,
            "batch_hash": batch_hash,
            "control_hash": control_hash,
            "external_approval": external_approval,
            "run_limits": run_limits,
            "artifact": artifact,
            "warning_codes": warning_codes,
            "error_codes": error_codes,
            "previous_event_hash": previous_event_hash,
        }
        return cls.model_validate({**values, "event_hash": _stable_hash(values)}, strict=True)

    def _content_payload(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            self.model_dump(mode="python", exclude={"event_hash"}),
        )

    def to_json(self) -> str:
        return _dump_model(self, cast(TypeAdapter[object], _EVENT_ADAPTER))

    @classmethod
    def from_json(cls, value: str) -> AgentRunEvent:
        parsed = _EVENT_ADAPTER.validate_json(
            _validated_json(value, label="AgentRunEvent"), strict=True
        )
        if type(parsed) is not cls:
            raise TypeError("AgentRunEvent JSON did not produce the exact contract type")
        return parsed


_GOAL_MODES: Final[dict[AgentRunGoal, frozenset[RuntimeMode]]] = {
    AgentRunGoal.MARKET_RESEARCH: frozenset(
        {
            RuntimeMode.RESEARCH,
            RuntimeMode.BACKTEST,
            RuntimeMode.PAPER,
            RuntimeMode.LIVE_ASSISTED,
        }
    ),
    AgentRunGoal.BACKTEST_REPORT: frozenset({RuntimeMode.BACKTEST}),
    AgentRunGoal.PORTFOLIO_REPORT: frozenset({RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED}),
    AgentRunGoal.ORDER_DRAFT: frozenset({RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED}),
    AgentRunGoal.PAPER_EXECUTION: frozenset({RuntimeMode.PAPER}),
    AgentRunGoal.LIVE_ASSISTED_EXECUTION: frozenset({RuntimeMode.LIVE_ASSISTED}),
}

_EXECUTION_GOALS: Final[frozenset[AgentRunGoal]] = frozenset(
    {AgentRunGoal.PAPER_EXECUTION, AgentRunGoal.LIVE_ASSISTED_EXECUTION}
)


class AgentRunSnapshot(BaseModel):
    """Complete immutable state obtained by folding one run's event chain."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["1"] = AGENT_RUNTIME_SCHEMA_VERSION
    runtime_version: ShortText = AGENT_RUNTIME_VERSION
    run_id: Identifier
    decision_id: Identifier
    decision_snapshot_hash: Sha256
    runtime_mode: RuntimeMode
    goal: AgentRunGoal
    state: AgentRunState
    created_at: datetime
    updated_at: datetime
    deadline: datetime
    approval_deadline: datetime | None = None
    completed_at: datetime | None = None
    terminal_reason: ShortText | None = None
    limits: AgentRunLimits
    revision: SequenceNumber
    tool_call_count: Counter = 0
    invalid_tool_call_count: Counter = 0
    reconciliation_attempt_count: Counter = 0
    cancel_requested: bool = False
    artifacts: Annotated[tuple[AgentArtifactRef, ...], Field(max_length=10_000)] = ()
    pending_call: AgentPendingToolCall | None = None
    events: Annotated[tuple[AgentRunEvent, ...], Field(min_length=1, max_length=1_000_001)]

    @field_validator("runtime_version", "terminal_reason")
    @classmethod
    def validate_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _exact_text(value, getattr(info, "field_name", "text"), maximum=512)

    @field_validator("run_id", "decision_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("decision_snapshot_hash")
    @classmethod
    def validate_decision_hash(cls, value: str) -> str:
        return _sha256(value, "decision_snapshot_hash")

    @field_validator(
        "created_at",
        "updated_at",
        "deadline",
        "approval_deadline",
        "completed_at",
    )
    @classmethod
    def normalize_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @field_validator("artifacts", "events")
    @classmethod
    def require_exact_tuples(cls, value: tuple[object, ...], info: object) -> tuple[object, ...]:
        if type(value) is not tuple:
            raise TypeError(f"{getattr(info, 'field_name', 'collection')} must be an exact tuple")
        return value

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.runtime_mode not in _GOAL_MODES[self.goal]:
            raise ValueError("run goal is not permitted in the fixed runtime mode")
        if self.deadline != self.created_at + timedelta(seconds=self.limits.timeout_seconds):
            raise ValueError("deadline must derive exactly from created_at and timeout_seconds")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.approval_deadline is not None and not (
            self.created_at < self.approval_deadline <= self.deadline
        ):
            raise ValueError("approval_deadline must fall inside the run window")
        terminal = self.state in TERMINAL_AGENT_RUN_STATES
        if terminal and (self.completed_at is None or self.terminal_reason is None):
            raise ValueError("terminal runs require completion time and reason")
        if not terminal and (self.completed_at is not None or self.terminal_reason is not None):
            raise ValueError("non-terminal runs cannot carry completion fields")
        if self.completed_at is not None and self.completed_at != self.updated_at:
            raise ValueError("completed_at must equal the final update time")
        if terminal and self.pending_call is not None:
            raise ValueError("terminal runs cannot retain a pending tool call")
        if self.state in {AgentRunState.PENDING_APPROVAL, AgentRunState.APPROVED}:
            if self.runtime_mode is not RuntimeMode.LIVE_ASSISTED:
                raise ValueError("approval states require LIVE_ASSISTED mode")
            if self.approval_deadline is None:
                raise ValueError("approval states require an approval_deadline")
        if (
            self.state
            in {
                AgentRunState.EXECUTING,
                AgentRunState.RECONCILING,
                AgentRunState.COMPLETED,
            }
            and self.goal not in _EXECUTION_GOALS
        ):
            raise ValueError("execution states require an execution run goal")
        if self.state is AgentRunState.REPORTED and self.goal in _EXECUTION_GOALS:
            raise ValueError("an execution goal cannot finish as a report-only success")
        if self.tool_call_count - self.reconciliation_attempt_count > self.limits.max_tool_calls:
            raise ValueError("non-reconciliation tool_call_count exceeds the run limit")
        if self.invalid_tool_call_count > self.limits.max_invalid_tool_calls:
            raise ValueError("invalid_tool_call_count exceeds the run limit")
        if self.reconciliation_attempt_count > self.limits.max_reconciliation_attempts:
            raise ValueError("reconciliation_attempt_count exceeds the run limit")

        events = self.events
        if self.revision != len(events) - 1 or events[-1].sequence != self.revision:
            raise ValueError("revision must equal the last contiguous event sequence")
        first = events[0]
        if (
            first.sequence != 0
            or first.kind is not AgentRunEventKind.RUN_CREATED
            or first.state_before is not AgentRunState.RECEIVED
            or first.state_after is not AgentRunState.RECEIVED
            or first.previous_event_hash is not None
            or first.run_limits != self.limits
        ):
            raise ValueError(
                "event chain must start with a RECEIVED RUN_CREATED event carrying run limits"
            )
        if self.created_at != first.occurred_at or self.updated_at != events[-1].occurred_at:
            raise ValueError("snapshot times must bind the first and last runtime events")
        previous: AgentRunEvent | None = None
        event_artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef] = {}
        reserved_events: list[AgentRunEvent] = []
        for index, event in enumerate(events):
            if (
                event.sequence != index
                or event.run_id != self.run_id
                or event.decision_id != self.decision_id
                or event.decision_snapshot_hash != self.decision_snapshot_hash
                or event.runtime_mode is not self.runtime_mode
                or event.goal is not self.goal
                or event.runtime_version != self.runtime_version
            ):
                raise ValueError("runtime event identity or sequence drifted from its run")
            if previous is not None and (
                event.previous_event_hash != previous.event_hash
                or event.state_before is not previous.state_after
                or event.occurred_at < previous.occurred_at
            ):
                raise ValueError("runtime event chain is discontinuous")
            if event.kind is AgentRunEventKind.TOOL_CALL_RESERVED:
                reserved_events.append(event)
            if event.artifact is not None:
                key = (event.artifact.kind, event.artifact.artifact_id)
                existing = event_artifacts.get(key)
                if existing is not None and existing != event.artifact:
                    raise ValueError("an artifact identity cannot bind different content")
                event_artifacts[key] = event.artifact
            previous = event
        if events[-1].state_after is not self.state:
            raise ValueError("snapshot state must equal the final event state")
        if self.tool_call_count != len(reserved_events):
            raise ValueError("tool_call_count must equal reserved tool-call events")
        observed_reconciliation_attempts = sum(
            event.tool_name == "reconcile_account" for event in reserved_events
        )
        if self.reconciliation_attempt_count != observed_reconciliation_attempts:
            raise ValueError("reconciliation count must equal reserved reconciliation calls")
        observed_invalid_calls = sum(
            event.kind is AgentRunEventKind.TOOL_CALL_REJECTED for event in events
        )
        if self.invalid_tool_call_count != observed_invalid_calls:
            raise ValueError("invalid-call count must equal rejected tool-call outcomes")

        artifact_map: dict[tuple[AgentArtifactKind, str], AgentArtifactRef] = {}
        for artifact in self.artifacts:
            key = (artifact.kind, artifact.artifact_id)
            if key in artifact_map:
                raise ValueError("snapshot artifacts must be unique by kind and identifier")
            artifact_map[key] = artifact
        if artifact_map != event_artifacts:
            raise ValueError("snapshot artifacts must exactly match event artifacts")

        cancel_events = tuple(
            event for event in events if event.kind is AgentRunEventKind.CANCELLATION_REQUESTED
        )
        if self.cancel_requested != bool(cancel_events):
            raise ValueError("cancel_requested must match the event chain")
        if self.pending_call is not None:
            matches = tuple(
                event
                for event in reserved_events
                if event.request_id == self.pending_call.request_id
                and event.tool_name == self.pending_call.tool_name
                and event.argument_hash == self.pending_call.argument_hash
                and event.idempotency_key_hash == self.pending_call.idempotency_key_hash
                and event.batch_hash == self.pending_call.batch_hash
                and event.occurred_at == self.pending_call.started_at
                and event.tool_effect is self.pending_call.effect
            )
            if len(matches) != 1:
                raise ValueError("pending_call must bind exactly one reserved event")
            completed = any(
                event.request_id == self.pending_call.request_id
                and event.kind
                in {
                    AgentRunEventKind.TOOL_CALL_SUCCEEDED,
                    AgentRunEventKind.TOOL_CALL_FAILED,
                    AgentRunEventKind.TOOL_CALL_REPLAYED,
                    AgentRunEventKind.TOOL_CALL_REJECTED,
                }
                for event in events[matches[0].sequence + 1 :]
            )
            if completed:
                raise ValueError("pending_call already has a recorded outcome")
        return self

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        decision_id: str,
        decision_snapshot_hash: str,
        runtime_mode: RuntimeMode,
        goal: AgentRunGoal,
        created_at: datetime,
        limits: AgentRunLimits | None = None,
        runtime_version: str = AGENT_RUNTIME_VERSION,
    ) -> AgentRunSnapshot:
        selected_limits = limits or AgentRunLimits()
        created = _utc(created_at)
        event = AgentRunEvent.build(
            run_id=run_id,
            decision_id=decision_id,
            decision_snapshot_hash=decision_snapshot_hash,
            runtime_mode=runtime_mode,
            goal=goal,
            sequence=0,
            kind=AgentRunEventKind.RUN_CREATED,
            trigger=AgentRunTrigger.SYSTEM,
            state_before=AgentRunState.RECEIVED,
            state_after=AgentRunState.RECEIVED,
            reason="RUN_CREATED",
            occurred_at=created,
            run_limits=selected_limits,
            runtime_version=runtime_version,
        )
        return cls(
            runtime_version=runtime_version,
            run_id=run_id,
            decision_id=decision_id,
            decision_snapshot_hash=decision_snapshot_hash,
            runtime_mode=runtime_mode,
            goal=goal,
            state=AgentRunState.RECEIVED,
            created_at=created,
            updated_at=created,
            deadline=created + timedelta(seconds=selected_limits.timeout_seconds),
            limits=selected_limits,
            revision=0,
            events=(event,),
        )

    def to_json(self) -> str:
        return _dump_model(self, cast(TypeAdapter[object], _RUN_ADAPTER))

    @classmethod
    def from_json(cls, value: str) -> AgentRunSnapshot:
        parsed = _RUN_ADAPTER.validate_json(
            _validated_json(value, label="AgentRunSnapshot"), strict=True
        )
        if type(parsed) is not cls:
            raise TypeError("AgentRunSnapshot JSON did not produce the exact contract type")
        return parsed


_EVENT_ADAPTER = TypeAdapter(AgentRunEvent)
_RUN_ADAPTER = TypeAdapter(AgentRunSnapshot)


__all__ = [
    "AGENT_RUNTIME_SCHEMA_VERSION",
    "AGENT_RUNTIME_VERSION",
    "TERMINAL_AGENT_RUN_STATES",
    "AgentArtifactKind",
    "AgentArtifactRef",
    "AgentExternalApproval",
    "AgentPendingToolCall",
    "AgentRunEvent",
    "AgentRunEventKind",
    "AgentRunGoal",
    "AgentRunLimits",
    "AgentRunSnapshot",
    "AgentRunState",
    "AgentRunTrigger",
]
