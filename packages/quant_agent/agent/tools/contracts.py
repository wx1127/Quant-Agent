"""Strict, immutable contracts for Agent-visible tools."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from quant_agent.agent.snapshots.contracts import DecisionSnapshot
from quant_agent.config import RuntimeMode
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import Provenance, ToolIssue, ToolResponse
from quant_agent.core.time import ensure_aware

_TOOL_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", re.ASCII)
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", re.ASCII)
_FLOATING_VERSIONS = frozenset({"current", "default", "latest"})


def _exact_text(value: str, field_name: str, *, max_length: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
        or not value.isprintable()
    ):
        raise ValueError(
            f"{field_name} must be printable, non-empty, bounded, "
            "and have no surrounding whitespace"
        )
    return value


def validate_tool_name(value: str) -> str:
    """Require one exact lower-ASCII snake-case identifier without normalization."""

    if type(value) is not str or len(value) > 64 or _TOOL_NAME.fullmatch(value) is None:
        raise ValueError(
            "tool name must be a lower-ASCII snake-case identifier of at most 64 chars"
        )
    return value


class ToolCategory(StrEnum):
    """Business-facing classification; it does not grant authority by itself."""

    DATA = "DATA"
    VALIDATION = "VALIDATION"
    ANALYSIS = "ANALYSIS"
    MAINLINE = "MAINLINE"
    LEADER = "LEADER"
    CANDIDATE = "CANDIDATE"
    BACKTEST = "BACKTEST"
    PORTFOLIO = "PORTFOLIO"
    RISK = "RISK"
    ORDER_DRAFT = "ORDER_DRAFT"
    EXECUTION = "EXECUTION"
    RECONCILIATION = "RECONCILIATION"
    REPORT = "REPORT"


class ToolCapability(StrEnum):
    """Trusted grants consumed by the registry, never values supplied by the model."""

    MARKET_DATA_READ = "MARKET_DATA_READ"
    MARKET_DATA_VALIDATE = "MARKET_DATA_VALIDATE"
    RESEARCH_ANALYSIS = "RESEARCH_ANALYSIS"
    BACKTEST_RUN = "BACKTEST_RUN"
    ACCOUNT_READ = "ACCOUNT_READ"
    PORTFOLIO_BUILD = "PORTFOLIO_BUILD"
    RISK_CHECK = "RISK_CHECK"
    ORDER_DRAFT_READ = "ORDER_DRAFT_READ"
    ORDER_DRAFT_CREATE = "ORDER_DRAFT_CREATE"
    PAPER_ORDER_SUBMIT = "PAPER_ORDER_SUBMIT"
    APPROVED_LIVE_ORDER_SUBMIT = "APPROVED_LIVE_ORDER_SUBMIT"
    ACCOUNT_RECONCILE = "ACCOUNT_RECONCILE"
    REPORT_READ = "REPORT_READ"


class ToolEffect(StrEnum):
    """Security-relevant side effect, orthogonal to the business category."""

    READ_ONLY = "READ_ONLY"
    ISOLATED_COMPUTE = "ISOLATED_COMPUTE"
    TARGET_ONLY = "TARGET_ONLY"
    ARTIFACT_WRITE = "ARTIFACT_WRITE"
    PAPER_EXECUTION_WRITE = "PAPER_EXECUTION_WRITE"
    LIVE_EXTERNAL_WRITE = "LIVE_EXTERNAL_WRITE"


class ToolDenialReason(StrEnum):
    """Stable, non-sensitive reasons recorded for rejected calls."""

    DECISION_UNAVAILABLE = "DECISION_UNAVAILABLE"
    DECISION_CONTEXT_MISMATCH = "DECISION_CONTEXT_MISMATCH"
    DECISION_FROM_FUTURE = "DECISION_FROM_FUTURE"
    INVALID_TOOL_NAME = "INVALID_TOOL_NAME"
    RESERVED_HUMAN_TOOL = "RESERVED_HUMAN_TOOL"
    TOOL_NOT_REGISTERED = "TOOL_NOT_REGISTERED"
    MODE_NOT_ALLOWED = "MODE_NOT_ALLOWED"
    CAPABILITY_NOT_GRANTED = "CAPABILITY_NOT_GRANTED"
    ACCOUNT_SCOPE_NOT_GRANTED = "ACCOUNT_SCOPE_NOT_GRANTED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    IDEMPOTENCY_KEY_REQUIRED = "IDEMPOTENCY_KEY_REQUIRED"
    EXTERNAL_AUTHORIZATION_REQUIRED = "EXTERNAL_AUTHORIZATION_REQUIRED"


class ToolArguments(BaseModel):
    """Required base for model-controlled arguments."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ToolOutput(BaseModel):
    """Required base for deterministic, schema-closed handler result data."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ToolCallContext(BaseModel):
    """Trusted transport context; mode/account/as-of are intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_id: str = Field(min_length=1, max_length=128)
    decision_id: str = Field(min_length=1, max_length=128)
    requested_at: datetime

    @field_validator("request_id", "decision_id")
    @classmethod
    def validate_identity(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identity")
        return _exact_text(value, field_name, max_length=128)

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at(cls, value: datetime) -> datetime:
        return ensure_aware(value).astimezone(UTC)


class ToolExecutionContext(BaseModel):
    """Registry-derived context passed to handlers after authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_id: str
    principal_id: str
    requested_at: datetime
    decision_snapshot: DecisionSnapshot

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _exact_text(value, "request_id", max_length=128)

    @field_validator("principal_id")
    @classmethod
    def validate_principal_id(cls, value: str) -> str:
        return _exact_text(value, "principal_id", max_length=256)

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at(cls, value: datetime) -> datetime:
        return ensure_aware(value).astimezone(UTC)


class VerifiedDecisionResolver(Protocol):
    """Resolve exactly one snapshot after revalidating its authoritative sources.

    Production composition should bind this protocol to
    ``DecisionSnapshotService.resume`` with current trusted inputs. A repository
    lookup alone proves stored identity, not that referenced sources remain valid.
    """

    def resolve(self, decision_id: str) -> DecisionSnapshot:
        """Return the exact, revalidated snapshot for ``decision_id``."""


class LiveExecutionAuthorizer(Protocol):
    """Trusted out-of-band lease for exactly one live submission.

    Entering the returned scope must atomically validate and reserve/consume the
    exact approval. The grant must remain valid until scope exit, covering both
    the synchronous ALLOWED audit and the handler call. Approval secrets never
    enter tool arguments or the returned context-manager value.
    """

    def authorization(
        self,
        context: ToolExecutionContext,
        arguments: ToolArguments,
    ) -> AbstractContextManager[None]:
        """Return an exact-call authorization scope or raise to deny."""


class ToolRegistryError(RuntimeError):
    """Base class for tool-registry failures."""


class ToolRegistrationError(ToolRegistryError):
    """Raised for an invalid, duplicate, unknown, or late registration."""


class ToolInvocationDenied(ToolRegistryError):
    """Safe external error carrying only structured denial metadata."""

    def __init__(
        self,
        reason: ToolDenialReason,
        *,
        mode: RuntimeMode,
        tool_name: str | None = None,
        validation_error_codes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(f"tool invocation denied: {reason.value}")
        self.reason = reason
        self.mode = mode
        self.tool_name = tool_name
        self.validation_error_codes = validation_error_codes


class ToolNotRegistered(ToolInvocationDenied):
    """Raised when a name is malformed, human-only, or has no registered handler."""


class ToolPermissionDenied(ToolInvocationDenied):
    """Raised when mode, capability, account, or approval policy rejects a call."""


class ToolArgumentsRejected(ToolInvocationDenied):
    """Raised after safe strict validation rejects model-controlled arguments."""


class ToolAuditUnavailable(ToolRegistryError):
    """Raised when a denial or authorization decision cannot be persisted."""

    def __init__(self, intended_denial: ToolInvocationDenied | None) -> None:
        super().__init__("tool invocation was not executed because audit persistence failed")
        self.intended_denial = intended_denial


class ToolExecutionFailed(ToolRegistryError):
    """Safe wrapper for an exception raised by a registered handler."""


class ToolOutputRejected(ToolRegistryError):
    """Raised when a handler does not return a snapshot-bound ToolResponse."""


class ToolDescriptor(BaseModel):
    """Agent-visible definition without a callable or hidden policy internals."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    version: str
    description: str
    category: ToolCategory
    capability: ToolCapability
    effect: ToolEffect
    argument_schema: dict[str, object]
    output_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class AgentTool[ArgumentsT: ToolArguments, OutputT: ToolOutput]:
    """One trusted handler plus its strict argument contract."""

    name: str
    version: str
    description: str
    arguments_type: type[ArgumentsT]
    output_type: type[OutputT]
    handler: Callable[[ToolExecutionContext, ArgumentsT], object]
    _adapter: TypeAdapter[ArgumentsT] = field(init=False, repr=False, compare=False)
    _output_adapter: TypeAdapter[OutputT] = field(init=False, repr=False, compare=False)
    _argument_schema_json: str = field(init=False, repr=False, compare=False)
    _output_schema_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            validate_tool_name(self.name)
            if _VERSION.fullmatch(self.version) is None:
                raise ValueError("tool version must be a bounded immutable identifier")
            if self.version.lower() in _FLOATING_VERSIONS:
                raise ValueError("tool version cannot be a floating alias")
            _exact_text(self.description, "description", max_length=500)
            if not isinstance(self.arguments_type, type) or not issubclass(
                self.arguments_type, ToolArguments
            ):
                raise ValueError("arguments_type must inherit ToolArguments")
            if not isinstance(self.output_type, type) or not issubclass(
                self.output_type, ToolOutput
            ):
                raise ValueError("output_type must inherit ToolOutput")
            for model_type in (self.arguments_type, self.output_type):
                config = model_type.model_config
                if (
                    config.get("extra") != "forbid"
                    or config.get("frozen") is not True
                    or (
                        model_type is self.output_type
                        and config.get("revalidate_instances") != "always"
                    )
                    or config.get("strict") is not True
                ):
                    raise ValueError("tool models cannot weaken strict/frozen/forbid settings")
            if not callable(self.handler):
                raise ValueError("tool handler must be callable")
            if inspect.iscoroutinefunction(self.handler):
                raise ValueError("async handlers require a separate future execution boundary")
            argument_schema_json = json.dumps(
                self.arguments_type.model_json_schema(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            output_schema_json = json.dumps(
                self.output_type.model_json_schema(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            adapter = TypeAdapter(self.arguments_type)
            output_adapter = TypeAdapter(self.output_type)
        except (TypeError, ValueError) as error:
            raise ToolRegistrationError("invalid Agent tool definition") from error
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_output_adapter", output_adapter)
        object.__setattr__(self, "_argument_schema_json", argument_schema_json)
        object.__setattr__(self, "_output_schema_json", output_schema_json)

    def validate_arguments(self, canonical_json: str) -> ArgumentsT:
        """Validate canonical JSON with Pydantic's strict JSON semantics."""

        return self._adapter.validate_json(canonical_json, strict=True)

    def validate_output_data(self, value: object) -> OutputT:
        """Require the exact registered output model and revalidate its fields."""

        if type(value) is not self.output_type:
            raise ToolOutputRejected("tool response data has an unsupported type")
        validation_failed = False
        try:
            encoded = self._output_adapter.dump_json(value, warnings="error")
            validated = self._output_adapter.validate_json(encoded, strict=True)
        except (TypeError, ValueError):
            validation_failed = True
        if validation_failed:
            raise ToolOutputRejected("tool response data failed strict validation") from None
        return validated

    def argument_schema(self) -> dict[str, Any]:
        """Return a detached copy of the schema frozen at tool construction."""

        value = json.loads(self._argument_schema_json)
        if type(value) is not dict:  # pragma: no cover - construction invariant
            raise AssertionError("argument schema must remain an object")
        return value

    def output_schema(self) -> dict[str, Any]:
        """Return a detached copy of the output schema frozen at construction."""

        value = json.loads(self._output_schema_json)
        if type(value) is not dict:  # pragma: no cover - construction invariant
            raise AssertionError("output schema must remain an object")
        return value

    def invoke(
        self,
        context: ToolExecutionContext,
        arguments: ArgumentsT,
    ) -> object:
        """Call the trusted synchronous handler with already validated arguments."""

        result = self.handler(context, arguments)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise ToolExecutionFailed("async handler results are not supported")
        return result

    def descriptor(
        self,
        *,
        category: ToolCategory,
        capability: ToolCapability,
        effect: ToolEffect,
    ) -> ToolDescriptor:
        """Build a fresh defensive schema copy for Agent discovery."""

        argument_schema = self.argument_schema()
        output_schema = self.output_schema()
        return ToolDescriptor(
            name=self.name,
            version=self.version,
            description=self.description,
            category=category,
            capability=capability,
            effect=effect,
            argument_schema=dict(argument_schema),
            output_schema=dict(output_schema),
        )


def validate_tool_response(
    value: object,
    *,
    context: ToolExecutionContext,
) -> ToolResponse[object]:
    """Require the common response contract to bind the exact decision context."""

    if not isinstance(value, ToolResponse):
        raise ToolOutputRejected("tool handler returned an unsupported response type")
    response = value
    if (
        type(response.ok) is not bool
        or type(response.request_id) is not str
        or type(response.decision_id) is not str
        or type(response.as_of) is not datetime
        or type(response.warnings) is not tuple
        or type(response.errors) is not tuple
        or type(response.provenance) is not Provenance
        or any(type(item) is not ToolIssue for item in (*response.warnings, *response.errors))
    ):
        raise ToolOutputRejected("tool handler returned malformed response fields") from None
    if (response.ok and response.errors) or (not response.ok and not response.errors):
        raise ToolOutputRejected("tool handler returned an inconsistent response") from None
    snapshot = context.decision_snapshot
    malformed = False
    try:
        response_as_of = ensure_aware(response.as_of).astimezone(UTC)
        _exact_text(response.provenance.service, "provenance service", max_length=128)
        _exact_text(response.provenance.version, "provenance version", max_length=128)
        if type(response.provenance.data_version) is not str:
            raise ValueError("provenance data version must be an exact string")
        for item in (*response.warnings, *response.errors):
            if type(item.code) is not ErrorCode:
                raise ValueError("tool issue code must be an exact ErrorCode")
            _exact_text(item.message, "tool issue message", max_length=2_000)
    except (TypeError, ValueError):
        malformed = True
    if malformed:
        raise ToolOutputRejected("tool handler returned malformed response fields") from None
    if (
        response.request_id != context.request_id
        or response.decision_id != snapshot.decision_id
        or response_as_of != snapshot.as_of
        or response.provenance.data_version != snapshot.data_version
    ):
        raise ToolOutputRejected("tool response does not bind the authorized decision context")
    return ToolResponse[object](
        ok=response.ok,
        request_id=response.request_id,
        decision_id=response.decision_id,
        as_of=response_as_of,
        data=response.data,
        warnings=tuple(
            ToolIssue(code=item.code, message=item.message) for item in response.warnings
        ),
        errors=tuple(ToolIssue(code=item.code, message=item.message) for item in response.errors),
        provenance=Provenance(
            service=response.provenance.service,
            version=response.provenance.version,
            data_version=response.provenance.data_version,
        ),
    )


__all__ = [
    "AgentTool",
    "LiveExecutionAuthorizer",
    "ToolArguments",
    "ToolArgumentsRejected",
    "ToolAuditUnavailable",
    "ToolCallContext",
    "ToolCapability",
    "ToolCategory",
    "ToolDenialReason",
    "ToolDescriptor",
    "ToolEffect",
    "ToolExecutionContext",
    "ToolExecutionFailed",
    "ToolInvocationDenied",
    "ToolNotRegistered",
    "ToolOutput",
    "ToolOutputRejected",
    "ToolPermissionDenied",
    "ToolRegistrationError",
    "ToolRegistryError",
    "VerifiedDecisionResolver",
    "validate_tool_name",
    "validate_tool_response",
]
