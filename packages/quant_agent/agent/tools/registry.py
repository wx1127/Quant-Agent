"""Server-owned Agent tool registry, permissions and argument validation."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from quant_agent.agent.audit_replay import InMemoryHarnessAuditStore, ToolAuditRecord
from quant_agent.agent.runtime import AgentRuntime, AgentState
from quant_agent.agent.snapshots import DecisionSnapshot
from quant_agent.config.models import RuntimeMode
from quant_agent.core.errors import ErrorCode, QuantAgentError
from quant_agent.core.responses import Provenance, ToolIssue, ToolResponse
from quant_agent.core.time import shanghai_now
from quant_agent.observability.audit import AuditEvent, AuditSink


class ToolCategory(StrEnum):
    DATA = "DATA"
    VALIDATION = "VALIDATION"
    ANALYSIS = "ANALYSIS"
    BACKTEST = "BACKTEST"
    PORTFOLIO = "PORTFOLIO"
    RISK = "RISK"
    ORDER_DRAFT = "ORDER_DRAFT"
    PAPER_EXECUTION = "PAPER_EXECUTION"
    RECONCILIATION = "RECONCILIATION"
    REPORT = "REPORT"


class ToolEffect(StrEnum):
    READ = "READ"
    WRITE = "WRITE"


ToolHandler = Callable[[BaseModel, "ToolCallContext"], Any]


@dataclass(frozen=True, slots=True)
class AgentTool:
    name: str
    category: ToolCategory
    effect: ToolEffect
    argument_model: type[BaseModel]
    handler: ToolHandler
    allowed_modes: frozenset[RuntimeMode]
    allowed_states: frozenset[AgentState]
    service: str
    version: str

    def __post_init__(self) -> None:
        if self.name == "approve_order_batch":
            raise ValueError("human approval must never be registered as an Agent tool")
        if self.effect is ToolEffect.WRITE and not self.allowed_modes <= {
            RuntimeMode.PAPER,
            RuntimeMode.LIVE_ASSISTED,
        }:
            raise ValueError("write tools cannot be enabled in research/backtest modes")


@dataclass(slots=True)
class ToolCallContext:
    request_id: str
    actor_id: str
    snapshot: DecisionSnapshot
    runtime: AgentRuntime
    audit_sink: AuditSink
    idempotency_key: str | None = None
    risk_approved_request_ids: set[str] = field(default_factory=set)
    replay_sink: InMemoryHarnessAuditStore | None = None
    audit_sequence: int = 0


class ToolRegistry:
    """Deny-by-default registry controlled by server configuration."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolCallContext,
    ) -> ToolResponse[Any]:
        context.runtime.reserve_tool_call()
        tool = self._tools.get(name)
        if tool is None:
            return self._denied(context, name, "tool is not registered", arguments)
        if context.snapshot.mode not in tool.allowed_modes:
            return self._denied(
                context, name, "tool is forbidden in current runtime mode", arguments
            )
        if context.runtime.state not in tool.allowed_states:
            return self._denied(
                context, name, "tool is forbidden in current Agent state", arguments
            )
        if tool.effect is ToolEffect.WRITE and not context.idempotency_key:
            return self._denied(context, name, "write tool requires an idempotency key", arguments)
        try:
            parsed = tool.argument_model.model_validate(arguments)
        except ValidationError as exc:
            return self._failed(
                context,
                tool,
                ErrorCode.INVALID_ARGUMENT,
                f"tool arguments failed validation: {exc.errors()[0]['msg']}",
                arguments,
            )
        try:
            data = tool.handler(parsed, context)
        except QuantAgentError as exc:
            return self._failed(context, tool, exc.code, exc.message, arguments)
        except Exception as exc:
            return self._failed(context, tool, ErrorCode.INTERNAL_ERROR, str(exc), arguments)
        response = ToolResponse[Any](
            ok=True,
            request_id=context.request_id,
            decision_id=context.snapshot.decision_id,
            as_of=context.snapshot.as_of,
            data=data,
            provenance=Provenance(
                service=tool.service,
                version=tool.version,
                data_version=context.snapshot.data_version,
            ),
        ).validate_consistency()
        self._audit(context, name, "allowed", arguments, response)
        return response

    def _denied(
        self,
        context: ToolCallContext,
        name: str,
        message: str,
        arguments: dict[str, Any],
    ) -> ToolResponse[Any]:
        response = ToolResponse[Any](
            ok=False,
            request_id=context.request_id,
            decision_id=context.snapshot.decision_id,
            as_of=context.snapshot.as_of,
            errors=(ToolIssue(code=ErrorCode.FORBIDDEN, message=message),),
            provenance=Provenance(
                service="agent-tool-registry",
                version="tool_registry_v1",
                data_version=context.snapshot.data_version,
            ),
        ).validate_consistency()
        self._audit(context, name, "denied", arguments, response)
        return response

    def _failed(
        self,
        context: ToolCallContext,
        tool: AgentTool,
        code: ErrorCode,
        message: str,
        arguments: dict[str, Any],
    ) -> ToolResponse[Any]:
        response = ToolResponse[Any](
            ok=False,
            request_id=context.request_id,
            decision_id=context.snapshot.decision_id,
            as_of=context.snapshot.as_of,
            errors=(ToolIssue(code=code, message=message),),
            provenance=Provenance(
                service=tool.service,
                version=tool.version,
                data_version=context.snapshot.data_version,
            ),
        ).validate_consistency()
        self._audit(context, tool.name, "failed", arguments, response)
        return response

    @staticmethod
    def _audit(
        context: ToolCallContext,
        name: str,
        result: str,
        arguments: dict[str, Any],
        response: ToolResponse[Any],
    ) -> None:
        canonical_args = json.dumps(arguments, default=str, sort_keys=True)
        canonical_response = response.model_dump_json()
        context.audit_sink.append(
            AuditEvent(
                event_type="agent_tool_call",
                actor_id=context.actor_id,
                action=name,
                result=result,
                request_id=context.request_id,
                decision_id=context.snapshot.decision_id,
                occurred_at=shanghai_now(),
                metadata={
                    "argument_hash": hashlib.sha256(canonical_args.encode()).hexdigest(),
                    "response_hash": hashlib.sha256(canonical_response.encode()).hexdigest(),
                    "state": context.runtime.state.value,
                },
            )
        )
        if context.replay_sink is not None:
            context.audit_sequence += 1
            context.replay_sink.append_tool_record(
                ToolAuditRecord.create(
                    decision_id=context.snapshot.decision_id,
                    sequence=context.audit_sequence,
                    tool_name=name,
                    occurred_at=shanghai_now(),
                    arguments=arguments,
                    response=response.model_dump(mode="json"),
                )
            )
