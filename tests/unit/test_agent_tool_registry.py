"""Adversarial tests for the fail-closed Agent tool registry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from itertools import product
from typing import Any

import pytest
from pydantic import BaseModel, Field, field_validator

from quant_agent.agent.snapshots.contracts import DecisionSnapshot, StrategySnapshotRef
from quant_agent.agent.tools.contracts import (
    AgentTool,
    ToolArguments,
    ToolArgumentsRejected,
    ToolAuditUnavailable,
    ToolCallContext,
    ToolCapability,
    ToolDenialReason,
    ToolExecutionContext,
    ToolExecutionFailed,
    ToolNotRegistered,
    ToolOutput,
    ToolOutputRejected,
    ToolPermissionDenied,
    ToolRegistrationError,
    ToolRegistryError,
)
from quant_agent.agent.tools.policy import (
    HUMAN_ONLY_TOOL_NAMES,
    MODE_TOOL_PERMISSIONS,
    V1_TOOL_POLICIES,
)
from quant_agent.agent.tools.registry import AgentToolRegistry, AgentToolRegistryBuilder
from quant_agent.config import RuntimeMode
from quant_agent.core import Provenance, ToolResponse
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import ToolIssue
from quant_agent.observability.audit import AuditEvent

AS_OF = datetime(2026, 9, 8, 7, 0, tzinfo=UTC)
REQUESTED_AT = AS_OF + timedelta(minutes=1)
DECISION_ID = "dec_20260908_0123456789abcdef0123456789abcdef"
ACCOUNT_ID = "paper-account-1"


class EmptyArguments(ToolArguments):
    pass


class WriteArguments(ToolArguments):
    idempotency_key: str
    order_batch_id: str


class ResultData(ToolOutput):
    tool_name: str
    call_count: int


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class RichArguments(ToolArguments):
    side: Side
    at: datetime
    amount: Decimal
    quantity: int
    enabled: bool


class _Resolver:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []

    def resolve(self, decision_id: str) -> DecisionSnapshot:
        self.calls.append(decision_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result  # type: ignore[return-value]


class _SequenceResolver:
    def __init__(self, results: tuple[DecisionSnapshot, ...]) -> None:
        self.results = results
        self.calls = 0

    def resolve(self, decision_id: str) -> DecisionSnapshot:
        result = self.results[self.calls]
        self.calls += 1
        return result


class _AuditSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        if self.fail:
            raise RuntimeError("TOPSECRET_AUDIT_FAILURE")
        self.events.append(event)


class _AuthorizationLease(AbstractContextManager[None]):
    def __init__(self, authorizer: _Authorizer) -> None:
        self.authorizer = authorizer

    def __enter__(self) -> None:
        self.authorizer.enter_calls += 1
        if not self.authorizer.available:
            raise RuntimeError("TOPSECRET_APPROVAL_DENIED")
        self.authorizer.available = False
        self.authorizer.reserved = True
        return None

    def __exit__(self, *exc_info: object) -> bool:
        self.authorizer.exit_calls += 1
        self.authorizer.reserved = False
        return False


class _Authorizer:
    def __init__(self, *, available: bool = True, fail_before_scope: bool = False) -> None:
        self.available = available
        self.fail_before_scope = fail_before_scope
        self.authorization_calls = 0
        self.enter_calls = 0
        self.exit_calls = 0
        self.reserved = False

    def authorization(
        self,
        context: ToolExecutionContext,
        arguments: ToolArguments,
    ) -> AbstractContextManager[None]:
        self.authorization_calls += 1
        if self.fail_before_scope:
            raise RuntimeError("TOPSECRET_AUTHORIZER_FAILURE")
        return _AuthorizationLease(self)


class _RevokingAuditSink(_AuditSink):
    def __init__(self, authorizer: _Authorizer) -> None:
        super().__init__()
        self.authorizer = authorizer

    def append(self, event: AuditEvent) -> None:
        super().append(event)
        if event.result == "ALLOWED":
            self.authorizer.available = False


def _snapshot(
    mode: RuntimeMode,
    *,
    decision_id: str = DECISION_ID,
    account_id: str = ACCOUNT_ID,
    as_of: datetime = AS_OF,
) -> DecisionSnapshot:
    reference = StrategySnapshotRef(
        strategy_name="etf-rotation",
        strategy_version="etf-rotation-v1",
        config_hash="1" * 64,
        parameter_version="params-v1",
        parameter_hash="2" * 64,
        registered_at=as_of - timedelta(days=1),
    )
    return DecisionSnapshot.build(
        decision_id=decision_id,
        mode=mode,
        market="CN_A",
        as_of=as_of,
        data_version="market_20260908_eod_v1",
        data_content_hash="3" * 64,
        strategy_refs=(reference,),
        risk_policy_version="portfolio-risk-v1",
        risk_policy_hash="4" * 64,
        account_id=account_id,
        account_snapshot_id=f"{account_id}:20260908T070000",
        account_snapshot_hash="5" * 64,
        code_commit="6" * 40,
        code_artifact_hash="7" * 64,
        agent_version="quant-agent-harness-v1",
        model_version="model-release-v1",
    )


def _context(
    *, decision_id: str = DECISION_ID, requested_at: datetime = REQUESTED_AT
) -> ToolCallContext:
    return ToolCallContext(
        request_id="req_agent_tool_registry",
        decision_id=decision_id,
        requested_at=requested_at,
    )


def _response(context: ToolExecutionContext, name: str, count: int) -> ToolResponse[ResultData]:
    snapshot = context.decision_snapshot
    return ToolResponse[ResultData](
        ok=True,
        request_id=context.request_id,
        decision_id=snapshot.decision_id,
        as_of=snapshot.as_of,
        data=ResultData(tool_name=name, call_count=count),
        provenance=Provenance(
            service="agent-tool-test",
            version="1.0.0",
            data_version=snapshot.data_version,
        ),
    )


def _tool(
    name: str,
    calls: list[tuple[ToolExecutionContext, ToolArguments]],
    *,
    arguments_type: type[ToolArguments] | None = None,
    output_type: type[ToolOutput] = ResultData,
    handler_override: Any = None,
) -> AgentTool[Any, Any]:
    selected_arguments = arguments_type
    if selected_arguments is None:
        policy = V1_TOOL_POLICIES.get(name)
        selected_arguments = (
            WriteArguments
            if policy is not None and policy.requires_idempotency_key
            else EmptyArguments
        )

    def handler(context: ToolExecutionContext, arguments: ToolArguments) -> object:
        calls.append((context, arguments))
        if handler_override is not None:
            return handler_override(context, arguments)
        return _response(context, name, len(calls))

    return AgentTool(
        name=name,
        version="1.0.0",
        description=f"Deterministic test adapter for {name}.",
        arguments_type=selected_arguments,
        output_type=output_type,
        handler=handler,
    )


def _arguments_for(name: str) -> dict[str, object]:
    if V1_TOOL_POLICIES[name].requires_idempotency_key:
        return {"idempotency_key": "idem-1", "order_batch_id": "batch-1"}
    return {}


def _build(
    *,
    mode: RuntimeMode,
    tools: tuple[AgentTool[Any, Any], ...],
    snapshot: DecisionSnapshot | None = None,
    resolver: Any = None,
    sink: _AuditSink | None = None,
    capabilities: frozenset[ToolCapability] | None = None,
    account_ids: frozenset[str] | None = None,
    authorizer: _Authorizer | None = None,
) -> tuple[AgentToolRegistry, Any, _AuditSink]:
    active_snapshot = snapshot or _snapshot(mode)
    active_resolver = resolver or _Resolver(active_snapshot)
    active_sink = sink or _AuditSink()
    builder = AgentToolRegistryBuilder(
        runtime_mode=mode,
        principal_id="agent-worker-1",
        granted_capabilities=(frozenset(ToolCapability) if capabilities is None else capabilities),
        granted_account_ids=(
            frozenset({active_snapshot.account_id}) if account_ids is None else account_ids
        ),
        decision_resolver=active_resolver,
        audit_sink=active_sink,
        live_execution_authorizer=authorizer,
    )
    for tool in tools:
        builder.register(tool)
    return builder.build(), active_resolver, active_sink


MATRIX_CASES = tuple(
    product(
        (
            RuntimeMode.RESEARCH,
            RuntimeMode.BACKTEST,
            RuntimeMode.PAPER,
            RuntimeMode.LIVE_ASSISTED,
        ),
        sorted(V1_TOOL_POLICIES),
    )
)


@pytest.mark.parametrize(("mode", "tool_name"), MATRIX_CASES)
def test_every_registered_tool_obeys_the_complete_mode_matrix(
    mode: RuntimeMode, tool_name: str
) -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    authorizer = _Authorizer()
    registry, resolver, sink = _build(
        mode=mode,
        tools=(_tool(tool_name, calls),),
        authorizer=authorizer,
    )

    if tool_name in MODE_TOOL_PERMISSIONS[mode]:
        result = registry.invoke(tool_name, _arguments_for(tool_name), _context())

        assert isinstance(result, ToolResponse)
        assert result.ok is True
        assert len(calls) == 1
        assert resolver.calls == [DECISION_ID]
        assert sink.events[-1].result == "ALLOWED"
    else:
        with pytest.raises(ToolPermissionDenied) as raised:
            registry.invoke(tool_name, "TOPSECRET_NOT_EVEN_JSON", _context())

        assert raised.value.reason is ToolDenialReason.MODE_NOT_ALLOWED
        assert calls == []
        assert resolver.calls == []
        assert sink.events[-1].metadata["denial_reason"] == "MODE_NOT_ALLOWED"


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("unknown_tool", ToolDenialReason.TOOL_NOT_REGISTERED),
        ("Bad Tool", ToolDenialReason.INVALID_TOOL_NAME),
        ("approve_order_batch", ToolDenialReason.RESERVED_HUMAN_TOOL),
    ],
)
def test_unknown_invalid_and_human_only_names_deny_before_decision_resolution(
    name: str, reason: ToolDenialReason
) -> None:
    resolver = _Resolver(RuntimeError("decision existence must remain hidden"))
    registry, _, sink = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(),
        resolver=resolver,
    )
    context = _context(decision_id="unknown-decision-id")

    with pytest.raises(ToolNotRegistered) as raised:
        registry.invoke(name, {}, context)

    assert raised.value.reason is reason
    assert resolver.calls == []
    serialized = sink.events[-1].model_dump_json()
    assert name not in serialized
    assert "tool_name_sha256" in serialized


def test_research_order_denial_precedes_payload_parsing_and_snapshot_resolution() -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, resolver, sink = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(_tool("create_order_draft", calls),),
    )
    raw = '{"approval_token":"TOPSECRET",'

    with pytest.raises(ToolPermissionDenied) as raised:
        registry.invoke("create_order_draft", raw, _context())

    assert raised.value.reason is ToolDenialReason.MODE_NOT_ALLOWED
    assert calls == []
    assert resolver.calls == []
    assert "TOPSECRET" not in sink.events[-1].model_dump_json()


@pytest.mark.parametrize(
    "mode",
    (
        RuntimeMode.RESEARCH,
        RuntimeMode.BACKTEST,
        RuntimeMode.PAPER,
        RuntimeMode.LIVE_ASSISTED,
    ),
)
def test_catalog_exactly_matches_registered_authorized_matrix_and_is_defensive(
    mode: RuntimeMode,
) -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    tools = tuple(_tool(name, calls) for name in V1_TOOL_POLICIES)
    registry, resolver, sink = _build(
        mode=mode,
        tools=tools,
        authorizer=_Authorizer(),
    )

    first = registry.catalog(_context())
    assert tuple(item.name for item in first) == tuple(sorted(MODE_TOOL_PERMISSIONS[mode]))
    assert resolver.calls == [DECISION_ID]
    assert sink.events == []
    assert all(not hasattr(item, "handler") for item in first)
    assert all(item.output_schema for item in first)

    first[0].argument_schema["mutated"] = True
    second = registry.catalog(_context())
    assert "mutated" not in second[0].argument_schema
    assert resolver.calls == [DECISION_ID, DECISION_ID]


def test_catalog_schema_is_frozen_against_argument_model_rebuild() -> None:
    class LocalArguments(ToolArguments):
        value: str = Field(description="original-safe-description")

    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    tool = _tool(
        "get_market_snapshot",
        calls,
        arguments_type=LocalArguments,
    )
    registry, _, _ = _build(mode=RuntimeMode.RESEARCH, tools=(tool,))
    original = registry.catalog(_context())[0].argument_schema

    LocalArguments.model_fields["value"].description = "approval_token=TOPSECRET"
    LocalArguments.model_rebuild(force=True)

    rebuilt = registry.catalog(_context())[0].argument_schema
    assert rebuilt == original
    assert "TOPSECRET" not in str(rebuilt)


def test_capability_and_account_scope_are_independent_fail_closed_checks() -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    tool = _tool("get_portfolio_snapshot", calls)
    no_capability, resolver_one, sink_one = _build(
        mode=RuntimeMode.PAPER,
        tools=(tool,),
        capabilities=frozenset(),
    )

    with pytest.raises(ToolPermissionDenied) as capability_error:
        no_capability.invoke("get_portfolio_snapshot", {}, _context())
    assert capability_error.value.reason is ToolDenialReason.CAPABILITY_NOT_GRANTED
    assert resolver_one.calls == []
    assert sink_one.events[-1].metadata["denial_reason"] == "CAPABILITY_NOT_GRANTED"

    no_account, resolver_two, sink_two = _build(
        mode=RuntimeMode.PAPER,
        tools=(tool,),
        account_ids=frozenset(),
    )
    with pytest.raises(ToolPermissionDenied) as account_error:
        no_account.invoke("get_portfolio_snapshot", {}, _context())
    assert account_error.value.reason is ToolDenialReason.ACCOUNT_SCOPE_NOT_GRANTED
    assert resolver_two.calls == [DECISION_ID]
    assert sink_two.events[-1].metadata["denial_reason"] == "ACCOUNT_SCOPE_NOT_GRANTED"
    assert calls == []


@pytest.mark.parametrize(
    ("resolver_result", "context", "reason"),
    [
        (
            RuntimeError("TOPSECRET_RESOLVER_FAILURE"),
            _context(),
            ToolDenialReason.DECISION_UNAVAILABLE,
        ),
        (
            _snapshot(RuntimeMode.PAPER),
            _context(),
            ToolDenialReason.DECISION_CONTEXT_MISMATCH,
        ),
        (
            _snapshot(RuntimeMode.RESEARCH, decision_id="different-decision"),
            _context(),
            ToolDenialReason.DECISION_CONTEXT_MISMATCH,
        ),
        (
            _snapshot(RuntimeMode.RESEARCH, as_of=REQUESTED_AT + timedelta(seconds=1)),
            _context(),
            ToolDenialReason.DECISION_FROM_FUTURE,
        ),
    ],
)
def test_decision_resolution_and_binding_fail_closed(
    resolver_result: object,
    context: ToolCallContext,
    reason: ToolDenialReason,
) -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    resolver = _Resolver(resolver_result)
    registry, _, sink = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(_tool("get_market_snapshot", calls),),
        resolver=resolver,
    )

    with pytest.raises(ToolPermissionDenied) as raised:
        registry.invoke("get_market_snapshot", {}, context)

    assert raised.value.reason is reason
    assert calls == []
    assert resolver.calls == [context.decision_id]
    serialized = sink.events[-1].model_dump_json()
    assert "TOPSECRET" not in serialized
    assert sink.events[-1].metadata["denial_reason"] == reason.value


def test_resolver_is_called_exactly_once_and_handler_receives_detached_snapshot() -> None:
    first = _snapshot(RuntimeMode.RESEARCH)
    second = _snapshot(RuntimeMode.RESEARCH, account_id="changed-account")
    resolver = _SequenceResolver((first, second))
    seen: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(_tool("get_market_snapshot", seen),),
        resolver=resolver,
    )

    registry.invoke("get_market_snapshot", {}, _context())

    assert resolver.calls == 1
    assert seen[0][0].decision_snapshot == first
    assert seen[0][0].decision_snapshot is not first


def test_strict_json_supports_native_enum_datetime_and_decimal_representations() -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                calls,
                arguments_type=RichArguments,
            ),
        ),
    )
    raw = '{"side":"BUY","at":"2026-09-08T07:00:00Z","amount":"12.34","quantity":3,"enabled":true}'

    registry.invoke("get_market_snapshot", raw, _context())

    arguments = calls[0][1]
    assert isinstance(arguments, RichArguments)
    assert arguments.side is Side.BUY
    assert arguments.at == AS_OF
    assert arguments.amount == Decimal("12.34")
    assert arguments.quantity == 3
    assert arguments.enabled is True


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        ("[]", "top_level_object_required"),
        ('{"side":"BUY","side":"SELL"}', "duplicate_json_key"),
        ('{"side":NaN}', "non_finite_number"),
        ({"side": float("inf")}, "non_finite_number"),
        ({1: "non-string-key"}, "invalid_object_key"),
        ({"side": object()}, "non_json_value"),
        ({"side": "x" * 10_001}, "string_too_long"),
    ],
)
def test_structurally_invalid_payloads_are_safely_rejected(raw: Any, expected_code: str) -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, _, sink = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                calls,
                arguments_type=RichArguments,
            ),
        ),
    )

    with pytest.raises(ToolArgumentsRejected) as raised:
        registry.invoke("get_market_snapshot", raw, _context())

    assert expected_code in raised.value.validation_error_codes
    assert calls == []
    assert sink.events[-1].metadata["denial_reason"] == "INVALID_ARGUMENTS"


def test_cycles_depth_size_and_strict_scalar_types_are_rejected() -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                calls,
                arguments_type=RichArguments,
            ),
        ),
    )
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    deep: dict[str, object] = {}
    cursor = deep
    for _ in range(14):
        child: dict[str, object] = {}
        cursor["next"] = child
        cursor = child
    payloads: tuple[object, ...] = (
        cycle,
        deep,
        {"items": list(range(1_001))},
        {
            "side": "BUY",
            "at": "2026-09-08T07:00:00Z",
            "amount": "12.34",
            "quantity": "3",
            "enabled": True,
        },
        {
            "side": "BUY",
            "at": "2026-09-08T07:00:00Z",
            "amount": "12.34",
            "quantity": 3,
            "enabled": 1,
        },
    )

    for payload in payloads:
        with pytest.raises(ToolArgumentsRejected):
            registry.invoke("get_market_snapshot", payload, _context())  # type: ignore[arg-type]
    assert calls == []


def test_validator_type_error_is_redacted_audited_and_never_reaches_handler() -> None:
    class ExplodingArguments(ToolArguments):
        value: str

        @field_validator("value")
        @classmethod
        def explode(cls, value: str) -> str:
            raise TypeError(f"TOPSECRET_VALIDATOR:{value}")

    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, _, sink = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                calls,
                arguments_type=ExplodingArguments,
            ),
        ),
    )

    with pytest.raises(ToolArgumentsRejected) as raised:
        registry.invoke(
            "get_market_snapshot",
            {"value": "TOPSECRET_MODEL_INPUT"},
            _context(),
        )

    assert raised.value.validation_error_codes == ("validator_failed",)
    assert raised.value.__cause__ is None
    assert calls == []
    assert "TOPSECRET" not in sink.events[-1].model_dump_json()
    assert "TOPSECRET" not in str(raised.value)


@pytest.mark.parametrize("idempotency_key", ["", " ", "x" * 129, "bad\nkey"])
def test_write_tool_requires_a_bounded_exact_idempotency_key(idempotency_key: str) -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, _, sink = _build(
        mode=RuntimeMode.PAPER,
        tools=(_tool("create_order_draft", calls),),
    )

    with pytest.raises(ToolArgumentsRejected) as raised:
        registry.invoke(
            "create_order_draft",
            {"idempotency_key": idempotency_key, "order_batch_id": "batch-1"},
            _context(),
        )

    assert raised.value.reason is ToolDenialReason.IDEMPOTENCY_KEY_REQUIRED
    assert calls == []
    assert sink.events[-1].metadata["denial_reason"] == "IDEMPOTENCY_KEY_REQUIRED"


def test_live_authorization_lease_covers_allowed_audit_and_handler_execution() -> None:
    authorizer = _Authorizer()
    sink = _RevokingAuditSink(authorizer)
    observations: list[bool] = []

    def handler(context: ToolExecutionContext, arguments: ToolArguments) -> object:
        observations.append(authorizer.reserved)
        return _response(context, "submit_approved_orders", 1)

    registry, _, _ = _build(
        mode=RuntimeMode.LIVE_ASSISTED,
        tools=(
            _tool(
                "submit_approved_orders",
                [],
                handler_override=handler,
            ),
        ),
        sink=sink,
        authorizer=authorizer,
    )

    registry.invoke(
        "submit_approved_orders",
        {"idempotency_key": "idem-live-1", "order_batch_id": "batch-live-1"},
        _context(),
    )

    assert observations == [True]
    assert authorizer.authorization_calls == 1
    assert authorizer.enter_calls == 1
    assert authorizer.exit_calls == 1
    assert authorizer.reserved is False
    assert sink.events[-1].result == "ALLOWED"


@pytest.mark.parametrize(
    "authorizer",
    (
        None,
        _Authorizer(available=False),
        _Authorizer(fail_before_scope=True),
    ),
)
def test_live_authorization_absence_or_failure_denies_without_handler(
    authorizer: _Authorizer | None,
) -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, resolver, sink = _build(
        mode=RuntimeMode.LIVE_ASSISTED,
        tools=(_tool("submit_approved_orders", calls),),
        authorizer=authorizer,
    )

    with pytest.raises(ToolPermissionDenied) as raised:
        registry.invoke(
            "submit_approved_orders",
            {"idempotency_key": "idem-live-1", "order_batch_id": "batch-live-1"},
            _context(),
        )

    assert raised.value.reason is ToolDenialReason.EXTERNAL_AUTHORIZATION_REQUIRED
    assert calls == []
    if authorizer is None:
        assert resolver.calls == []
    assert sink.events[-1].metadata["denial_reason"] == "EXTERNAL_AUTHORIZATION_REQUIRED"
    assert "TOPSECRET" not in sink.events[-1].model_dump_json()


def test_audit_failure_stops_both_allowed_and_denied_calls() -> None:
    allowed_calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    allowed, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(_tool("get_market_snapshot", allowed_calls),),
        sink=_AuditSink(fail=True),
    )
    with pytest.raises(ToolAuditUnavailable) as allowed_error:
        allowed.invoke("get_market_snapshot", {}, _context())
    assert allowed_error.value.intended_denial is None
    assert allowed_calls == []

    denied_calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    denied, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(_tool("create_order_draft", denied_calls),),
        sink=_AuditSink(fail=True),
    )
    with pytest.raises(ToolAuditUnavailable) as denied_error:
        denied.invoke("create_order_draft", {}, _context())
    assert denied_error.value.intended_denial is not None
    assert denied_error.value.intended_denial.reason is ToolDenialReason.MODE_NOT_ALLOWED
    assert denied_calls == []

    live_calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    authorizer = _Authorizer()
    live, _, _ = _build(
        mode=RuntimeMode.LIVE_ASSISTED,
        tools=(_tool("submit_approved_orders", live_calls),),
        sink=_AuditSink(fail=True),
        authorizer=authorizer,
    )
    with pytest.raises(ToolAuditUnavailable):
        live.invoke(
            "submit_approved_orders",
            {"idempotency_key": "idem-live-1", "order_batch_id": "batch-live-1"},
            _context(),
        )
    assert live_calls == []
    assert authorizer.enter_calls == 1
    assert authorizer.exit_calls == 1
    assert authorizer.reserved is False


def test_invalid_arguments_are_hashed_without_raw_secret_in_audit_or_exception() -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    registry, _, sink = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                calls,
                arguments_type=RichArguments,
            ),
        ),
    )
    raw = {"unexpected": "TOPSECRET_MODEL_VALUE"}

    with pytest.raises(ToolArgumentsRejected) as raised:
        registry.invoke("get_market_snapshot", raw, _context())

    event_json = sink.events[-1].model_dump_json()
    assert "TOPSECRET_MODEL_VALUE" not in event_json
    assert "TOPSECRET_MODEL_VALUE" not in str(raised.value)
    assert len(sink.events[-1].metadata["argument_sha256"]) == 64
    assert sink.events[-1].metadata["validation_error_codes"]
    assert calls == []


def test_handler_exception_is_generic_and_allowed_audit_precedes_execution() -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    sink = _AuditSink()

    def explode(context: ToolExecutionContext, arguments: ToolArguments) -> object:
        assert sink.events[-1].result == "ALLOWED"
        raise RuntimeError("TOPSECRET_HANDLER_FAILURE")

    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                calls,
                handler_override=explode,
            ),
        ),
        sink=sink,
    )

    with pytest.raises(ToolExecutionFailed) as raised:
        registry.invoke("get_market_snapshot", {}, _context())

    assert "TOPSECRET" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert len(calls) == 1
    assert sink.events[-1].result == "ALLOWED"


@pytest.mark.parametrize(
    "change",
    ("request_id", "decision_id", "as_of", "data_version", "inconsistent"),
)
def test_handler_response_must_bind_exact_context_and_consistency(change: str) -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []

    def wrong_response(context: ToolExecutionContext, arguments: ToolArguments) -> object:
        response = _response(context, "get_market_snapshot", 1)
        if change == "request_id":
            return response.model_copy(update={"request_id": "different-request"})
        if change == "decision_id":
            return response.model_copy(update={"decision_id": "different-decision"})
        if change == "as_of":
            return response.model_copy(update={"as_of": response.as_of + timedelta(seconds=1)})
        if change == "data_version":
            provenance = response.provenance.model_copy(update={"data_version": "different-v1"})
            return response.model_copy(update={"provenance": provenance})
        issue = ToolIssue(code=ErrorCode.INTERNAL_ERROR, message="generic failure")
        return response.model_copy(update={"errors": (issue,)})

    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                calls,
                handler_override=wrong_response,
            ),
        ),
    )

    with pytest.raises(ToolOutputRejected) as raised:
        registry.invoke("get_market_snapshot", {}, _context())

    assert raised.value.__cause__ is None


def test_output_requires_exact_closed_type_and_revalidates_corrupted_instance() -> None:
    corrupted = ResultData(tool_name="get_market_snapshot", call_count=1)
    object.__setattr__(corrupted, "call_count", "TOPSECRET_CORRUPTION")

    def corrupted_response(context: ToolExecutionContext, arguments: ToolArguments) -> object:
        response = _response(context, "get_market_snapshot", 1)
        return response.model_copy(update={"data": corrupted})

    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                [],
                handler_override=corrupted_response,
            ),
        ),
    )
    with pytest.raises(ToolOutputRejected) as corrupted_error:
        registry.invoke("get_market_snapshot", {}, _context())
    assert "TOPSECRET" not in str(corrupted_error.value)
    assert corrupted_error.value.__cause__ is None

    def dict_response(context: ToolExecutionContext, arguments: ToolArguments) -> object:
        response = _response(context, "get_market_snapshot", 1)
        return response.model_copy(update={"data": {"approval_token": "TOPSECRET"}})

    registry_two, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                [],
                handler_override=dict_response,
            ),
        ),
    )
    with pytest.raises(ToolOutputRejected):
        registry_two.invoke("get_market_snapshot", {}, _context())


def test_response_subclass_fields_are_removed_by_standard_response_normalization() -> None:
    class ExtendedResponse(ToolResponse[ResultData]):
        extra_secret: str

    def extended_response(context: ToolExecutionContext, arguments: ToolArguments) -> object:
        baseline = _response(context, "get_market_snapshot", 1)
        return ExtendedResponse(
            **baseline.model_dump(),
            extra_secret="TOPSECRET_RESPONSE_SUBCLASS",
        )

    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(
            _tool(
                "get_market_snapshot",
                [],
                handler_override=extended_response,
            ),
        ),
    )

    result = registry.invoke("get_market_snapshot", {}, _context())

    assert type(result) is ToolResponse[object]
    assert not hasattr(result, "extra_secret")
    assert "TOPSECRET_RESPONSE_SUBCLASS" not in result.model_dump_json()


def test_builder_rejects_unknown_reserved_duplicate_and_late_registration() -> None:
    calls: list[tuple[ToolExecutionContext, ToolArguments]] = []
    snapshot = _snapshot(RuntimeMode.RESEARCH)
    builder = AgentToolRegistryBuilder(
        runtime_mode=RuntimeMode.RESEARCH,
        principal_id="agent-worker-1",
        granted_capabilities=frozenset(ToolCapability),
        granted_account_ids=frozenset({snapshot.account_id}),
        decision_resolver=_Resolver(snapshot),
        audit_sink=_AuditSink(),
    )

    with pytest.raises(ToolRegistrationError, match="central"):
        builder.register(_tool("unknown_tool", calls))
    with pytest.raises(ToolRegistrationError, match="human-only"):
        builder.register(_tool("approve_order_batch", calls))

    valid = _tool("get_market_snapshot", calls)
    builder.register(valid)
    with pytest.raises(ToolRegistrationError, match="duplicate"):
        builder.register(valid)

    registry = builder.build()
    with pytest.raises(ToolRegistrationError, match="sealed"):
        builder.register(_tool("validate_market_data", calls))
    with pytest.raises(ToolRegistrationError, match="sealed"):
        builder.build()
    assert registry.registered_tool_names == ("get_market_snapshot",)
    with pytest.raises(AttributeError, match="immutable"):
        registry._runtime_mode = RuntimeMode.PAPER  # type: ignore[misc]


def test_builder_duplicate_registration_is_atomic_under_concurrency() -> None:
    snapshot = _snapshot(RuntimeMode.RESEARCH)
    builder = AgentToolRegistryBuilder(
        runtime_mode=RuntimeMode.RESEARCH,
        principal_id="agent-worker-1",
        granted_capabilities=frozenset(ToolCapability),
        granted_account_ids=frozenset({snapshot.account_id}),
        decision_resolver=_Resolver(snapshot),
        audit_sink=_AuditSink(),
    )
    tools = (_tool("get_market_snapshot", []), _tool("get_market_snapshot", []))

    def register(tool: AgentTool[Any, Any]) -> str:
        try:
            builder.register(tool)
        except ToolRegistrationError:
            return "denied"
        return "registered"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(register, tools))

    assert sorted(outcomes) == ["denied", "registered"]
    assert builder.build().registered_tool_names == ("get_market_snapshot",)


def test_concurrent_permission_denials_each_reach_the_shared_audit_sink() -> None:
    registry, resolver, sink = _build(mode=RuntimeMode.RESEARCH, tools=())

    def deny(index: int) -> str:
        context = ToolCallContext(
            request_id=f"req_concurrent_{index}",
            decision_id="unknown-decision",
            requested_at=REQUESTED_AT,
        )
        with pytest.raises(ToolNotRegistered):
            registry.invoke("unknown_tool", {}, context)
        return context.request_id

    with ThreadPoolExecutor(max_workers=16) as executor:
        request_ids = tuple(executor.map(deny, range(100)))

    assert resolver.calls == []
    assert len(sink.events) == 100
    assert {event.request_id for event in sink.events} == set(request_ids)
    assert all(event.result == "DENIED" for event in sink.events)


@pytest.mark.parametrize(
    "name",
    sorted(HUMAN_ONLY_TOOL_NAMES),
)
def test_every_human_only_name_is_structurally_unregistrable(name: str) -> None:
    snapshot = _snapshot(RuntimeMode.RESEARCH)
    builder = AgentToolRegistryBuilder(
        runtime_mode=RuntimeMode.RESEARCH,
        principal_id="agent-worker-1",
        granted_capabilities=frozenset(ToolCapability),
        granted_account_ids=frozenset({snapshot.account_id}),
        decision_resolver=_Resolver(snapshot),
        audit_sink=_AuditSink(),
    )
    with pytest.raises(ToolRegistrationError, match="human-only"):
        builder.register(_tool(name, []))


def test_builder_rejects_unsafe_input_and_output_schemas() -> None:
    class OpenNested(BaseModel):
        value: int

    class NestedArguments(ToolArguments):
        nested: OpenNested

    class MappingArguments(ToolArguments):
        metadata: dict[str, str]

    class AnyArguments(ToolArguments):
        payload: Any

    class UnicodeAliasArguments(ToolArguments):
        value: str = Field(alias="approv\u0430l_t\u043eken")

    class ContextArguments(ToolArguments):
        account_id: str

    class SecretOutput(ToolOutput):
        approval_token: str

    class MappingOutput(ToolOutput):
        metadata: dict[str, str]

    class AnyOutput(ToolOutput):
        payload: Any

    snapshot = _snapshot(RuntimeMode.RESEARCH)

    for arguments_type, output_type in (
        (NestedArguments, ResultData),
        (MappingArguments, ResultData),
        (AnyArguments, ResultData),
        (UnicodeAliasArguments, ResultData),
        (ContextArguments, ResultData),
        (EmptyArguments, SecretOutput),
        (EmptyArguments, MappingOutput),
        (EmptyArguments, AnyOutput),
    ):
        builder = AgentToolRegistryBuilder(
            runtime_mode=RuntimeMode.RESEARCH,
            principal_id="agent-worker-1",
            granted_capabilities=frozenset(ToolCapability),
            granted_account_ids=frozenset({snapshot.account_id}),
            decision_resolver=_Resolver(snapshot),
            audit_sink=_AuditSink(),
        )
        with pytest.raises(ToolRegistrationError):
            builder.register(
                _tool(
                    "get_market_snapshot",
                    [],
                    arguments_type=arguments_type,
                    output_type=output_type,
                )
            )


def test_builder_requires_exact_idempotency_field_contract() -> None:
    class MissingIdempotency(ToolArguments):
        order_batch_id: str

    class OptionalIdempotency(ToolArguments):
        idempotency_key: str = "default"

    class IntegerIdempotency(ToolArguments):
        idempotency_key: int

    snapshot = _snapshot(RuntimeMode.PAPER)
    for arguments_type in (MissingIdempotency, OptionalIdempotency, IntegerIdempotency):
        builder = AgentToolRegistryBuilder(
            runtime_mode=RuntimeMode.PAPER,
            principal_id="agent-worker-1",
            granted_capabilities=frozenset(ToolCapability),
            granted_account_ids=frozenset({snapshot.account_id}),
            decision_resolver=_Resolver(snapshot),
            audit_sink=_AuditSink(),
        )
        with pytest.raises(ToolRegistrationError, match="idempotency"):
            builder.register(
                _tool(
                    "create_order_draft",
                    [],
                    arguments_type=arguments_type,
                )
            )


@pytest.mark.parametrize("runtime_mode", [RuntimeMode.LIVE_AUTO, "RESEARCH", None])
def test_builder_rejects_live_auto_and_untyped_modes(runtime_mode: object) -> None:
    snapshot = _snapshot(RuntimeMode.RESEARCH)
    with pytest.raises(ToolRegistrationError, match="runtime mode"):
        AgentToolRegistryBuilder(
            runtime_mode=runtime_mode,  # type: ignore[arg-type]
            principal_id="agent-worker-1",
            granted_capabilities=frozenset(ToolCapability),
            granted_account_ids=frozenset({snapshot.account_id}),
            decision_resolver=_Resolver(snapshot),
            audit_sink=_AuditSink(),
        )


def test_context_revalidation_rejects_model_copy_bypass_and_wrong_object() -> None:
    registry, _, _ = _build(
        mode=RuntimeMode.RESEARCH,
        tools=(_tool("get_market_snapshot", []),),
    )
    invalid = _context().model_copy(update={"requested_at": REQUESTED_AT.replace(tzinfo=None)})

    with pytest.raises(ToolRegistryError, match="revalidation"):
        registry.invoke("get_market_snapshot", {}, invalid)
    with pytest.raises(ToolRegistryError, match="exact ToolCallContext"):
        registry.invoke("get_market_snapshot", {}, object())  # type: ignore[arg-type]
