from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from quant_agent.agent.audit_replay import InMemoryHarnessAuditStore
from quant_agent.agent.runtime import AgentRuntime, AgentState
from quant_agent.agent.snapshots import DecisionSnapshot
from quant_agent.agent.tools.adapters import (
    register_portfolio_execution_tools,
    register_research_tools,
)
from quant_agent.agent.tools.registry import (
    AgentTool,
    ToolCallContext,
    ToolCategory,
    ToolEffect,
    ToolRegistry,
)
from quant_agent.config.models import RuntimeMode
from quant_agent.core.errors import ErrorCode
from quant_agent.observability.audit import AuditEvent

NOW = datetime(2026, 7, 30, 15, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


class MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def make_context(
    *,
    mode: RuntimeMode = RuntimeMode.RESEARCH,
    state: AgentState = AgentState.SNAPSHOT_READY,
    idempotency_key: str | None = None,
) -> tuple[ToolCallContext, MemoryAuditSink]:
    account = (
        "account_snapshot_1" if mode in {RuntimeMode.PAPER, RuntimeMode.LIVE_ASSISTED} else None
    )
    item = DecisionSnapshot(
        decision_id="dec_20260730_tools",
        mode=mode,
        market="CN_A",
        as_of=NOW,
        data_version="market_v1",
        strategy_version="strategy_v1",
        parameter_version="parameters_v1",
        risk_policy_version="risk_v1",
        account_snapshot_id=account,
        code_commit="abc123",
    )
    runtime = AgentRuntime(
        decision_id=item.decision_id,
        deadline=NOW + timedelta(days=3650),
        max_tool_calls=20,
    )
    paths = {
        AgentState.SNAPSHOT_READY: (AgentState.SNAPSHOT_READY,),
        AgentState.ANALYZED: (AgentState.SNAPSHOT_READY, AgentState.ANALYZED),
        AgentState.RISK_CHECKED: (
            AgentState.SNAPSHOT_READY,
            AgentState.ANALYZED,
            AgentState.RISK_CHECKED,
        ),
        AgentState.EXECUTING: (
            AgentState.SNAPSHOT_READY,
            AgentState.ANALYZED,
            AgentState.RISK_CHECKED,
            AgentState.PENDING_APPROVAL,
            AgentState.EXECUTING,
        ),
        AgentState.RECONCILING: (
            AgentState.SNAPSHOT_READY,
            AgentState.ANALYZED,
            AgentState.RISK_CHECKED,
            AgentState.PENDING_APPROVAL,
            AgentState.EXECUTING,
            AgentState.RECONCILING,
        ),
    }
    for target in paths[state]:
        runtime.transition(target, reason="test setup", now=NOW)
    audit = MemoryAuditSink()
    return (
        ToolCallContext("req_1", "agent", item, runtime, audit, idempotency_key),
        audit,
    )


def test_registry_denies_unregistered_wrong_mode_and_records_audit() -> None:
    registry = ToolRegistry()
    context, audit = make_context()
    replay = InMemoryHarnessAuditStore()
    context.replay_sink = replay
    result = registry.invoke("unknown", {"token": "secret"}, context)
    assert not result.ok
    assert result.errors[0].code is ErrorCode.FORBIDDEN
    assert audit.events[-1].result == "denied"
    record = replay.tool_records(context.snapshot.decision_id)[0]
    assert record.arguments["token"] == "[REDACTED]"

    registry.register(
        AgentTool(
            "paper_write",
            ToolCategory.PAPER_EXECUTION,
            ToolEffect.WRITE,
            Arguments,
            lambda args, context: {"value": args.value},
            frozenset({RuntimeMode.PAPER}),
            frozenset({AgentState.EXECUTING}),
            "paper",
            "v1",
        )
    )
    result = registry.invoke("paper_write", {"value": 1}, context)
    assert not result.ok
    assert "runtime mode" in result.errors[0].message


def test_registry_validates_arguments_and_write_idempotency() -> None:
    registry = ToolRegistry()
    registry.register(
        AgentTool(
            "paper_write",
            ToolCategory.PAPER_EXECUTION,
            ToolEffect.WRITE,
            Arguments,
            lambda args, context: {"value": args.value},
            frozenset({RuntimeMode.PAPER}),
            frozenset({AgentState.EXECUTING}),
            "paper",
            "v1",
        )
    )
    context, _ = make_context(mode=RuntimeMode.PAPER, state=AgentState.EXECUTING)
    assert not registry.invoke("paper_write", {"value": 1}, context).ok
    context.idempotency_key = "idem_1"
    invalid = registry.invoke("paper_write", {"value": 1, "extra": True}, context)
    assert invalid.errors[0].code is ErrorCode.INVALID_ARGUMENT
    result = registry.invoke("paper_write", {"value": 3}, context)
    assert result.ok and result.data == {"value": 3}
    assert result.decision_id == context.snapshot.decision_id
    assert result.provenance.data_version == "market_v1"


def test_approval_tool_is_never_registerable() -> None:
    try:
        AgentTool(
            "approve_order_batch",
            ToolCategory.ORDER_DRAFT,
            ToolEffect.WRITE,
            Arguments,
            lambda args, context: None,
            frozenset({RuntimeMode.PAPER}),
            frozenset({AgentState.RISK_CHECKED}),
            "approval",
            "v1",
        )
    except ValueError as exc:
        assert "human approval" in str(exc)
    else:
        raise AssertionError("approval tool must be rejected")


def handler(
    arguments: dict[str, Any], *, decision_id: str, as_of: datetime, data_version: str
) -> dict[str, Any]:
    return {
        "arguments": arguments,
        "decision_id": decision_id,
        "as_of": as_of.isoformat(),
        "data_version": data_version,
    }


def passed_risk_handler(
    arguments: dict[str, Any], *, decision_id: str, as_of: datetime, data_version: str
) -> dict[str, Any]:
    return {
        "passed": True,
        "request_id": "risk_1",
        "decision_id": decision_id,
        "as_of": as_of.isoformat(),
        "data_version": data_version,
    }


def test_research_adapters_are_standard_read_only_responses() -> None:
    registry = ToolRegistry()
    register_research_tools(
        registry,
        {
            "get_market_snapshot": handler,
            "detect_market_regime": handler,
            "rank_market_themes": handler,
            "rank_theme_leaders": handler,
            "rank_stock_candidates": handler,
            "explain_candidate": handler,
        },
    )
    assert "approve_order_batch" not in registry.names
    context, _ = make_context()
    result = registry.invoke("detect_market_regime", {}, context)
    assert result.ok
    assert result.data["decision_id"] == context.snapshot.decision_id
    assert result.as_of == NOW


def test_portfolio_adapter_requires_passed_risk_and_paper_isolation() -> None:
    registry = ToolRegistry()
    register_portfolio_execution_tools(
        registry,
        {
            "check_portfolio_risk": passed_risk_handler,
            "create_order_draft": handler,
            "submit_paper_orders": handler,
            "reconcile_account": handler,
        },
    )
    context, _ = make_context(
        mode=RuntimeMode.PAPER,
        state=AgentState.ANALYZED,
        idempotency_key="idem_1",
    )
    context.runtime.transition(AgentState.RISK_CHECKED, reason="unverified", now=NOW)
    failed = registry.invoke(
        "create_order_draft",
        {"risk_passed": False, "risk_request_id": "risk_1", "payload": {}},
        context,
    )
    assert not failed.ok
    assert failed.errors[0].code is ErrorCode.INVALID_ARGUMENT
    fabricated = registry.invoke(
        "create_order_draft",
        {"risk_passed": True, "risk_request_id": "risk_1", "payload": {}},
        context,
    )
    assert fabricated.errors[0].code is ErrorCode.RISK_REJECTED
    context.runtime = make_context(
        mode=RuntimeMode.PAPER,
        state=AgentState.ANALYZED,
        idempotency_key="idem_1",
    )[0].runtime
    risk_result = registry.invoke("check_portfolio_risk", {"payload": {}}, context)
    assert risk_result.ok
    context.runtime.transition(AgentState.RISK_CHECKED, reason="risk passed", now=NOW)
    passed = registry.invoke(
        "create_order_draft",
        {"risk_passed": True, "risk_request_id": "risk_1", "payload": {}},
        context,
    )
    assert passed.ok
    live_context, _ = make_context(
        mode=RuntimeMode.LIVE_ASSISTED,
        state=AgentState.EXECUTING,
        idempotency_key="idem_2",
    )
    assert not registry.invoke("submit_paper_orders", {"payload": {}}, live_context).ok
