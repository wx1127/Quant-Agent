from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from quant_agent.agent.runtime import AgentRuntime, AgentState
from quant_agent.agent.snapshots import (
    DecisionSnapshot,
    InMemoryDecisionSnapshotStore,
)
from quant_agent.config.models import RuntimeMode

NOW = datetime(2026, 7, 30, 15, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


def snapshot(*, decision_id: str = "dec_20260730_test") -> DecisionSnapshot:
    return DecisionSnapshot(
        decision_id=decision_id,
        mode=RuntimeMode.RESEARCH,
        market="CN_A",
        as_of=NOW,
        data_version="market_v1",
        strategy_version="strategy_v1",
        parameter_version="parameters_v1",
        risk_policy_version="risk_v1",
        account_snapshot_id=None,
        code_commit="abc123",
    )


def test_decision_snapshot_roundtrip_hash_and_append_only_store() -> None:
    item = snapshot()
    restored = DecisionSnapshot.from_json(item.to_json())
    assert restored == item
    assert restored.content_hash == item.content_hash
    store = InMemoryDecisionSnapshotStore()
    store.save(item)
    store.save(item)
    assert store.get(item.decision_id) == item
    with pytest.raises(ValueError, match="immutable"):
        store.save(replace(item, data_version="market_v2"))


def test_trading_snapshot_requires_account_and_create_generates_id() -> None:
    with pytest.raises(ValueError, match="account snapshot"):
        replace(snapshot(), mode=RuntimeMode.PAPER)
    created = DecisionSnapshot.create(
        mode=RuntimeMode.PAPER,
        market="CN_A",
        as_of=NOW,
        data_version="market_v1",
        strategy_version="strategy_v1",
        parameter_version="parameters_v1",
        risk_policy_version="risk_v1",
        account_snapshot_id="acct_snapshot_1",
        code_commit="abc123",
    )
    assert created.decision_id.startswith("dec_20260730_")


def test_runtime_enforces_transitions_call_budget_and_trace() -> None:
    runtime = AgentRuntime(
        decision_id="dec_1", deadline=NOW + timedelta(minutes=5), max_tool_calls=1
    )
    runtime.transition(AgentState.SNAPSHOT_READY, reason="snapshot locked", now=NOW)
    assert runtime.reserve_tool_call(now=NOW) == 1
    with pytest.raises(RuntimeError, match="maximum"):
        runtime.reserve_tool_call(now=NOW)
    with pytest.raises(ValueError, match="illegal"):
        runtime.transition(AgentState.EXECUTING, reason="skip controls", now=NOW)
    assert runtime.trace[0].from_state is AgentState.RECEIVED


def test_runtime_timeout_and_cancel_are_explicit_terminal_states() -> None:
    expired = AgentRuntime(decision_id="dec_1", deadline=NOW)
    with pytest.raises(TimeoutError):
        expired.check_deadline(NOW + timedelta(seconds=1))
    assert expired.state is AgentState.TIMED_OUT
    runtime = AgentRuntime(decision_id="dec_2", deadline=NOW + timedelta(minutes=1))
    runtime.cancel(reason="user cancelled", now=NOW)
    assert runtime.state is AgentState.CANCELLED
    with pytest.raises(ValueError, match="terminal"):
        runtime.cancel(reason="again", now=NOW)
