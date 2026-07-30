from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quant_agent.agent.audit_replay import (
    HarnessReplayService,
    InMemoryHarnessAuditStore,
    ReplayDifferenceKind,
    ToolAuditRecord,
)
from quant_agent.agent.runtime import AgentState, StateTransition
from quant_agent.agent.snapshots import DecisionSnapshot, InMemoryDecisionSnapshotStore
from quant_agent.config.models import RuntimeMode

NOW = datetime(2026, 7, 30, 15, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


def make_replay() -> tuple[HarnessReplayService, InMemoryHarnessAuditStore]:
    snapshot = DecisionSnapshot(
        decision_id="dec_20260730_replay",
        mode=RuntimeMode.RESEARCH,
        market="CN_A",
        as_of=NOW,
        data_version="frozen_market_v1",
        strategy_version="strategy_v1",
        parameter_version="parameters_v1",
        risk_policy_version="risk_v1",
        account_snapshot_id=None,
        code_commit="abc123",
    )
    snapshots = InMemoryDecisionSnapshotStore()
    snapshots.save(snapshot)
    audit = InMemoryHarnessAuditStore()
    audit.append_tool_record(
        ToolAuditRecord.create(
            decision_id=snapshot.decision_id,
            sequence=1,
            tool_name="detect_market_regime",
            occurred_at=NOW,
            arguments={"token": "secret", "as_of": NOW.isoformat()},
            response={"score": 68.2, "data_version": "frozen_market_v1"},
        )
    )
    audit.save_state_trace(
        snapshot.decision_id,
        (
            StateTransition(
                1,
                AgentState.RECEIVED,
                AgentState.SNAPSHOT_READY,
                NOW,
                "snapshot locked",
            ),
        ),
    )
    audit.save_final_response(snapshot.decision_id, {"summary": "range strong"})
    return HarnessReplayService(snapshots, audit), audit


def test_replay_reconstructs_snapshot_records_and_redacts_secrets() -> None:
    service, _ = make_replay()
    replay = service.replay("dec_20260730_replay")
    assert replay.snapshot.data_version == "frozen_market_v1"
    assert replay.tool_records[0].arguments["token"] == "[REDACTED]"
    assert replay.state_trace[0].to_state is AgentState.SNAPSHOT_READY
    assert not service.compare(
        "dec_20260730_replay",
        [{"score": 68.2, "data_version": "frozen_market_v1"}],
    )


def test_replay_reports_hash_differences_and_missing_fields() -> None:
    service, _ = make_replay()
    differences = service.compare("dec_20260730_replay", [{"score": 99.0}])
    assert differences[0].kind is ReplayDifferenceKind.HASH_MISMATCH

    snapshot = service._snapshots.get("dec_20260730_replay")
    snapshots = InMemoryDecisionSnapshotStore()
    snapshots.save(snapshot)
    incomplete = HarnessReplayService(snapshots, InMemoryHarnessAuditStore())
    with pytest.raises(ValueError, match="critical audit fields"):
        incomplete.replay(snapshot.decision_id)


def test_audit_store_rejects_non_contiguous_sequence_and_tampering() -> None:
    _, audit = make_replay()
    with pytest.raises(ValueError, match="sequence"):
        audit.append_tool_record(
            ToolAuditRecord.create(
                decision_id="dec_20260730_replay",
                sequence=3,
                tool_name="rank_market_themes",
                occurred_at=NOW,
                arguments={},
                response={},
            )
        )
    original = audit.tool_records("dec_20260730_replay")[0]
    tampered = replace(original, response={"score": -1})
    with pytest.raises(ValueError, match="hash mismatch"):
        tampered.verify()
