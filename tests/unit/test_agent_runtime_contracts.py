"""Strict contracts, workflow rules, and CAS repository tests for Agent runs."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
from pydantic import ValidationError

from quant_agent.agent.runtime.contracts import (
    TERMINAL_AGENT_RUN_STATES,
    AgentArtifactKind,
    AgentArtifactRef,
    AgentExternalApproval,
    AgentPendingToolCall,
    AgentRunEvent,
    AgentRunEventKind,
    AgentRunGoal,
    AgentRunLimits,
    AgentRunSnapshot,
    AgentRunState,
    AgentRunTrigger,
)
from quant_agent.agent.runtime.replay import replay_agent_run
from quant_agent.agent.runtime.repository import (
    AgentRunConflict,
    AgentRunNotFound,
    InMemoryAgentRunRepository,
)
from quant_agent.agent.runtime.rules import (
    GOAL_MODE_COMPATIBILITY,
    AgentGoalModeMismatch,
    InvalidAgentStateTransition,
    legal_transition_targets,
    report_ready,
    stage_tool_names,
    validate_goal_mode,
    validate_transition,
)
from quant_agent.agent.tools.contracts import ToolEffect
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ

LOCAL_NOW = datetime(2026, 9, 9, 15, 0, tzinfo=SHANGHAI_TZ)

_TOOL_EVENT_KINDS = frozenset(
    {
        AgentRunEventKind.TOOL_CALL_RESERVED,
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
    }
)
_TOOL_OUTCOME_EVENT_KINDS = _TOOL_EVENT_KINDS - {AgentRunEventKind.TOOL_CALL_RESERVED}
_APPROVAL_CONTROL_EVENT_KINDS = frozenset(
    {
        AgentRunEventKind.APPROVAL_RECORDED,
        AgentRunEventKind.APPROVAL_REJECTED,
    }
)
_NON_TOOL_EVENT_KINDS = tuple(kind for kind in AgentRunEventKind if kind not in _TOOL_EVENT_KINDS)
_BATCH_HASH_FORBIDDEN_EVENT_KINDS = tuple(
    kind
    for kind in AgentRunEventKind
    if kind not in _TOOL_EVENT_KINDS | _APPROVAL_CONTROL_EVENT_KINDS
)
_OUTCOME_CODE_FORBIDDEN_EVENT_KINDS = tuple(
    kind for kind in AgentRunEventKind if kind not in _TOOL_OUTCOME_EVENT_KINDS
)
_EVENT_REVALIDATION_PATHS = (
    "strict-payload",
    "copied-instance",
    "copied-json",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _start(
    *,
    run_id: str = "run_contract_001",
    decision_id: str = "decision_contract_001",
    decision_hash: str | None = None,
    goal: AgentRunGoal = AgentRunGoal.MARKET_RESEARCH,
    mode: RuntimeMode = RuntimeMode.RESEARCH,
    limits: AgentRunLimits | None = None,
) -> AgentRunSnapshot:
    return AgentRunSnapshot.start(
        run_id=run_id,
        decision_id=decision_id,
        decision_snapshot_hash=decision_hash or _digest("decision"),
        runtime_mode=mode,
        goal=goal,
        created_at=LOCAL_NOW,
        limits=limits,
    )


def _approval_event(**changes: object) -> AgentRunEvent:
    approval = AgentExternalApproval(
        approval_id="approval_event_001",
        approval_hash=_digest("approval-event-control"),
        decision_id="decision_approval_event_001",
        batch_hash=_digest("approval-event-batch"),
        approved_at=LOCAL_NOW,
        expires_at=LOCAL_NOW + timedelta(minutes=5),
    )
    values: dict[str, object] = {
        "run_id": "run_approval_event_001",
        "decision_id": approval.decision_id,
        "decision_snapshot_hash": _digest("approval-event-decision"),
        "runtime_mode": RuntimeMode.LIVE_ASSISTED,
        "goal": AgentRunGoal.LIVE_ASSISTED_EXECUTION,
        "sequence": 1,
        "kind": AgentRunEventKind.APPROVAL_RECORDED,
        "trigger": AgentRunTrigger.HUMAN,
        "state_before": AgentRunState.PENDING_APPROVAL,
        "state_after": AgentRunState.APPROVED,
        "reason": "APPROVAL_RECORDED",
        "occurred_at": LOCAL_NOW + timedelta(seconds=1),
        "batch_hash": approval.batch_hash,
        "control_hash": approval.approval_hash,
        "external_approval": approval,
        "previous_event_hash": _digest("approval-event-previous"),
    }
    values.update(changes)
    return AgentRunEvent.build(**values)  # type: ignore[arg-type]


def _event_for_kind(kind: AgentRunEventKind) -> AgentRunEvent:
    """Build one valid contract-level baseline for every event kind."""

    if kind is AgentRunEventKind.RUN_CREATED:
        return _start(
            run_id="run_isolation_created_001",
            decision_id="decision_isolation_created_001",
        ).events[0]
    if kind is AgentRunEventKind.APPROVAL_RECORDED:
        return _approval_event()

    suffix = kind.value.lower()
    values: dict[str, object] = {
        "run_id": f"run_isolation_{suffix}",
        "decision_id": f"decision_isolation_{suffix}",
        "decision_snapshot_hash": _digest(f"decision-isolation-{suffix}"),
        "runtime_mode": RuntimeMode.RESEARCH,
        "goal": AgentRunGoal.MARKET_RESEARCH,
        "sequence": 1,
        "kind": kind,
        "trigger": AgentRunTrigger.SYSTEM,
        "state_before": AgentRunState.RECEIVED,
        "state_after": (
            AgentRunState.SNAPSHOT_READY
            if kind is AgentRunEventKind.STATE_TRANSITION
            else AgentRunState.RECEIVED
        ),
        "reason": f"ISOLATION_{kind.value}",
        "occurred_at": LOCAL_NOW,
        "previous_event_hash": _digest(f"previous-isolation-{suffix}"),
    }
    if kind in _TOOL_EVENT_KINDS:
        values.update(
            trigger=AgentRunTrigger.TOOL,
            request_id=f"request_isolation_{suffix}",
            tool_name="get_market_snapshot",
            tool_effect=ToolEffect.READ_ONLY,
            argument_hash=_digest(f"arguments-isolation-{suffix}"),
        )
        if kind in {
            AgentRunEventKind.TOOL_CALL_SUCCEEDED,
            AgentRunEventKind.TOOL_CALL_REPLAYED,
        }:
            values["response_hash"] = _digest(f"response-isolation-{suffix}")
        elif kind in {
            AgentRunEventKind.TOOL_CALL_FAILED,
            AgentRunEventKind.TOOL_CALL_REJECTED,
        }:
            values["error_codes"] = ("EXPECTED_TOOL_ERROR",)
    elif kind is AgentRunEventKind.APPROVAL_REJECTED:
        values.update(
            runtime_mode=RuntimeMode.LIVE_ASSISTED,
            goal=AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            trigger=AgentRunTrigger.HUMAN,
            state_before=AgentRunState.PENDING_APPROVAL,
            state_after=AgentRunState.REJECTED,
            control_hash=_digest("approval-rejected-control"),
        )
    return AgentRunEvent.build(**values)  # type: ignore[arg-type]


def _event_build_values(event: AgentRunEvent) -> dict[str, object]:
    return event.model_dump(
        mode="python",
        exclude={"schema_version", "event_hash"},
    )


def _revalidate_event_mutation(
    event: AgentRunEvent,
    changes: dict[str, object],
    path: str,
) -> AgentRunEvent:
    if path == "strict-payload":
        values = event.model_dump(mode="python")
        values.update(changes)
        return AgentRunEvent.model_validate(values, strict=True)

    tampered = event.model_copy(update=changes)
    if path == "copied-instance":
        return AgentRunEvent.model_validate(tampered, strict=True)
    if path == "copied-json":
        return AgentRunEvent.from_json(tampered.model_dump_json())
    raise AssertionError(f"unknown event revalidation path: {path}")


def _append(
    snapshot: AgentRunSnapshot,
    state_after: AgentRunState,
    *,
    kind: AgentRunEventKind = AgentRunEventKind.STATE_TRANSITION,
    trigger: AgentRunTrigger = AgentRunTrigger.SYSTEM,
    reason: str = "TEST_TRANSITION",
    control_hash: str | None = None,
) -> AgentRunSnapshot:
    occurred_at = snapshot.updated_at + timedelta(seconds=1)
    event = AgentRunEvent.build(
        run_id=snapshot.run_id,
        decision_id=snapshot.decision_id,
        decision_snapshot_hash=snapshot.decision_snapshot_hash,
        runtime_mode=snapshot.runtime_mode,
        goal=snapshot.goal,
        sequence=snapshot.revision + 1,
        kind=kind,
        trigger=trigger,
        state_before=snapshot.state,
        state_after=state_after,
        reason=reason,
        occurred_at=occurred_at,
        control_hash=control_hash,
        previous_event_hash=snapshot.events[-1].event_hash,
    )
    values = snapshot.model_dump(mode="python")
    values.update(
        {
            "state": state_after,
            "updated_at": occurred_at,
            "revision": snapshot.revision + 1,
            "events": (*snapshot.events, event),
            "cancel_requested": (
                snapshot.cancel_requested or kind is AgentRunEventKind.CANCELLATION_REQUESTED
            ),
        }
    )
    if state_after in TERMINAL_AGENT_RUN_STATES:
        values.update(completed_at=occurred_at, terminal_reason=reason)
    return AgentRunSnapshot.model_validate(values, strict=True)


def _snapshot_with_completed_read_call() -> AgentRunSnapshot:
    snapshot = _append(_start(), AgentRunState.SNAPSHOT_READY)
    started_at = snapshot.updated_at + timedelta(seconds=1)
    argument_hash = _digest("market-arguments")
    pending = AgentPendingToolCall(
        request_id="request_market_001",
        tool_name="get_market_snapshot",
        argument_hash=argument_hash,
        started_at=started_at,
        effect=ToolEffect.READ_ONLY,
    )
    reserved = AgentRunEvent.build(
        run_id=snapshot.run_id,
        decision_id=snapshot.decision_id,
        decision_snapshot_hash=snapshot.decision_snapshot_hash,
        runtime_mode=snapshot.runtime_mode,
        goal=snapshot.goal,
        sequence=snapshot.revision + 1,
        kind=AgentRunEventKind.TOOL_CALL_RESERVED,
        trigger=AgentRunTrigger.TOOL,
        state_before=snapshot.state,
        state_after=snapshot.state,
        reason="TOOL_CALL_RESERVED",
        occurred_at=started_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        previous_event_hash=snapshot.events[-1].event_hash,
    )
    reserved_values = snapshot.model_dump(mode="python")
    reserved_values.update(
        updated_at=started_at,
        revision=snapshot.revision + 1,
        tool_call_count=1,
        pending_call=pending,
        events=(*snapshot.events, reserved),
    )
    with_pending = AgentRunSnapshot.model_validate(reserved_values, strict=True)

    response_hash = _digest("market-response")
    artifact = AgentArtifactRef(
        kind=AgentArtifactKind.MARKET_SNAPSHOT,
        artifact_id="market_snapshot_001",
        content_hash=_digest("market-content"),
        tool_name=pending.tool_name,
        response_hash=response_hash,
    )
    finished_at = started_at + timedelta(seconds=1)
    succeeded = AgentRunEvent.build(
        run_id=with_pending.run_id,
        decision_id=with_pending.decision_id,
        decision_snapshot_hash=with_pending.decision_snapshot_hash,
        runtime_mode=with_pending.runtime_mode,
        goal=with_pending.goal,
        sequence=with_pending.revision + 1,
        kind=AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        trigger=AgentRunTrigger.TOOL,
        state_before=with_pending.state,
        state_after=AgentRunState.DATA_VALIDATED,
        reason="MARKET_SNAPSHOT_READY",
        occurred_at=finished_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        response_hash=response_hash,
        artifact=artifact,
        warning_codes=("PARTIAL_SOURCE_WARNING",),
        previous_event_hash=with_pending.events[-1].event_hash,
    )
    completed_values = with_pending.model_dump(mode="python")
    completed_values.update(
        state=AgentRunState.DATA_VALIDATED,
        updated_at=finished_at,
        revision=with_pending.revision + 1,
        pending_call=None,
        artifacts=(artifact,),
        events=(*with_pending.events, succeeded),
    )
    return AgentRunSnapshot.model_validate(completed_values, strict=True)


def _snapshot_with_rejected_call() -> AgentRunSnapshot:
    snapshot = _append(_start(), AgentRunState.SNAPSHOT_READY)
    started_at = snapshot.updated_at + timedelta(seconds=1)
    argument_hash = _digest("rejected-arguments")
    pending = AgentPendingToolCall(
        request_id="request_rejected_001",
        tool_name="get_market_snapshot",
        argument_hash=argument_hash,
        started_at=started_at,
        effect=ToolEffect.READ_ONLY,
    )
    reserved = AgentRunEvent.build(
        run_id=snapshot.run_id,
        decision_id=snapshot.decision_id,
        decision_snapshot_hash=snapshot.decision_snapshot_hash,
        runtime_mode=snapshot.runtime_mode,
        goal=snapshot.goal,
        sequence=snapshot.revision + 1,
        kind=AgentRunEventKind.TOOL_CALL_RESERVED,
        trigger=AgentRunTrigger.TOOL,
        state_before=snapshot.state,
        state_after=snapshot.state,
        reason="TOOL_CALL_RESERVED",
        occurred_at=started_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        previous_event_hash=snapshot.events[-1].event_hash,
    )
    rejected_at = started_at + timedelta(seconds=1)
    rejected = AgentRunEvent.build(
        run_id=snapshot.run_id,
        decision_id=snapshot.decision_id,
        decision_snapshot_hash=snapshot.decision_snapshot_hash,
        runtime_mode=snapshot.runtime_mode,
        goal=snapshot.goal,
        sequence=snapshot.revision + 2,
        kind=AgentRunEventKind.TOOL_CALL_REJECTED,
        trigger=AgentRunTrigger.TOOL,
        state_before=snapshot.state,
        state_after=snapshot.state,
        reason="TOOL_CALL_REJECTED",
        occurred_at=rejected_at,
        request_id=pending.request_id,
        tool_name=pending.tool_name,
        tool_effect=pending.effect,
        argument_hash=pending.argument_hash,
        error_codes=("OUT_OF_STAGE",),
        previous_event_hash=reserved.event_hash,
    )
    values = snapshot.model_dump(mode="python")
    values.update(
        updated_at=rejected_at,
        revision=snapshot.revision + 2,
        tool_call_count=1,
        invalid_tool_call_count=1,
        events=(*snapshot.events, reserved, rejected),
    )
    return AgentRunSnapshot.model_validate(values, strict=True)


_VALID_GOAL_MODES = tuple(
    (goal, mode)
    for goal, modes in GOAL_MODE_COMPATIBILITY.items()
    for mode in sorted(modes, key=lambda item: item.value)
)
_INVALID_GOAL_MODES = tuple(
    (goal, mode)
    for goal in AgentRunGoal
    for mode in RuntimeMode
    if mode not in GOAL_MODE_COMPATIBILITY[goal]
)


@pytest.mark.parametrize(
    "field,value",
    (
        ("max_tool_calls", 0),
        ("max_tool_calls", 1_001),
        ("max_invalid_tool_calls", 0),
        ("max_reconciliation_attempts", 101),
        ("timeout_seconds", 604_801),
        ("approval_timeout_seconds", 86_401),
        ("max_tool_calls", True),
        ("timeout_seconds", "300"),
    ),
)
def test_limits_are_strict_positive_and_bounded(field: str, value: object) -> None:
    values = AgentRunLimits().model_dump(mode="python")
    values[field] = value

    with pytest.raises(ValidationError):
        AgentRunLimits.model_validate(values, strict=True)


def test_models_are_frozen_and_forbid_extra_fields() -> None:
    limits = AgentRunLimits()

    with pytest.raises(ValidationError):
        limits.max_tool_calls = 99
    with pytest.raises(ValidationError):
        AgentRunLimits.model_validate({**limits.model_dump(), "extra": 1}, strict=True)
    with pytest.raises(ValidationError, match="max_invalid_tool_calls"):
        AgentRunLimits(max_tool_calls=1, max_invalid_tool_calls=2)


def test_artifact_hash_time_and_expiry_scope_are_strict() -> None:
    expires_at = LOCAL_NOW + timedelta(minutes=5)
    draft = AgentArtifactRef(
        kind=AgentArtifactKind.ORDER_DRAFT,
        artifact_id="draft_001",
        content_hash=_digest("batch"),
        tool_name="create_order_draft",
        response_hash=_digest("response"),
        truncated=False,
        expires_at=expires_at,
    )

    assert draft.expires_at == expires_at.astimezone(UTC)
    with pytest.raises(ValidationError, match="ORDER_DRAFT"):
        AgentArtifactRef(
            kind=AgentArtifactKind.MARKET_SNAPSHOT,
            artifact_id="market_001",
            content_hash=_digest("market"),
            tool_name="get_market_snapshot",
            response_hash=_digest("market-response"),
            expires_at=expires_at,
        )
    with pytest.raises(ValidationError):
        AgentArtifactRef(
            kind=AgentArtifactKind.ORDER_DRAFT,
            artifact_id="draft_001",
            content_hash="A" * 64,
            tool_name="CreateOrderDraft",
            response_hash=_digest("response"),
        )
    with pytest.raises(ValidationError):
        AgentArtifactRef(
            kind=AgentArtifactKind.ORDER_DRAFT,
            artifact_id="draft_001",
            content_hash=_digest("batch"),
            tool_name="create_order_draft",
            response_hash=_digest("response"),
            expires_at=datetime(2026, 9, 9, 15, 5),
        )


@pytest.mark.parametrize(
    "effect,idempotency_hash,batch_hash",
    (
        (ToolEffect.ARTIFACT_WRITE, None, None),
        (ToolEffect.PAPER_EXECUTION_WRITE, _digest("key"), None),
        (ToolEffect.LIVE_EXTERNAL_WRITE, None, _digest("batch")),
    ),
)
def test_pending_write_calls_require_stable_identities(
    effect: ToolEffect,
    idempotency_hash: str | None,
    batch_hash: str | None,
) -> None:
    with pytest.raises(ValidationError):
        AgentPendingToolCall(
            request_id="request_001",
            tool_name="submit_paper_orders",
            argument_hash=_digest("arguments"),
            idempotency_key_hash=idempotency_hash,
            batch_hash=batch_hash,
            started_at=LOCAL_NOW,
            effect=effect,
        )


def test_pending_read_call_normalizes_aware_time() -> None:
    pending = AgentPendingToolCall(
        request_id="request_001",
        tool_name="get_market_snapshot",
        argument_hash=_digest("arguments"),
        started_at=LOCAL_NOW,
        effect=ToolEffect.READ_ONLY,
    )

    assert pending.started_at == LOCAL_NOW.astimezone(UTC)


def test_external_approval_is_hash_bound_and_has_a_strict_window() -> None:
    approval = AgentExternalApproval(
        approval_id="approval_001",
        approval_hash=_digest("approval"),
        decision_id="decision_contract_001",
        batch_hash=_digest("batch"),
        approved_at=LOCAL_NOW,
        expires_at=LOCAL_NOW + timedelta(minutes=5),
    )

    assert approval.approved_at.tzinfo is UTC
    assert AgentExternalApproval.model_validate_json(approval.model_dump_json()) == approval
    with pytest.raises(ValidationError, match="expire after"):
        AgentExternalApproval(
            **{
                **approval.model_dump(mode="python"),
                "expires_at": approval.approved_at,
            }
        )
    with pytest.raises(ValidationError):
        AgentExternalApproval(
            **{
                **approval.model_dump(mode="python"),
                "approval_hash": "not-a-hash",
            }
        )


def test_approval_event_retains_complete_strict_evidence_in_json() -> None:
    event = _approval_event()
    restored = AgentRunEvent.from_json(event.to_json())

    assert restored == event
    assert restored.external_approval is not None
    assert restored.external_approval.approval_hash == restored.control_hash
    assert restored.external_approval.decision_id == restored.decision_id
    assert restored.external_approval.batch_hash == restored.batch_hash
    assert restored.external_approval.approved_at <= restored.occurred_at
    assert restored.occurred_at < restored.external_approval.expires_at

    python_payload = restored.external_approval.model_dump(mode="python")
    python_payload["approved_at"] = restored.external_approval.approved_at.isoformat()
    with pytest.raises(ValidationError):
        AgentExternalApproval.model_validate(python_payload, strict=True)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-evidence",
        "control-hash",
        "decision-id",
        "batch-hash",
        "before-approved-at",
        "at-expiry",
        "evidence-on-rejection",
    ),
)
def test_approval_event_rejects_unbound_or_misplaced_evidence(mutation: str) -> None:
    baseline = _approval_event()
    approval = baseline.external_approval
    assert approval is not None
    changes: dict[str, object]
    if mutation == "missing-evidence":
        changes = {"external_approval": None}
    elif mutation == "control-hash":
        changes = {"control_hash": _digest("different-control")}
    elif mutation == "decision-id":
        changes = {"decision_id": "decision_different_001"}
    elif mutation == "batch-hash":
        changes = {"batch_hash": _digest("different-batch")}
    elif mutation == "before-approved-at":
        changes = {"occurred_at": approval.approved_at - timedelta(microseconds=1)}
    elif mutation == "at-expiry":
        changes = {"occurred_at": approval.expires_at}
    else:
        changes = {
            "kind": AgentRunEventKind.APPROVAL_REJECTED,
            "state_after": AgentRunState.REJECTED,
        }

    with pytest.raises(ValidationError, match="approval"):
        _approval_event(**changes)


def test_approval_event_json_rejects_nested_evidence_tampering() -> None:
    event = _approval_event()
    payload = json.loads(event.to_json())
    payload["external_approval"]["approval_id"] = "approval_tampered_001"

    with pytest.raises(ValidationError, match="event_hash"):
        AgentRunEvent.from_json(json.dumps(payload))


@pytest.mark.parametrize(("goal", "mode"), _VALID_GOAL_MODES)
def test_goal_mode_matrix_accepts_only_explicit_pairs(
    goal: AgentRunGoal,
    mode: RuntimeMode,
) -> None:
    validate_goal_mode(goal, mode)
    assert _start(goal=goal, mode=mode).runtime_mode is mode


@pytest.mark.parametrize(("goal", "mode"), _INVALID_GOAL_MODES)
def test_goal_mode_matrix_rejects_every_other_pair(
    goal: AgentRunGoal,
    mode: RuntimeMode,
) -> None:
    with pytest.raises(AgentGoalModeMismatch):
        validate_goal_mode(goal, mode)
    with pytest.raises(ValidationError, match="not permitted"):
        _start(goal=goal, mode=mode)


def test_goal_mode_rule_rejects_non_enum_inputs() -> None:
    with pytest.raises(AgentGoalModeMismatch, match="exact AgentRunGoal"):
        validate_goal_mode("MARKET_RESEARCH", RuntimeMode.RESEARCH)  # type: ignore[arg-type]
    with pytest.raises(AgentGoalModeMismatch, match="unsupported"):
        validate_goal_mode(AgentRunGoal.MARKET_RESEARCH, "RESEARCH")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("goal", "mode", "before", "after", "kind"),
    (
        (
            AgentRunGoal.MARKET_RESEARCH,
            RuntimeMode.RESEARCH,
            AgentRunState.RECEIVED,
            AgentRunState.SNAPSHOT_READY,
            AgentRunEventKind.STATE_TRANSITION,
        ),
        (
            AgentRunGoal.BACKTEST_REPORT,
            RuntimeMode.BACKTEST,
            AgentRunState.DATA_VALIDATED,
            AgentRunState.ANALYZED,
            AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        ),
        (
            AgentRunGoal.PORTFOLIO_REPORT,
            RuntimeMode.PAPER,
            AgentRunState.ANALYZED,
            AgentRunState.PORTFOLIO_READY,
            AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        ),
        (
            AgentRunGoal.ORDER_DRAFT,
            RuntimeMode.PAPER,
            AgentRunState.RISK_CHECKED,
            AgentRunState.DRAFT_READY,
            AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        ),
        (
            AgentRunGoal.PAPER_EXECUTION,
            RuntimeMode.PAPER,
            AgentRunState.DRAFT_READY,
            AgentRunState.EXECUTING,
            AgentRunEventKind.STATE_TRANSITION,
        ),
        (
            AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            RuntimeMode.LIVE_ASSISTED,
            AgentRunState.PENDING_APPROVAL,
            AgentRunState.APPROVED,
            AgentRunEventKind.APPROVAL_RECORDED,
        ),
        (
            AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            RuntimeMode.LIVE_ASSISTED,
            AgentRunState.EXECUTING,
            AgentRunState.RECONCILING,
            AgentRunEventKind.STATE_TRANSITION,
        ),
        (
            AgentRunGoal.PAPER_EXECUTION,
            RuntimeMode.PAPER,
            AgentRunState.RECONCILING,
            AgentRunState.COMPLETED,
            AgentRunEventKind.STATE_TRANSITION,
        ),
        (
            AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            RuntimeMode.LIVE_ASSISTED,
            AgentRunState.PENDING_APPROVAL,
            AgentRunState.REJECTED,
            AgentRunEventKind.APPROVAL_REJECTED,
        ),
        (
            AgentRunGoal.ORDER_DRAFT,
            RuntimeMode.PAPER,
            AgentRunState.DRAFT_READY,
            AgentRunState.EXPIRED,
            AgentRunEventKind.TIMEOUT_RECORDED,
        ),
    ),
)
def test_representative_happy_and_control_transitions_are_legal(
    goal: AgentRunGoal,
    mode: RuntimeMode,
    before: AgentRunState,
    after: AgentRunState,
    kind: AgentRunEventKind,
) -> None:
    assert after in legal_transition_targets(before, goal=goal, mode=mode)
    validate_transition(before, after, goal=goal, mode=mode, event_kind=kind)


@pytest.mark.parametrize(
    ("before", "after", "kind"),
    (
        (
            AgentRunState.RECEIVED,
            AgentRunState.ANALYZED,
            AgentRunEventKind.STATE_TRANSITION,
        ),
        (
            AgentRunState.RECONCILING,
            AgentRunState.CANCELLED,
            AgentRunEventKind.CANCELLATION_REQUESTED,
        ),
        (
            AgentRunState.PENDING_APPROVAL,
            AgentRunState.APPROVED,
            AgentRunEventKind.STATE_TRANSITION,
        ),
        (
            AgentRunState.SNAPSHOT_READY,
            AgentRunState.TIMED_OUT,
            AgentRunEventKind.STATE_TRANSITION,
        ),
        (
            AgentRunState.SNAPSHOT_READY,
            AgentRunState.DATA_VALIDATED,
            AgentRunEventKind.RUN_CREATED,
        ),
    ),
)
def test_illegal_or_wrongly_typed_transitions_fail_closed(
    before: AgentRunState,
    after: AgentRunState,
    kind: AgentRunEventKind,
) -> None:
    with pytest.raises(InvalidAgentStateTransition):
        validate_transition(
            before,
            after,
            goal=AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            mode=RuntimeMode.LIVE_ASSISTED,
            event_kind=kind,
        )


@pytest.mark.parametrize(
    "kind",
    (
        AgentRunEventKind.TOOL_CALL_RESERVED,
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
    ),
)
def test_only_observation_event_kinds_may_self_loop(kind: AgentRunEventKind) -> None:
    validate_transition(
        AgentRunState.SNAPSHOT_READY,
        AgentRunState.SNAPSHOT_READY,
        goal=AgentRunGoal.MARKET_RESEARCH,
        mode=RuntimeMode.RESEARCH,
        event_kind=kind,
    )
    with pytest.raises(InvalidAgentStateTransition, match="same-state"):
        validate_transition(
            AgentRunState.SNAPSHOT_READY,
            AgentRunState.SNAPSHOT_READY,
            goal=AgentRunGoal.MARKET_RESEARCH,
            mode=RuntimeMode.RESEARCH,
            event_kind=AgentRunEventKind.STATE_TRANSITION,
        )


@pytest.mark.parametrize("state", (AgentRunState.EXECUTING, AgentRunState.RECONCILING))
def test_cancellation_self_loop_is_reserved_for_the_execution_boundary(
    state: AgentRunState,
) -> None:
    validate_transition(
        state,
        state,
        goal=AgentRunGoal.PAPER_EXECUTION,
        mode=RuntimeMode.PAPER,
        event_kind=AgentRunEventKind.CANCELLATION_REQUESTED,
    )
    with pytest.raises(InvalidAgentStateTransition, match="execution boundary"):
        validate_transition(
            AgentRunState.SNAPSHOT_READY,
            AgentRunState.SNAPSHOT_READY,
            goal=AgentRunGoal.MARKET_RESEARCH,
            mode=RuntimeMode.RESEARCH,
            event_kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        )


@pytest.mark.parametrize(
    "terminal",
    sorted(TERMINAL_AGENT_RUN_STATES, key=lambda item: item.value),
)
def test_terminal_states_are_absorbing(terminal: AgentRunState) -> None:
    assert not legal_transition_targets(
        terminal,
        goal=AgentRunGoal.MARKET_RESEARCH,
        mode=RuntimeMode.RESEARCH,
    )
    with pytest.raises(InvalidAgentStateTransition, match="absorbing"):
        validate_transition(
            terminal,
            terminal,
            goal=AgentRunGoal.MARKET_RESEARCH,
            mode=RuntimeMode.RESEARCH,
            event_kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        )


@pytest.mark.parametrize(
    ("state", "goal", "mode", "expected"),
    (
        (
            AgentRunState.RECEIVED,
            AgentRunGoal.MARKET_RESEARCH,
            RuntimeMode.RESEARCH,
            frozenset(),
        ),
        (
            AgentRunState.SNAPSHOT_READY,
            AgentRunGoal.MARKET_RESEARCH,
            RuntimeMode.RESEARCH,
            frozenset({"get_market_snapshot", "validate_market_data"}),
        ),
        (
            AgentRunState.DATA_VALIDATED,
            AgentRunGoal.BACKTEST_REPORT,
            RuntimeMode.BACKTEST,
            frozenset({"run_backtest"}),
        ),
        (
            AgentRunState.PORTFOLIO_READY,
            AgentRunGoal.PAPER_EXECUTION,
            RuntimeMode.PAPER,
            frozenset({"check_portfolio_risk"}),
        ),
        (
            AgentRunState.RISK_CHECKED,
            AgentRunGoal.PAPER_EXECUTION,
            RuntimeMode.PAPER,
            frozenset({"create_order_draft"}),
        ),
        (
            AgentRunState.EXECUTING,
            AgentRunGoal.PAPER_EXECUTION,
            RuntimeMode.PAPER,
            frozenset({"submit_paper_orders"}),
        ),
        (
            AgentRunState.EXECUTING,
            AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            RuntimeMode.LIVE_ASSISTED,
            frozenset({"submit_approved_orders"}),
        ),
        (
            AgentRunState.RECONCILING,
            AgentRunGoal.PAPER_EXECUTION,
            RuntimeMode.PAPER,
            frozenset({"reconcile_account"}),
        ),
    ),
)
def test_stage_catalog_is_closed_by_state_and_goal(
    state: AgentRunState,
    goal: AgentRunGoal,
    mode: RuntimeMode,
    expected: frozenset[str],
) -> None:
    assert stage_tool_names(state, goal=goal, mode=mode) == expected


@pytest.mark.parametrize(
    ("goal", "ready_state"),
    (
        (AgentRunGoal.MARKET_RESEARCH, AgentRunState.ANALYZED),
        (AgentRunGoal.BACKTEST_REPORT, AgentRunState.ANALYZED),
        (AgentRunGoal.PORTFOLIO_REPORT, AgentRunState.RISK_CHECKED),
        (AgentRunGoal.ORDER_DRAFT, AgentRunState.DRAFT_READY),
    ),
)
def test_report_readiness_is_goal_specific(
    goal: AgentRunGoal,
    ready_state: AgentRunState,
) -> None:
    assert report_ready(goal, ready_state)
    assert not report_ready(goal, AgentRunState.RECEIVED)


def test_event_build_is_deterministic_utc_and_json_roundtrips() -> None:
    event = _start().events[0]
    rebuilt = AgentRunEvent.build(
        run_id=event.run_id,
        decision_id=event.decision_id,
        decision_snapshot_hash=event.decision_snapshot_hash,
        runtime_mode=event.runtime_mode,
        goal=event.goal,
        sequence=event.sequence,
        kind=event.kind,
        trigger=event.trigger,
        state_before=event.state_before,
        state_after=event.state_after,
        reason=event.reason,
        occurred_at=LOCAL_NOW,
        run_limits=event.run_limits,
    )

    assert event == rebuilt
    assert event.occurred_at == LOCAL_NOW.astimezone(UTC)
    assert len(event.event_hash) == 64
    assert AgentRunEvent.from_json(event.to_json()) == event
    assert event.to_json() == AgentRunEvent.from_json(event.to_json()).to_json()


def test_event_json_rejects_content_tampering_and_duplicate_fields() -> None:
    event = _start().events[0]
    payload = json.loads(event.to_json())
    payload["reason"] = "TAMPERED"

    with pytest.raises(ValidationError, match="event_hash"):
        AgentRunEvent.from_json(json.dumps(payload))
    duplicate = '{"run_id":"duplicate",' + event.to_json()[1:]
    with pytest.raises(ValueError, match="duplicate JSON field"):
        AgentRunEvent.from_json(duplicate)


@pytest.mark.parametrize("kind", _NON_TOOL_EVENT_KINDS, ids=lambda kind: kind.value)
def test_event_build_rejects_request_id_on_every_non_tool_kind(
    kind: AgentRunEventKind,
) -> None:
    values = _event_build_values(_event_for_kind(kind))
    values["request_id"] = "request_injected_001"

    with pytest.raises(ValidationError, match="non-tool events cannot carry tool-call fields"):
        AgentRunEvent.build(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("path", _EVENT_REVALIDATION_PATHS)
@pytest.mark.parametrize("kind", _NON_TOOL_EVENT_KINDS, ids=lambda kind: kind.value)
def test_event_revalidation_rejects_request_id_on_every_non_tool_kind(
    kind: AgentRunEventKind,
    path: str,
) -> None:
    event = _event_for_kind(kind)

    with pytest.raises(ValidationError, match="non-tool events cannot carry tool-call fields"):
        _revalidate_event_mutation(
            event,
            {"request_id": "request_injected_001"},
            path,
        )


@pytest.mark.parametrize(
    "kind",
    _BATCH_HASH_FORBIDDEN_EVENT_KINDS,
    ids=lambda kind: kind.value,
)
def test_event_build_rejects_batch_hash_outside_tool_or_approval_control_kinds(
    kind: AgentRunEventKind,
) -> None:
    values = _event_build_values(_event_for_kind(kind))
    values["batch_hash"] = _digest("injected-batch")

    with pytest.raises(
        ValidationError,
        match="only tool or approval-control events may carry a batch hash",
    ):
        AgentRunEvent.build(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("path", _EVENT_REVALIDATION_PATHS)
@pytest.mark.parametrize(
    "kind",
    _BATCH_HASH_FORBIDDEN_EVENT_KINDS,
    ids=lambda kind: kind.value,
)
def test_event_revalidation_rejects_batch_hash_outside_allowlist(
    kind: AgentRunEventKind,
    path: str,
) -> None:
    event = _event_for_kind(kind)

    with pytest.raises(
        ValidationError,
        match="only tool or approval-control events may carry a batch hash",
    ):
        _revalidate_event_mutation(
            event,
            {"batch_hash": _digest("injected-batch")},
            path,
        )


@pytest.mark.parametrize(
    ("field", "codes"),
    (
        ("warning_codes", ("INJECTED_WARNING",)),
        ("error_codes", ("INJECTED_ERROR",)),
    ),
    ids=("warning-codes", "error-codes"),
)
@pytest.mark.parametrize(
    "kind",
    _OUTCOME_CODE_FORBIDDEN_EVENT_KINDS,
    ids=lambda kind: kind.value,
)
def test_event_build_rejects_codes_outside_tool_outcomes(
    kind: AgentRunEventKind,
    field: str,
    codes: tuple[str, ...],
) -> None:
    values = _event_build_values(_event_for_kind(kind))
    values[field] = codes

    with pytest.raises(
        ValidationError,
        match="only tool outcomes may carry warning or error codes",
    ):
        AgentRunEvent.build(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("path", _EVENT_REVALIDATION_PATHS)
@pytest.mark.parametrize(
    ("field", "codes"),
    (
        ("warning_codes", ("INJECTED_WARNING",)),
        ("error_codes", ("INJECTED_ERROR",)),
    ),
    ids=("warning-codes", "error-codes"),
)
@pytest.mark.parametrize(
    "kind",
    _OUTCOME_CODE_FORBIDDEN_EVENT_KINDS,
    ids=lambda kind: kind.value,
)
def test_event_revalidation_rejects_codes_outside_tool_outcomes(
    kind: AgentRunEventKind,
    field: str,
    codes: tuple[str, ...],
    path: str,
) -> None:
    event = _event_for_kind(kind)

    with pytest.raises(
        ValidationError,
        match="only tool outcomes may carry warning or error codes",
    ):
        _revalidate_event_mutation(event, {field: codes}, path)


def test_event_field_allowlists_retain_intended_tool_and_approval_evidence() -> None:
    tool_values = _event_build_values(_event_for_kind(AgentRunEventKind.TOOL_CALL_RESERVED))
    tool_values["batch_hash"] = _digest("tool-batch")
    tool_event = AgentRunEvent.build(**tool_values)  # type: ignore[arg-type]

    rejected_values = _event_build_values(_event_for_kind(AgentRunEventKind.APPROVAL_REJECTED))
    rejected_values["batch_hash"] = _digest("rejected-approval-batch")
    approval_rejected = AgentRunEvent.build(**rejected_values)  # type: ignore[arg-type]

    succeeded_values = _event_build_values(_event_for_kind(AgentRunEventKind.TOOL_CALL_SUCCEEDED))
    succeeded_values["warning_codes"] = ("EXPECTED_TOOL_WARNING",)
    tool_outcome = AgentRunEvent.build(**succeeded_values)  # type: ignore[arg-type]
    approval_recorded = _approval_event()

    assert tool_event.batch_hash == _digest("tool-batch")
    assert approval_recorded.batch_hash == approval_recorded.external_approval.batch_hash
    assert approval_rejected.batch_hash == _digest("rejected-approval-batch")
    assert tool_outcome.warning_codes == ("EXPECTED_TOOL_WARNING",)


@pytest.mark.parametrize(
    "changes",
    (
        {
            "kind": AgentRunEventKind.STATE_TRANSITION,
            "sequence": 0,
            "previous_event_hash": None,
        },
        {
            "kind": AgentRunEventKind.APPROVAL_RECORDED,
            "sequence": 1,
            "previous_event_hash": _digest("previous"),
        },
        {
            "kind": AgentRunEventKind.CANCELLATION_REQUESTED,
            "sequence": 1,
            "previous_event_hash": _digest("previous"),
            "control_hash": _digest("forbidden-control"),
        },
    ),
)
def test_event_contract_rejects_wrong_kind_specific_evidence(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "run_id": "run_event_001",
        "decision_id": "decision_event_001",
        "decision_snapshot_hash": _digest("decision-event"),
        "runtime_mode": RuntimeMode.RESEARCH,
        "goal": AgentRunGoal.MARKET_RESEARCH,
        "sequence": 1,
        "kind": AgentRunEventKind.CANCELLATION_REQUESTED,
        "trigger": AgentRunTrigger.SYSTEM,
        "state_before": AgentRunState.SNAPSHOT_READY,
        "state_after": AgentRunState.SNAPSHOT_READY,
        "reason": "TEST_EVENT",
        "occurred_at": LOCAL_NOW,
        "previous_event_hash": _digest("previous"),
    }
    values.update(changes)

    with pytest.raises(ValidationError):
        AgentRunEvent.build(**values)  # type: ignore[arg-type]


def test_snapshot_with_event_chain_pending_call_and_artifact_roundtrips() -> None:
    snapshot = _snapshot_with_completed_read_call()
    restored = AgentRunSnapshot.from_json(snapshot.to_json())

    assert restored == snapshot
    assert restored is not snapshot
    assert restored.created_at.tzinfo is UTC
    assert restored.tool_call_count == 1
    assert restored.pending_call is None
    assert restored.artifacts[0].kind is AgentArtifactKind.MARKET_SNAPSHOT
    assert restored.events[-1].previous_event_hash == restored.events[-2].event_hash


def test_snapshot_retains_non_default_limits_in_genesis_event() -> None:
    limits = AgentRunLimits(
        max_tool_calls=7,
        max_invalid_tool_calls=2,
        max_reconciliation_attempts=4,
        timeout_seconds=90,
        approval_timeout_seconds=30,
    )
    snapshot = _start(limits=limits)

    assert snapshot.limits == limits
    assert snapshot.events[0].run_limits == limits
    assert snapshot.deadline == snapshot.created_at + timedelta(seconds=90)


@pytest.mark.parametrize(
    "mutation",
    ("state", "revision", "tool-count", "event-prefix", "completion"),
)
def test_snapshot_rejects_derived_state_or_chain_tampering(mutation: str) -> None:
    snapshot = _snapshot_with_completed_read_call()
    values = snapshot.model_dump(mode="python")
    if mutation == "state":
        values["state"] = AgentRunState.ANALYZED
    elif mutation == "revision":
        values["revision"] = snapshot.revision + 1
    elif mutation == "tool-count":
        values["tool_call_count"] = 0
    elif mutation == "event-prefix":
        last = snapshot.events[-1]
        broken = AgentRunEvent.build(
            **{
                **last.model_dump(mode="python", exclude={"schema_version", "event_hash"}),
                "previous_event_hash": _digest("wrong-previous"),
            }
        )
        values["events"] = (*snapshot.events[:-1], broken)
    else:
        values["completed_at"] = snapshot.updated_at

    with pytest.raises(ValidationError):
        AgentRunSnapshot.model_validate(values, strict=True)


def test_snapshot_cannot_underreport_rejected_tool_calls() -> None:
    snapshot = _snapshot_with_rejected_call()
    values = snapshot.model_dump(mode="python")
    values["invalid_tool_call_count"] = 0

    assert snapshot.invalid_tool_call_count == 1
    assert sum(event.kind is AgentRunEventKind.TOOL_CALL_REJECTED for event in snapshot.events) == 1
    with pytest.raises(ValidationError, match="rejected tool-call outcomes"):
        AgentRunSnapshot.model_validate(values, strict=True)


def test_snapshot_json_rejects_hash_tampering_and_duplicate_fields() -> None:
    snapshot = _snapshot_with_completed_read_call()
    payload = json.loads(snapshot.to_json())
    payload["events"][-1]["event_hash"] = "f" * 64

    with pytest.raises(ValidationError, match="event_hash"):
        AgentRunSnapshot.from_json(json.dumps(payload))
    duplicate = '{"run_id":"duplicate",' + snapshot.to_json()[1:]
    with pytest.raises(ValueError, match="duplicate JSON field"):
        AgentRunSnapshot.from_json(duplicate)


def test_repository_create_get_and_exact_retries_are_defensive() -> None:
    repository = InMemoryAgentRunRepository()
    snapshot = _start()

    created = repository.create(snapshot)
    replayed = repository.create(AgentRunSnapshot.from_json(snapshot.to_json()))
    loaded = repository.get(snapshot.run_id)

    assert created == replayed == loaded == snapshot
    assert created is not snapshot
    assert loaded is not created
    with pytest.raises(ValidationError):
        loaded.revision = 99
    assert repository.get(snapshot.run_id) == snapshot


def test_repository_create_conflicts_and_get_fails_closed() -> None:
    repository = InMemoryAgentRunRepository()
    repository.create(_start())

    with pytest.raises(AgentRunConflict, match="already exists"):
        repository.create(_start(decision_id="different_decision"))
    with pytest.raises(AgentRunNotFound):
        repository.get("missing_run")
    for invalid in ("", " run_contract_001 ", "\n", True):
        with pytest.raises(ValueError):
            repository.get(invalid)  # type: ignore[arg-type]


def test_repository_create_rejects_unreplayable_and_derived_field_drift() -> None:
    unreplayable = _append(
        _start(run_id="run_unreplayable_create_001"),
        AgentRunState.DATA_VALIDATED,
        reason="SKIPPED_REQUIRED_STAGE",
    )
    with pytest.raises(AgentRunConflict, match="replay"):
        InMemoryAgentRunRepository().create(unreplayable)

    baseline = _start(run_id="run_derived_create_001")
    values = baseline.model_dump(mode="python")
    values["approval_deadline"] = baseline.created_at + timedelta(seconds=10)
    drifted = AgentRunSnapshot.model_validate(values, strict=True)
    assert drifted != replay_agent_run(drifted.events)

    with pytest.raises(AgentRunConflict, match="replay"):
        InMemoryAgentRunRepository().create(drifted)


def test_repository_atomically_saves_multiple_events_and_replays_exact_result() -> None:
    repository = InMemoryAgentRunRepository()
    base = repository.create(_start())
    first = _append(base, AgentRunState.SNAPSHOT_READY, reason="SNAPSHOT_READY")
    second = _append(
        first,
        AgentRunState.CANCELLED,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
        reason="CANCELLED_BEFORE_EXECUTION",
    )

    saved = repository.save(second, expected_revision=base.revision)
    replayed = repository.save(second, expected_revision=base.revision)

    assert replay_agent_run(second.events) == second
    assert saved == replayed == second
    assert repository.get(base.run_id) == second


def test_repository_save_rejects_unreplayable_and_derived_field_drift() -> None:
    repository = InMemoryAgentRunRepository()
    base = repository.create(_start(run_id="run_replay_save_001"))
    unreplayable = _append(
        base,
        AgentRunState.DATA_VALIDATED,
        reason="SKIPPED_REQUIRED_STAGE",
    )

    with pytest.raises(AgentRunConflict, match="replay"):
        repository.save(unreplayable, expected_revision=base.revision)
    assert repository.get(base.run_id) == base

    valid = _append(base, AgentRunState.SNAPSHOT_READY, reason="SNAPSHOT_READY")
    values = valid.model_dump(mode="python")
    values["approval_deadline"] = base.created_at + timedelta(seconds=10)
    drifted = AgentRunSnapshot.model_validate(values, strict=True)
    assert replay_agent_run(drifted.events) == valid
    assert drifted != valid

    with pytest.raises(AgentRunConflict, match="replay"):
        repository.save(drifted, expected_revision=base.revision)
    assert repository.get(base.run_id) == base


def test_repository_rejects_stale_cas_bad_prefix_and_identity_drift() -> None:
    repository = InMemoryAgentRunRepository()
    base = repository.create(_start())
    current = _append(base, AgentRunState.SNAPSHOT_READY, reason="WINNER")
    repository.save(current, expected_revision=base.revision)

    stale = _append(base, AgentRunState.SNAPSHOT_READY, reason="STALE")
    with pytest.raises(AgentRunConflict, match="revision changed"):
        repository.save(stale, expected_revision=base.revision)

    divergent_prefix = _append(
        stale,
        AgentRunState.CANCELLED,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
        reason="DIVERGENT",
    )
    with pytest.raises(AgentRunConflict, match="prior event chain"):
        repository.save(divergent_prefix, expected_revision=current.revision)

    other_base = _start(decision_id="different_decision")
    identity_drift = _append(other_base, AgentRunState.SNAPSHOT_READY)
    identity_drift = _append(
        identity_drift,
        AgentRunState.CANCELLED,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
        reason="IDENTITY_DRIFT",
    )
    with pytest.raises(AgentRunConflict, match="identity"):
        repository.save(identity_drift, expected_revision=current.revision)


@pytest.mark.parametrize("expected_revision", (-1, True, 1.0, "0"))
def test_repository_rejects_invalid_cas_tokens(expected_revision: object) -> None:
    repository = InMemoryAgentRunRepository()
    candidate = _append(repository.create(_start()), AgentRunState.SNAPSHOT_READY)

    with pytest.raises(ValueError, match="non-negative integer"):
        repository.save(
            candidate,
            expected_revision=expected_revision,  # type: ignore[arg-type]
        )


def test_repository_lock_allows_exactly_one_conflicting_cas_winner() -> None:
    repository = InMemoryAgentRunRepository()
    base = repository.create(_start())
    candidates = (
        _append(base, AgentRunState.SNAPSHOT_READY, reason="CANDIDATE_A"),
        _append(base, AgentRunState.SNAPSHOT_READY, reason="CANDIDATE_B"),
    )
    barrier = Barrier(2)

    def attempt(candidate: AgentRunSnapshot) -> str:
        barrier.wait()
        try:
            repository.save(candidate, expected_revision=base.revision)
        except AgentRunConflict:
            return "conflict"
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(attempt, candidates))

    assert sorted(outcomes) == ["conflict", "saved"]
    assert repository.get(base.run_id) in candidates
