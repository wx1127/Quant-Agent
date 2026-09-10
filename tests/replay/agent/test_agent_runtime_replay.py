"""Adversarial replay tests for the explicit Agent event state machine."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from quant_agent.agent.runtime.contracts import (
    AGENT_RUNTIME_VERSION,
    AgentArtifactKind,
    AgentArtifactRef,
    AgentExternalApproval,
    AgentRunEvent,
    AgentRunEventKind,
    AgentRunGoal,
    AgentRunLimits,
    AgentRunSnapshot,
    AgentRunState,
    AgentRunTrigger,
)
from quant_agent.agent.runtime.replay import AgentRunReplayError, replay_agent_run
from quant_agent.agent.tools.contracts import ToolEffect
from quant_agent.agent.tools.policy import V1_TOOL_POLICIES
from quant_agent.config import RuntimeMode
from quant_agent.core.errors import ErrorCode

NOW = datetime(2026, 9, 10, 1, 0, tzinfo=UTC)
LOCKED_BATCH_HASH = hashlib.sha256(b"locked-order-batch").hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _Trace:
    """Build hash-valid events without pre-validating their replay semantics."""

    def __init__(
        self,
        *,
        goal: AgentRunGoal = AgentRunGoal.MARKET_RESEARCH,
        mode: RuntimeMode = RuntimeMode.RESEARCH,
        limits: AgentRunLimits | None = None,
        runtime_version: str = AGENT_RUNTIME_VERSION,
    ) -> None:
        self.goal = goal
        self.mode = mode
        self.limits = limits or AgentRunLimits()
        self.state = AgentRunState.RECEIVED
        self.at = NOW
        self.events: list[AgentRunEvent] = []
        self.pending: AgentRunEvent | None = None
        self.add(
            AgentRunEventKind.RUN_CREATED,
            AgentRunState.RECEIVED,
            trigger=AgentRunTrigger.SYSTEM,
            reason="RUN_CREATED",
            sequence=0,
            previous_event_hash=None,
            run_limits=self.limits,
            runtime_version=runtime_version,
        )

    @property
    def chain(self) -> tuple[AgentRunEvent, ...]:
        return tuple(self.events)

    def add(
        self,
        kind: AgentRunEventKind,
        state_after: AgentRunState,
        *,
        trigger: AgentRunTrigger,
        reason: str,
        at: datetime | None = None,
        sequence: int | None = None,
        previous_event_hash: str | None = None,
        state_before: AgentRunState | None = None,
        run_id: str = "run_replay_001",
        decision_id: str = "decision_replay_001",
        decision_snapshot_hash: str | None = None,
        runtime_version: str = AGENT_RUNTIME_VERSION,
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
    ) -> AgentRunEvent:
        occurred_at = at or (self.at if not self.events else self.at + timedelta(seconds=1))
        selected_sequence = len(self.events) if sequence is None else sequence
        if previous_event_hash is None and self.events:
            previous_event_hash = self.events[-1].event_hash
        event = AgentRunEvent.build(
            run_id=run_id,
            decision_id=decision_id,
            decision_snapshot_hash=decision_snapshot_hash or _digest("decision-snapshot"),
            runtime_mode=self.mode,
            goal=self.goal,
            sequence=selected_sequence,
            kind=kind,
            trigger=trigger,
            state_before=state_before or self.state,
            state_after=state_after,
            reason=reason,
            occurred_at=occurred_at,
            request_id=request_id,
            tool_name=tool_name,
            tool_effect=tool_effect,
            argument_hash=argument_hash,
            response_hash=response_hash,
            idempotency_key_hash=idempotency_key_hash,
            batch_hash=batch_hash,
            control_hash=control_hash,
            external_approval=external_approval,
            run_limits=run_limits,
            artifact=artifact,
            warning_codes=warning_codes,
            error_codes=error_codes,
            previous_event_hash=previous_event_hash,
            runtime_version=runtime_version,
        )
        self.events.append(event)
        self.state = state_after
        self.at = occurred_at
        return event

    def transition(
        self,
        state_after: AgentRunState,
        *,
        kind: AgentRunEventKind = AgentRunEventKind.STATE_TRANSITION,
        trigger: AgentRunTrigger = AgentRunTrigger.SYSTEM,
        reason: str = "STATE_TRANSITION",
        at: datetime | None = None,
        batch_hash: str | None = None,
        control_hash: str | None = None,
        external_approval: AgentExternalApproval | None = None,
    ) -> AgentRunEvent:
        return self.add(
            kind,
            state_after,
            trigger=trigger,
            reason=reason,
            at=at,
            batch_hash=batch_hash,
            control_hash=control_hash,
            external_approval=external_approval,
        )

    def reserve(
        self,
        tool_name: str,
        *,
        request_id: str | None = None,
        batch_hash: str | None = None,
        effect: ToolEffect | None = None,
        argument_hash: str | None = None,
        at: datetime | None = None,
        omit_idempotency_key: bool = False,
    ) -> AgentRunEvent:
        selected_request = request_id or f"request_{len(self.events):03d}"
        selected_argument = argument_hash or _digest(f"arguments:{selected_request}")
        policy = V1_TOOL_POLICIES.get(tool_name)
        selected_effect = (
            effect
            if effect is not None
            else policy.effect
            if policy is not None
            else ToolEffect.READ_ONLY
        )
        idempotency_hash = (
            _digest(f"idempotency:{selected_request}")
            if selected_effect
            in {
                ToolEffect.ARTIFACT_WRITE,
                ToolEffect.PAPER_EXECUTION_WRITE,
                ToolEffect.LIVE_EXTERNAL_WRITE,
            }
            and not omit_idempotency_key
            else None
        )
        event = self.add(
            AgentRunEventKind.TOOL_CALL_RESERVED,
            self.state,
            trigger=AgentRunTrigger.TOOL,
            reason="TOOL_CALL_RESERVED",
            at=at,
            request_id=selected_request,
            tool_name=tool_name,
            tool_effect=selected_effect,
            argument_hash=selected_argument,
            idempotency_key_hash=idempotency_hash,
            batch_hash=batch_hash,
        )
        self.pending = event
        return event

    def outcome(
        self,
        *,
        state_after: AgentRunState | None = None,
        kind: AgentRunEventKind = AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        reason: str = "TOOL_OUTCOME",
        artifact_kind: AgentArtifactKind | None = None,
        artifact_id: str | None = None,
        content_hash: str | None = None,
        expires_at: datetime | None = None,
        request_id: str | None = None,
        tool_name: str | None = None,
        argument_hash: str | None = None,
        batch_hash: str | None = None,
        error_codes: tuple[str, ...] | None = None,
        at: datetime | None = None,
    ) -> AgentRunEvent:
        if self.pending is None:
            raise AssertionError("test trace has no pending reservation")
        pending = self.pending
        selected_request = request_id or pending.request_id
        selected_tool = tool_name or pending.tool_name
        selected_argument = argument_hash or pending.argument_hash
        selected_batch = pending.batch_hash if batch_hash is None else batch_hash
        response_hash = (
            _digest(f"response:{len(self.events)}")
            if kind
            in {
                AgentRunEventKind.TOOL_CALL_SUCCEEDED,
                AgentRunEventKind.TOOL_CALL_REPLAYED,
            }
            or artifact_kind is not None
            else None
        )
        artifact = None
        if artifact_kind is not None:
            if selected_tool is None or response_hash is None:
                raise AssertionError("artifact outcome requires tool and response identity")
            artifact = AgentArtifactRef(
                kind=artifact_kind,
                artifact_id=artifact_id or f"artifact_{len(self.events):03d}",
                content_hash=content_hash or _digest(f"content:{len(self.events)}"),
                tool_name=selected_tool,
                response_hash=response_hash,
                expires_at=expires_at,
            )
        selected_error_codes = (
            error_codes
            if error_codes is not None
            else (
                ("EXPECTED_TEST_FAILURE",)
                if kind is AgentRunEventKind.TOOL_CALL_FAILED
                else (
                    ("RUNTIME_STAGE_REJECTED",)
                    if kind is AgentRunEventKind.TOOL_CALL_REJECTED
                    else ()
                )
            )
        )
        event = self.add(
            kind,
            state_after or self.state,
            trigger=AgentRunTrigger.TOOL,
            reason=reason,
            at=at,
            request_id=selected_request,
            tool_name=selected_tool,
            tool_effect=pending.tool_effect,
            argument_hash=selected_argument,
            response_hash=response_hash,
            idempotency_key_hash=pending.idempotency_key_hash,
            batch_hash=selected_batch,
            artifact=artifact,
            error_codes=selected_error_codes,
        )
        self.pending = None
        return event

    def success(
        self,
        tool_name: str,
        artifact_kind: AgentArtifactKind,
        *,
        state_after: AgentRunState | None = None,
        content_hash: str | None = None,
        expires_at: datetime | None = None,
        batch_hash: str | None = None,
        artifact_id: str | None = None,
    ) -> AgentRunEvent:
        self.reserve(tool_name, batch_hash=batch_hash)
        return self.outcome(
            state_after=state_after,
            artifact_kind=artifact_kind,
            content_hash=content_hash,
            expires_at=expires_at,
            artifact_id=artifact_id,
        )


def _approval_evidence(
    trace: _Trace,
    *,
    approval_hash: str,
    batch_hash: str = LOCKED_BATCH_HASH,
    approved_at: datetime | None = None,
) -> AgentExternalApproval:
    return AgentExternalApproval(
        approval_id=f"approval_{len(trace.events):03d}",
        approval_hash=approval_hash,
        decision_id=trace.events[0].decision_id,
        batch_hash=batch_hash,
        approved_at=approved_at or trace.at,
        expires_at=trace.at + timedelta(minutes=30),
    )


def _data_validated(trace: _Trace) -> _Trace:
    trace.transition(AgentRunState.SNAPSHOT_READY)
    trace.success("get_market_snapshot", AgentArtifactKind.MARKET_SNAPSHOT)
    trace.success(
        "validate_market_data",
        AgentArtifactKind.DATA_QUALITY,
        state_after=AgentRunState.DATA_VALIDATED,
    )
    return trace


def _analyzed(trace: _Trace) -> _Trace:
    _data_validated(trace)
    if trace.goal is AgentRunGoal.BACKTEST_REPORT:
        trace.success(
            "run_backtest",
            AgentArtifactKind.BACKTEST_REPORT,
            state_after=AgentRunState.ANALYZED,
        )
    else:
        trace.success(
            "rank_stock_candidates",
            AgentArtifactKind.STOCK_CANDIDATES,
            state_after=AgentRunState.ANALYZED,
        )
    return trace


def _portfolio_ready(trace: _Trace) -> _Trace:
    _analyzed(trace)
    trace.success("get_portfolio_snapshot", AgentArtifactKind.PORTFOLIO_SNAPSHOT)
    trace.success(
        "build_target_portfolio",
        AgentArtifactKind.TARGET_PORTFOLIO,
        state_after=AgentRunState.PORTFOLIO_READY,
    )
    return trace


def _risk_checked(trace: _Trace) -> _Trace:
    _portfolio_ready(trace)
    trace.success(
        "check_portfolio_risk",
        AgentArtifactKind.RISK_CHECK,
        state_after=AgentRunState.RISK_CHECKED,
    )
    return trace


def _draft_ready(trace: _Trace, *, expires_at: datetime | None = None) -> _Trace:
    _risk_checked(trace)
    trace.success(
        "create_order_draft",
        AgentArtifactKind.ORDER_DRAFT,
        state_after=AgentRunState.DRAFT_READY,
        content_hash=LOCKED_BATCH_HASH,
        expires_at=expires_at or NOW + timedelta(minutes=30),
        artifact_id="order_draft_001",
    )
    return trace


def _live_pending(*, limits: AgentRunLimits | None = None) -> _Trace:
    trace = _draft_ready(
        _Trace(
            goal=AgentRunGoal.LIVE_ASSISTED_EXECUTION,
            mode=RuntimeMode.LIVE_ASSISTED,
            limits=limits,
        )
    )
    trace.transition(AgentRunState.PENDING_APPROVAL, reason="APPROVAL_REQUIRED")
    return trace


def _paper_executing(*, limits: AgentRunLimits | None = None) -> _Trace:
    trace = _draft_ready(
        _Trace(
            goal=AgentRunGoal.PAPER_EXECUTION,
            mode=RuntimeMode.PAPER,
            limits=limits,
        )
    )
    trace.transition(AgentRunState.EXECUTING)
    return trace


def _paper_reconciling(*, limits: AgentRunLimits | None = None) -> _Trace:
    trace = _paper_executing(limits=limits)
    trace.success(
        "submit_paper_orders",
        AgentArtifactKind.EXECUTION_RECEIPT,
        state_after=AgentRunState.RECONCILING,
        batch_hash=LOCKED_BATCH_HASH,
    )
    return trace


def _research_report() -> _Trace:
    trace = _analyzed(_Trace())
    trace.success(
        "generate_decision_report",
        AgentArtifactKind.DECISION_REPORT,
        state_after=AgentRunState.REPORTED,
    )
    return trace


def _backtest_report() -> _Trace:
    trace = _analyzed(_Trace(goal=AgentRunGoal.BACKTEST_REPORT, mode=RuntimeMode.BACKTEST))
    trace.success(
        "generate_decision_report",
        AgentArtifactKind.DECISION_REPORT,
        state_after=AgentRunState.REPORTED,
    )
    return trace


def _live_approved(*, enter_execution: bool = True) -> _Trace:
    trace = _live_pending()
    approval_hash = _digest("approval-control")
    trace.transition(
        AgentRunState.APPROVED,
        kind=AgentRunEventKind.APPROVAL_RECORDED,
        trigger=AgentRunTrigger.HUMAN,
        reason="APPROVED",
        batch_hash=LOCKED_BATCH_HASH,
        control_hash=approval_hash,
        external_approval=_approval_evidence(trace, approval_hash=approval_hash),
    )
    if enter_execution:
        trace.transition(AgentRunState.EXECUTING)
    return trace


def _live_completed() -> _Trace:
    trace = _live_approved()
    trace.success(
        "submit_approved_orders",
        AgentArtifactKind.EXECUTION_RECEIPT,
        state_after=AgentRunState.RECONCILING,
        batch_hash=LOCKED_BATCH_HASH,
    )
    trace.success(
        "reconcile_account",
        AgentArtifactKind.RECONCILIATION,
        state_after=AgentRunState.COMPLETED,
        batch_hash=LOCKED_BATCH_HASH,
    )
    return trace


def _cancellable_trace_at(state: AgentRunState) -> _Trace:
    if state is AgentRunState.RECEIVED:
        trace = _Trace()
    elif state is AgentRunState.SNAPSHOT_READY:
        trace = _Trace()
        trace.transition(AgentRunState.SNAPSHOT_READY)
    elif state is AgentRunState.DATA_VALIDATED:
        trace = _data_validated(_Trace())
    elif state is AgentRunState.ANALYZED:
        trace = _analyzed(_Trace())
    elif state is AgentRunState.PORTFOLIO_READY:
        trace = _portfolio_ready(_Trace(goal=AgentRunGoal.PORTFOLIO_REPORT, mode=RuntimeMode.PAPER))
    elif state is AgentRunState.RISK_CHECKED:
        trace = _risk_checked(_Trace(goal=AgentRunGoal.PORTFOLIO_REPORT, mode=RuntimeMode.PAPER))
    elif state is AgentRunState.DRAFT_READY:
        trace = _draft_ready(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
    elif state is AgentRunState.PENDING_APPROVAL:
        trace = _live_pending()
    elif state is AgentRunState.APPROVED:
        trace = _live_approved(enter_execution=False)
    elif state is AgentRunState.EXECUTING:
        trace = _draft_ready(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
        trace.transition(AgentRunState.EXECUTING)
    else:  # pragma: no cover - the test matrix is intentionally closed
        raise AssertionError(f"unsupported cancellable test state: {state.value}")
    assert trace.state is state
    return trace


def _trace_at_earlier_authorization_deadline(
    state: AgentRunState,
) -> tuple[_Trace, datetime]:
    if state in {AgentRunState.DRAFT_READY, AgentRunState.PENDING_APPROVAL}:
        draft_expiry = NOW + timedelta(seconds=60)
        goal = (
            AgentRunGoal.PAPER_EXECUTION
            if state is AgentRunState.DRAFT_READY
            else AgentRunGoal.LIVE_ASSISTED_EXECUTION
        )
        mode = (
            RuntimeMode.PAPER if state is AgentRunState.DRAFT_READY else RuntimeMode.LIVE_ASSISTED
        )
        trace = _draft_ready(_Trace(goal=goal, mode=mode), expires_at=draft_expiry)
        if state is AgentRunState.PENDING_APPROVAL:
            trace.transition(AgentRunState.PENDING_APPROVAL)
        deadline = draft_expiry
    else:
        limits = AgentRunLimits(approval_timeout_seconds=30)
        trace = _live_pending(limits=limits)
        deadline = trace.at + timedelta(seconds=limits.approval_timeout_seconds)
        approval_hash = _digest(f"approval-for-{state.value}")
        trace.transition(
            AgentRunState.APPROVED,
            kind=AgentRunEventKind.APPROVAL_RECORDED,
            trigger=AgentRunTrigger.HUMAN,
            batch_hash=LOCKED_BATCH_HASH,
            control_hash=approval_hash,
            external_approval=_approval_evidence(trace, approval_hash=approval_hash),
        )
        if state is AgentRunState.EXECUTING:
            trace.transition(AgentRunState.EXECUTING)
    assert trace.state is state
    assert deadline < NOW + timedelta(seconds=trace.limits.timeout_seconds)
    return trace, deadline


def _trace_at_exhausted_budget(
    scenario: str,
) -> tuple[_Trace, AgentRunEventKind, AgentRunState, str, str | None]:
    if scenario == "invalid-pre-execution":
        trace = _Trace(
            limits=AgentRunLimits(
                max_tool_calls=3,
                max_invalid_tool_calls=1,
                max_reconciliation_attempts=1,
            )
        )
        trace.reserve("get_market_snapshot", request_id="invalid_threshold")
        trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
        return (
            trace,
            AgentRunEventKind.LIMIT_REACHED,
            AgentRunState.LIMIT_EXCEEDED,
            "get_market_snapshot",
            None,
        )
    if scenario == "model-pre-execution":
        trace = _data_validated(
            _Trace(
                limits=AgentRunLimits(
                    max_tool_calls=2,
                    max_invalid_tool_calls=1,
                    max_reconciliation_attempts=1,
                )
            )
        )
        return (
            trace,
            AgentRunEventKind.LIMIT_REACHED,
            AgentRunState.LIMIT_EXCEEDED,
            "rank_stock_candidates",
            None,
        )
    if scenario == "invalid-executing":
        limits = AgentRunLimits(
            max_tool_calls=20,
            max_invalid_tool_calls=1,
            max_reconciliation_attempts=1,
        )
    else:
        limits = AgentRunLimits(
            max_tool_calls=8,
            max_invalid_tool_calls=2,
            max_reconciliation_attempts=1,
        )
    trace = _paper_executing(limits=limits)
    trace.reserve("submit_paper_orders", batch_hash=LOCKED_BATCH_HASH)
    trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
    return (
        trace,
        AgentRunEventKind.INCIDENT_RECORDED,
        AgentRunState.INCIDENT,
        "submit_paper_orders",
        LOCKED_BATCH_HASH,
    )


def _unsafe_execution_denial_trace(
    scenario: str,
) -> tuple[_Trace, str, str | None]:
    reconciling = scenario.startswith("reconciling")
    trace = _paper_reconciling() if reconciling else _paper_executing()
    if scenario.endswith("invalid-pending"):
        trace.reserve(
            "submit_paper_orders",
            batch_hash=LOCKED_BATCH_HASH,
            omit_idempotency_key=True,
        )
    else:
        trace.reserve("get_market_snapshot")
    trace.outcome(
        kind=AgentRunEventKind.TOOL_CALL_REJECTED,
        reason="PRE_DISPATCH_POLICY_DENIED",
        error_codes=("REGISTRY_POLICY_DENIED",),
    )
    return (
        trace,
        "reconcile_account" if reconciling else "submit_paper_orders",
        LOCKED_BATCH_HASH,
    )


def _pending_execution_at_boundary(
    write_mode: str,
    boundary_kind: str,
) -> tuple[_Trace, datetime, str]:
    if write_mode == "paper":
        if boundary_kind == "authorization":
            boundary = NOW + timedelta(seconds=60)
            trace = _draft_ready(
                _Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER),
                expires_at=boundary,
            )
            trace.transition(AgentRunState.EXECUTING)
        else:
            trace = _paper_executing()
            boundary = NOW + timedelta(seconds=trace.limits.timeout_seconds)
        tool_name = "submit_paper_orders"
    else:
        if boundary_kind == "authorization":
            trace, boundary = _trace_at_earlier_authorization_deadline(AgentRunState.EXECUTING)
        else:
            trace = _live_approved()
            boundary = NOW + timedelta(seconds=trace.limits.timeout_seconds)
        tool_name = "submit_approved_orders"
    trace.reserve(tool_name, batch_hash=LOCKED_BATCH_HASH)
    assert trace.at < boundary
    return trace, boundary, tool_name


@pytest.mark.parametrize(
    "trace",
    (
        _Trace(),
        _research_report(),
        _backtest_report(),
        _live_completed(),
    ),
    ids=("genesis", "research-report", "backtest-report", "live-assisted-completed"),
)
def test_successful_replay_is_deterministic_and_roundtrippable(trace: _Trace) -> None:
    snapshot = replay_agent_run(trace.chain)

    assert snapshot.events == trace.chain
    assert snapshot.state is trace.state
    assert snapshot.revision == len(trace.events) - 1
    assert snapshot.created_at == NOW
    assert snapshot.deadline == NOW + timedelta(seconds=trace.limits.timeout_seconds)
    assert AgentRunSnapshot.from_json(snapshot.to_json()) == snapshot
    assert replay_agent_run(snapshot.events) == snapshot


def test_replay_accepts_an_out_of_stage_rejection_but_counts_it() -> None:
    trace = _Trace()
    trace.reserve("get_market_snapshot")
    trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.RECEIVED
    assert snapshot.tool_call_count == 1
    assert snapshot.invalid_tool_call_count == 1
    assert snapshot.pending_call is None


def test_replay_preserves_one_unresolved_pending_call() -> None:
    trace = _Trace()
    trace.transition(AgentRunState.SNAPSHOT_READY)
    reservation = trace.reserve("get_market_snapshot")

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.pending_call is not None
    assert snapshot.pending_call.request_id == reservation.request_id
    assert snapshot.pending_call.started_at == reservation.occurred_at


@pytest.mark.parametrize("bad_input", ((), [], (object(),)))
def test_replay_requires_a_non_empty_exact_event_tuple(bad_input: object) -> None:
    with pytest.raises(AgentRunReplayError):
        replay_agent_run(bad_input)  # type: ignore[arg-type]


def test_replay_revalidates_each_event_hash_before_using_it() -> None:
    trace = _research_report()
    tampered = trace.events[-1].model_copy(update={"reason": "TAMPERED_AFTER_HASH"})

    with pytest.raises(AgentRunReplayError, match="content-hash revalidation"):
        replay_agent_run((*trace.chain[:-1], tampered))


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("runtime-version", "invalid RUN_CREATED origin"),
        ("goal-mode", "failed closed"),
        ("sequence", "identity or sequence drifted"),
        ("identity", "identity or sequence drifted"),
        ("previous-hash", "chain is discontinuous"),
        ("time-regression", "chain is discontinuous"),
        ("state-before", "state_before"),
        ("illegal-transition", "illegal transition"),
        ("untrusted-trigger", "untrusted trigger"),
        ("terminal-continued", "illegal transition"),
    ),
)
def test_replay_rejects_structurally_invalid_sequences(case: str, message: str) -> None:
    if case == "runtime-version":
        trace = _Trace(runtime_version="agent-runtime-v0")
    elif case == "goal-mode":
        trace = _Trace(mode=RuntimeMode.LIVE_AUTO)
    else:
        trace = _Trace()
        if case == "sequence":
            trace.add(
                AgentRunEventKind.STATE_TRANSITION,
                AgentRunState.SNAPSHOT_READY,
                trigger=AgentRunTrigger.SYSTEM,
                reason="GAPPED_SEQUENCE",
                sequence=2,
            )
        elif case == "identity":
            trace.add(
                AgentRunEventKind.STATE_TRANSITION,
                AgentRunState.SNAPSHOT_READY,
                trigger=AgentRunTrigger.SYSTEM,
                reason="CHANGED_RUN",
                run_id="different_run",
            )
        elif case == "previous-hash":
            trace.add(
                AgentRunEventKind.STATE_TRANSITION,
                AgentRunState.SNAPSHOT_READY,
                trigger=AgentRunTrigger.SYSTEM,
                reason="WRONG_PREVIOUS_HASH",
                previous_event_hash=_digest("wrong-previous"),
            )
        elif case == "time-regression":
            trace.add(
                AgentRunEventKind.STATE_TRANSITION,
                AgentRunState.SNAPSHOT_READY,
                trigger=AgentRunTrigger.SYSTEM,
                reason="CLOCK_REGRESSION",
                at=NOW - timedelta(seconds=1),
            )
        elif case == "state-before":
            trace.add(
                AgentRunEventKind.TOOL_CALL_REJECTED,
                AgentRunState.SNAPSHOT_READY,
                trigger=AgentRunTrigger.TOOL,
                reason="WRONG_BEFORE",
                state_before=AgentRunState.SNAPSHOT_READY,
                request_id="request_wrong_before",
                tool_name="get_market_snapshot",
                tool_effect=ToolEffect.READ_ONLY,
                argument_hash=_digest("wrong-before"),
                error_codes=("REJECTED",),
            )
        elif case == "illegal-transition":
            trace.add(
                AgentRunEventKind.STATE_TRANSITION,
                AgentRunState.ANALYZED,
                trigger=AgentRunTrigger.SYSTEM,
                reason="SKIPPED_STAGES",
            )
        elif case == "untrusted-trigger":
            trace.add(
                AgentRunEventKind.STATE_TRANSITION,
                AgentRunState.SNAPSHOT_READY,
                trigger=AgentRunTrigger.HUMAN,
                reason="UNTRUSTED_TRIGGER",
            )
        else:
            trace.transition(
                AgentRunState.CANCELLED,
                kind=AgentRunEventKind.CANCELLATION_REQUESTED,
                trigger=AgentRunTrigger.CANCELLATION,
            )
            trace.add(
                AgentRunEventKind.CANCELLATION_REQUESTED,
                AgentRunState.CANCELLED,
                trigger=AgentRunTrigger.CANCELLATION,
                reason="AFTER_TERMINAL",
            )

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("outcome-without-reservation", "exact pending reservation"),
        ("event-before-outcome", "must be resolved"),
        ("duplicate-request", "duplicate or overlaps"),
        ("mismatched-request", "exact pending reservation"),
        ("out-of-stage-success", "out-of-stage"),
        ("failed-advancement", "failed tool outcome advanced"),
        ("success-without-artifact", "required artifact"),
        ("wrong-artifact-kind", "wrong kind"),
        ("artifact-on-rejection", "content-hash revalidation"),
        ("missing-prerequisite", "out-of-stage"),
        ("repeat-market-snapshot", "out-of-stage"),
    ),
)
def test_replay_rejects_invalid_tool_payload_sequences(case: str, message: str) -> None:
    trace = _Trace()
    if case == "outcome-without-reservation":
        trace.add(
            AgentRunEventKind.TOOL_CALL_FAILED,
            AgentRunState.RECEIVED,
            trigger=AgentRunTrigger.TOOL,
            reason="ORPHAN_OUTCOME",
            request_id="orphan",
            tool_name="get_market_snapshot",
            tool_effect=ToolEffect.READ_ONLY,
            argument_hash=_digest("orphan"),
            error_codes=("ORPHAN",),
        )
    elif case == "event-before-outcome":
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.reserve("get_market_snapshot")
        trace.transition(AgentRunState.DATA_VALIDATED)
    elif case == "duplicate-request":
        trace.reserve("get_market_snapshot", request_id="duplicate_request")
        trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
        trace.reserve("get_market_snapshot", request_id="duplicate_request")
    elif case == "mismatched-request":
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.reserve("get_market_snapshot")
        trace.outcome(
            request_id="different_request", artifact_kind=AgentArtifactKind.MARKET_SNAPSHOT
        )
    elif case == "out-of-stage-success":
        trace.success("get_market_snapshot", AgentArtifactKind.MARKET_SNAPSHOT)
    elif case == "failed-advancement":
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.reserve("get_market_snapshot")
        trace.outcome(
            kind=AgentRunEventKind.TOOL_CALL_FAILED,
            state_after=AgentRunState.DATA_VALIDATED,
        )
    elif case == "success-without-artifact":
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.reserve("get_market_snapshot")
        trace.outcome()
    elif case == "wrong-artifact-kind":
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.reserve("get_market_snapshot")
        trace.outcome(artifact_kind=AgentArtifactKind.DATA_QUALITY)
    elif case == "artifact-on-rejection":
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.reserve("get_market_snapshot")
        successful = trace.outcome(
            artifact_kind=AgentArtifactKind.MARKET_SNAPSHOT,
        )
        trace.events[-1] = successful.model_copy(
            update={
                "kind": AgentRunEventKind.TOOL_CALL_REJECTED,
                "error_codes": ("FORGED_REJECTED_ARTIFACT",),
            }
        )
    elif case == "missing-prerequisite":
        analyzed = _analyzed(
            _Trace(
                goal=AgentRunGoal.PORTFOLIO_REPORT,
                mode=RuntimeMode.PAPER,
            )
        )
        trace = analyzed
        trace.success(
            "build_target_portfolio",
            AgentArtifactKind.TARGET_PORTFOLIO,
            state_after=AgentRunState.PORTFOLIO_READY,
        )
    else:
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.success("get_market_snapshot", AgentArtifactKind.MARKET_SNAPSHOT)
        trace.success("get_market_snapshot", AgentArtifactKind.MARKET_SNAPSHOT)

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


def test_replay_rejects_order_draft_without_expiry() -> None:
    trace = _risk_checked(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
    trace.success(
        "create_order_draft",
        AgentArtifactKind.ORDER_DRAFT,
        state_after=AgentRunState.DRAFT_READY,
        content_hash=LOCKED_BATCH_HASH,
    )

    with pytest.raises(AgentRunReplayError, match="retain its expiry"):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize("expiry_relation", ("equal", "past"))
@pytest.mark.parametrize("followup", ("missing", "delayed", "same-time"))
def test_already_expired_draft_outcome_requires_immediate_expiry_transition(
    expiry_relation: str,
    followup: str,
) -> None:
    trace = _risk_checked(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
    trace.reserve("create_order_draft")
    outcome_at = trace.at + timedelta(seconds=1)
    draft_expiry = outcome_at - (
        timedelta(microseconds=1) if expiry_relation == "past" else timedelta(0)
    )
    trace.outcome(
        state_after=AgentRunState.DRAFT_READY,
        artifact_kind=AgentArtifactKind.ORDER_DRAFT,
        content_hash=LOCKED_BATCH_HASH,
        expires_at=draft_expiry,
        at=outcome_at,
    )
    if followup != "missing":
        trace.transition(
            AgentRunState.EXPIRED,
            kind=AgentRunEventKind.TIMEOUT_RECORDED,
            trigger=AgentRunTrigger.TIMEOUT,
            at=(outcome_at if followup == "same-time" else outcome_at + timedelta(microseconds=1)),
        )

    if followup == "same-time":
        snapshot = replay_agent_run(trace.chain)
        assert snapshot.state is AgentRunState.EXPIRED
        assert snapshot.events[-2].occurred_at == snapshot.events[-1].occurred_at
    else:
        with pytest.raises(AgentRunReplayError):
            replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    ("replace_identity", "message"),
    ((True, "changed content"), (False, "replace its locked")),
)
def test_replay_rejects_changed_or_multiple_locked_drafts(
    replace_identity: bool,
    message: str,
) -> None:
    trace = _draft_ready(_Trace(goal=AgentRunGoal.ORDER_DRAFT, mode=RuntimeMode.PAPER))
    trace.reserve("get_order_draft", batch_hash=LOCKED_BATCH_HASH)
    trace.outcome(
        artifact_kind=AgentArtifactKind.ORDER_DRAFT,
        artifact_id="order_draft_001" if replace_identity else "order_draft_002",
        content_hash=_digest("different-batch"),
        expires_at=NOW + timedelta(minutes=20),
    )

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("pending-without-draft", "missing required artifact"),
        ("pending-after-expiry", "approval window expired"),
        ("approval-batch", "does not bind"),
        ("approval-deadline", "at or after"),
        ("rejection-batch", "unrelated"),
        ("approval-wrong-transition", "illegal transition"),
    ),
)
def test_replay_rejects_invalid_approval_sequences(case: str, message: str) -> None:
    if case == "pending-without-draft":
        trace = _risk_checked(
            _Trace(
                goal=AgentRunGoal.LIVE_ASSISTED_EXECUTION,
                mode=RuntimeMode.LIVE_ASSISTED,
            )
        )
        trace.transition(AgentRunState.DRAFT_READY)
        trace.transition(AgentRunState.PENDING_APPROVAL)
    elif case == "pending-after-expiry":
        trace = _draft_ready(
            _Trace(
                goal=AgentRunGoal.LIVE_ASSISTED_EXECUTION,
                mode=RuntimeMode.LIVE_ASSISTED,
            ),
            expires_at=NOW + timedelta(seconds=20),
        )
        trace.transition(
            AgentRunState.PENDING_APPROVAL,
            at=NOW + timedelta(seconds=21),
        )
    else:
        trace = _live_pending()
        if case == "approval-batch":
            other_batch = _digest("other-batch")
            approval_hash = _digest("approval")
            trace.transition(
                AgentRunState.APPROVED,
                kind=AgentRunEventKind.APPROVAL_RECORDED,
                trigger=AgentRunTrigger.HUMAN,
                batch_hash=other_batch,
                control_hash=approval_hash,
                external_approval=_approval_evidence(
                    trace,
                    approval_hash=approval_hash,
                    batch_hash=other_batch,
                ),
            )
        elif case == "approval-deadline":
            deadline = NOW + timedelta(seconds=trace.limits.timeout_seconds)
            approval_hash = _digest("approval")
            trace.transition(
                AgentRunState.APPROVED,
                kind=AgentRunEventKind.APPROVAL_RECORDED,
                trigger=AgentRunTrigger.HUMAN,
                at=deadline,
                batch_hash=LOCKED_BATCH_HASH,
                control_hash=approval_hash,
                external_approval=_approval_evidence(
                    trace,
                    approval_hash=approval_hash,
                ),
            )
        elif case == "rejection-batch":
            trace.transition(
                AgentRunState.REJECTED,
                kind=AgentRunEventKind.APPROVAL_REJECTED,
                trigger=AgentRunTrigger.HUMAN,
                batch_hash=_digest("other-batch"),
                control_hash=_digest("rejection"),
            )
        else:
            trace.transition(AgentRunState.APPROVED)

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


def test_successful_live_approval_binds_the_locked_batch() -> None:
    snapshot = replay_agent_run(_live_completed().chain)

    approval = next(
        event for event in snapshot.events if event.kind is AgentRunEventKind.APPROVAL_RECORDED
    )
    draft = next(
        artifact
        for artifact in snapshot.artifacts
        if artifact.kind is AgentArtifactKind.ORDER_DRAFT
    )
    evidence = approval.external_approval
    assert evidence is not None
    assert approval.control_hash == evidence.approval_hash
    assert approval.decision_id == evidence.decision_id
    assert approval.batch_hash == evidence.batch_hash == draft.content_hash == LOCKED_BATCH_HASH


def test_approval_cannot_predate_entry_into_pending_approval() -> None:
    trace = _live_pending()
    pending_approval_since = trace.at
    approval_hash = _digest("stale-approval-control")
    trace.transition(
        AgentRunState.APPROVED,
        kind=AgentRunEventKind.APPROVAL_RECORDED,
        trigger=AgentRunTrigger.HUMAN,
        batch_hash=LOCKED_BATCH_HASH,
        control_hash=approval_hash,
        external_approval=_approval_evidence(
            trace,
            approval_hash=approval_hash,
            approved_at=pending_approval_since - timedelta(microseconds=1),
        ),
    )

    with pytest.raises(AgentRunReplayError):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("early-timeout", "before the immutable run deadline"),
        ("early-expiry", "before every bound expiry threshold"),
        ("early-limit", "before an immutable threshold"),
        ("invalid-budget", "exhausted tool budget must immediately"),
        ("total-budget", "exhausted tool budget must immediately"),
        ("reconciliation-budget", "failed reconciliation must immediately"),
    ),
)
def test_replay_rejects_false_deadlines_or_exhausted_budgets(
    case: str,
    message: str,
) -> None:
    if case == "early-timeout":
        trace = _Trace()
        trace.transition(
            AgentRunState.TIMED_OUT,
            kind=AgentRunEventKind.TIMEOUT_RECORDED,
            trigger=AgentRunTrigger.TIMEOUT,
        )
    elif case == "early-expiry":
        trace = _live_pending()
        trace.transition(
            AgentRunState.EXPIRED,
            kind=AgentRunEventKind.TIMEOUT_RECORDED,
            trigger=AgentRunTrigger.TIMEOUT,
        )
    elif case == "early-limit":
        trace = _Trace()
        trace.transition(
            AgentRunState.LIMIT_EXCEEDED,
            kind=AgentRunEventKind.LIMIT_REACHED,
        )
    elif case == "invalid-budget":
        trace = _Trace(
            limits=AgentRunLimits(
                max_tool_calls=3,
                max_invalid_tool_calls=1,
                max_reconciliation_attempts=1,
            )
        )
        for index in range(2):
            trace.reserve("get_market_snapshot", request_id=f"invalid_{index}")
            trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
    elif case == "total-budget":
        trace = _Trace(
            limits=AgentRunLimits(
                max_tool_calls=1,
                max_invalid_tool_calls=1,
                max_reconciliation_attempts=1,
            )
        )
        _data_validated(trace)
        trace.reserve("rank_stock_candidates")
    else:
        trace = _paper_reconciling(
            limits=AgentRunLimits(
                max_tool_calls=20,
                max_invalid_tool_calls=2,
                max_reconciliation_attempts=1,
            )
        )
        trace.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
        trace.outcome(kind=AgentRunEventKind.TOOL_CALL_FAILED)
        trace.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


def test_replay_accepts_timeout_and_limit_only_at_bound_thresholds() -> None:
    timeout_trace = _Trace()
    timeout_trace.transition(
        AgentRunState.TIMED_OUT,
        kind=AgentRunEventKind.TIMEOUT_RECORDED,
        trigger=AgentRunTrigger.TIMEOUT,
        at=NOW + timedelta(seconds=timeout_trace.limits.timeout_seconds),
    )
    limited_trace = _Trace(
        limits=AgentRunLimits(
            max_tool_calls=1,
            max_invalid_tool_calls=1,
            max_reconciliation_attempts=1,
        )
    )
    limited_trace.reserve("get_market_snapshot")
    limited_trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
    limited_trace.transition(
        AgentRunState.LIMIT_EXCEEDED,
        kind=AgentRunEventKind.LIMIT_REACHED,
        at=limited_trace.at,
    )

    assert replay_agent_run(timeout_trace.chain).state is AgentRunState.TIMED_OUT
    assert replay_agent_run(limited_trace.chain).state is AgentRunState.LIMIT_EXCEEDED


@pytest.mark.parametrize(
    "scenario",
    (
        "invalid-pre-execution",
        "model-pre-execution",
        "invalid-executing",
        "model-executing",
    ),
)
@pytest.mark.parametrize("continuation", ("omitted", "reserve"))
def test_exhausted_budget_requires_an_immediate_terminal_event(
    scenario: str,
    continuation: str,
) -> None:
    trace, _, _, next_tool, batch_hash = _trace_at_exhausted_budget(scenario)
    if continuation == "reserve":
        trace.reserve(next_tool, batch_hash=batch_hash)

    with pytest.raises(
        AgentRunReplayError,
        match=r"exhausted tool budget.*immediately|not closed by a terminal event",
    ):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    "scenario",
    (
        "invalid-pre-execution",
        "model-pre-execution",
        "invalid-executing",
        "model-executing",
    ),
)
def test_exhausted_budget_closes_in_the_context_specific_terminal_state(
    scenario: str,
) -> None:
    trace, expected_kind, expected_state, _, _ = _trace_at_exhausted_budget(scenario)
    exhausted_at = trace.at
    trace.transition(
        expected_state,
        kind=expected_kind,
        reason="BUDGET_EXHAUSTED",
        at=exhausted_at,
    )

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is expected_state
    assert snapshot.events[-2].occurred_at == snapshot.events[-1].occurred_at


def test_replay_requires_the_central_tool_effect() -> None:
    trace = _Trace()
    trace.transition(AgentRunState.SNAPSHOT_READY)
    trace.reserve("get_market_snapshot", effect=ToolEffect.ISOLATED_COMPUTE)

    with pytest.raises(AgentRunReplayError, match="central v1 policy"):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    ("error_code", "expected_state"),
    (
        (ErrorCode.INVALID_ARGUMENT, AgentRunState.SNAPSHOT_READY),
        (ErrorCode.UNAUTHORIZED, AgentRunState.REJECTED),
        (ErrorCode.FORBIDDEN, AgentRunState.REJECTED),
        (ErrorCode.NOT_FOUND, AgentRunState.SNAPSHOT_READY),
        (ErrorCode.CONFLICT, AgentRunState.SNAPSHOT_READY),
        (ErrorCode.DATA_UNAVAILABLE, AgentRunState.SNAPSHOT_READY),
        (ErrorCode.DATA_INVALID, AgentRunState.DATA_INVALID),
        (ErrorCode.RISK_REJECTED, AgentRunState.REJECTED),
        (ErrorCode.APPROVAL_REQUIRED, AgentRunState.REJECTED),
        (ErrorCode.APPROVAL_EXPIRED, AgentRunState.FAILED),
        (ErrorCode.KILL_SWITCH_ACTIVE, AgentRunState.INCIDENT),
        (ErrorCode.INTERNAL_ERROR, AgentRunState.FAILED),
    ),
)
def test_normal_failed_error_code_matches_service_state_projection(
    error_code: ErrorCode,
    expected_state: AgentRunState,
) -> None:
    valid = _Trace()
    valid.transition(AgentRunState.SNAPSHOT_READY)
    valid.reserve("get_market_snapshot")
    valid.outcome(
        kind=AgentRunEventKind.TOOL_CALL_FAILED,
        state_after=expected_state,
        error_codes=(error_code.value,),
    )
    assert replay_agent_run(valid.chain).state is expected_state

    invalid = _Trace()
    invalid.transition(AgentRunState.SNAPSHOT_READY)
    invalid.reserve("get_market_snapshot")
    invalid.outcome(
        kind=AgentRunEventKind.TOOL_CALL_FAILED,
        state_after=(
            AgentRunState.FAILED
            if expected_state is AgentRunState.SNAPSHOT_READY
            else AgentRunState.SNAPSHOT_READY
        ),
        error_codes=(error_code.value,),
    )
    with pytest.raises(AgentRunReplayError, match="fail-closed error projection"):
        replay_agent_run(invalid.chain)


@pytest.mark.parametrize(
    ("error_code", "terminal_state"),
    (
        (ErrorCode.KILL_SWITCH_ACTIVE.value, AgentRunState.INCIDENT),
        ("TOOL_AUDIT_UNAVAILABLE", AgentRunState.FAILED),
    ),
)
def test_kill_switch_or_audit_failure_cannot_self_loop_then_retry(
    error_code: str,
    terminal_state: AgentRunState,
) -> None:
    invalid = _Trace()
    invalid.transition(AgentRunState.SNAPSHOT_READY)
    invalid.reserve("get_market_snapshot")
    invalid.outcome(
        kind=AgentRunEventKind.TOOL_CALL_FAILED,
        error_codes=(error_code,),
    )
    invalid.reserve("get_market_snapshot")
    with pytest.raises(AgentRunReplayError, match="fail-closed error projection"):
        replay_agent_run(invalid.chain)

    valid = _Trace()
    valid.transition(AgentRunState.SNAPSHOT_READY)
    valid.reserve("get_market_snapshot")
    valid.outcome(
        kind=AgentRunEventKind.TOOL_CALL_FAILED,
        state_after=terminal_state,
        error_codes=(error_code,),
    )
    assert replay_agent_run(valid.chain).state is terminal_state


@pytest.mark.parametrize(
    ("continuation", "message"),
    (
        ("missing", "not closed by REJECTED"),
        ("delayed", "must immediately enter REJECTED"),
        ("retry", "must immediately enter REJECTED"),
    ),
)
def test_registry_authorization_rejection_requires_same_time_rejected_transition(
    continuation: str,
    message: str,
) -> None:
    trace = _Trace()
    trace.transition(AgentRunState.SNAPSHOT_READY)
    trace.reserve("get_market_snapshot")
    trace.outcome(
        kind=AgentRunEventKind.TOOL_CALL_REJECTED,
        reason="REGISTRY_AUTHORIZATION_REJECTED",
        error_codes=("REGISTRY_CAPABILITY_NOT_GRANTED",),
    )
    if continuation == "delayed":
        trace.transition(
            AgentRunState.REJECTED,
            at=trace.at + timedelta(microseconds=1),
        )
    elif continuation == "retry":
        trace.reserve("get_market_snapshot")

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


def test_registry_authorization_rejection_accepts_same_time_rejected_transition() -> None:
    trace = _Trace()
    trace.transition(AgentRunState.SNAPSHOT_READY)
    trace.reserve("get_market_snapshot")
    trace.outcome(
        kind=AgentRunEventKind.TOOL_CALL_REJECTED,
        reason="REGISTRY_AUTHORIZATION_REJECTED",
        error_codes=("REGISTRY_CAPABILITY_NOT_GRANTED",),
    )
    rejected_at = trace.at
    trace.transition(AgentRunState.REJECTED, at=rejected_at)

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.REJECTED
    assert snapshot.events[-2].occurred_at == snapshot.events[-1].occurred_at


@pytest.mark.parametrize(
    "reservation_case",
    (
        "artifact-missing-idempotency",
        "paper-missing-idempotency",
        "paper-missing-batch",
        "live-missing-idempotency",
        "live-missing-batch",
    ),
)
@pytest.mark.parametrize(
    "outcome_kind",
    (
        AgentRunEventKind.TOOL_CALL_REJECTED,
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
    ),
)
def test_incomplete_write_identity_can_only_be_immediately_rejected(
    reservation_case: str,
    outcome_kind: AgentRunEventKind,
) -> None:
    if reservation_case.startswith("artifact"):
        trace = _risk_checked(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
        reservation = trace.reserve(
            "create_order_draft",
            omit_idempotency_key=True,
        )
    elif reservation_case.startswith("paper"):
        trace = _draft_ready(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
        trace.transition(AgentRunState.EXECUTING)
        reservation = trace.reserve(
            "submit_paper_orders",
            batch_hash=(None if reservation_case.endswith("batch") else LOCKED_BATCH_HASH),
            omit_idempotency_key=reservation_case.endswith("idempotency"),
        )
    else:
        trace = _live_approved()
        reservation = trace.reserve(
            "submit_approved_orders",
            batch_hash=(None if reservation_case.endswith("batch") else LOCKED_BATCH_HASH),
            omit_idempotency_key=reservation_case.endswith("idempotency"),
        )

    if reservation_case.endswith("idempotency"):
        assert reservation.idempotency_key_hash is None
    else:
        assert reservation.batch_hash is None
    trace.outcome(kind=outcome_kind)
    assert trace.events[-2] == reservation

    if outcome_kind is AgentRunEventKind.TOOL_CALL_REJECTED:
        if not reservation_case.startswith("artifact"):
            trace.transition(
                AgentRunState.INCIDENT,
                kind=AgentRunEventKind.INCIDENT_RECORDED,
                reason="INVALID_EXECUTION_RESERVATION",
                at=trace.at,
            )
        snapshot = replay_agent_run(trace.chain)
        assert snapshot.invalid_tool_call_count == 1
        assert snapshot.pending_call is None
        assert snapshot.state is (
            AgentRunState.RISK_CHECKED
            if reservation_case.startswith("artifact")
            else AgentRunState.INCIDENT
        )
    else:
        with pytest.raises(AgentRunReplayError, match="invalid reservation"):
            replay_agent_run(trace.chain)


def test_replay_rejects_a_changed_write_idempotency_identity() -> None:
    trace = _risk_checked(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
    trace.reserve("create_order_draft", request_id="draft_attempt_one")
    trace.outcome(
        kind=AgentRunEventKind.TOOL_CALL_FAILED,
        error_codes=(ErrorCode.INVALID_ARGUMENT.value,),
    )
    trace.reserve("create_order_draft", request_id="draft_attempt_two")
    trace.outcome(
        kind=AgentRunEventKind.TOOL_CALL_FAILED,
        error_codes=(ErrorCode.INVALID_ARGUMENT.value,),
    )

    with pytest.raises(AgentRunReplayError, match="changed its locked idempotency"):
        replay_agent_run(trace.chain)


def test_cancelled_execution_rejection_requires_incident_and_blocks_resubmit() -> None:
    trace = _paper_executing()
    trace.reserve(
        "submit_paper_orders",
        request_id="submission_before_cancel",
        batch_hash=LOCKED_BATCH_HASH,
    )
    trace.transition(
        AgentRunState.EXECUTING,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
    )
    trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
    with pytest.raises(AgentRunReplayError):
        replay_agent_run(trace.chain)

    rejected_at = trace.at
    trace.transition(
        AgentRunState.INCIDENT,
        kind=AgentRunEventKind.INCIDENT_RECORDED,
        reason="UNSAFE_REJECTION_AFTER_CANCELLATION",
        at=rejected_at,
    )
    incident = replay_agent_run(trace.chain)
    assert incident.state is AgentRunState.INCIDENT
    assert incident.cancel_requested
    assert incident.pending_call is None
    assert incident.events[-2].occurred_at == incident.events[-1].occurred_at

    trace.reserve(
        "submit_paper_orders",
        request_id="submission_after_cancel",
        batch_hash=LOCKED_BATCH_HASH,
    )

    with pytest.raises(AgentRunReplayError, match="illegal transition"):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    "scenario",
    (
        "executing-invalid-pending",
        "executing-stage-invalid",
        "reconciling-invalid-pending",
        "reconciling-stage-invalid",
    ),
)
@pytest.mark.parametrize("continuation", ("omitted", "retry"))
def test_unsafe_execution_denial_requires_an_immediate_incident(
    scenario: str,
    continuation: str,
) -> None:
    trace, retry_tool, retry_batch = _unsafe_execution_denial_trace(scenario)
    if continuation == "retry":
        trace.reserve(retry_tool, batch_hash=retry_batch)

    with pytest.raises(AgentRunReplayError):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    "scenario",
    (
        "executing-invalid-pending",
        "executing-stage-invalid",
        "reconciling-invalid-pending",
        "reconciling-stage-invalid",
    ),
)
def test_unsafe_execution_denial_accepts_same_time_incident_closure(
    scenario: str,
) -> None:
    trace, _, _ = _unsafe_execution_denial_trace(scenario)
    denied_at = trace.at
    trace.transition(
        AgentRunState.INCIDENT,
        kind=AgentRunEventKind.INCIDENT_RECORDED,
        reason="PRE_DISPATCH_DENIAL_AFTER_EXECUTION",
        at=denied_at,
    )

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.INCIDENT
    assert snapshot.events[-2].occurred_at == snapshot.events[-1].occurred_at


@pytest.mark.parametrize("state", (AgentRunState.EXECUTING, AgentRunState.RECONCILING))
def test_stage_valid_argument_rejection_remains_retryable_within_budget(
    state: AgentRunState,
) -> None:
    trace = _paper_reconciling() if state is AgentRunState.RECONCILING else _paper_executing()
    tool_name = "reconcile_account" if state is AgentRunState.RECONCILING else "submit_paper_orders"
    trace.reserve(tool_name, batch_hash=LOCKED_BATCH_HASH)
    trace.outcome(
        kind=AgentRunEventKind.TOOL_CALL_REJECTED,
        reason="TOOL_ARGUMENTS_REJECTED",
        error_codes=("REGISTRY_INVALID_ARGUMENTS",),
    )
    rejected = replay_agent_run(trace.chain)
    assert rejected.state is state
    assert rejected.invalid_tool_call_count == 1

    retry = trace.reserve(tool_name, batch_hash=LOCKED_BATCH_HASH)
    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is state
    assert snapshot.pending_call is not None
    assert snapshot.pending_call.request_id == retry.request_id


@pytest.mark.parametrize(
    ("outcome_kind", "required_state", "message"),
    (
        (
            AgentRunEventKind.TOOL_CALL_SUCCEEDED,
            AgentRunState.RECONCILING,
            "successful execution must immediately enter RECONCILING",
        ),
        (
            AgentRunEventKind.TOOL_CALL_REPLAYED,
            AgentRunState.RECONCILING,
            "successful execution must immediately enter RECONCILING",
        ),
        (
            AgentRunEventKind.TOOL_CALL_FAILED,
            AgentRunState.INCIDENT,
            "uncertain execution failure must immediately enter INCIDENT",
        ),
    ),
)
def test_execution_outcome_must_directly_enter_its_safe_state(
    outcome_kind: AgentRunEventKind,
    required_state: AgentRunState,
    message: str,
) -> None:
    invalid = _paper_executing()
    invalid.reserve("submit_paper_orders", batch_hash=LOCKED_BATCH_HASH)
    invalid.outcome(
        kind=outcome_kind,
        state_after=AgentRunState.EXECUTING,
        artifact_kind=(
            AgentArtifactKind.EXECUTION_RECEIPT
            if outcome_kind
            in {
                AgentRunEventKind.TOOL_CALL_SUCCEEDED,
                AgentRunEventKind.TOOL_CALL_REPLAYED,
            }
            else None
        ),
    )
    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(invalid.chain)

    valid = _paper_executing()
    valid.reserve("submit_paper_orders", batch_hash=LOCKED_BATCH_HASH)
    valid.outcome(
        kind=outcome_kind,
        state_after=required_state,
        artifact_kind=(
            AgentArtifactKind.EXECUTION_RECEIPT
            if required_state is AgentRunState.RECONCILING
            else None
        ),
    )

    assert replay_agent_run(valid.chain).state is required_state


@pytest.mark.parametrize(
    "outcome_kind",
    (
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
    ),
)
def test_successful_reconciliation_must_close_as_completed_or_incident(
    outcome_kind: AgentRunEventKind,
) -> None:
    invalid = _paper_reconciling()
    invalid.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
    invalid.outcome(
        kind=outcome_kind,
        state_after=AgentRunState.RECONCILING,
        artifact_kind=AgentArtifactKind.RECONCILIATION,
    )
    with pytest.raises(AgentRunReplayError, match="must close the execution outcome"):
        replay_agent_run(invalid.chain)

    for safe_state in (AgentRunState.COMPLETED, AgentRunState.INCIDENT):
        valid = _paper_reconciling()
        valid.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
        valid.outcome(
            kind=outcome_kind,
            state_after=safe_state,
            artifact_kind=AgentArtifactKind.RECONCILIATION,
        )
        assert replay_agent_run(valid.chain).state is safe_state


def test_failed_reconciliation_must_immediately_enter_incident() -> None:
    invalid = _paper_reconciling()
    invalid.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
    invalid.outcome(kind=AgentRunEventKind.TOOL_CALL_FAILED)
    with pytest.raises(
        AgentRunReplayError,
        match="failed reconciliation must immediately enter INCIDENT",
    ):
        replay_agent_run(invalid.chain)

    valid = _paper_reconciling()
    valid.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
    valid.outcome(
        kind=AgentRunEventKind.TOOL_CALL_FAILED,
        state_after=AgentRunState.INCIDENT,
    )
    assert replay_agent_run(valid.chain).state is AgentRunState.INCIDENT


@pytest.mark.parametrize(
    ("followup", "message"),
    (
        ("direct", "rejected tool outcome must be recorded before"),
        ("missing", "rejected reconciliation was not closed"),
        ("delayed", "rejected reconciliation must immediately enter INCIDENT"),
    ),
)
def test_rejected_reconciliation_requires_an_immediate_incident_event(
    followup: str,
    message: str,
) -> None:
    trace = _paper_reconciling()
    trace.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
    trace.outcome(
        kind=AgentRunEventKind.TOOL_CALL_REJECTED,
        state_after=(AgentRunState.INCIDENT if followup == "direct" else AgentRunState.RECONCILING),
    )
    if followup == "delayed":
        trace.transition(
            AgentRunState.INCIDENT,
            kind=AgentRunEventKind.INCIDENT_RECORDED,
            at=trace.at + timedelta(microseconds=1),
        )

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


def test_rejected_reconciliation_accepts_only_same_time_incident_followup() -> None:
    trace = _paper_reconciling()
    trace.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
    trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
    rejected_at = trace.at
    trace.transition(
        AgentRunState.INCIDENT,
        kind=AgentRunEventKind.INCIDENT_RECORDED,
        at=rejected_at,
    )

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.INCIDENT
    assert snapshot.events[-2].occurred_at == snapshot.events[-1].occurred_at


def test_replay_rejects_an_out_of_stage_reservation_left_pending() -> None:
    trace = _Trace()
    trace.reserve("get_market_snapshot")

    with pytest.raises(AgentRunReplayError, match="cannot remain pending"):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize("terminal_kind", ("cancel", "timeout"))
def test_terminal_interruptions_clear_a_pending_call(terminal_kind: str) -> None:
    trace = _Trace()
    trace.transition(AgentRunState.SNAPSHOT_READY)
    trace.reserve("get_market_snapshot")
    if terminal_kind == "cancel":
        trace.transition(
            AgentRunState.CANCELLED,
            kind=AgentRunEventKind.CANCELLATION_REQUESTED,
            trigger=AgentRunTrigger.CANCELLATION,
        )
    else:
        trace.transition(
            AgentRunState.TIMED_OUT,
            kind=AgentRunEventKind.TIMEOUT_RECORDED,
            trigger=AgentRunTrigger.TIMEOUT,
            at=NOW + timedelta(seconds=trace.limits.timeout_seconds),
        )

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.pending_call is None
    assert snapshot.state in {AgentRunState.CANCELLED, AgentRunState.TIMED_OUT}


@pytest.mark.parametrize("write_mode", ("paper", "live"))
@pytest.mark.parametrize("boundary_kind", ("run", "authorization"))
def test_timeout_or_expiry_cannot_mask_a_pending_execution_write(
    write_mode: str,
    boundary_kind: str,
) -> None:
    trace, boundary, _ = _pending_execution_at_boundary(write_mode, boundary_kind)
    trace.transition(
        (AgentRunState.TIMED_OUT if boundary_kind == "run" else AgentRunState.EXPIRED),
        kind=AgentRunEventKind.TIMEOUT_RECORDED,
        trigger=AgentRunTrigger.TIMEOUT,
        at=boundary,
    )

    with pytest.raises(AgentRunReplayError):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize("write_mode", ("paper", "live"))
@pytest.mark.parametrize("boundary_kind", ("run", "authorization"))
def test_pending_execution_write_accepts_incident_at_expiry_boundary(
    write_mode: str,
    boundary_kind: str,
) -> None:
    trace, boundary, _ = _pending_execution_at_boundary(write_mode, boundary_kind)
    trace.transition(
        AgentRunState.INCIDENT,
        kind=AgentRunEventKind.INCIDENT_RECORDED,
        trigger=(AgentRunTrigger.TIMEOUT if boundary_kind == "run" else AgentRunTrigger.SYSTEM),
        at=boundary,
    )

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.INCIDENT
    assert snapshot.pending_call is None


@pytest.mark.parametrize("reservation_kind", ("out-of-stage", "malformed"))
@pytest.mark.parametrize("interruption", ("cancel", "timeout", "incident"))
def test_terminal_interruption_cannot_mask_an_invalid_pending_reservation(
    reservation_kind: str,
    interruption: str,
) -> None:
    if reservation_kind == "out-of-stage":
        trace = _Trace()
        trace.reserve("get_market_snapshot")
    else:
        trace = _risk_checked(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
        trace.reserve("create_order_draft", omit_idempotency_key=True)

    if interruption == "cancel":
        trace.transition(
            AgentRunState.CANCELLED,
            kind=AgentRunEventKind.CANCELLATION_REQUESTED,
            trigger=AgentRunTrigger.CANCELLATION,
        )
    elif interruption == "timeout":
        trace.transition(
            AgentRunState.TIMED_OUT,
            kind=AgentRunEventKind.TIMEOUT_RECORDED,
            trigger=AgentRunTrigger.TIMEOUT,
            at=NOW + timedelta(seconds=trace.limits.timeout_seconds),
        )
    else:
        trace.transition(
            AgentRunState.INCIDENT,
            kind=AgentRunEventKind.INCIDENT_RECORDED,
        )

    with pytest.raises(AgentRunReplayError):
        replay_agent_run(trace.chain)


def test_pre_execution_cancellation_observation_cannot_leave_call_pending() -> None:
    trace = _Trace()
    trace.transition(AgentRunState.SNAPSHOT_READY)
    trace.reserve("get_market_snapshot")
    trace.transition(
        AgentRunState.SNAPSHOT_READY,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
    )

    with pytest.raises(AgentRunReplayError, match="illegal transition"):
        replay_agent_run(trace.chain)


def test_existing_locked_draft_can_satisfy_a_read_replay_without_new_artifact() -> None:
    trace = _draft_ready(_Trace(goal=AgentRunGoal.ORDER_DRAFT, mode=RuntimeMode.PAPER))
    trace.reserve("get_order_draft", batch_hash=LOCKED_BATCH_HASH)
    trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REPLAYED)

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.DRAFT_READY
    assert sum(item.kind is AgentArtifactKind.ORDER_DRAFT for item in snapshot.artifacts) == 1


def test_wrong_draft_batch_is_stage_invalid_but_may_be_rejected() -> None:
    trace = _draft_ready(_Trace(goal=AgentRunGoal.ORDER_DRAFT, mode=RuntimeMode.PAPER))
    trace.reserve("get_order_draft", batch_hash=_digest("unrelated-batch"))
    trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.invalid_tool_call_count == 1


@pytest.mark.parametrize(
    ("goal", "mode", "message"),
    (
        (
            AgentRunGoal.MARKET_RESEARCH,
            RuntimeMode.RESEARCH,
            "ANALYZED requires",
        ),
        (
            AgentRunGoal.PORTFOLIO_REPORT,
            RuntimeMode.PAPER,
            "missing required artifact STOCK_CANDIDATES",
        ),
    ),
)
def test_analyzed_milestone_requires_goal_specific_evidence(
    goal: AgentRunGoal,
    mode: RuntimeMode,
    message: str,
) -> None:
    trace = _data_validated(_Trace(goal=goal, mode=mode))
    trace.transition(AgentRunState.ANALYZED)

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("tool-after-deadline", "reserved at or after"),
        ("execution-call-expired", "authorization expired"),
        ("execution-unlock-expired", "execution was unlocked"),
    ),
)
def test_replay_enforces_deadlines_before_dispatch_and_execution(
    case: str,
    message: str,
) -> None:
    if case == "tool-after-deadline":
        trace = _Trace()
        trace.transition(AgentRunState.SNAPSHOT_READY)
        trace.reserve(
            "get_market_snapshot",
            at=NOW + timedelta(seconds=trace.limits.timeout_seconds),
        )
    else:
        limits = AgentRunLimits(approval_timeout_seconds=30)
        trace = _live_pending(limits=limits)
        approval_deadline = trace.at + timedelta(seconds=limits.approval_timeout_seconds)
        approval_hash = _digest("approval-deadline-test")
        trace.transition(
            AgentRunState.APPROVED,
            kind=AgentRunEventKind.APPROVAL_RECORDED,
            trigger=AgentRunTrigger.HUMAN,
            batch_hash=LOCKED_BATCH_HASH,
            control_hash=approval_hash,
            external_approval=_approval_evidence(trace, approval_hash=approval_hash),
        )
        if case == "execution-call-expired":
            trace.transition(AgentRunState.EXECUTING)
            trace.reserve(
                "submit_approved_orders",
                batch_hash=LOCKED_BATCH_HASH,
                at=approval_deadline,
            )
        else:
            trace.transition(AgentRunState.EXECUTING, at=approval_deadline)

    with pytest.raises(AgentRunReplayError, match=message):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    "state",
    (
        AgentRunState.DRAFT_READY,
        AgentRunState.PENDING_APPROVAL,
        AgentRunState.APPROVED,
        AgentRunState.EXECUTING,
    ),
)
def test_non_reconciliation_reservation_obeys_earliest_authorization_expiry(
    state: AgentRunState,
) -> None:
    before, expiry = _trace_at_earlier_authorization_deadline(state)
    tool_name = "submit_approved_orders" if state is AgentRunState.EXECUTING else "get_order_draft"
    before.reserve(
        tool_name,
        batch_hash=LOCKED_BATCH_HASH,
        at=expiry - timedelta(microseconds=1),
    )
    before_snapshot = replay_agent_run(before.chain)
    assert before_snapshot.pending_call is not None
    assert before_snapshot.pending_call.tool_name == tool_name

    expired, expiry = _trace_at_earlier_authorization_deadline(state)
    expired.reserve(tool_name, batch_hash=LOCKED_BATCH_HASH, at=expiry)
    with pytest.raises(AgentRunReplayError):
        replay_agent_run(expired.chain)


@pytest.mark.parametrize("transition_case", ("progress", "data-failure"))
def test_non_reconciliation_state_transition_cannot_occur_after_run_deadline(
    transition_case: str,
) -> None:
    trace = _Trace()
    if transition_case == "progress":
        target = AgentRunState.SNAPSHOT_READY
    else:
        trace.transition(AgentRunState.SNAPSHOT_READY)
        target = AgentRunState.DATA_INVALID
    trace.transition(
        target,
        at=NOW + timedelta(seconds=trace.limits.timeout_seconds + 1),
    )

    with pytest.raises(AgentRunReplayError):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    "state",
    (
        AgentRunState.RECEIVED,
        AgentRunState.SNAPSHOT_READY,
        AgentRunState.DATA_VALIDATED,
        AgentRunState.ANALYZED,
        AgentRunState.PORTFOLIO_READY,
        AgentRunState.RISK_CHECKED,
        AgentRunState.DRAFT_READY,
        AgentRunState.PENDING_APPROVAL,
        AgentRunState.APPROVED,
        AgentRunState.EXECUTING,
    ),
)
@pytest.mark.parametrize("seconds_after_deadline", (0, 1), ids=("at", "after"))
def test_cancellation_cannot_mask_an_expired_run_deadline(
    state: AgentRunState,
    seconds_after_deadline: int,
) -> None:
    trace = _cancellable_trace_at(state)
    run_deadline = NOW + timedelta(seconds=trace.limits.timeout_seconds)
    trace.transition(
        AgentRunState.CANCELLED,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
        at=run_deadline + timedelta(seconds=seconds_after_deadline),
    )

    with pytest.raises(AgentRunReplayError, match="cannot mask an expired runtime deadline"):
        replay_agent_run(trace.chain)


@pytest.mark.parametrize(
    "state",
    (
        AgentRunState.DRAFT_READY,
        AgentRunState.PENDING_APPROVAL,
        AgentRunState.APPROVED,
        AgentRunState.EXECUTING,
    ),
)
@pytest.mark.parametrize("seconds_after_deadline", (0, 1), ids=("at", "after"))
def test_cancellation_cannot_mask_an_earlier_authorization_deadline(
    state: AgentRunState,
    seconds_after_deadline: int,
) -> None:
    trace, authorization_deadline = _trace_at_earlier_authorization_deadline(state)
    trace.transition(
        AgentRunState.CANCELLED,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
        at=authorization_deadline + timedelta(seconds=seconds_after_deadline),
    )

    with pytest.raises(AgentRunReplayError, match="cannot mask an expired runtime deadline"):
        replay_agent_run(trace.chain)


def test_reconciling_cancellation_remains_observational_after_all_deadlines() -> None:
    trace = _paper_reconciling()
    draft = next(
        artifact
        for artifact in replay_agent_run(trace.chain).artifacts
        if artifact.kind is AgentArtifactKind.ORDER_DRAFT
    )
    assert draft.expires_at is not None
    after_all_deadlines = max(
        NOW + timedelta(seconds=trace.limits.timeout_seconds),
        draft.expires_at,
    ) + timedelta(seconds=1)
    reservation = trace.reserve(
        "reconcile_account",
        batch_hash=LOCKED_BATCH_HASH,
        at=after_all_deadlines,
    )
    trace.transition(
        AgentRunState.RECONCILING,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
        at=after_all_deadlines + timedelta(seconds=1),
    )

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.RECONCILING
    assert snapshot.cancel_requested
    assert snapshot.pending_call is not None
    assert snapshot.pending_call.request_id == reservation.request_id
    assert any(
        artifact.kind is AgentArtifactKind.EXECUTION_RECEIPT for artifact in snapshot.artifacts
    )


@pytest.mark.parametrize("early", (True, False), ids=("early", "at-deadline"))
def test_timeout_incident_requires_the_run_deadline(early: bool) -> None:
    trace = _Trace()
    occurred_at = NOW + timedelta(seconds=trace.limits.timeout_seconds)
    if early:
        occurred_at -= timedelta(seconds=1)
    trace.transition(
        AgentRunState.INCIDENT,
        kind=AgentRunEventKind.INCIDENT_RECORDED,
        trigger=AgentRunTrigger.TIMEOUT,
        at=occurred_at,
    )

    if early:
        with pytest.raises(AgentRunReplayError, match=r"timeout incident.*before"):
            replay_agent_run(trace.chain)
    else:
        assert replay_agent_run(trace.chain).state is AgentRunState.INCIDENT


def test_reconciliation_limit_is_independent_and_can_end_in_incident() -> None:
    limits = AgentRunLimits(
        max_tool_calls=20,
        max_invalid_tool_calls=2,
        max_reconciliation_attempts=1,
    )
    trace = _paper_reconciling(limits=limits)
    trace.reserve("reconcile_account", batch_hash=LOCKED_BATCH_HASH)
    trace.outcome(kind=AgentRunEventKind.TOOL_CALL_REJECTED)
    trace.transition(
        AgentRunState.INCIDENT,
        kind=AgentRunEventKind.INCIDENT_RECORDED,
        reason="RECONCILIATION_LIMIT_REACHED",
        at=trace.at,
    )

    snapshot = replay_agent_run(trace.chain)

    assert snapshot.state is AgentRunState.INCIDENT
    assert snapshot.reconciliation_attempt_count == 1


def test_reconciliation_budget_cannot_fund_extra_non_reconciliation_calls() -> None:
    limits = AgentRunLimits(
        max_tool_calls=2,
        max_invalid_tool_calls=1,
        max_reconciliation_attempts=5,
    )
    trace = _analyzed(_Trace(limits=limits))

    assert (
        sum(
            event.kind is AgentRunEventKind.TOOL_CALL_RESERVED
            and event.tool_name != "reconcile_account"
            for event in trace.events
        )
        == limits.max_tool_calls + 1
    )
    assert limits.max_tool_calls + 1 <= (limits.max_tool_calls + limits.max_reconciliation_attempts)
    with pytest.raises(AgentRunReplayError, match="exhausted tool budget must immediately"):
        replay_agent_run(trace.chain)


def test_failed_tool_artifact_cannot_be_promoted_to_a_trusted_prerequisite() -> None:
    trace = _Trace()
    trace.transition(AgentRunState.SNAPSHOT_READY)
    trace.reserve("get_market_snapshot")
    successful = trace.outcome(
        artifact_kind=AgentArtifactKind.MARKET_SNAPSHOT,
    )
    failed_with_artifact = successful.model_copy(
        update={
            "kind": AgentRunEventKind.TOOL_CALL_FAILED,
            "error_codes": ("FORGED_FAILED_ARTIFACT",),
        }
    )
    trace.events[-1] = failed_with_artifact
    trace.success(
        "validate_market_data",
        AgentArtifactKind.DATA_QUALITY,
        state_after=AgentRunState.DATA_VALIDATED,
    )

    with pytest.raises(AgentRunReplayError, match="content-hash revalidation"):
        replay_agent_run(trace.chain)


def test_cancellation_requires_terminal_pre_execution_but_is_observational_afterward() -> None:
    pre_execution = _Trace()
    pre_execution.transition(AgentRunState.SNAPSHOT_READY)
    pre_execution.transition(
        AgentRunState.SNAPSHOT_READY,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
    )
    with pytest.raises(AgentRunReplayError, match="illegal transition"):
        replay_agent_run(pre_execution.chain)

    cancelled = _Trace()
    cancelled.transition(AgentRunState.SNAPSHOT_READY)
    cancelled.transition(
        AgentRunState.CANCELLED,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
    )
    assert replay_agent_run(cancelled.chain).state is AgentRunState.CANCELLED

    executing = _draft_ready(_Trace(goal=AgentRunGoal.PAPER_EXECUTION, mode=RuntimeMode.PAPER))
    executing.transition(AgentRunState.EXECUTING)
    executing.reserve("submit_paper_orders", batch_hash=LOCKED_BATCH_HASH)
    executing.transition(
        AgentRunState.EXECUTING,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
    )
    executing_snapshot = replay_agent_run(executing.chain)
    assert executing_snapshot.state is AgentRunState.EXECUTING
    assert executing_snapshot.cancel_requested
    assert executing_snapshot.pending_call is not None

    reconciling = _paper_reconciling()
    reconciling.transition(
        AgentRunState.RECONCILING,
        kind=AgentRunEventKind.CANCELLATION_REQUESTED,
        trigger=AgentRunTrigger.CANCELLATION,
    )
    reconciling_snapshot = replay_agent_run(reconciling.chain)
    assert reconciling_snapshot.state is AgentRunState.RECONCILING
    assert reconciling_snapshot.cancel_requested
