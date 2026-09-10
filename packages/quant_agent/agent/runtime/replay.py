"""Pure, fail-closed replay for one explicit Agent runtime event chain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, NoReturn

from pydantic import ValidationError

from quant_agent.agent.runtime.contracts import (
    AGENT_RUNTIME_VERSION,
    TERMINAL_AGENT_RUN_STATES,
    AgentArtifactKind,
    AgentArtifactRef,
    AgentPendingToolCall,
    AgentRunEvent,
    AgentRunEventKind,
    AgentRunGoal,
    AgentRunLimits,
    AgentRunSnapshot,
    AgentRunState,
    AgentRunTrigger,
)
from quant_agent.agent.runtime.rules import (
    InvalidAgentStateTransition,
    stage_tool_names,
    validate_goal_mode,
    validate_transition,
)
from quant_agent.agent.tools.contracts import ToolEffect
from quant_agent.agent.tools.policy import V1_TOOL_POLICIES
from quant_agent.core.errors import ErrorCode


class AgentRunReplayError(ValueError):
    """Raised when an event sequence cannot prove one exact runtime state."""


@dataclass(frozen=True, slots=True)
class _PendingReservation:
    """Replay-only reservation, including calls rejected before dispatch."""

    request_id: str
    tool_name: str
    argument_hash: str
    idempotency_key_hash: str | None
    batch_hash: str | None
    started_at: datetime
    effect: ToolEffect


_TOOL_OUTCOMES: Final[frozenset[AgentRunEventKind]] = frozenset(
    {
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
    }
)
_FAILED_TOOL_OUTCOMES: Final[frozenset[AgentRunEventKind]] = frozenset(
    {
        AgentRunEventKind.TOOL_CALL_FAILED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
    }
)
_WRITE_EFFECTS: Final[frozenset[ToolEffect]] = frozenset(
    {
        ToolEffect.ARTIFACT_WRITE,
        ToolEffect.PAPER_EXECUTION_WRITE,
        ToolEffect.LIVE_EXTERNAL_WRITE,
    }
)
_SAFE_ARGUMENT_REJECTION_CODES: Final[frozenset[str]] = frozenset(
    {
        "REGISTRY_INVALID_ARGUMENTS",
        "REGISTRY_IDEMPOTENCY_KEY_REQUIRED",
    }
)
_RECOVERABLE_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        ErrorCode.INVALID_ARGUMENT.value,
        ErrorCode.NOT_FOUND.value,
        ErrorCode.CONFLICT.value,
        ErrorCode.DATA_UNAVAILABLE.value,
    }
)
_RESEARCH_ARTIFACTS: Final[frozenset[AgentArtifactKind]] = frozenset(
    {
        AgentArtifactKind.MARKET_REGIME,
        AgentArtifactKind.MARKET_THEMES,
        AgentArtifactKind.THEME_LEADERS,
        AgentArtifactKind.STOCK_CANDIDATES,
        AgentArtifactKind.CANDIDATE_EXPLANATION,
    }
)
_EXPECTED_ARTIFACT: Final[dict[str, AgentArtifactKind]] = {
    "get_market_snapshot": AgentArtifactKind.MARKET_SNAPSHOT,
    "validate_market_data": AgentArtifactKind.DATA_QUALITY,
    "detect_market_regime": AgentArtifactKind.MARKET_REGIME,
    "rank_market_themes": AgentArtifactKind.MARKET_THEMES,
    "rank_theme_leaders": AgentArtifactKind.THEME_LEADERS,
    "rank_stock_candidates": AgentArtifactKind.STOCK_CANDIDATES,
    "explain_candidate": AgentArtifactKind.CANDIDATE_EXPLANATION,
    "run_backtest": AgentArtifactKind.BACKTEST_REPORT,
    "get_portfolio_snapshot": AgentArtifactKind.PORTFOLIO_SNAPSHOT,
    "build_target_portfolio": AgentArtifactKind.TARGET_PORTFOLIO,
    "check_portfolio_risk": AgentArtifactKind.RISK_CHECK,
    "create_order_draft": AgentArtifactKind.ORDER_DRAFT,
    "get_order_draft": AgentArtifactKind.ORDER_DRAFT,
    "submit_paper_orders": AgentArtifactKind.EXECUTION_RECEIPT,
    "submit_approved_orders": AgentArtifactKind.EXECUTION_RECEIPT,
    "reconcile_account": AgentArtifactKind.RECONCILIATION,
    "generate_decision_report": AgentArtifactKind.DECISION_REPORT,
}
_PREREQUISITE_ARTIFACT: Final[dict[str, AgentArtifactKind]] = {
    "validate_market_data": AgentArtifactKind.MARKET_SNAPSHOT,
    "explain_candidate": AgentArtifactKind.STOCK_CANDIDATES,
    "build_target_portfolio": AgentArtifactKind.PORTFOLIO_SNAPSHOT,
    "check_portfolio_risk": AgentArtifactKind.TARGET_PORTFOLIO,
    "create_order_draft": AgentArtifactKind.RISK_CHECK,
    "get_order_draft": AgentArtifactKind.ORDER_DRAFT,
    "submit_paper_orders": AgentArtifactKind.ORDER_DRAFT,
    "submit_approved_orders": AgentArtifactKind.ORDER_DRAFT,
    "reconcile_account": AgentArtifactKind.EXECUTION_RECEIPT,
}
_KIND_TRIGGERS: Final[dict[AgentRunEventKind, frozenset[AgentRunTrigger]]] = {
    AgentRunEventKind.RUN_CREATED: frozenset({AgentRunTrigger.SYSTEM}),
    AgentRunEventKind.STATE_TRANSITION: frozenset({AgentRunTrigger.SYSTEM}),
    AgentRunEventKind.TOOL_CALL_RESERVED: frozenset({AgentRunTrigger.TOOL}),
    AgentRunEventKind.TOOL_CALL_SUCCEEDED: frozenset({AgentRunTrigger.TOOL}),
    AgentRunEventKind.TOOL_CALL_FAILED: frozenset({AgentRunTrigger.TOOL}),
    AgentRunEventKind.TOOL_CALL_REPLAYED: frozenset({AgentRunTrigger.TOOL}),
    AgentRunEventKind.TOOL_CALL_REJECTED: frozenset({AgentRunTrigger.TOOL}),
    AgentRunEventKind.CANCELLATION_REQUESTED: frozenset({AgentRunTrigger.CANCELLATION}),
    AgentRunEventKind.APPROVAL_RECORDED: frozenset({AgentRunTrigger.HUMAN}),
    AgentRunEventKind.APPROVAL_REJECTED: frozenset({AgentRunTrigger.HUMAN}),
    AgentRunEventKind.TIMEOUT_RECORDED: frozenset({AgentRunTrigger.TIMEOUT}),
    AgentRunEventKind.LIMIT_REACHED: frozenset({AgentRunTrigger.SYSTEM}),
    AgentRunEventKind.INCIDENT_RECORDED: frozenset(
        {
            AgentRunTrigger.SYSTEM,
            AgentRunTrigger.TIMEOUT,
            AgentRunTrigger.KILL_SWITCH,
            AgentRunTrigger.CANCELLATION,
        }
    ),
}


def _fail(message: str) -> NoReturn:
    raise AgentRunReplayError(message)


def _strict_event(value: object) -> AgentRunEvent:
    if type(value) is not AgentRunEvent:
        _fail("runtime replay accepts only exact AgentRunEvent values")
    source = value
    try:
        event = AgentRunEvent.from_json(source.to_json())
    except (TypeError, ValueError, ValidationError) as error:
        raise AgentRunReplayError(
            "runtime event failed strict content-hash revalidation"
        ) from error
    return event


def _artifact_by_kind(
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
    kind: AgentArtifactKind,
) -> tuple[AgentArtifactRef, ...]:
    return tuple(item for item in artifacts.values() if item.kind is kind)


def _has_artifact(
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
    kind: AgentArtifactKind,
) -> bool:
    return bool(_artifact_by_kind(artifacts, kind))


def _locked_draft(
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
) -> AgentArtifactRef | None:
    drafts = _artifact_by_kind(artifacts, AgentArtifactKind.ORDER_DRAFT)
    if not drafts:
        return None
    hashes = {item.content_hash for item in drafts}
    if len(hashes) != 1:
        _fail("one Agent run cannot lock multiple order-draft batches")
    return drafts[0]


def _reservation_matches(event: AgentRunEvent, pending: _PendingReservation) -> bool:
    return (
        event.request_id == pending.request_id
        and event.tool_name == pending.tool_name
        and event.tool_effect is pending.effect
        and event.argument_hash == pending.argument_hash
        and event.idempotency_key_hash == pending.idempotency_key_hash
        and event.batch_hash == pending.batch_hash
    )


def _validate_tool_effect(event: AgentRunEvent) -> None:
    if event.tool_name is None or event.tool_effect is None:
        _fail("tool event omitted its immutable name or effect")
    policy = V1_TOOL_POLICIES.get(event.tool_name)
    expected = policy.effect if policy is not None else ToolEffect.READ_ONLY
    if event.tool_effect is not expected:
        _fail("tool event effect does not match the central v1 policy")


def _tool_is_stage_valid(
    event: AgentRunEvent,
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
) -> bool:
    if event.tool_name is None:
        return False
    allowed = stage_tool_names(
        event.state_before,
        goal=event.goal,
        mode=event.runtime_mode,
    )
    if event.tool_name not in allowed:
        return False
    required = _PREREQUISITE_ARTIFACT.get(event.tool_name)
    if required is not None and not _has_artifact(artifacts, required):
        return False
    if event.tool_name == "get_market_snapshot" and _has_artifact(
        artifacts, AgentArtifactKind.MARKET_SNAPSHOT
    ):
        return False
    if event.tool_name in {
        "get_order_draft",
        "submit_paper_orders",
        "submit_approved_orders",
        "reconcile_account",
    }:
        draft = _locked_draft(artifacts)
        if draft is None or event.batch_hash != draft.content_hash:
            return False
    return True


def _validate_artifact_event(
    event: AgentRunEvent,
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
) -> None:
    artifact = event.artifact
    if artifact is None:
        return
    if event.tool_name is None:
        _fail("a runtime artifact must belong to a tool outcome")
    expected = _EXPECTED_ARTIFACT.get(event.tool_name)
    if expected is None or artifact.kind is not expected:
        _fail("tool outcome carried an artifact of the wrong kind")
    if event.kind not in {
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
    }:
        _fail("only a successful validated tool response may introduce an artifact")
    if artifact.kind is AgentArtifactKind.ORDER_DRAFT and artifact.expires_at is None:
        _fail("an order-draft artifact must retain its expiry")
    key = (artifact.kind, artifact.artifact_id)
    previous = artifacts.get(key)
    if previous is not None and previous != artifact:
        _fail("an artifact identity changed content during replay")
    if artifact.kind is AgentArtifactKind.ORDER_DRAFT:
        locked = _locked_draft(artifacts)
        if locked is not None and locked.content_hash != artifact.content_hash:
            _fail("one Agent run cannot replace its locked order-draft batch")
    artifacts.setdefault(key, artifact)


def _validate_success_artifact(
    event: AgentRunEvent,
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
) -> None:
    if event.kind not in {
        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
        AgentRunEventKind.TOOL_CALL_REPLAYED,
    }:
        return
    if event.tool_name is None:
        _fail("successful tool outcome omitted its tool identity")
    expected = _EXPECTED_ARTIFACT.get(event.tool_name)
    if expected is None:
        _fail("successful tool outcome has no reviewed artifact contract")
    if event.artifact is None and not _has_artifact(artifacts, expected):
        _fail("successful tool outcome omitted its required artifact")


def _validate_milestone(
    event: AgentRunEvent,
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
    *,
    approval_recorded: bool,
) -> None:
    state = event.state_after
    if state in TERMINAL_AGENT_RUN_STATES and state not in {
        AgentRunState.REPORTED,
        AgentRunState.COMPLETED,
    }:
        return
    required: AgentArtifactKind | None = None
    if state is AgentRunState.DATA_VALIDATED:
        required = AgentArtifactKind.DATA_QUALITY
    elif state is AgentRunState.ANALYZED:
        if event.goal is AgentRunGoal.BACKTEST_REPORT:
            required = AgentArtifactKind.BACKTEST_REPORT
        elif event.goal is not AgentRunGoal.MARKET_RESEARCH:
            required = AgentArtifactKind.STOCK_CANDIDATES
        elif not any(_has_artifact(artifacts, kind) for kind in _RESEARCH_ARTIFACTS):
            _fail("ANALYZED requires at least one validated research artifact")
    elif state is AgentRunState.PORTFOLIO_READY:
        required = AgentArtifactKind.TARGET_PORTFOLIO
    elif state is AgentRunState.RISK_CHECKED:
        required = AgentArtifactKind.RISK_CHECK
    elif state in {
        AgentRunState.DRAFT_READY,
        AgentRunState.PENDING_APPROVAL,
        AgentRunState.APPROVED,
        AgentRunState.EXECUTING,
    }:
        required = AgentArtifactKind.ORDER_DRAFT
    elif state is AgentRunState.RECONCILING:
        required = AgentArtifactKind.EXECUTION_RECEIPT
    elif state is AgentRunState.REPORTED:
        required = AgentArtifactKind.DECISION_REPORT
    elif state is AgentRunState.COMPLETED:
        required = AgentArtifactKind.RECONCILIATION
    if required is not None and not _has_artifact(artifacts, required):
        _fail(f"{state.value} is missing required artifact {required.value}")
    if (
        state
        in {
            AgentRunState.APPROVED,
            AgentRunState.EXECUTING,
            AgentRunState.RECONCILING,
            AgentRunState.COMPLETED,
        }
        and event.goal is AgentRunGoal.LIVE_ASSISTED_EXECUTION
        and not approval_recorded
    ):
        _fail("live-assisted execution reached an execution state without approval")


def _validate_control_event(
    event: AgentRunEvent,
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
    *,
    approval_deadline: datetime | None,
    approval_pending_since: datetime | None,
) -> None:
    if event.kind not in {
        AgentRunEventKind.APPROVAL_RECORDED,
        AgentRunEventKind.APPROVAL_REJECTED,
    }:
        return
    if approval_deadline is None:
        _fail("approval event appeared before an approval deadline was established")
    if event.occurred_at >= approval_deadline:
        _fail("approval control event occurred at or after its deadline")
    if event.kind is AgentRunEventKind.APPROVAL_RECORDED:
        approval = event.external_approval
        if approval is None:
            _fail("recorded approval omitted its complete trusted evidence")
        if approval_pending_since is None or approval.approved_at < approval_pending_since:
            _fail("recorded approval predates this pending-approval window")
        draft = _locked_draft(artifacts)
        if (
            draft is None
            or event.batch_hash != draft.content_hash
            or approval.batch_hash != draft.content_hash
            or approval.decision_id != event.decision_id
            or approval.approval_hash != event.control_hash
        ):
            _fail("recorded approval does not bind the locked order-draft batch")
    elif event.batch_hash is not None:
        draft = _locked_draft(artifacts)
        if draft is None or event.batch_hash != draft.content_hash:
            _fail("approval rejection carries an unrelated order-draft batch")


def _validate_deadline_event(
    event: AgentRunEvent,
    *,
    deadline: datetime,
    approval_deadline: datetime | None,
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef],
) -> None:
    draft = _locked_draft(artifacts)
    if (
        event.kind is AgentRunEventKind.STATE_TRANSITION
        and event.state_before is not AgentRunState.RECONCILING
        and event.occurred_at >= deadline
    ):
        _fail("a workflow transition occurred at or after the immutable run deadline")
    if (
        event.kind is AgentRunEventKind.CANCELLATION_REQUESTED
        and event.state_before is not AgentRunState.RECONCILING
    ):
        cancellation_deadlines = [deadline]
        if event.state_before in {
            AgentRunState.DRAFT_READY,
            AgentRunState.PENDING_APPROVAL,
            AgentRunState.APPROVED,
            AgentRunState.EXECUTING,
        }:
            if draft is not None and draft.expires_at is not None:
                cancellation_deadlines.append(draft.expires_at)
            if approval_deadline is not None:
                cancellation_deadlines.append(approval_deadline)
        if event.occurred_at >= min(cancellation_deadlines):
            _fail("cancellation cannot mask an expired runtime deadline")
    if event.kind is AgentRunEventKind.TOOL_CALL_RESERVED:
        if event.tool_name != "reconcile_account" and event.occurred_at >= deadline:
            _fail("a tool call was reserved at or after the immutable run deadline")
        if event.tool_name != "reconcile_account" and event.state_before in {
            AgentRunState.DRAFT_READY,
            AgentRunState.PENDING_APPROVAL,
            AgentRunState.APPROVED,
            AgentRunState.EXECUTING,
        }:
            authorization_expiries = [deadline]
            if draft is not None and draft.expires_at is not None:
                authorization_expiries.append(draft.expires_at)
            if approval_deadline is not None:
                authorization_expiries.append(approval_deadline)
            if event.occurred_at >= min(authorization_expiries):
                _fail("a tool call was reserved after its bound authorization expired")
        if event.tool_name in {"submit_paper_orders", "submit_approved_orders"}:
            expiries = [deadline]
            if draft is not None and draft.expires_at is not None:
                expiries.append(draft.expires_at)
            if approval_deadline is not None:
                expiries.append(approval_deadline)
            if event.occurred_at >= min(expiries):
                _fail("an execution call was reserved after its bound authorization expired")
    if (
        event.state_after is AgentRunState.EXECUTING
        and event.state_before is not AgentRunState.EXECUTING
    ):
        expiries = [deadline]
        if draft is not None and draft.expires_at is not None:
            expiries.append(draft.expires_at)
        if approval_deadline is not None:
            expiries.append(approval_deadline)
        if event.occurred_at >= min(expiries):
            _fail("execution was unlocked after its bound authorization expired")
    if (
        event.kind is AgentRunEventKind.INCIDENT_RECORDED
        and event.trigger is AgentRunTrigger.TIMEOUT
    ):
        if event.occurred_at < deadline:
            _fail("a timeout incident was recorded before the immutable run deadline")
        return
    if event.kind is not AgentRunEventKind.TIMEOUT_RECORDED and event.state_after not in {
        AgentRunState.TIMED_OUT,
        AgentRunState.EXPIRED,
    }:
        return
    if event.state_after is AgentRunState.EXPIRED:
        candidates: list[datetime] = []
        if approval_deadline is not None:
            candidates.append(approval_deadline)
        if draft is not None and draft.expires_at is not None:
            candidates.append(draft.expires_at)
        if not candidates or event.occurred_at < min(candidates):
            _fail("expiry was recorded before every bound expiry threshold")
    elif event.occurred_at < deadline:
        _fail("run timeout was recorded before the immutable run deadline")


def _validate_limit_event(
    event: AgentRunEvent,
    *,
    limits: AgentRunLimits,
    tool_call_count: int,
    invalid_tool_call_count: int,
    reconciliation_attempt_count: int,
) -> None:
    if event.kind not in {
        AgentRunEventKind.LIMIT_REACHED,
        AgentRunEventKind.TOOL_CALL_REJECTED,
    } or event.state_after not in {
        AgentRunState.LIMIT_EXCEEDED,
        AgentRunState.INCIDENT,
    }:
        return
    exhausted = invalid_tool_call_count >= limits.max_invalid_tool_calls
    if event.state_before is AgentRunState.RECONCILING:
        exhausted = exhausted or reconciliation_attempt_count >= limits.max_reconciliation_attempts
    else:
        exhausted = exhausted or tool_call_count >= limits.max_tool_calls
    if not exhausted:
        _fail("a call limit was recorded before an immutable threshold was reached")


def _expected_failed_state(event: AgentRunEvent) -> AgentRunState:
    """Mirror the service's fail-closed error-code projection during replay."""

    if event.tool_name in {
        "submit_paper_orders",
        "submit_approved_orders",
        "reconcile_account",
    }:
        return AgentRunState.INCIDENT
    codes = frozenset(event.error_codes)
    if ErrorCode.KILL_SWITCH_ACTIVE.value in codes:
        return AgentRunState.INCIDENT
    if ErrorCode.RISK_REJECTED.value in codes:
        return AgentRunState.REJECTED
    if ErrorCode.DATA_INVALID.value in codes:
        return AgentRunState.DATA_INVALID
    if ErrorCode.APPROVAL_EXPIRED.value in codes:
        return AgentRunState.FAILED
    if codes.intersection(
        {
            ErrorCode.UNAUTHORIZED.value,
            ErrorCode.FORBIDDEN.value,
        }
    ):
        return AgentRunState.REJECTED
    if ErrorCode.APPROVAL_REQUIRED.value in codes:
        return AgentRunState.REJECTED
    if codes.intersection(_RECOVERABLE_FAILURE_CODES):
        return event.state_before
    return AgentRunState.FAILED


def _replay(events: tuple[AgentRunEvent, ...]) -> AgentRunSnapshot:
    if type(events) is not tuple or not events:
        _fail("runtime replay requires one non-empty exact event tuple")
    detached = tuple(_strict_event(item) for item in events)
    first = detached[0]
    if (
        first.sequence != 0
        or first.kind is not AgentRunEventKind.RUN_CREATED
        or first.trigger is not AgentRunTrigger.SYSTEM
        or first.state_before is not AgentRunState.RECEIVED
        or first.state_after is not AgentRunState.RECEIVED
        or first.previous_event_hash is not None
        or first.run_limits is None
        or first.runtime_version != AGENT_RUNTIME_VERSION
    ):
        _fail("runtime event chain has an invalid RUN_CREATED origin")
    validate_goal_mode(first.goal, first.runtime_mode)
    limits = first.run_limits
    created_at = first.occurred_at
    deadline = created_at + timedelta(seconds=limits.timeout_seconds)

    state = AgentRunState.RECEIVED
    previous: AgentRunEvent | None = None
    pending: _PendingReservation | None = None
    pending_was_stage_valid = False
    pending_was_strict = False
    seen_requests: set[str] = set()
    write_identities: dict[str, tuple[str | None, str | None]] = {}
    artifacts: dict[tuple[AgentArtifactKind, str], AgentArtifactRef] = {}
    tool_call_count = 0
    invalid_tool_call_count = 0
    reconciliation_attempt_count = 0
    cancel_requested = False
    approval_recorded = False
    approval_deadline = None
    approval_pending_since = None
    overdue_tool_outcome_at: datetime | None = None
    expired_draft_outcome_at: datetime | None = None
    unsafe_rejection: tuple[datetime, AgentRunState] | None = None
    authorization_rejection_at: datetime | None = None
    exhausted_budget_followup: (
        tuple[
            datetime,
            AgentRunEventKind,
            AgentRunState,
        ]
        | None
    ) = None

    identity = (
        first.runtime_version,
        first.run_id,
        first.decision_id,
        first.decision_snapshot_hash,
        first.runtime_mode,
        first.goal,
    )
    for index, event in enumerate(detached):
        if (
            event.sequence != index
            or (
                event.runtime_version,
                event.run_id,
                event.decision_id,
                event.decision_snapshot_hash,
                event.runtime_mode,
                event.goal,
            )
            != identity
        ):
            _fail("runtime event identity or sequence drifted within the chain")
        if index and event.kind is AgentRunEventKind.RUN_CREATED:
            _fail("RUN_CREATED may appear only at sequence zero")
        if event.trigger not in _KIND_TRIGGERS[event.kind]:
            _fail("runtime event kind has an untrusted trigger")
        if event.state_before is not state:
            _fail("runtime event state_before does not match replayed state")
        if previous is not None and (
            event.previous_event_hash != previous.event_hash
            or event.occurred_at < previous.occurred_at
        ):
            _fail("runtime event chain is discontinuous")
        if overdue_tool_outcome_at is not None:
            if not (
                event.kind is AgentRunEventKind.TIMEOUT_RECORDED
                and event.trigger is AgentRunTrigger.TIMEOUT
                and event.state_after is AgentRunState.TIMED_OUT
                and event.occurred_at == overdue_tool_outcome_at
            ):
                _fail("an overdue tool outcome must immediately enter TIMED_OUT")
            overdue_tool_outcome_at = None
        if expired_draft_outcome_at is not None:
            if not (
                event.kind is AgentRunEventKind.TIMEOUT_RECORDED
                and event.trigger is AgentRunTrigger.TIMEOUT
                and event.state_after is AgentRunState.EXPIRED
                and event.occurred_at == expired_draft_outcome_at
            ):
                _fail("an already-expired order draft must immediately enter EXPIRED")
            expired_draft_outcome_at = None
        if unsafe_rejection is not None:
            unsafe_rejection_at, unsafe_rejection_state = unsafe_rejection
            if not (
                event.kind is AgentRunEventKind.INCIDENT_RECORDED
                and event.trigger is AgentRunTrigger.SYSTEM
                and event.state_after is AgentRunState.INCIDENT
                and event.occurred_at == unsafe_rejection_at
            ):
                if unsafe_rejection_state is AgentRunState.RECONCILING:
                    _fail("a rejected reconciliation must immediately enter INCIDENT")
                _fail("an unsafe execution-stage rejection must immediately enter INCIDENT")
            unsafe_rejection = None
        if authorization_rejection_at is not None:
            if not (
                event.kind is AgentRunEventKind.STATE_TRANSITION
                and event.trigger is AgentRunTrigger.SYSTEM
                and event.state_after is AgentRunState.REJECTED
                and event.occurred_at == authorization_rejection_at
            ):
                _fail("a registry authorization rejection must immediately enter REJECTED")
            authorization_rejection_at = None
        if exhausted_budget_followup is not None:
            exhausted_at, expected_kind, expected_state = exhausted_budget_followup
            if not (
                event.kind is expected_kind
                and event.trigger is AgentRunTrigger.SYSTEM
                and event.state_after is expected_state
                and event.occurred_at == exhausted_at
            ):
                _fail("an exhausted tool budget must immediately enter its terminal state")
            exhausted_budget_followup = None
        try:
            validate_transition(
                event.state_before,
                event.state_after,
                goal=event.goal,
                mode=event.runtime_mode,
                event_kind=event.kind,
            )
        except InvalidAgentStateTransition as error:
            raise AgentRunReplayError("runtime event contains an illegal transition") from error

        if event.kind is AgentRunEventKind.CANCELLATION_REQUESTED:
            unsafe_to_claim_cancelled = event.state_before is AgentRunState.RECONCILING or (
                pending is not None
                and pending.effect
                in {
                    ToolEffect.PAPER_EXECUTION_WRITE,
                    ToolEffect.LIVE_EXTERNAL_WRITE,
                }
            )
            expected_cancellation_state = (
                event.state_before if unsafe_to_claim_cancelled else AgentRunState.CANCELLED
            )
            if event.state_after is not expected_cancellation_state:
                _fail("cancellation state does not match the pending execution boundary")

        if event.kind is AgentRunEventKind.TOOL_CALL_RESERVED or event.kind in _TOOL_OUTCOMES:
            _validate_tool_effect(event)
        if pending is not None and event.kind not in _TOOL_OUTCOMES:
            if not pending_was_stage_valid or not pending_was_strict:
                _fail("an invalid reservation must be immediately rejected")
            if event.kind is AgentRunEventKind.TIMEOUT_RECORDED and pending.effect in {
                ToolEffect.PAPER_EXECUTION_WRITE,
                ToolEffect.LIVE_EXTERNAL_WRITE,
            }:
                _fail("an execution write with an unknown outcome must enter INCIDENT")
            terminal_interruption = (
                event.state_after in TERMINAL_AGENT_RUN_STATES
                and event.kind
                in {
                    AgentRunEventKind.CANCELLATION_REQUESTED,
                    AgentRunEventKind.TIMEOUT_RECORDED,
                    AgentRunEventKind.INCIDENT_RECORDED,
                }
            )
            cancellation_observation = (
                event.kind is AgentRunEventKind.CANCELLATION_REQUESTED
                and event.state_after is event.state_before
            )
            if not terminal_interruption and not cancellation_observation:
                _fail("a reserved tool call must be resolved before another runtime event")
        if event.kind is AgentRunEventKind.TOOL_CALL_RESERVED:
            if invalid_tool_call_count >= limits.max_invalid_tool_calls:
                _fail("a tool call was reserved after the invalid-call budget was exhausted")
            if state is AgentRunState.RECONCILING:
                if reconciliation_attempt_count >= limits.max_reconciliation_attempts:
                    _fail("a reconciliation was reserved after its independent budget expired")
            elif tool_call_count >= limits.max_tool_calls:
                _fail("a tool call was reserved after the model-call budget was exhausted")
            if pending is not None or event.request_id in seen_requests:
                _fail("tool reservation is duplicate or overlaps a pending call")
            if (
                event.request_id is None
                or event.tool_name is None
                or event.tool_effect is None
                or event.argument_hash is None
            ):
                _fail("tool reservation omitted its immutable identity")
            request_id = event.request_id
            tool_name = event.tool_name
            if cancel_requested and event.tool_effect in {
                ToolEffect.PAPER_EXECUTION_WRITE,
                ToolEffect.LIVE_EXTERNAL_WRITE,
            }:
                _fail("a cancellation request permanently closes execution submission")
            pending = _PendingReservation(
                request_id=request_id,
                tool_name=tool_name,
                argument_hash=event.argument_hash,
                idempotency_key_hash=event.idempotency_key_hash,
                batch_hash=event.batch_hash,
                started_at=event.occurred_at,
                effect=event.tool_effect,
            )
            pending_was_stage_valid = _tool_is_stage_valid(event, artifacts)
            try:
                AgentPendingToolCall(
                    request_id=pending.request_id,
                    tool_name=pending.tool_name,
                    argument_hash=pending.argument_hash,
                    idempotency_key_hash=pending.idempotency_key_hash,
                    batch_hash=pending.batch_hash,
                    started_at=pending.started_at,
                    effect=pending.effect,
                )
                pending_was_strict = True
            except (TypeError, ValueError, ValidationError):
                pending_was_strict = False
            seen_requests.add(request_id)
            tool_call_count += 1
            if tool_name == "reconcile_account":
                reconciliation_attempt_count += 1
        elif event.kind in _TOOL_OUTCOMES:
            if pending is None or not _reservation_matches(event, pending):
                _fail("tool outcome does not resolve the exact pending reservation")
            reservation_is_strict = pending_was_strict
            if not reservation_is_strict and event.kind is not AgentRunEventKind.TOOL_CALL_REJECTED:
                _fail("a dispatched tool outcome resolved an invalid reservation")
            if (
                event.kind is not AgentRunEventKind.TOOL_CALL_REJECTED
                and pending.effect in _WRITE_EFFECTS
            ):
                identity_pair = (
                    pending.idempotency_key_hash,
                    pending.batch_hash,
                )
                previous_identity = write_identities.setdefault(
                    pending.tool_name,
                    identity_pair,
                )
                if previous_identity != identity_pair:
                    _fail("a write stage changed its locked idempotency or batch identity")
            if (
                not pending_was_stage_valid
                and event.kind is not AgentRunEventKind.TOOL_CALL_REJECTED
            ):
                _fail("an out-of-stage tool reservation produced a non-rejection outcome")
            if event.kind in _FAILED_TOOL_OUTCOMES and event.state_after not in {
                event.state_before,
                AgentRunState.REJECTED,
                AgentRunState.DATA_INVALID,
                AgentRunState.EXPIRED,
                AgentRunState.LIMIT_EXCEEDED,
                AgentRunState.FAILED,
                AgentRunState.INCIDENT,
            }:
                _fail("a failed tool outcome advanced a successful workflow milestone")
            if (
                event.kind is AgentRunEventKind.TOOL_CALL_REJECTED
                and event.state_after is not event.state_before
            ):
                _fail("a rejected tool outcome must be recorded before its terminal transition")
            if pending.tool_name in {"submit_paper_orders", "submit_approved_orders"}:
                if (
                    event.kind
                    in {
                        AgentRunEventKind.TOOL_CALL_SUCCEEDED,
                        AgentRunEventKind.TOOL_CALL_REPLAYED,
                    }
                    and event.state_after is not AgentRunState.RECONCILING
                ):
                    _fail("a successful execution must immediately enter RECONCILING")
                if (
                    event.kind is AgentRunEventKind.TOOL_CALL_FAILED
                    and event.state_after is not AgentRunState.INCIDENT
                ):
                    _fail("an uncertain execution failure must immediately enter INCIDENT")
            if pending.tool_name == "reconcile_account":
                if event.kind in {
                    AgentRunEventKind.TOOL_CALL_SUCCEEDED,
                    AgentRunEventKind.TOOL_CALL_REPLAYED,
                } and event.state_after not in {
                    AgentRunState.COMPLETED,
                    AgentRunState.INCIDENT,
                }:
                    _fail("a successful reconciliation must close the execution outcome")
                if (
                    event.kind is AgentRunEventKind.TOOL_CALL_FAILED
                    and event.state_after is not AgentRunState.INCIDENT
                ):
                    _fail("a failed reconciliation must immediately enter INCIDENT")
            if (
                event.kind is AgentRunEventKind.TOOL_CALL_FAILED
                and event.occurred_at < deadline
                and event.state_after is not _expected_failed_state(event)
            ):
                _fail("a failed tool outcome does not match its fail-closed error projection")
            if event.kind is AgentRunEventKind.TOOL_CALL_REJECTED:
                invalid_tool_call_count += 1
                safe_argument_rejection = (
                    not cancel_requested
                    and reservation_is_strict
                    and pending_was_stage_valid
                    and len(event.error_codes) == 1
                    and event.error_codes[0] in _SAFE_ARGUMENT_REJECTION_CODES
                )
                safe_pre_execution_rejection = (
                    event.error_codes == ("RUNTIME_STAGE_REJECTED",) or safe_argument_rejection
                )
                budget_exhausted = invalid_tool_call_count >= limits.max_invalid_tool_calls or (
                    event.state_before is not AgentRunState.RECONCILING
                    and tool_call_count >= limits.max_tool_calls
                )
                if event.state_before in {
                    AgentRunState.EXECUTING,
                    AgentRunState.RECONCILING,
                } and not (safe_argument_rejection or budget_exhausted):
                    unsafe_rejection = (event.occurred_at, event.state_before)
                elif (
                    event.state_before
                    not in {
                        AgentRunState.EXECUTING,
                        AgentRunState.RECONCILING,
                    }
                    and not safe_pre_execution_rejection
                    and not budget_exhausted
                    and event.occurred_at < deadline
                ):
                    authorization_rejection_at = event.occurred_at
            normal_outcome_overdue = (
                pending.tool_name
                not in {
                    "submit_paper_orders",
                    "submit_approved_orders",
                    "reconcile_account",
                }
                and event.occurred_at >= deadline
            )
            if (
                event.state_after not in TERMINAL_AGENT_RUN_STATES
                and event.state_after is not AgentRunState.RECONCILING
                and not normal_outcome_overdue
                and (
                    invalid_tool_call_count >= limits.max_invalid_tool_calls
                    or tool_call_count >= limits.max_tool_calls
                )
            ):
                after_execution = event.state_before in {
                    AgentRunState.EXECUTING,
                    AgentRunState.RECONCILING,
                }
                exhausted_budget_followup = (
                    event.occurred_at,
                    (
                        AgentRunEventKind.INCIDENT_RECORDED
                        if after_execution
                        else AgentRunEventKind.LIMIT_REACHED
                    ),
                    (AgentRunState.INCIDENT if after_execution else AgentRunState.LIMIT_EXCEEDED),
                )
            _validate_artifact_event(event, artifacts)
            _validate_success_artifact(event, artifacts)
            if (
                event.artifact is not None
                and event.artifact.kind is AgentArtifactKind.ORDER_DRAFT
                and event.artifact.expires_at is not None
                and event.artifact.expires_at <= event.occurred_at
                and event.state_after not in TERMINAL_AGENT_RUN_STATES
                and overdue_tool_outcome_at is None
                and exhausted_budget_followup is None
            ):
                expired_draft_outcome_at = event.occurred_at
            pending = None
            pending_was_stage_valid = False
            pending_was_strict = False
        elif event.artifact is not None:
            _fail("non-tool event introduced an artifact")

        if (
            event.kind in _TOOL_OUTCOMES
            and event.tool_name
            not in {
                "submit_paper_orders",
                "submit_approved_orders",
                "reconcile_account",
            }
            and event.occurred_at >= deadline
        ):
            if event.state_after is not event.state_before:
                _fail("an overdue tool outcome advanced instead of timing out")
            overdue_tool_outcome_at = event.occurred_at

        if (
            pending is not None
            and event.kind
            in {
                AgentRunEventKind.CANCELLATION_REQUESTED,
                AgentRunEventKind.TIMEOUT_RECORDED,
                AgentRunEventKind.INCIDENT_RECORDED,
            }
            and event.state_after in TERMINAL_AGENT_RUN_STATES
        ):
            pending = None
            pending_was_stage_valid = False
            pending_was_strict = False

        if event.kind is AgentRunEventKind.CANCELLATION_REQUESTED:
            cancel_requested = True
        if event.kind in {
            AgentRunEventKind.APPROVAL_RECORDED,
            AgentRunEventKind.APPROVAL_REJECTED,
        }:
            _validate_control_event(
                event,
                artifacts,
                approval_deadline=approval_deadline,
                approval_pending_since=approval_pending_since,
            )
            approval_recorded = event.kind is AgentRunEventKind.APPROVAL_RECORDED
            if approval_recorded:
                approval = event.external_approval
                if approval is None:  # pragma: no cover - strict event invariant
                    _fail("recorded approval omitted its complete trusted evidence")
                if approval_deadline is None:  # pragma: no cover - validated above
                    _fail("recorded approval has no runtime deadline")
                approval_deadline = min(approval_deadline, approval.expires_at)

        if (
            event.state_after is AgentRunState.PENDING_APPROVAL
            and event.state_before is not AgentRunState.PENDING_APPROVAL
        ):
            draft = _locked_draft(artifacts)
            if draft is None or draft.expires_at is None:
                _fail("PENDING_APPROVAL requires an expiring locked order draft")
            approval_deadline = min(
                draft.expires_at,
                deadline,
                event.occurred_at + timedelta(seconds=limits.approval_timeout_seconds),
            )
            if approval_deadline <= event.occurred_at:
                _fail("PENDING_APPROVAL was entered after its approval window expired")
            approval_pending_since = event.occurred_at

        _validate_deadline_event(
            event,
            deadline=deadline,
            approval_deadline=approval_deadline,
            artifacts=artifacts,
        )
        _validate_limit_event(
            event,
            limits=limits,
            tool_call_count=tool_call_count,
            invalid_tool_call_count=invalid_tool_call_count,
            reconciliation_attempt_count=reconciliation_attempt_count,
        )
        if tool_call_count - reconciliation_attempt_count > limits.max_tool_calls:
            _fail("replayed non-reconciliation call count exceeds the immutable run budget")
        if invalid_tool_call_count > limits.max_invalid_tool_calls:
            _fail("replayed invalid-call count exceeds the immutable run budget")
        if reconciliation_attempt_count > limits.max_reconciliation_attempts:
            _fail("replayed reconciliation count exceeds the immutable run budget")

        _validate_milestone(
            event,
            artifacts,
            approval_recorded=approval_recorded,
        )
        state = event.state_after
        previous = event

    if pending is not None and not pending_was_stage_valid:
        _fail("an out-of-stage reservation cannot remain pending")
    if overdue_tool_outcome_at is not None:
        _fail("an overdue tool outcome was not closed by a timeout event")
    if expired_draft_outcome_at is not None:
        _fail("an already-expired order draft was not closed by an expiry event")
    if unsafe_rejection is not None:
        if unsafe_rejection[1] is AgentRunState.RECONCILING:
            _fail("a rejected reconciliation was not closed by an incident event")
        _fail("an unsafe execution-stage rejection was not closed by an incident event")
    if authorization_rejection_at is not None:
        _fail("a registry authorization rejection was not closed by REJECTED")
    if exhausted_budget_followup is not None:
        _fail("an exhausted tool budget was not closed by a terminal event")

    pending_call: AgentPendingToolCall | None = None
    if pending is not None:
        try:
            pending_call = AgentPendingToolCall(
                request_id=pending.request_id,
                tool_name=pending.tool_name,
                argument_hash=pending.argument_hash,
                idempotency_key_hash=pending.idempotency_key_hash,
                batch_hash=pending.batch_hash,
                started_at=pending.started_at,
                effect=pending.effect,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise AgentRunReplayError(
                "pending reservation cannot produce a valid pending tool call"
            ) from error
        locked_identity = write_identities.get(pending.tool_name)
        if locked_identity is not None and locked_identity != (
            pending.idempotency_key_hash,
            pending.batch_hash,
        ):
            _fail("a pending write changed its locked idempotency or batch identity")

    final = detached[-1]
    terminal = state in TERMINAL_AGENT_RUN_STATES
    try:
        snapshot = AgentRunSnapshot.model_validate(
            {
                "runtime_version": first.runtime_version,
                "run_id": first.run_id,
                "decision_id": first.decision_id,
                "decision_snapshot_hash": first.decision_snapshot_hash,
                "runtime_mode": first.runtime_mode,
                "goal": first.goal,
                "state": state,
                "created_at": created_at,
                "updated_at": final.occurred_at,
                "deadline": deadline,
                "approval_deadline": approval_deadline,
                "completed_at": final.occurred_at if terminal else None,
                "terminal_reason": final.reason if terminal else None,
                "limits": limits,
                "revision": final.sequence,
                "tool_call_count": tool_call_count,
                "invalid_tool_call_count": invalid_tool_call_count,
                "reconciliation_attempt_count": reconciliation_attempt_count,
                "cancel_requested": cancel_requested,
                "artifacts": tuple(artifacts.values()),
                "pending_call": pending_call,
                "events": detached,
            },
            strict=True,
        )
        return AgentRunSnapshot.from_json(snapshot.to_json())
    except (TypeError, ValueError, ValidationError) as error:
        raise AgentRunReplayError(
            "replayed event chain cannot produce a valid AgentRunSnapshot"
        ) from error


def replay_agent_run(events: tuple[AgentRunEvent, ...]) -> AgentRunSnapshot:
    """Fold a complete v1 event chain without consulting tools or external state."""

    try:
        return _replay(events)
    except AgentRunReplayError:
        raise
    except Exception as error:
        raise AgentRunReplayError("runtime replay failed closed") from error


__all__ = ["AgentRunReplayError", "replay_agent_run"]
