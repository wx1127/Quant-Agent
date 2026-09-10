"""State-gated facade over the centrally authorized Agent tool registry.

The runtime is deliberately the only object intended for model-facing adapters.
It narrows the registry catalog by workflow state, records a reservation before
dispatch, derives state changes from validated tool responses, and never accepts
an arbitrary next-state value from the model.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Protocol, cast
from uuid import uuid4

from quant_agent.agent.snapshots.contracts import DecisionSnapshot
from quant_agent.agent.tools.contracts import (
    ToolArgumentsRejected,
    ToolAuditUnavailable,
    ToolCallContext,
    ToolDescriptor,
    ToolEffect,
    ToolExecutionFailed,
    ToolInvocationDenied,
    ToolNotRegistered,
    ToolOutput,
    ToolOutputRejected,
    ToolPermissionDenied,
    validate_tool_name,
)
from quant_agent.agent.tools.policy import V1_TOOL_POLICIES
from quant_agent.agent.tools.portfolio_execution.contracts import (
    OrderDraftOutput,
    PaperExecutionOutput,
    PortfolioRiskOutput,
    PortfolioSnapshotOutput,
    ReconciliationOutput,
    TargetPortfolioOutput,
)
from quant_agent.agent.tools.research.contracts import (
    CandidateExplanationOutput,
    CandidateRankingOutput,
    LeaderRankingOutput,
    MarketDataQualityOutput,
    MarketRegimeOutput,
    MarketSnapshotOutput,
    ThemeRankingOutput,
)
from quant_agent.config import RuntimeMode
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import ToolResponse
from quant_agent.core.time import ensure_aware
from quant_agent.reconciliation import ReconciliationStatus
from quant_agent.risk import RiskCheckStatus

from .contracts import (
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
from .repository import AgentRunRepository
from .rules import report_ready, stage_tool_names, validate_goal_mode, validate_transition

_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_EXECUTION_TOOLS = frozenset({"submit_paper_orders", "submit_approved_orders"})
_WRITE_EFFECTS = frozenset(
    {
        ToolEffect.ARTIFACT_WRITE,
        ToolEffect.PAPER_EXECUTION_WRITE,
        ToolEffect.LIVE_EXTERNAL_WRITE,
    }
)
_ANALYSIS_TOOLS = frozenset(
    {
        "detect_market_regime",
        "rank_market_themes",
        "rank_theme_leaders",
        "rank_stock_candidates",
        "explain_candidate",
    }
)


class RuntimeToolRegistry(Protocol):
    """Narrow interface implemented by :class:`AgentToolRegistry`."""

    @property
    def runtime_mode(self) -> RuntimeMode: ...

    def catalog(self, context: ToolCallContext) -> tuple[ToolDescriptor, ...]: ...

    def invoke(
        self,
        tool_name: str,
        raw_arguments: str | dict[str, object],
        context: ToolCallContext,
    ) -> object: ...


class ReconciliationIncidentHandler(Protocol):
    """Trusted adapter that resolves the full reconciliation before activation.

    A detached or truncated model-visible stop signal is not sufficient to
    activate a kill switch.  Implementations must retrieve the authoritative
    reconciliation by identity, verify ``result_hash``, and invoke the existing
    kill-switch service idempotently.
    """

    def handle_verified_reconciliation(
        self,
        *,
        decision_id: str,
        reconciliation_id: str,
        result_hash: str,
        requested_at: datetime,
    ) -> str: ...


class AgentRuntimeError(RuntimeError):
    """Base class for safe runtime failures."""


class AgentRunTerminal(AgentRuntimeError):
    """Raised when a non-replay call targets an absorbing terminal state."""


class AgentToolStateDenied(AgentRuntimeError):
    """Raised before the registry when a tool is unavailable in this stage."""

    def __init__(self, *, tool_name: str, state: AgentRunState, reason: str) -> None:
        super().__init__(f"tool invocation rejected in {state.value}: {reason}")
        self.tool_name = tool_name
        self.state = state
        self.reason = reason


class AgentToolCallLimitExceeded(AgentRuntimeError):
    """Raised after the server-owned model or reconciliation budget is exhausted."""


class AgentRequestConflict(AgentRuntimeError):
    """Raised when a request identity is reused with a different call."""


class AgentRuntimeOutputInvalid(AgentRuntimeError):
    """Raised when an otherwise authorized response breaks cross-tool bindings."""


class AgentApprovalInvalid(AgentRuntimeError):
    """Raised when trusted approval evidence does not bind the pending draft."""


class AgentRuntimeClockError(AgentRuntimeError):
    """Raised when the injected wall clock is naive or moves backwards."""


class AgentRuntimeRepositoryConflict(AgentRuntimeError):
    """Raised when an internal runtime save loses an unexpected CAS race."""


def _utc(value: datetime) -> datetime:
    try:
        return ensure_aware(value).astimezone(UTC)
    except (TypeError, ValueError) as error:
        raise AgentRuntimeClockError("runtime clock must return an aware datetime") from error


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_text(value: str) -> str:
    return _hash_bytes(value.encode("utf-8"))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object field")
        result[key] = value
    return result


def _argument_identity(
    raw_arguments: str | dict[str, object],
) -> tuple[str, dict[str, object] | None]:
    """Return a secret-safe digest plus a parsed object when parsing is safe."""

    try:
        if type(raw_arguments) is str:
            parsed = json.loads(raw_arguments, object_pairs_hook=_unique_object)
            canonical = _canonical_json(parsed)
        elif type(raw_arguments) is dict:
            canonical = _canonical_json(raw_arguments)
            parsed = json.loads(canonical, object_pairs_hook=_unique_object)
        else:
            raise TypeError("arguments must be an exact string or dict")
        if type(parsed) is not dict:
            return _hash_text(canonical), None
        return _hash_text(canonical), cast(dict[str, object], parsed)
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeError):
        if type(raw_arguments) is str:
            material = raw_arguments.encode("utf-8", errors="replace")
        else:
            material = f"unsupported:{type(raw_arguments).__name__}".encode()
        return _hash_bytes(material), None


def _response_hash(response: ToolResponse[object]) -> str:
    rendered = _canonical_json(response.model_dump(mode="json"))
    return _hash_text(rendered)


def _codes(response: ToolResponse[object], *, errors: bool) -> tuple[str, ...]:
    issues = response.errors if errors else response.warnings
    return tuple(sorted({item.code.value for item in issues}))


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _artifact(
    *,
    kind: AgentArtifactKind,
    artifact_id: str,
    content_hash: str,
    tool_name: str,
    response_hash: str,
    data: ToolOutput,
    expires_at: datetime | None = None,
) -> AgentArtifactRef:
    truncated = any(
        type(getattr(data, field_name, None)) is bool
        and getattr(data, field_name)
        and field_name.endswith("_truncated")
        for field_name in type(data).model_fields
    )
    return AgentArtifactRef(
        kind=kind,
        artifact_id=artifact_id,
        content_hash=content_hash,
        tool_name=tool_name,
        response_hash=response_hash,
        truncated=truncated,
        expires_at=expires_at,
    )


def _find_artifact(
    snapshot: AgentRunSnapshot,
    kind: AgentArtifactKind,
) -> AgentArtifactRef | None:
    values = tuple(item for item in snapshot.artifacts if item.kind is kind)
    return values[-1] if values else None


def _tool_effect(tool_name: str) -> ToolEffect:
    policy = V1_TOOL_POLICIES.get(tool_name)
    return policy.effect if policy is not None else ToolEffect.READ_ONLY


def _safe_tool_name(value: str) -> str:
    try:
        return validate_tool_name(value)
    except (TypeError, ValueError):
        return "invalid_tool_name"


class AgentRuntime:
    """Serialize one run at a time and enforce the explicit Agent workflow."""

    def __init__(
        self,
        *,
        registry: RuntimeToolRegistry,
        repository: AgentRunRepository,
        clock: Callable[[], datetime] | None = None,
        incident_handler: ReconciliationIncidentHandler | None = None,
    ) -> None:
        if not isinstance(registry.runtime_mode, RuntimeMode):
            raise TypeError("registry runtime mode must be a RuntimeMode")
        self._registry = registry
        self._repository = repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._incident_handler = incident_handler
        self._locks_guard = RLock()
        self._locks: dict[str, RLock] = {}
        self._decisions: dict[str, DecisionSnapshot] = {}
        self._responses: dict[tuple[str, str], tuple[str, str, ToolResponse[object]]] = {}
        self._handled_incidents: set[tuple[str, str]] = set()

    def start_run(
        self,
        *,
        decision_snapshot: DecisionSnapshot,
        goal: AgentRunGoal,
        limits: AgentRunLimits | None = None,
        run_id: str | None = None,
    ) -> AgentRunSnapshot:
        """Start and bind a run to one exact immutable decision snapshot."""

        if type(decision_snapshot) is not DecisionSnapshot:
            raise TypeError("decision_snapshot must be an exact DecisionSnapshot")
        decision = DecisionSnapshot.from_json(decision_snapshot.to_json())
        if type(goal) is not AgentRunGoal:
            raise TypeError("goal must be an exact AgentRunGoal")
        validate_goal_mode(goal, decision.mode)
        if decision.mode is not self._registry.runtime_mode:
            raise ValueError("decision mode does not match the fixed registry mode")
        selected_run_id = run_id or f"run_{uuid4().hex}"
        now = self._now()
        if now < decision.as_of:
            raise AgentRuntimeClockError("run cannot start before the decision as_of")
        started = AgentRunSnapshot.start(
            run_id=selected_run_id,
            decision_id=decision.decision_id,
            decision_snapshot_hash=decision.content_hash,
            runtime_mode=decision.mode,
            goal=goal,
            created_at=now,
            limits=limits,
        )
        bound = self._append_event(
            started,
            kind=AgentRunEventKind.STATE_TRANSITION,
            trigger=AgentRunTrigger.SYSTEM,
            state_after=AgentRunState.SNAPSHOT_READY,
            reason="DECISION_SNAPSHOT_BOUND",
            occurred_at=now,
        )
        stored = self._repository.create(bound)
        with self._locks_guard:
            previous = self._decisions.get(stored.run_id)
            if previous is not None and previous != decision:
                raise AgentRequestConflict("run identity is already bound to another decision")
            self._decisions[stored.run_id] = decision
        return stored

    def get_run(self, run_id: str) -> AgentRunSnapshot:
        """Return the current run, applying cooperative deadline transitions."""

        with self._lock_for(run_id):
            snapshot = self._repository.get(run_id)
            return self._apply_time_gate(snapshot, self._now())

    def catalog(self, run_id: str) -> tuple[ToolDescriptor, ...]:
        """Return registry-authorized tools further narrowed by runtime state."""

        with self._lock_for(run_id):
            now = self._now()
            snapshot = self._apply_time_gate(self._repository.get(run_id), now)
            if snapshot.state in TERMINAL_AGENT_RUN_STATES or snapshot.pending_call is not None:
                return ()
            if self._model_budget_exhausted(snapshot):
                return ()
            allowed = self._available_tool_names(snapshot)
            context = ToolCallContext(
                request_id=f"catalog_{snapshot.run_id}_{snapshot.revision}",
                decision_id=snapshot.decision_id,
                requested_at=max(now, snapshot.created_at),
            )
            return tuple(
                descriptor
                for descriptor in self._registry.catalog(context)
                if descriptor.name in allowed
            )

    def invoke_tool(
        self,
        run_id: str,
        tool_name: str,
        raw_arguments: str | dict[str, object],
        *,
        request_id: str | None = None,
    ) -> ToolResponse[object]:
        """Reserve, authorize, invoke, validate, and record one Agent tool call."""

        selected_request_id = request_id or f"tool_{uuid4().hex}"
        safe_name = _safe_tool_name(tool_name)
        argument_hash, parsed_arguments = _argument_identity(raw_arguments)
        signature = _hash_text(f"{safe_name}:{argument_hash}")

        with self._lock_for(run_id):
            replay = self._responses.get((run_id, selected_request_id))
            if replay is not None:
                previous_tool, previous_signature, response = replay
                if previous_tool != safe_name or previous_signature != signature:
                    raise AgentRequestConflict(
                        "request_id is already bound to a different tool call"
                    )
                return response

            now = self._now()
            snapshot = self._apply_time_gate(self._repository.get(run_id), now)
            if self._request_seen(snapshot, selected_request_id):
                raise AgentRequestConflict(
                    "request_id already exists but its response is unavailable for replay"
                )
            if snapshot.state in TERMINAL_AGENT_RUN_STATES:
                raise AgentRunTerminal(
                    f"run is terminal: {snapshot.state.value}/{snapshot.terminal_reason}"
                )
            self._ensure_budget(snapshot, occurred_at=now)

            effect = _tool_effect(safe_name)
            denial = self._tool_denial_reason(
                snapshot,
                safe_name,
                parsed_arguments,
            )
            if denial is not None:
                if snapshot.cancel_requested and effect in {
                    ToolEffect.PAPER_EXECUTION_WRITE,
                    ToolEffect.LIVE_EXTERNAL_WRITE,
                }:
                    incident = self._append_event(
                        snapshot,
                        kind=AgentRunEventKind.INCIDENT_RECORDED,
                        trigger=AgentRunTrigger.SYSTEM,
                        state_after=AgentRunState.INCIDENT,
                        reason="EXECUTION_ATTEMPT_AFTER_CANCELLATION",
                        occurred_at=now,
                    )
                    updated = self._save(snapshot, incident)
                    raise AgentToolStateDenied(
                        tool_name=safe_name,
                        state=updated.state,
                        reason=denial,
                    )
                updated = self._record_pre_dispatch_rejection(
                    snapshot,
                    request_id=selected_request_id,
                    tool_name=safe_name,
                    effect=effect,
                    argument_hash=argument_hash,
                    parsed_arguments=parsed_arguments,
                    reason=denial,
                    occurred_at=now,
                )
                if updated.state is AgentRunState.LIMIT_EXCEEDED:
                    raise AgentToolCallLimitExceeded(denial)
                raise AgentToolStateDenied(
                    tool_name=safe_name,
                    state=updated.state,
                    reason=denial,
                )

            idempotency_hash = self._idempotency_hash(parsed_arguments)
            batch_hash = self._batch_hash(parsed_arguments)
            self._verify_locked_write_identity(
                snapshot,
                tool_name=safe_name,
                effect=effect,
                idempotency_key_hash=idempotency_hash,
                batch_hash=batch_hash,
            )
            pending = AgentPendingToolCall(
                request_id=selected_request_id,
                tool_name=safe_name,
                argument_hash=argument_hash,
                idempotency_key_hash=idempotency_hash,
                batch_hash=batch_hash,
                started_at=now,
                effect=effect,
            )
            reserved = self._append_event(
                snapshot,
                kind=AgentRunEventKind.TOOL_CALL_RESERVED,
                trigger=AgentRunTrigger.TOOL,
                state_after=snapshot.state,
                reason="TOOL_CALL_RESERVED",
                occurred_at=now,
                request_id=selected_request_id,
                tool_name=safe_name,
                tool_effect=effect,
                argument_hash=argument_hash,
                idempotency_key_hash=idempotency_hash,
                batch_hash=batch_hash,
                pending_call=pending,
                tool_call_count=snapshot.tool_call_count + 1,
                reconciliation_attempt_count=(
                    snapshot.reconciliation_attempt_count
                    + (1 if safe_name == "reconcile_account" else 0)
                ),
            )
            reserved = self._save(snapshot, reserved)
            context = ToolCallContext(
                request_id=selected_request_id,
                decision_id=reserved.decision_id,
                requested_at=now,
            )
            try:
                raw_response = self._registry.invoke(safe_name, raw_arguments, context)
            except Exception as error:
                self._record_registry_exception(reserved, error)
                raise
            if not isinstance(raw_response, ToolResponse):
                runtime_error = AgentRuntimeOutputInvalid(
                    "registry returned a non-ToolResponse value"
                )
                self._record_registry_exception(reserved, runtime_error)
                raise runtime_error
            response = cast(ToolResponse[object], raw_response)
            self._record_response(reserved, response)
            self._responses[(run_id, selected_request_id)] = (
                safe_name,
                signature,
                response,
            )
            return response

    def request_cancel(self, run_id: str) -> AgentRunSnapshot:
        """Cooperatively cancel before execution, or only flag an active execution."""

        with self._lock_for(run_id):
            now = self._now()
            snapshot = self._apply_time_gate(self._repository.get(run_id), now)
            if snapshot.cancel_requested or snapshot.state in TERMINAL_AGENT_RUN_STATES:
                return snapshot
            unsafe_to_claim_cancelled = snapshot.state is AgentRunState.RECONCILING or (
                snapshot.pending_call is not None
                and snapshot.pending_call.effect
                in {
                    ToolEffect.PAPER_EXECUTION_WRITE,
                    ToolEffect.LIVE_EXTERNAL_WRITE,
                }
            )
            target = snapshot.state if unsafe_to_claim_cancelled else AgentRunState.CANCELLED
            updated = self._append_event(
                snapshot,
                kind=AgentRunEventKind.CANCELLATION_REQUESTED,
                trigger=AgentRunTrigger.CANCELLATION,
                state_after=target,
                reason=(
                    "CANCEL_REQUESTED_RECONCILIATION_REQUIRED"
                    if unsafe_to_claim_cancelled
                    else "CANCELLED_BEFORE_EXECUTION"
                ),
                occurred_at=now,
                cancel_requested=True,
            )
            return self._save(snapshot, updated)

    def record_external_approval(
        self,
        run_id: str,
        approval: AgentExternalApproval,
    ) -> AgentRunSnapshot:
        """Accept trusted, hash-bound approval and expose the live submit stage."""

        if type(approval) is not AgentExternalApproval:
            raise TypeError("approval must be an exact AgentExternalApproval")
        approval = AgentExternalApproval.model_validate_json(approval.model_dump_json())
        with self._lock_for(run_id):
            now = self._now()
            snapshot = self._apply_time_gate(self._repository.get(run_id), now)
            if snapshot.state in TERMINAL_AGENT_RUN_STATES:
                raise AgentRunTerminal(f"run is terminal: {snapshot.state.value}")
            draft = _find_artifact(snapshot, AgentArtifactKind.ORDER_DRAFT)
            pending_since = next(
                (
                    event.occurred_at
                    for event in reversed(snapshot.events)
                    if event.state_before is not AgentRunState.PENDING_APPROVAL
                    and event.state_after is AgentRunState.PENDING_APPROVAL
                ),
                None,
            )
            if (
                snapshot.state is not AgentRunState.PENDING_APPROVAL
                or draft is None
                or snapshot.approval_deadline is None
                or pending_since is None
            ):
                raise AgentApprovalInvalid("run is not waiting for approval of a locked draft")
            if (
                approval.decision_id != snapshot.decision_id
                or approval.batch_hash != draft.content_hash
                or approval.approved_at < pending_since
                or approval.approved_at > now
                or approval.expires_at <= now
                or (snapshot.approval_deadline is not None and now >= snapshot.approval_deadline)
            ):
                raise AgentApprovalInvalid("approval does not bind the pending decision and draft")
            effective_deadline = min(snapshot.approval_deadline, approval.expires_at)
            approved = self._append_event(
                snapshot,
                kind=AgentRunEventKind.APPROVAL_RECORDED,
                trigger=AgentRunTrigger.HUMAN,
                state_after=AgentRunState.APPROVED,
                reason="EXTERNAL_APPROVAL_VERIFIED",
                occurred_at=now,
                batch_hash=approval.batch_hash,
                control_hash=approval.approval_hash,
                external_approval=approval,
                approval_deadline=effective_deadline,
            )
            executing = self._append_event(
                approved,
                kind=AgentRunEventKind.STATE_TRANSITION,
                trigger=AgentRunTrigger.SYSTEM,
                state_after=AgentRunState.EXECUTING,
                reason="LIVE_EXECUTION_READY",
                occurred_at=now,
            )
            return self._save(snapshot, executing)

    def reject_external_approval(
        self,
        run_id: str,
        *,
        rejection_hash: str,
    ) -> AgentRunSnapshot:
        """Record one trusted human rejection without storing free-form text."""

        if not _is_sha256(rejection_hash):
            raise ValueError("rejection_hash must be a lowercase SHA-256 digest")
        with self._lock_for(run_id):
            now = self._now()
            snapshot = self._apply_time_gate(self._repository.get(run_id), now)
            if snapshot.state is not AgentRunState.PENDING_APPROVAL:
                if snapshot.state in TERMINAL_AGENT_RUN_STATES:
                    raise AgentRunTerminal(f"run is terminal: {snapshot.state.value}")
                raise AgentApprovalInvalid("run is not waiting for an external approval")
            updated = self._append_event(
                snapshot,
                kind=AgentRunEventKind.APPROVAL_REJECTED,
                trigger=AgentRunTrigger.HUMAN,
                state_after=AgentRunState.REJECTED,
                reason="EXTERNAL_APPROVAL_REJECTED",
                occurred_at=now,
                control_hash=rejection_hash,
            )
            return self._save(snapshot, updated)

    def _lock_for(self, run_id: str) -> RLock:
        if type(run_id) is not str or not run_id or run_id != run_id.strip():
            raise ValueError("run_id must be an exact non-empty string")
        with self._locks_guard:
            return self._locks.setdefault(run_id, RLock())

    def _now(self, snapshot: AgentRunSnapshot | None = None) -> datetime:
        now = _utc(self._clock())
        if snapshot is not None and now < snapshot.updated_at:
            raise AgentRuntimeClockError("runtime clock moved backwards")
        return now

    def _save(
        self,
        before: AgentRunSnapshot,
        after: AgentRunSnapshot,
    ) -> AgentRunSnapshot:
        try:
            return self._repository.save(after, expected_revision=before.revision)
        except Exception as error:
            raise AgentRuntimeRepositoryConflict("runtime state update lost its CAS") from error

    def _append_event(
        self,
        snapshot: AgentRunSnapshot,
        *,
        kind: AgentRunEventKind,
        trigger: AgentRunTrigger,
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
        artifact: AgentArtifactRef | None = None,
        warning_codes: tuple[str, ...] = (),
        error_codes: tuple[str, ...] = (),
        pending_call: AgentPendingToolCall | object | None = None,
        approval_deadline: datetime | object | None = None,
        tool_call_count: int | None = None,
        invalid_tool_call_count: int | None = None,
        reconciliation_attempt_count: int | None = None,
        cancel_requested: bool | None = None,
    ) -> AgentRunSnapshot:
        occurred = _utc(occurred_at)
        if occurred < snapshot.updated_at:
            raise AgentRuntimeClockError("runtime event time cannot move backwards")
        if state_after is not snapshot.state:
            validate_transition(
                snapshot.state,
                state_after,
                goal=snapshot.goal,
                mode=snapshot.runtime_mode,
                event_kind=kind,
            )
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
            occurred_at=occurred,
            request_id=request_id,
            tool_name=tool_name,
            tool_effect=tool_effect,
            argument_hash=argument_hash,
            response_hash=response_hash,
            idempotency_key_hash=idempotency_key_hash,
            batch_hash=batch_hash,
            control_hash=control_hash,
            external_approval=external_approval,
            artifact=artifact,
            warning_codes=warning_codes,
            error_codes=error_codes,
            previous_event_hash=snapshot.events[-1].event_hash,
        )
        artifacts = snapshot.artifacts
        if artifact is not None:
            matching = tuple(
                item
                for item in artifacts
                if (item.kind, item.artifact_id) == (artifact.kind, artifact.artifact_id)
            )
            if matching and matching[0] != artifact:
                raise AgentRuntimeOutputInvalid(
                    "an artifact identity changed within the same Agent run"
                )
            if not matching:
                artifacts = (*artifacts, artifact)

        values = snapshot.model_dump(mode="python")
        values.update(
            {
                "state": state_after,
                "updated_at": occurred,
                "revision": snapshot.revision + 1,
                "events": (*snapshot.events, event),
                "artifacts": artifacts,
            }
        )
        if pending_call is not None:
            values["pending_call"] = pending_call
        elif kind in {
            AgentRunEventKind.TOOL_CALL_SUCCEEDED,
            AgentRunEventKind.TOOL_CALL_FAILED,
            AgentRunEventKind.TOOL_CALL_REJECTED,
        }:
            values["pending_call"] = None
        if approval_deadline is not None:
            values["approval_deadline"] = approval_deadline
        if tool_call_count is not None:
            values["tool_call_count"] = tool_call_count
        if invalid_tool_call_count is not None:
            values["invalid_tool_call_count"] = invalid_tool_call_count
        if reconciliation_attempt_count is not None:
            values["reconciliation_attempt_count"] = reconciliation_attempt_count
        if cancel_requested is not None:
            values["cancel_requested"] = cancel_requested
        if state_after in TERMINAL_AGENT_RUN_STATES:
            values["completed_at"] = occurred
            values["terminal_reason"] = reason
            values["pending_call"] = None
        return AgentRunSnapshot.model_validate(values, strict=True)

    def _apply_time_gate(
        self,
        snapshot: AgentRunSnapshot,
        now: datetime,
    ) -> AgentRunSnapshot:
        if snapshot.state in TERMINAL_AGENT_RUN_STATES:
            return snapshot
        now = _utc(now)
        if now < snapshot.updated_at:
            raise AgentRuntimeClockError("runtime clock moved backwards")
        if snapshot.state in {
            AgentRunState.DRAFT_READY,
            AgentRunState.PENDING_APPROVAL,
            AgentRunState.APPROVED,
            AgentRunState.EXECUTING,
        }:
            draft = _find_artifact(snapshot, AgentArtifactKind.ORDER_DRAFT)
            authorization_expiries = [
                value
                for value in (
                    draft.expires_at if draft is not None else None,
                    snapshot.approval_deadline,
                )
                if value is not None
            ]
            if authorization_expiries and now >= min(authorization_expiries):
                authorization_deadline = min(authorization_expiries)
                outcome_unknown = (
                    snapshot.pending_call is not None
                    and snapshot.pending_call.effect
                    in {
                        ToolEffect.PAPER_EXECUTION_WRITE,
                        ToolEffect.LIVE_EXTERNAL_WRITE,
                    }
                )
                expired = self._append_event(
                    snapshot,
                    kind=(
                        AgentRunEventKind.INCIDENT_RECORDED
                        if outcome_unknown
                        else AgentRunEventKind.TIMEOUT_RECORDED
                    ),
                    trigger=(
                        AgentRunTrigger.SYSTEM if outcome_unknown else AgentRunTrigger.TIMEOUT
                    ),
                    state_after=(
                        AgentRunState.INCIDENT if outcome_unknown else AgentRunState.EXPIRED
                    ),
                    reason=(
                        "EXECUTION_AUTHORIZATION_EXPIRED_WITH_PENDING_WRITE"
                        if outcome_unknown
                        else "APPROVAL_WINDOW_EXPIRED"
                        if snapshot.state is AgentRunState.PENDING_APPROVAL
                        else "APPROVAL_EXPIRED_BEFORE_SUBMISSION"
                        if snapshot.approval_deadline == authorization_deadline
                        else "ORDER_DRAFT_EXPIRED_BEFORE_EXECUTION"
                    ),
                    occurred_at=now,
                )
                return self._save(snapshot, expired)
        if now < snapshot.deadline:
            return snapshot
        if snapshot.state is AgentRunState.RECONCILING:
            # A receipt already exists. Global model time cannot authorize
            # skipping the independent recovery/reconciliation budget.
            return snapshot
        execution_uncertain = (
            snapshot.pending_call is not None
            and snapshot.pending_call.effect
            in {
                ToolEffect.PAPER_EXECUTION_WRITE,
                ToolEffect.LIVE_EXTERNAL_WRITE,
            }
        )
        target = AgentRunState.INCIDENT if execution_uncertain else AgentRunState.TIMED_OUT
        reason = (
            "RUN_DEADLINE_EXCEEDED_AFTER_EXECUTION_BOUNDARY"
            if execution_uncertain
            else "RUN_DEADLINE_EXCEEDED"
        )
        timed_out = self._append_event(
            snapshot,
            kind=(
                AgentRunEventKind.INCIDENT_RECORDED
                if execution_uncertain
                else AgentRunEventKind.TIMEOUT_RECORDED
            ),
            trigger=AgentRunTrigger.TIMEOUT,
            state_after=target,
            reason=reason,
            occurred_at=now,
        )
        return self._save(snapshot, timed_out)

    @staticmethod
    def _request_seen(snapshot: AgentRunSnapshot, request_id: str) -> bool:
        return any(event.request_id == request_id for event in snapshot.events)

    @staticmethod
    def _idempotency_hash(arguments: Mapping[str, object] | None) -> str | None:
        if arguments is None:
            return None
        value = arguments.get("idempotency_key")
        if type(value) is not str or not value:
            return None
        return _hash_text(value)

    @staticmethod
    def _batch_hash(arguments: Mapping[str, object] | None) -> str | None:
        if arguments is None:
            return None
        value = arguments.get("batch_hash")
        return cast(str, value) if _is_sha256(value) else None

    def _available_tool_names(self, snapshot: AgentRunSnapshot) -> frozenset[str]:
        candidates = stage_tool_names(
            snapshot.state,
            goal=snapshot.goal,
            mode=snapshot.runtime_mode,
        )
        available: set[str] = set()
        for name in candidates:
            if self._prerequisite_failure(snapshot, name) is None:
                available.add(name)
        return frozenset(available)

    def _prerequisite_failure(
        self,
        snapshot: AgentRunSnapshot,
        tool_name: str,
    ) -> str | None:
        policy = V1_TOOL_POLICIES.get(tool_name)
        if (
            snapshot.cancel_requested
            and policy is not None
            and policy.effect
            in {
                ToolEffect.PAPER_EXECUTION_WRITE,
                ToolEffect.LIVE_EXTERNAL_WRITE,
            }
        ):
            return "execution submission is closed after a cancellation request"
        required: AgentArtifactKind | None = None
        if tool_name == "validate_market_data":
            required = AgentArtifactKind.MARKET_SNAPSHOT
        elif tool_name == "explain_candidate":
            required = AgentArtifactKind.STOCK_CANDIDATES
        elif tool_name == "build_target_portfolio":
            required = AgentArtifactKind.PORTFOLIO_SNAPSHOT
        elif tool_name == "check_portfolio_risk":
            required = AgentArtifactKind.TARGET_PORTFOLIO
        elif tool_name == "create_order_draft":
            required = AgentArtifactKind.RISK_CHECK
        elif tool_name in {"get_order_draft", *_EXECUTION_TOOLS}:
            required = AgentArtifactKind.ORDER_DRAFT
        elif tool_name == "reconcile_account":
            required = AgentArtifactKind.EXECUTION_RECEIPT
        if required is not None and _find_artifact(snapshot, required) is None:
            return f"required artifact is missing: {required.value}"
        if (
            tool_name == "get_market_snapshot"
            and _find_artifact(snapshot, AgentArtifactKind.MARKET_SNAPSHOT) is not None
        ):
            return "the market snapshot is already locked"
        return None

    def _tool_denial_reason(
        self,
        snapshot: AgentRunSnapshot,
        tool_name: str,
        parsed_arguments: Mapping[str, object] | None,
    ) -> str | None:
        if tool_name not in self._available_tool_names(snapshot):
            prerequisite = self._prerequisite_failure(snapshot, tool_name)
            return prerequisite or "tool is unavailable in the current runtime stage"
        policy = V1_TOOL_POLICIES.get(tool_name)
        if policy is None:
            return "tool is absent from the central runtime policy"
        if parsed_arguments is None:
            return "tool arguments are not one valid JSON object"
        if policy.requires_idempotency_key and self._idempotency_hash(parsed_arguments) is None:
            return "write tools require a non-empty idempotency key"
        if tool_name in {"get_order_draft", *_EXECUTION_TOOLS, "reconcile_account"}:
            locked = _find_artifact(snapshot, AgentArtifactKind.ORDER_DRAFT)
            if locked is None:
                return "the run has no locked order draft"
            if self._batch_hash(parsed_arguments) != locked.content_hash:
                return "batch_hash does not match the run's locked order draft"
        return None

    @staticmethod
    def _verify_locked_write_identity(
        snapshot: AgentRunSnapshot,
        *,
        tool_name: str,
        effect: ToolEffect,
        idempotency_key_hash: str | None,
        batch_hash: str | None,
    ) -> None:
        if effect not in _WRITE_EFFECTS:
            return
        outcomes = {
            event.request_id: event.kind
            for event in snapshot.events
            if event.kind
            in {
                AgentRunEventKind.TOOL_CALL_SUCCEEDED,
                AgentRunEventKind.TOOL_CALL_FAILED,
                AgentRunEventKind.TOOL_CALL_REPLAYED,
                AgentRunEventKind.TOOL_CALL_REJECTED,
            }
            and event.request_id is not None
        }
        previous = next(
            (
                event
                for event in snapshot.events
                if event.kind is AgentRunEventKind.TOOL_CALL_RESERVED
                and event.tool_name == tool_name
                and event.idempotency_key_hash is not None
                and outcomes.get(event.request_id or "") is not AgentRunEventKind.TOOL_CALL_REJECTED
            ),
            None,
        )
        if previous is not None and (
            previous.idempotency_key_hash != idempotency_key_hash
            or previous.batch_hash != batch_hash
        ):
            raise AgentRequestConflict(
                "a write stage cannot change its locked idempotency or batch identity"
            )

    def _model_budget_exhausted(self, snapshot: AgentRunSnapshot) -> bool:
        if snapshot.invalid_tool_call_count >= snapshot.limits.max_invalid_tool_calls:
            return True
        if snapshot.state is AgentRunState.RECONCILING:
            return (
                snapshot.reconciliation_attempt_count >= snapshot.limits.max_reconciliation_attempts
            )
        return snapshot.tool_call_count >= snapshot.limits.max_tool_calls

    def _ensure_budget(
        self,
        snapshot: AgentRunSnapshot,
        *,
        occurred_at: datetime,
    ) -> None:
        if not self._model_budget_exhausted(snapshot):
            return
        after_execution = snapshot.state in {
            AgentRunState.EXECUTING,
            AgentRunState.RECONCILING,
        }
        target = AgentRunState.INCIDENT if after_execution else AgentRunState.LIMIT_EXCEEDED
        limited = self._append_event(
            snapshot,
            kind=(
                AgentRunEventKind.INCIDENT_RECORDED
                if after_execution
                else AgentRunEventKind.LIMIT_REACHED
            ),
            trigger=AgentRunTrigger.SYSTEM,
            state_after=target,
            reason=(
                "RECONCILIATION_BUDGET_EXHAUSTED"
                if after_execution
                else "TOOL_CALL_BUDGET_EXHAUSTED"
            ),
            occurred_at=occurred_at,
        )
        self._save(snapshot, limited)
        raise AgentToolCallLimitExceeded(limited.terminal_reason or "call limit exceeded")

    def _record_pre_dispatch_rejection(
        self,
        snapshot: AgentRunSnapshot,
        *,
        request_id: str,
        tool_name: str,
        effect: ToolEffect,
        argument_hash: str,
        parsed_arguments: Mapping[str, object] | None,
        reason: str,
        occurred_at: datetime,
    ) -> AgentRunSnapshot:
        next_call_count = snapshot.tool_call_count + 1
        next_invalid_count = snapshot.invalid_tool_call_count + 1
        reserved = self._append_event(
            snapshot,
            kind=AgentRunEventKind.TOOL_CALL_RESERVED,
            trigger=AgentRunTrigger.TOOL,
            state_after=snapshot.state,
            reason="TOOL_CALL_RESERVED",
            occurred_at=occurred_at,
            request_id=request_id,
            tool_name=tool_name,
            tool_effect=effect,
            argument_hash=argument_hash,
            idempotency_key_hash=self._idempotency_hash(parsed_arguments),
            batch_hash=self._batch_hash(parsed_arguments),
            tool_call_count=next_call_count,
            reconciliation_attempt_count=(
                snapshot.reconciliation_attempt_count
                + (1 if tool_name == "reconcile_account" else 0)
            ),
        )
        after_execution = snapshot.state in {
            AgentRunState.EXECUTING,
            AgentRunState.RECONCILING,
        }
        exhausted = after_execution or (
            next_invalid_count >= snapshot.limits.max_invalid_tool_calls
            or (
                snapshot.state is not AgentRunState.RECONCILING
                and next_call_count >= snapshot.limits.max_tool_calls
            )
        )
        target = (
            AgentRunState.INCIDENT
            if exhausted and after_execution
            else AgentRunState.LIMIT_EXCEEDED
            if exhausted
            else snapshot.state
        )
        rejected = self._append_event(
            reserved,
            kind=AgentRunEventKind.TOOL_CALL_REJECTED,
            trigger=AgentRunTrigger.TOOL,
            state_after=reserved.state,
            reason="TOOL_STAGE_REJECTED",
            occurred_at=occurred_at,
            request_id=request_id,
            tool_name=tool_name,
            tool_effect=effect,
            argument_hash=argument_hash,
            idempotency_key_hash=self._idempotency_hash(parsed_arguments),
            batch_hash=self._batch_hash(parsed_arguments),
            error_codes=("RUNTIME_STAGE_REJECTED",),
            invalid_tool_call_count=next_invalid_count,
        )
        if exhausted:
            rejected = self._append_event(
                rejected,
                kind=(
                    AgentRunEventKind.INCIDENT_RECORDED
                    if target is AgentRunState.INCIDENT
                    else AgentRunEventKind.LIMIT_REACHED
                ),
                trigger=AgentRunTrigger.SYSTEM,
                state_after=target,
                reason="INVALID_TOOL_CALL_LIMIT_REACHED",
                occurred_at=occurred_at,
            )
        return self._save(snapshot, rejected)

    def _record_registry_exception(
        self,
        reserved: AgentRunSnapshot,
        error: Exception,
    ) -> AgentRunSnapshot:
        pending = reserved.pending_call
        if pending is None:
            raise AgentRuntimeOutputInvalid("reserved tool call lost its pending identity")
        invalid = isinstance(error, (ToolArgumentsRejected, ToolInvocationDenied))
        next_invalid = reserved.invalid_tool_call_count + (1 if invalid else 0)
        execution_unknown = pending.tool_name in _EXECUTION_TOOLS and isinstance(
            error,
            (ToolExecutionFailed, ToolOutputRejected, AgentRuntimeOutputInvalid),
        )
        reconciliation_failed = pending.tool_name == "reconcile_account" and not isinstance(
            error,
            ToolArgumentsRejected,
        )

        if (
            (
                reserved.state is AgentRunState.EXECUTING
                and not isinstance(error, ToolArgumentsRejected)
            )
            or execution_unknown
            or reconciliation_failed
        ):
            target = AgentRunState.INCIDENT
            reason = (
                "EXECUTION_OUTCOME_UNKNOWN"
                if execution_unknown or reserved.state is AgentRunState.EXECUTING
                else "RECONCILIATION_EXECUTION_FAILED"
            )
        elif isinstance(error, ToolPermissionDenied | ToolNotRegistered):
            target = AgentRunState.REJECTED
            reason = "REGISTRY_AUTHORIZATION_REJECTED"
        elif isinstance(error, ToolArgumentsRejected):
            target = reserved.state
            reason = "TOOL_ARGUMENTS_REJECTED"
        elif isinstance(error, ToolAuditUnavailable):
            target = AgentRunState.FAILED
            reason = "TOOL_AUDIT_UNAVAILABLE"
        else:
            target = AgentRunState.FAILED
            reason = "TOOL_EXECUTION_FAILED"

        exhausted = next_invalid >= reserved.limits.max_invalid_tool_calls or (
            reserved.state is not AgentRunState.RECONCILING
            and reserved.tool_call_count >= reserved.limits.max_tool_calls
            and target is reserved.state
        )
        if exhausted:
            if reserved.state in {AgentRunState.EXECUTING, AgentRunState.RECONCILING}:
                target = AgentRunState.INCIDENT
                reason = "INVALID_CALL_LIMIT_AFTER_EXECUTION"
            else:
                target = AgentRunState.LIMIT_EXCEEDED
                reason = "INVALID_TOOL_CALL_LIMIT_REACHED"
        kind = (
            AgentRunEventKind.TOOL_CALL_REJECTED
            if isinstance(error, ToolInvocationDenied)
            else AgentRunEventKind.TOOL_CALL_FAILED
        )
        code = self._exception_code(error)
        now = self._now(reserved)
        if now >= reserved.deadline and pending.tool_name not in {
            *_EXECUTION_TOOLS,
            "reconcile_account",
        }:
            # The registry exception is still evidence about the reserved call,
            # but it cannot advance a normal workflow after the run deadline.
            # Persist the same-state outcome first, then close through the one
            # authoritative timeout transition used by successful responses.
            target = AgentRunState.TIMED_OUT
            reason = "RUN_DEADLINE_EXCEEDED_DURING_TOOL_CALL"
        event_target = (
            reserved.state
            if kind is AgentRunEventKind.TOOL_CALL_REJECTED
            or target
            in {AgentRunState.LIMIT_EXCEEDED, AgentRunState.TIMED_OUT, AgentRunState.EXPIRED}
            else target
        )
        updated = self._append_event(
            reserved,
            kind=kind,
            trigger=AgentRunTrigger.TOOL,
            state_after=event_target,
            reason=reason if event_target is target else "TOOL_CALL_REJECTED",
            occurred_at=now,
            request_id=pending.request_id,
            tool_name=pending.tool_name,
            tool_effect=pending.effect,
            argument_hash=pending.argument_hash,
            idempotency_key_hash=pending.idempotency_key_hash,
            batch_hash=pending.batch_hash,
            error_codes=(code,),
            invalid_tool_call_count=next_invalid,
        )
        if target is not event_target:
            if target is AgentRunState.LIMIT_EXCEEDED:
                transition_kind = AgentRunEventKind.LIMIT_REACHED
            elif target in {AgentRunState.TIMED_OUT, AgentRunState.EXPIRED}:
                transition_kind = AgentRunEventKind.TIMEOUT_RECORDED
            elif target is AgentRunState.INCIDENT:
                transition_kind = AgentRunEventKind.INCIDENT_RECORDED
            else:
                transition_kind = AgentRunEventKind.STATE_TRANSITION
            updated = self._append_event(
                updated,
                kind=transition_kind,
                trigger=(
                    AgentRunTrigger.TIMEOUT
                    if transition_kind is AgentRunEventKind.TIMEOUT_RECORDED
                    else AgentRunTrigger.SYSTEM
                ),
                state_after=target,
                reason=reason,
                occurred_at=now,
            )
        return self._save(reserved, updated)

    @staticmethod
    def _exception_code(error: Exception) -> str:
        if isinstance(error, ToolArgumentsRejected):
            return f"REGISTRY_{error.reason.value}"
        if isinstance(error, ToolInvocationDenied):
            return f"REGISTRY_{error.reason.value}"
        if isinstance(error, ToolAuditUnavailable):
            return "TOOL_AUDIT_UNAVAILABLE"
        if isinstance(error, ToolOutputRejected):
            return "TOOL_OUTPUT_REJECTED"
        if isinstance(error, ToolExecutionFailed):
            return "TOOL_EXECUTION_FAILED"
        if isinstance(error, AgentRuntimeOutputInvalid):
            return "RUNTIME_OUTPUT_INVALID"
        return "RUNTIME_INTERNAL_ERROR"

    def _record_response(
        self,
        reserved: AgentRunSnapshot,
        response: ToolResponse[object],
    ) -> AgentRunSnapshot:
        pending = reserved.pending_call
        if pending is None:
            raise AgentRuntimeOutputInvalid("reserved tool call has no pending identity")
        try:
            self._validate_response_identity(reserved, pending, response)
            response_digest = _response_hash(response)
            projected = (
                self._project_artifact(
                    reserved,
                    pending.tool_name,
                    response,
                    response_digest,
                )
                if response.ok
                else None
            )
            target, reason = self._response_target(
                reserved,
                pending.tool_name,
                response,
            )
            validate_transition(
                reserved.state,
                target,
                goal=reserved.goal,
                mode=reserved.runtime_mode,
                event_kind=(
                    AgentRunEventKind.TOOL_CALL_SUCCEEDED
                    if response.ok
                    else AgentRunEventKind.TOOL_CALL_FAILED
                ),
            )
            artifact = self._deduplicate_artifact(reserved, projected)
        except (TypeError, ValueError, AgentRuntimeOutputInvalid) as error:
            self._record_registry_exception(
                reserved,
                error
                if isinstance(error, AgentRuntimeOutputInvalid)
                else AgentRuntimeOutputInvalid("tool response binding validation failed"),
            )
            if isinstance(error, AgentRuntimeOutputInvalid):
                raise
            raise AgentRuntimeOutputInvalid("tool response binding validation failed") from None

        now = self._now(reserved)
        if now >= reserved.deadline and pending.tool_name not in {
            *_EXECUTION_TOOLS,
            "reconcile_account",
        }:
            target = AgentRunState.TIMED_OUT
            reason = "RUN_DEADLINE_EXCEEDED_DURING_TOOL_CALL"
        kind = (
            AgentRunEventKind.TOOL_CALL_SUCCEEDED
            if response.ok
            else AgentRunEventKind.TOOL_CALL_FAILED
        )
        event_target = (
            reserved.state
            if target
            in {
                AgentRunState.TIMED_OUT,
                AgentRunState.EXPIRED,
                AgentRunState.LIMIT_EXCEEDED,
            }
            else target
        )
        updated = self._append_event(
            reserved,
            kind=kind,
            trigger=AgentRunTrigger.TOOL,
            state_after=event_target,
            reason=reason if event_target is target else "TOOL_RESPONSE_RECORDED",
            occurred_at=now,
            request_id=pending.request_id,
            tool_name=pending.tool_name,
            tool_effect=pending.effect,
            argument_hash=pending.argument_hash,
            response_hash=response_digest,
            idempotency_key_hash=pending.idempotency_key_hash,
            batch_hash=pending.batch_hash,
            artifact=artifact,
            warning_codes=_codes(response, errors=False),
            error_codes=_codes(response, errors=True),
        )
        if target is not event_target:
            updated = self._append_event(
                updated,
                kind=(
                    AgentRunEventKind.LIMIT_REACHED
                    if target is AgentRunState.LIMIT_EXCEEDED
                    else AgentRunEventKind.TIMEOUT_RECORDED
                ),
                trigger=(
                    AgentRunTrigger.SYSTEM
                    if target is AgentRunState.LIMIT_EXCEEDED
                    else AgentRunTrigger.TIMEOUT
                ),
                state_after=target,
                reason=reason,
                occurred_at=now,
            )
        updated = self._append_post_call_limit(updated, now)
        updated = self._append_automatic_transitions(updated, response, now)
        stored = self._save(reserved, updated)
        if (
            response.ok
            and stored.state is AgentRunState.INCIDENT
            and pending.tool_name == "reconcile_account"
            and isinstance(response.data, ReconciliationOutput)
            and response.data.stop_signal.required
        ):
            self._notify_reconciliation_incident(stored, response.data, now)
        return stored

    def _validate_response_identity(
        self,
        snapshot: AgentRunSnapshot,
        pending: AgentPendingToolCall,
        response: ToolResponse[object],
    ) -> None:
        decision = self._decision_for(snapshot)
        if (
            response.request_id != pending.request_id
            or response.decision_id != snapshot.decision_id
            or response.as_of != decision.as_of
            or response.provenance.data_version != decision.data_version
            or type(response.ok) is not bool
            or (response.ok and response.errors)
            or (not response.ok and not response.errors)
        ):
            raise AgentRuntimeOutputInvalid("tool response does not bind its reserved call")
        if response.ok and response.data is None:
            raise AgentRuntimeOutputInvalid("successful tool response omitted typed data")

    def _project_artifact(
        self,
        snapshot: AgentRunSnapshot,
        tool_name: str,
        response: ToolResponse[object],
        response_digest: str,
    ) -> AgentArtifactRef | None:
        data = response.data
        if data is None:
            return None
        if not isinstance(data, ToolOutput):
            raise AgentRuntimeOutputInvalid("tool response data is not a closed ToolOutput")
        self._validate_domain_bindings(snapshot, tool_name, data)

        if tool_name == "get_market_snapshot":
            if not isinstance(data, MarketSnapshotOutput):
                raise AgentRuntimeOutputInvalid("market snapshot output type changed")
            return _artifact(
                kind=AgentArtifactKind.MARKET_SNAPSHOT,
                artifact_id=data.data_version,
                content_hash=data.data_content_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name == "validate_market_data":
            if not isinstance(data, MarketDataQualityOutput):
                raise AgentRuntimeOutputInvalid("data-quality output type changed")
            return _artifact(
                kind=AgentArtifactKind.DATA_QUALITY,
                artifact_id=f"quality:{snapshot.decision_id}",
                content_hash=response_digest,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        research: dict[str, tuple[type[ToolOutput], AgentArtifactKind]] = {
            "detect_market_regime": (MarketRegimeOutput, AgentArtifactKind.MARKET_REGIME),
            "rank_market_themes": (ThemeRankingOutput, AgentArtifactKind.MARKET_THEMES),
            "rank_theme_leaders": (LeaderRankingOutput, AgentArtifactKind.THEME_LEADERS),
            "rank_stock_candidates": (
                CandidateRankingOutput,
                AgentArtifactKind.STOCK_CANDIDATES,
            ),
            "explain_candidate": (
                CandidateExplanationOutput,
                AgentArtifactKind.CANDIDATE_EXPLANATION,
            ),
        }
        research_contract = research.get(tool_name)
        if research_contract is not None:
            expected, kind = research_contract
            if not isinstance(data, expected):
                raise AgentRuntimeOutputInvalid("research output type changed")
            content_hash = getattr(data, "result_hash", None)
            if not _is_sha256(content_hash):
                raise AgentRuntimeOutputInvalid("research output omitted its result hash")
            selected_candidate = getattr(data, "candidate", None)
            selected_exclusion = getattr(data, "exclusion", None)
            instrument_id = getattr(
                selected_candidate if selected_candidate is not None else selected_exclusion,
                "instrument_id",
                None,
            )
            artifact_id = (
                instrument_id
                if tool_name == "explain_candidate" and type(instrument_id) is str
                else f"{kind.value.lower()}:{content_hash}"
            )
            return _artifact(
                kind=kind,
                artifact_id=artifact_id,
                content_hash=cast(str, content_hash),
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name == "get_portfolio_snapshot":
            if not isinstance(data, PortfolioSnapshotOutput):
                raise AgentRuntimeOutputInvalid("portfolio snapshot output type changed")
            return _artifact(
                kind=AgentArtifactKind.PORTFOLIO_SNAPSHOT,
                artifact_id=data.account_snapshot_id,
                content_hash=data.content_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name == "build_target_portfolio":
            if not isinstance(data, TargetPortfolioOutput):
                raise AgentRuntimeOutputInvalid("target portfolio output type changed")
            return _artifact(
                kind=AgentArtifactKind.TARGET_PORTFOLIO,
                artifact_id=f"target:{data.result_hash}",
                content_hash=data.result_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name == "check_portfolio_risk":
            if not isinstance(data, PortfolioRiskOutput):
                raise AgentRuntimeOutputInvalid("portfolio risk output type changed")
            return _artifact(
                kind=AgentArtifactKind.RISK_CHECK,
                artifact_id=f"risk:{data.result_hash}",
                content_hash=data.result_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name in {"create_order_draft", "get_order_draft"}:
            if not isinstance(data, OrderDraftOutput):
                raise AgentRuntimeOutputInvalid("order draft output type changed")
            return _artifact(
                kind=AgentArtifactKind.ORDER_DRAFT,
                artifact_id=data.batch_id,
                content_hash=data.batch_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
                expires_at=data.expires_at,
            )
        if tool_name in _EXECUTION_TOOLS:
            if not isinstance(data, PaperExecutionOutput):
                # The live adapter will receive its own closed contract in the
                # live-execution milestone.  Until then, no placeholder handler
                # is accepted merely because its fields look similar.
                raise AgentRuntimeOutputInvalid("execution receipt output type is unsupported")
            return _artifact(
                kind=AgentArtifactKind.EXECUTION_RECEIPT,
                artifact_id=data.receipt_id,
                content_hash=data.receipt_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name == "reconcile_account":
            if not isinstance(data, ReconciliationOutput):
                raise AgentRuntimeOutputInvalid("reconciliation output type changed")
            return _artifact(
                kind=AgentArtifactKind.RECONCILIATION,
                artifact_id=data.reconciliation_id,
                content_hash=data.result_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name == "run_backtest":
            content_hash = self._generic_result_hash(data, response_digest)
            return _artifact(
                kind=AgentArtifactKind.BACKTEST_REPORT,
                artifact_id=f"backtest:{content_hash}",
                content_hash=content_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        if tool_name == "generate_decision_report":
            content_hash = self._generic_result_hash(data, response_digest)
            return _artifact(
                kind=AgentArtifactKind.DECISION_REPORT,
                artifact_id=f"report:{content_hash}",
                content_hash=content_hash,
                tool_name=tool_name,
                response_hash=response_digest,
                data=data,
            )
        raise AgentRuntimeOutputInvalid("runtime has no artifact contract for this tool")

    @staticmethod
    def _generic_result_hash(data: ToolOutput, fallback: str) -> str:
        for field_name in ("result_hash", "report_hash", "content_hash"):
            value = getattr(data, field_name, None)
            if _is_sha256(value):
                return cast(str, value)
        return fallback

    def _decision_for(self, snapshot: AgentRunSnapshot) -> DecisionSnapshot:
        with self._locks_guard:
            decision = self._decisions.get(snapshot.run_id)
        if decision is None:
            raise AgentRuntimeOutputInvalid(
                "decision snapshot is unavailable for cross-tool binding validation"
            )
        if (
            decision.decision_id != snapshot.decision_id
            or decision.content_hash != snapshot.decision_snapshot_hash
            or decision.mode is not snapshot.runtime_mode
        ):
            raise AgentRuntimeOutputInvalid("cached decision snapshot identity drifted")
        return decision

    def _validate_domain_bindings(
        self,
        snapshot: AgentRunSnapshot,
        tool_name: str,
        data: ToolOutput,
    ) -> None:
        decision = self._decision_for(snapshot)
        for field_name, expected in (
            ("decision_id", decision.decision_id),
            ("as_of", decision.as_of),
            ("data_version", decision.data_version),
            ("account_id", decision.account_id),
            ("account_snapshot_id", decision.account_snapshot_id),
            ("account_snapshot_hash", decision.account_snapshot_hash),
        ):
            actual = getattr(data, field_name, expected)
            if actual != expected:
                raise AgentRuntimeOutputInvalid(
                    f"{tool_name} output does not bind the decision {field_name}"
                )

        if isinstance(data, MarketSnapshotOutput) and (
            data.data_content_hash != decision.data_content_hash
        ):
            raise AgentRuntimeOutputInvalid("market snapshot hash drifted from the decision")
        if isinstance(data, MarketDataQualityOutput):
            market = _find_artifact(snapshot, AgentArtifactKind.MARKET_SNAPSHOT)
            if (
                market is None
                or data.data_content_hash != decision.data_content_hash
                or data.data_content_hash != market.content_hash
            ):
                raise AgentRuntimeOutputInvalid("quality result does not bind the locked data")
        if isinstance(data, PortfolioSnapshotOutput) and (
            data.content_hash != decision.account_snapshot_hash
        ):
            raise AgentRuntimeOutputInvalid("portfolio snapshot drifted from the decision")
        if isinstance(data, TargetPortfolioOutput) and (
            data.account_snapshot_hash != decision.account_snapshot_hash
        ):
            raise AgentRuntimeOutputInvalid("target portfolio drifted from the account snapshot")
        if isinstance(data, PortfolioRiskOutput):
            target = _find_artifact(snapshot, AgentArtifactKind.TARGET_PORTFOLIO)
            if (
                target is None
                or data.request.portfolio_proposal_hash != target.content_hash
                or data.request.policy_hash != decision.risk_policy_hash
            ):
                raise AgentRuntimeOutputInvalid("risk result does not bind target and policy")
        if isinstance(data, OrderDraftOutput):
            target = _find_artifact(snapshot, AgentArtifactKind.TARGET_PORTFOLIO)
            risk = _find_artifact(snapshot, AgentArtifactKind.RISK_CHECK)
            locked_draft = _find_artifact(snapshot, AgentArtifactKind.ORDER_DRAFT)
            if (
                target is None
                or risk is None
                or data.portfolio_proposal_hash != target.content_hash
                or data.risk_result_hash != risk.content_hash
                or data.risk_policy_hash != decision.risk_policy_hash
                or data.runtime_mode is not snapshot.runtime_mode
                or data.risk_status not in {RiskCheckStatus.PASS, RiskCheckStatus.WARN}
                or (
                    tool_name == "get_order_draft"
                    and (
                        locked_draft is None
                        or data.batch_hash != locked_draft.content_hash
                        or data.batch_id != locked_draft.artifact_id
                    )
                )
            ):
                raise AgentRuntimeOutputInvalid("order draft does not bind executable risk inputs")
        if isinstance(data, PaperExecutionOutput):
            draft = _find_artifact(snapshot, AgentArtifactKind.ORDER_DRAFT)
            if (
                draft is None
                or data.batch_hash != draft.content_hash
                or data.account_before.account_id != decision.account_id
                or data.account_after.account_id != decision.account_id
                or data.account_before.source_snapshot_id != decision.account_snapshot_id
                or data.account_after.source_snapshot_id != decision.account_snapshot_id
                or data.account_before.source_snapshot_hash != decision.account_snapshot_hash
                or data.account_after.source_snapshot_hash != decision.account_snapshot_hash
            ):
                raise AgentRuntimeOutputInvalid("execution receipt changed the locked draft batch")
        if isinstance(data, ReconciliationOutput):
            draft = _find_artifact(snapshot, AgentArtifactKind.ORDER_DRAFT)
            receipt = _find_artifact(snapshot, AgentArtifactKind.EXECUTION_RECEIPT)
            if (
                draft is None
                or receipt is None
                or data.batch_hash != draft.content_hash
                or data.draft_hash != draft.content_hash
                or data.receipt_id != receipt.artifact_id
                or data.receipt_hash != receipt.content_hash
            ):
                raise AgentRuntimeOutputInvalid(
                    "reconciliation does not bind the locked draft and receipt"
                )

    @staticmethod
    def _deduplicate_artifact(
        snapshot: AgentRunSnapshot,
        artifact: AgentArtifactRef | None,
    ) -> AgentArtifactRef | None:
        if artifact is None:
            return None
        matches = tuple(
            item
            for item in snapshot.artifacts
            if (item.kind, item.artifact_id) == (artifact.kind, artifact.artifact_id)
        )
        if not matches:
            return artifact
        existing = matches[0]
        if (
            existing.content_hash != artifact.content_hash
            or existing.expires_at != artifact.expires_at
        ):
            raise AgentRuntimeOutputInvalid("an immutable artifact identity changed content")
        # A fresh request produces a fresh response hash, but it must not replace
        # the content-addressed artifact already pinned by the run.
        return None

    def _response_target(
        self,
        snapshot: AgentRunSnapshot,
        tool_name: str,
        response: ToolResponse[object],
    ) -> tuple[AgentRunState, str]:
        if not response.ok:
            return self._failed_response_target(snapshot, tool_name, response)
        data = response.data
        if tool_name == "get_market_snapshot":
            return snapshot.state, "MARKET_SNAPSHOT_LOCKED"
        if tool_name == "validate_market_data":
            if not isinstance(data, MarketDataQualityOutput) or not data.qualified:
                return AgentRunState.DATA_INVALID, "MARKET_DATA_QUALITY_REJECTED"
            return AgentRunState.DATA_VALIDATED, "MARKET_DATA_VALIDATED"
        if tool_name in _ANALYSIS_TOOLS:
            if snapshot.goal is AgentRunGoal.MARKET_RESEARCH:
                return AgentRunState.ANALYZED, "RESEARCH_ANALYSIS_COMPLETED"
            if tool_name == "rank_stock_candidates":
                return AgentRunState.ANALYZED, "CANDIDATE_ANALYSIS_COMPLETED"
            return snapshot.state, "RESEARCH_ARTIFACT_RECORDED"
        if tool_name == "run_backtest":
            return AgentRunState.ANALYZED, "BACKTEST_COMPLETED"
        if tool_name == "get_portfolio_snapshot":
            return snapshot.state, "PORTFOLIO_SNAPSHOT_LOCKED"
        if tool_name == "build_target_portfolio":
            return AgentRunState.PORTFOLIO_READY, "TARGET_PORTFOLIO_BUILT"
        if tool_name == "check_portfolio_risk":
            if not isinstance(data, PortfolioRiskOutput):
                raise AgentRuntimeOutputInvalid("risk tool omitted its typed result")
            if (
                data.status in {RiskCheckStatus.PASS, RiskCheckStatus.WARN}
                and data.allows_execution
            ):
                return AgentRunState.RISK_CHECKED, "PORTFOLIO_RISK_ACCEPTED"
            if data.status is RiskCheckStatus.REJECT:
                return AgentRunState.REJECTED, "PORTFOLIO_RISK_REJECTED"
            return AgentRunState.DATA_INVALID, "PORTFOLIO_RISK_FAILED_CLOSED"
        if tool_name == "create_order_draft":
            return AgentRunState.DRAFT_READY, "ORDER_DRAFT_LOCKED"
        if tool_name == "get_order_draft":
            return snapshot.state, "ORDER_DRAFT_RETRIEVED"
        if tool_name in _EXECUTION_TOOLS:
            return AgentRunState.RECONCILING, "EXECUTION_RECEIPT_LOCKED"
        if tool_name == "reconcile_account":
            if not isinstance(data, ReconciliationOutput):
                raise AgentRuntimeOutputInvalid("reconciliation tool omitted its typed result")
            if (
                data.status
                in {
                    ReconciliationStatus.MATCHED,
                    ReconciliationStatus.RECONCILED_WITH_VARIANCE,
                }
                and not data.stop_signal.required
            ):
                return AgentRunState.COMPLETED, "RECONCILIATION_COMPLETED"
            return AgentRunState.INCIDENT, "RECONCILIATION_INCIDENT"
        if tool_name == "generate_decision_report":
            if not report_ready(snapshot.goal, snapshot.state):
                raise AgentRuntimeOutputInvalid("report completed before its required milestone")
            return AgentRunState.REPORTED, "DECISION_REPORT_COMPLETED"
        raise AgentRuntimeOutputInvalid("tool has no runtime state projection")

    @staticmethod
    def _failed_response_target(
        snapshot: AgentRunSnapshot,
        tool_name: str,
        response: ToolResponse[object],
    ) -> tuple[AgentRunState, str]:
        codes = {item.code for item in response.errors}
        if tool_name in _EXECUTION_TOOLS or tool_name == "reconcile_account":
            return AgentRunState.INCIDENT, "EXECUTION_OR_RECONCILIATION_FAILED"
        if ErrorCode.KILL_SWITCH_ACTIVE in codes:
            return AgentRunState.INCIDENT, "KILL_SWITCH_BLOCKED_EXECUTION"
        if ErrorCode.RISK_REJECTED in codes:
            return AgentRunState.REJECTED, "PORTFOLIO_RISK_REJECTED"
        if ErrorCode.DATA_INVALID in codes:
            return AgentRunState.DATA_INVALID, "DECISION_BOUND_DATA_INVALID"
        if ErrorCode.APPROVAL_EXPIRED in codes:
            raise AgentRuntimeOutputInvalid(
                "tool response cannot authoritatively expire runtime approval"
            )
        if codes.intersection({ErrorCode.UNAUTHORIZED, ErrorCode.FORBIDDEN}):
            return AgentRunState.REJECTED, "TOOL_AUTHORIZATION_REJECTED"
        if ErrorCode.APPROVAL_REQUIRED in codes:
            return AgentRunState.REJECTED, "UNEXPECTED_APPROVAL_REQUIREMENT"
        if codes.intersection(
            {
                ErrorCode.INVALID_ARGUMENT,
                ErrorCode.NOT_FOUND,
                ErrorCode.CONFLICT,
                ErrorCode.DATA_UNAVAILABLE,
            }
        ):
            return snapshot.state, "RECOVERABLE_TOOL_FAILURE"
        return AgentRunState.FAILED, "NON_RECOVERABLE_TOOL_FAILURE"

    def _append_automatic_transitions(
        self,
        snapshot: AgentRunSnapshot,
        response: ToolResponse[object],
        now: datetime,
    ) -> AgentRunSnapshot:
        if not response.ok or snapshot.state in TERMINAL_AGENT_RUN_STATES:
            return snapshot
        data = response.data
        if (
            not isinstance(data, OrderDraftOutput)
            or snapshot.state is not AgentRunState.DRAFT_READY
        ):
            return snapshot
        approval_deadline = min(
            data.expires_at,
            snapshot.deadline,
            now + timedelta(seconds=snapshot.limits.approval_timeout_seconds),
        )
        if approval_deadline <= now:
            return self._append_event(
                snapshot,
                kind=AgentRunEventKind.TIMEOUT_RECORDED,
                trigger=AgentRunTrigger.TIMEOUT,
                state_after=AgentRunState.EXPIRED,
                reason="ORDER_DRAFT_EXPIRED_BEFORE_EXECUTION",
                occurred_at=now,
            )
        if snapshot.goal is AgentRunGoal.PAPER_EXECUTION:
            return self._append_event(
                snapshot,
                kind=AgentRunEventKind.STATE_TRANSITION,
                trigger=AgentRunTrigger.SYSTEM,
                state_after=AgentRunState.EXECUTING,
                reason="PAPER_EXECUTION_READY",
                occurred_at=now,
            )
        if snapshot.goal is AgentRunGoal.LIVE_ASSISTED_EXECUTION:
            return self._append_event(
                snapshot,
                kind=AgentRunEventKind.STATE_TRANSITION,
                trigger=AgentRunTrigger.SYSTEM,
                state_after=AgentRunState.PENDING_APPROVAL,
                reason="LIVE_APPROVAL_REQUIRED",
                occurred_at=now,
                approval_deadline=approval_deadline,
            )
        return snapshot

    def _append_post_call_limit(
        self,
        snapshot: AgentRunSnapshot,
        now: datetime,
    ) -> AgentRunSnapshot:
        if snapshot.state in TERMINAL_AGENT_RUN_STATES:
            return snapshot
        if snapshot.state is AgentRunState.RECONCILING:
            if snapshot.reconciliation_attempt_count < snapshot.limits.max_reconciliation_attempts:
                return snapshot
            return self._append_event(
                snapshot,
                kind=AgentRunEventKind.INCIDENT_RECORDED,
                trigger=AgentRunTrigger.SYSTEM,
                state_after=AgentRunState.INCIDENT,
                reason="RECONCILIATION_BUDGET_EXHAUSTED",
                occurred_at=now,
            )
        if snapshot.tool_call_count < snapshot.limits.max_tool_calls:
            return snapshot
        return self._append_event(
            snapshot,
            kind=AgentRunEventKind.LIMIT_REACHED,
            trigger=AgentRunTrigger.SYSTEM,
            state_after=AgentRunState.LIMIT_EXCEEDED,
            reason="TOOL_CALL_BUDGET_EXHAUSTED",
            occurred_at=now,
        )

    def _notify_reconciliation_incident(
        self,
        snapshot: AgentRunSnapshot,
        result: ReconciliationOutput,
        requested_at: datetime,
    ) -> None:
        handler = self._incident_handler
        if handler is None:
            return
        key = (snapshot.run_id, result.result_hash)
        with self._locks_guard:
            if key in self._handled_incidents:
                return
            self._handled_incidents.add(key)
        try:
            activation_reference = handler.handle_verified_reconciliation(
                decision_id=snapshot.decision_id,
                reconciliation_id=result.reconciliation_id,
                result_hash=result.result_hash,
                requested_at=requested_at,
            )
            if (
                type(activation_reference) is not str
                or not activation_reference
                or activation_reference != activation_reference.strip()
            ):
                raise ValueError("incident handler returned an invalid reference")
        except Exception:
            # The run is already durably INCIDENT.  The handler owns its own
            # durable retry/outbox; a model-visible exception must not make the
            # reconciliation result look less severe or invite order retries.
            return


__all__ = [
    "AgentApprovalInvalid",
    "AgentRequestConflict",
    "AgentRunTerminal",
    "AgentRuntime",
    "AgentRuntimeClockError",
    "AgentRuntimeError",
    "AgentRuntimeOutputInvalid",
    "AgentRuntimeRepositoryConflict",
    "AgentToolCallLimitExceeded",
    "AgentToolStateDenied",
    "ReconciliationIncidentHandler",
    "RuntimeToolRegistry",
]
