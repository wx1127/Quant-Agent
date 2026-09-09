"""Atomic repository boundary for the global and account kill switch."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from threading import RLock
from typing import Protocol

from quant_agent.core.time import ensure_aware, shanghai_now
from quant_agent.regime.contracts import stable_hash

from .contracts import (
    KillSwitchActivationRequest,
    KillSwitchActor,
    KillSwitchActorKind,
    KillSwitchActorRole,
    KillSwitchAuditEvent,
    KillSwitchEventType,
    KillSwitchGateDecision,
    KillSwitchGateReason,
    KillSwitchGateRequest,
    KillSwitchGateStatus,
    KillSwitchIncident,
    KillSwitchRecoveryRequest,
    KillSwitchScope,
    KillSwitchState,
    KillSwitchStatus,
    KillSwitchTransition,
)


class KillSwitchRepositoryError(RuntimeError):
    """Base class for authoritative kill-switch repository failures."""


class KillSwitchStateNotFound(KillSwitchRepositoryError):
    """Raised when a scope has not been explicitly initialized."""


class KillSwitchIncidentNotFound(KillSwitchRepositoryError):
    """Raised when an incident identity is unknown."""


class KillSwitchRepositoryConflict(KillSwitchRepositoryError):
    """Raised when immutable repository facts disagree."""


class KillSwitchIdempotencyConflict(KillSwitchRepositoryConflict):
    """Raised when an operation key is reused for different content."""


class KillSwitchConcurrentUpdate(KillSwitchRepositoryConflict):
    """Raised when a recovery approval no longer matches current state."""


class KillSwitchIncidentConflict(KillSwitchRepositoryConflict):
    """Raised when one incident identity is presented with different content."""


class KillSwitchAuthorizationError(KillSwitchRepositoryError):
    """Raised when an actor is not authorized for an administrative operation."""


class KillSwitchRepository(Protocol):
    """Trusted internal persistence; never expose this interface as an Agent tool.

    Administrative callers must use KillSwitchService so recovery is authenticated.
    Repository access and clock injection belong to application composition only.
    """

    def initialize(
        self,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        actor: KillSwitchActor,
        changed_at: datetime,
        idempotency_key: str | None = None,
    ) -> KillSwitchTransition:
        """Explicitly initialize one GLOBAL or ACCOUNT state as inactive."""

    def activate(self, request: KillSwitchActivationRequest) -> KillSwitchTransition:
        """Atomically persist an incident, active state, and chained audit event."""

    def recover(self, request: KillSwitchRecoveryRequest) -> KillSwitchTransition:
        """CAS-recover one active state using a bound, unexpired approval."""

    def get_state(
        self,
        scope: KillSwitchScope,
        account_id: str | None = None,
    ) -> KillSwitchState:
        """Return the current immutable state for one initialized scope."""

    def get_incident(self, incident_id: str) -> KillSwitchIncident:
        """Return an immutable incident by identity."""

    def events(
        self,
        *,
        scope: KillSwitchScope | None = None,
        account_id: str | None = None,
    ) -> tuple[KillSwitchAuditEvent, ...]:
        """Return a stable snapshot of the authoritative audit chain."""

    def guard_new_order(
        self,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> AbstractContextManager[KillSwitchGateDecision]:
        """Hold a linearization lock while a caller submits a new order."""


_StateKey = tuple[KillSwitchScope, str | None]
_OperationKey = tuple[KillSwitchScope, str | None, str]


class InMemoryKillSwitchRepository:
    """Process-local store with an independent event hash chain per state scope."""

    def __init__(self, *, clock: Callable[[], datetime] = shanghai_now) -> None:
        self._lock = RLock()
        self._active_order_guards = 0
        self._clock = clock
        self._states: dict[_StateKey, KillSwitchState] = {}
        self._incidents: dict[str, KillSwitchIncident] = {}
        self._events: list[KillSwitchAuditEvent] = []
        self._events_by_hash: dict[str, KillSwitchAuditEvent] = {}
        self._events_by_id: dict[str, KillSwitchAuditEvent] = {}
        self._latest_event_hashes: dict[_StateKey, str] = {}
        self._operations: dict[
            _OperationKey,
            tuple[str, KillSwitchTransition],
        ] = {}
        self._gate_requests: dict[tuple[str, str], str] = {}
        self._blocked_gate_decisions: dict[
            tuple[str, str],
            KillSwitchGateDecision,
        ] = {}

    def initialize(
        self,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        actor: KillSwitchActor,
        changed_at: datetime,
        idempotency_key: str | None = None,
    ) -> KillSwitchTransition:
        """Explicitly create an inactive state and its first state event."""

        if not (
            (actor.kind is KillSwitchActorKind.SYSTEM and actor.role is KillSwitchActorRole.SYSTEM)
            or (
                actor.kind is KillSwitchActorKind.HUMAN
                and actor.role is KillSwitchActorRole.RISK_ADMIN
            )
        ):
            raise KillSwitchAuthorizationError(
                "kill-switch initialization requires SYSTEM or human RISK_ADMIN authority"
            )
        state_after = KillSwitchState.initial(
            scope=scope,
            account_id=account_id,
            actor=actor,
            changed_at=changed_at,
        )
        effective_key = (
            self._non_empty(idempotency_key, "idempotency_key")
            if idempotency_key is not None
            else f"initialize:{state_after.state_id}"
        )
        operation_hash = stable_hash(
            {
                "operation": "INITIALIZE",
                "idempotency_key": effective_key,
                "state_hash": state_after.state_hash,
            }
        )
        operation_key = self._operation_key(
            state_after.scope,
            state_after.account_id,
            effective_key,
        )
        with self._lock:
            self._reject_reentrant_mutation()
            committed = self._committed_operation(operation_key, operation_hash)
            if committed is not None:
                return committed
            state_key = self._state_key(state_after.scope, state_after.account_id)
            if state_key in self._states:
                raise KillSwitchRepositoryConflict(
                    f"kill switch is already initialized: {state_after.state_id}"
                )
            event = KillSwitchAuditEvent.build(
                event_type=KillSwitchEventType.INITIALIZED,
                state_before=None,
                state_after=state_after,
                actor=actor,
                operation_hash=operation_hash,
                incident=None,
                reason_codes=(),
                evidence_hashes=(),
                occurred_at=changed_at,
                previous_event_hash=None,
            )
            transition = KillSwitchTransition.build(
                operation_hash=operation_hash,
                state_before=None,
                state_after=state_after,
                incident=None,
                audit_event=event,
            )
            self._validate_event_append(event)
            self._states[state_key] = state_after
            self._append_event(event)
            self._operations[operation_key] = (operation_hash, transition)
            return transition

    def activate(self, request: KillSwitchActivationRequest) -> KillSwitchTransition:
        """Activate an initialized scope or append a trigger to an active scope."""

        incident = request.incident
        operation_key = self._operation_key(
            incident.scope,
            incident.account_id,
            request.idempotency_key,
        )
        with self._lock:
            self._reject_reentrant_mutation()
            committed = self._committed_operation(operation_key, request.request_hash)
            if committed is not None:
                return committed
            state_key = self._state_key(incident.scope, incident.account_id)
            state_before = self._states.get(state_key)
            if state_before is None:
                raise KillSwitchStateNotFound(
                    f"kill switch is not initialized: {incident.scope.value}:{incident.account_id}"
                )
            existing_incident = self._incidents.get(incident.incident_id)
            if existing_incident is not None and existing_incident != incident:
                raise KillSwitchIncidentConflict(
                    "incident identity is already stored with different content"
                )
            activation_at = max(request.requested_at, state_before.changed_at)
            state_after = KillSwitchState.activate(
                previous=state_before,
                incident=incident,
                actor=request.actor,
                changed_at=activation_at,
            )
            event_type = (
                KillSwitchEventType.TRIGGER_ADDED
                if state_before.status is KillSwitchStatus.ACTIVE
                else KillSwitchEventType.ACTIVATED
            )
            event = KillSwitchAuditEvent.build(
                event_type=event_type,
                state_before=state_before,
                state_after=state_after,
                actor=request.actor,
                operation_hash=request.request_hash,
                incident=incident,
                reason_codes=incident.reason_codes,
                evidence_hashes=incident.evidence_hashes,
                occurred_at=activation_at,
                previous_event_hash=self._latest_event_hash(
                    state_after.scope,
                    state_after.account_id,
                ),
            )
            transition = KillSwitchTransition.build(
                operation_hash=request.request_hash,
                state_before=state_before,
                state_after=state_after,
                incident=incident,
                audit_event=event,
            )
            self._validate_event_append(event)
            self._incidents.setdefault(incident.incident_id, incident)
            self._states[state_key] = state_after
            self._append_event(event)
            self._operations[operation_key] = (request.request_hash, transition)
            return transition

    def recover(self, request: KillSwitchRecoveryRequest) -> KillSwitchTransition:
        """Recover only when the approval is a current-state compare-and-swap token."""

        approval = request.approval
        operation_key = self._operation_key(
            approval.scope,
            approval.account_id,
            request.idempotency_key,
        )
        with self._lock:
            self._reject_reentrant_mutation()
            committed = self._committed_operation(operation_key, request.request_hash)
            if committed is not None:
                return committed
            processed_at = ensure_aware(self._clock())
            if processed_at < request.requested_at:
                raise KillSwitchAuthorizationError(
                    "recovery processing time cannot precede its request"
                )
            if processed_at >= approval.expires_at:
                raise KillSwitchAuthorizationError("recovery approval expired before atomic commit")
            state_key = self._state_key(approval.scope, approval.account_id)
            state_before = self._states.get(state_key)
            if state_before is None:
                raise KillSwitchStateNotFound(
                    f"kill switch is not initialized: {approval.scope.value}:{approval.account_id}"
                )
            if (
                state_before.status is not KillSwitchStatus.ACTIVE
                or state_before.state_hash != approval.expected_state_hash
                or state_before.revision != approval.expected_revision
                or state_before.latest_incident_hash != approval.expected_latest_incident_hash
            ):
                raise KillSwitchConcurrentUpdate(
                    "recovery approval no longer matches the current active state"
                )
            state_after = KillSwitchState.recover(
                previous=state_before,
                actor=request.actor,
                changed_at=processed_at,
            )
            event = KillSwitchAuditEvent.build(
                event_type=KillSwitchEventType.RECOVERED,
                state_before=state_before,
                state_after=state_after,
                actor=request.actor,
                operation_hash=request.request_hash,
                incident=None,
                reason_codes=("APPROVED_RECOVERY",),
                evidence_hashes=(approval.approval_hash,),
                occurred_at=processed_at,
                previous_event_hash=self._latest_event_hash(
                    state_after.scope,
                    state_after.account_id,
                ),
            )
            transition = KillSwitchTransition.build(
                operation_hash=request.request_hash,
                state_before=state_before,
                state_after=state_after,
                incident=None,
                audit_event=event,
            )
            self._validate_event_append(event)
            self._states[state_key] = state_after
            self._append_event(event)
            self._operations[operation_key] = (request.request_hash, transition)
            return transition

    def get_state(
        self,
        scope: KillSwitchScope,
        account_id: str | None = None,
    ) -> KillSwitchState:
        """Return the current state; absent state is a distinct closed-world error."""

        key = self._state_key(scope, account_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                raise KillSwitchStateNotFound(
                    f"kill switch is not initialized: {scope.value}:{key[1]}"
                )
            return state

    def get_incident(self, incident_id: str) -> KillSwitchIncident:
        """Return one incident without exposing mutable repository storage."""

        normalized_id = self._non_empty(incident_id, "incident_id")
        with self._lock:
            incident = self._incidents.get(normalized_id)
            if incident is None:
                raise KillSwitchIncidentNotFound(f"unknown kill-switch incident: {normalized_id}")
            return incident

    def events(
        self,
        *,
        scope: KillSwitchScope | None = None,
        account_id: str | None = None,
    ) -> tuple[KillSwitchAuditEvent, ...]:
        """Return events in append order, optionally filtered by scope identity."""

        if scope is None:
            if account_id is not None:
                raise ValueError("account_id filtering requires ACCOUNT scope")
            with self._lock:
                return tuple(self._events)
        if not isinstance(scope, KillSwitchScope):
            raise ValueError("kill-switch scope is invalid")
        normalized_account: str | None
        if scope is KillSwitchScope.GLOBAL:
            if account_id is not None:
                raise ValueError("GLOBAL event filtering cannot carry account_id")
            normalized_account = None
        elif account_id is None:
            normalized_account = None
        else:
            normalized_account = self._non_empty(account_id, "account_id")
        with self._lock:
            return tuple(
                event
                for event in self._events
                if event.scope is scope
                and (normalized_account is None or event.account_id == normalized_account)
            )

    @contextmanager
    def guard_new_order(
        self,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> Iterator[KillSwitchGateDecision]:
        """Check and hold the same lock until the guarded submission completes."""

        with self._lock:
            gate_key = (request.account_id, request.idempotency_key)
            committed_hash = self._gate_requests.get(gate_key)
            if committed_hash is not None and committed_hash != request.operation_hash:
                raise KillSwitchIdempotencyConflict(
                    "gate idempotency key belongs to a different new-order request"
                )
            committed_block = self._blocked_gate_decisions.get(gate_key)
            if committed_block is not None:
                yield committed_block
                return
            global_state = self._states.get((KillSwitchScope.GLOBAL, None))
            account_state = self._states.get((KillSwitchScope.ACCOUNT, request.account_id))
            if global_state is None or account_state is None:
                decision = KillSwitchGateDecision.build(
                    request=request,
                    status=KillSwitchGateStatus.BLOCKED,
                    global_state_hash=(
                        global_state.state_hash if global_state is not None else None
                    ),
                    account_state_hash=(
                        account_state.state_hash if account_state is not None else None
                    ),
                    reason_codes=(KillSwitchGateReason.STATE_UNAVAILABLE,),
                )
                self._gate_requests[gate_key] = request.operation_hash
                self._blocked_gate_decisions[gate_key] = decision
                yield decision
                return
            if (
                request.checked_at < global_state.changed_at
                or request.checked_at < account_state.changed_at
            ):
                decision = KillSwitchGateDecision.build(
                    request=request,
                    status=KillSwitchGateStatus.BLOCKED,
                    global_state_hash=global_state.state_hash,
                    account_state_hash=account_state.state_hash,
                    reason_codes=(KillSwitchGateReason.STATE_UNAVAILABLE,),
                )
                self._gate_requests[gate_key] = request.operation_hash
                self._blocked_gate_decisions[gate_key] = decision
                yield decision
                return

            blocking_states = tuple(
                state
                for state in (global_state, account_state)
                if state.status is KillSwitchStatus.ACTIVE
            )
            if not blocking_states:
                decision = KillSwitchGateDecision.build(
                    request=request,
                    status=KillSwitchGateStatus.ALLOWED,
                    global_state_hash=global_state.state_hash,
                    account_state_hash=account_state.state_hash,
                )
                self._gate_requests[gate_key] = request.operation_hash
                self._active_order_guards += 1
                try:
                    yield decision
                finally:
                    self._active_order_guards -= 1
                return

            reasons: list[KillSwitchGateReason] = []
            if global_state.status is KillSwitchStatus.ACTIVE:
                reasons.append(KillSwitchGateReason.GLOBAL_ACTIVE)
            if account_state.status is KillSwitchStatus.ACTIVE:
                reasons.append(KillSwitchGateReason.ACCOUNT_ACTIVE)
            primary_state = blocking_states[0]
            previous_event_hash = self._latest_event_hash(
                primary_state.scope,
                primary_state.account_id,
            )
            operation_hash = stable_hash(
                {
                    "operation": "ORDER_BLOCKED",
                    "request_hash": request.request_hash,
                    "global_state_hash": global_state.state_hash,
                    "account_state_hash": account_state.state_hash,
                    "previous_event_hash": previous_event_hash,
                }
            )
            event = KillSwitchAuditEvent.build(
                event_type=KillSwitchEventType.ORDER_BLOCKED,
                state_before=primary_state,
                state_after=primary_state,
                actor=actor,
                operation_hash=operation_hash,
                incident=None,
                reason_codes=tuple(reason.value for reason in reasons),
                evidence_hashes=tuple(state.state_hash for state in blocking_states),
                occurred_at=max(request.checked_at, primary_state.changed_at),
                previous_event_hash=previous_event_hash,
            )
            decision = KillSwitchGateDecision.build(
                request=request,
                status=KillSwitchGateStatus.BLOCKED,
                global_state_hash=global_state.state_hash,
                account_state_hash=account_state.state_hash,
                blocking_state_hashes=tuple(state.state_hash for state in blocking_states),
                reason_codes=tuple(reasons),
                incident_ids=tuple(
                    incident_id
                    for state in blocking_states
                    for incident_id in state.active_incident_ids
                ),
                audit_event_hash=event.event_hash,
            )
            self._validate_event_append(event)
            self._append_event(event)
            self._gate_requests[gate_key] = request.operation_hash
            self._blocked_gate_decisions[gate_key] = decision
            yield decision

    def _reject_reentrant_mutation(self) -> None:
        # Other threads cannot hold this RLock concurrently. A nonzero count
        # here therefore means a callback is trying to change state inside its
        # own allowed order guard, before that order has committed.
        if self._active_order_guards:
            raise KillSwitchRepositoryConflict(
                "kill-switch mutation cannot reenter an active new-order guard"
            )

    @staticmethod
    def _non_empty(value: str, field_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be non-empty")
        return normalized

    @classmethod
    def _state_key(
        cls,
        scope: KillSwitchScope,
        account_id: str | None,
    ) -> _StateKey:
        if not isinstance(scope, KillSwitchScope):
            raise ValueError("kill-switch scope is invalid")
        if scope is KillSwitchScope.GLOBAL:
            if account_id is not None:
                raise ValueError("GLOBAL kill switch cannot carry account_id")
            return (scope, None)
        if account_id is None:
            raise ValueError("ACCOUNT kill switch requires account_id")
        return (scope, cls._non_empty(account_id, "account_id"))

    @classmethod
    def _operation_key(
        cls,
        scope: KillSwitchScope,
        account_id: str | None,
        idempotency_key: str,
    ) -> _OperationKey:
        state_key = cls._state_key(scope, account_id)
        return (
            state_key[0],
            state_key[1],
            cls._non_empty(idempotency_key, "idempotency_key"),
        )

    def _committed_operation(
        self,
        key: _OperationKey,
        operation_hash: str,
    ) -> KillSwitchTransition | None:
        committed = self._operations.get(key)
        if committed is None:
            return None
        committed_hash, transition = committed
        if committed_hash != operation_hash:
            raise KillSwitchIdempotencyConflict(
                "kill-switch idempotency key belongs to different operation content"
            )
        return transition

    def _latest_event_hash(
        self,
        scope: KillSwitchScope,
        account_id: str | None,
    ) -> str | None:
        return self._latest_event_hashes.get(self._state_key(scope, account_id))

    def _append_event(self, event: KillSwitchAuditEvent) -> None:
        self._validate_event_append(event)
        if event.event_hash in self._events_by_hash:
            return
        state_key = self._state_key(event.scope, event.account_id)
        self._events.append(event)
        self._events_by_hash[event.event_hash] = event
        self._events_by_id[event.event_id] = event
        self._latest_event_hashes[state_key] = event.event_hash

    def _validate_event_append(self, event: KillSwitchAuditEvent) -> None:
        state_key = self._state_key(event.scope, event.account_id)
        expected_previous = self._latest_event_hashes.get(state_key)
        if event.previous_event_hash != expected_previous:
            raise KillSwitchRepositoryConflict(
                "audit event does not extend its scope-local hash chain"
            )
        existing = self._events_by_hash.get(event.event_hash)
        if existing is not None:
            if existing == event:
                return
            raise KillSwitchRepositoryConflict(
                "audit event hash is already stored with different content"
            )
        existing_id = self._events_by_id.get(event.event_id)
        if existing_id is not None and existing_id != event:
            raise KillSwitchRepositoryConflict(
                "audit event identity is already stored with different content"
            )


__all__ = [
    "InMemoryKillSwitchRepository",
    "KillSwitchAuthorizationError",
    "KillSwitchConcurrentUpdate",
    "KillSwitchIdempotencyConflict",
    "KillSwitchIncidentConflict",
    "KillSwitchIncidentNotFound",
    "KillSwitchRepository",
    "KillSwitchRepositoryConflict",
    "KillSwitchRepositoryError",
    "KillSwitchStateNotFound",
]
