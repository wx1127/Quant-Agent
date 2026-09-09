"""Transactional orchestration for fail-closed new-order controls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Protocol

from quant_agent.observability.audit import AuditEvent, AuditSink

from .contracts import (
    KillSwitchActivationRequest,
    KillSwitchActor,
    KillSwitchAuditEvent,
    KillSwitchGateDecision,
    KillSwitchGateReason,
    KillSwitchGateRequest,
    KillSwitchGateStatus,
    KillSwitchRecoveryRequest,
    KillSwitchScope,
    KillSwitchTransition,
)
from .repository import KillSwitchAuthorizationError, KillSwitchRepository

if TYPE_CHECKING:
    from quant_agent.reconciliation import ReconciliationResult


class KillSwitchOrderBlocked(RuntimeError):
    """Raised before submission with the complete fail-closed gate decision."""

    def __init__(
        self,
        decision: KillSwitchGateDecision,
        *,
        cause: Exception | None = None,
    ) -> None:
        reasons = ",".join(reason.value for reason in decision.reason_codes)
        super().__init__(f"new order blocked by kill switch: {reasons}")
        self.decision = decision
        self.cause = cause


class KillSwitchRecoveryAuthorizer(Protocol):
    """Trusted application boundary for recovery authentication and approval.

    Verify both the requesting principal and the approval against independent
    trusted records, including the approver, scope, state revision and approval
    hash. A caller-supplied RISK_ADMIN label or content hash is not authorization.
    """

    def authorize_recovery(self, request: KillSwitchRecoveryRequest) -> bool:
        """Return true only for an authenticated, authorized recovery request."""


class KillSwitchService:
    """Expose atomic state operations and a lock-spanning new-order guard."""

    def __init__(
        self,
        *,
        repository: KillSwitchRepository,
        audit_sink: AuditSink | None = None,
        recovery_authorizer: KillSwitchRecoveryAuthorizer | None = None,
    ) -> None:
        self._repository = repository
        self._audit_sink = audit_sink
        self._recovery_authorizer = recovery_authorizer
        self._projection_lock = RLock()
        self._projected_event_hashes: set[str] = set()

    @property
    def repository(self) -> KillSwitchRepository:
        """Trusted composition/diagnostics access; never expose to Agent tools."""

        return self._repository

    def initialize(
        self,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        actor: KillSwitchActor,
        changed_at: datetime,
        idempotency_key: str | None = None,
    ) -> KillSwitchTransition:
        """Initialize a scope, then best-effort project its committed event."""

        transition = self._repository.initialize(
            scope=scope,
            account_id=account_id,
            actor=actor,
            changed_at=changed_at,
            idempotency_key=idempotency_key,
        )
        request_id = idempotency_key or transition.operation_hash
        self._project_event(
            transition.audit_event,
            actor=actor,
            request_id=request_id,
            decision_id=transition.transition_hash,
        )
        return transition

    def activate(self, request: KillSwitchActivationRequest) -> KillSwitchTransition:
        """Commit activation before attempting any external audit projection."""

        transition = self._repository.activate(request)
        self._project_event(
            transition.audit_event,
            actor=request.actor,
            request_id=request.request_id,
            decision_id=transition.transition_hash,
        )
        return transition

    def activate_from_reconciliation(
        self,
        *,
        result: ReconciliationResult,
        actor: KillSwitchActor,
        request_id: str,
        idempotency_key: str,
        requested_at: datetime,
    ) -> KillSwitchTransition:
        """Activate from a complete reconciliation result, never a detached signal."""

        from .adapters import activation_request_from_reconciliation

        request = activation_request_from_reconciliation(
            result=result,
            actor=actor,
            request_id=request_id,
            idempotency_key=idempotency_key,
            requested_at=requested_at,
        )
        return self.activate(request)

    def recover(self, request: KillSwitchRecoveryRequest) -> KillSwitchTransition:
        """Commit approved recovery before attempting external projection."""

        authorizer = self._recovery_authorizer
        try:
            authorized = authorizer is not None and authorizer.authorize_recovery(request)
        except Exception as error:
            raise KillSwitchAuthorizationError(
                "recovery authorization provider failed closed"
            ) from error
        if authorized is not True:
            raise KillSwitchAuthorizationError(
                "recovery requires an authenticated external authorization"
            )
        transition = self._repository.recover(request)
        self._project_event(
            transition.audit_event,
            actor=request.actor,
            request_id=request.request_id,
            decision_id=transition.transition_hash,
        )
        return transition

    @contextmanager
    def guard_new_order(
        self,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> Iterator[KillSwitchGateDecision]:
        """Yield only while both switches are inactive under the repository lock."""

        decision: KillSwitchGateDecision | None = None
        caller_entered = False
        try:
            with self._repository.guard_new_order(request, actor) as candidate:
                decision = candidate
                if candidate.allowed:
                    caller_entered = True
                    yield candidate
        except Exception as error:
            if caller_entered:
                raise
            decision = self._repository_failure_decision(request)
            raise KillSwitchOrderBlocked(decision, cause=error) from error

        if decision is None:
            failure = RuntimeError("repository guard returned no gate decision")
            blocked = self._repository_failure_decision(request)
            raise KillSwitchOrderBlocked(blocked, cause=failure) from failure
        if decision.allowed:
            return

        self._project_blocked_event(
            decision,
            actor=actor,
            request_id=request.request_id,
        )
        raise KillSwitchOrderBlocked(decision)

    @staticmethod
    def _repository_failure_decision(
        request: KillSwitchGateRequest,
    ) -> KillSwitchGateDecision:
        return KillSwitchGateDecision.build(
            request=request,
            status=KillSwitchGateStatus.BLOCKED,
            global_state_hash=None,
            account_state_hash=None,
            reason_codes=(KillSwitchGateReason.REPOSITORY_FAILURE,),
        )

    def _project_blocked_event(
        self,
        decision: KillSwitchGateDecision,
        *,
        actor: KillSwitchActor,
        request_id: str,
    ) -> None:
        target_hash = decision.audit_event_hash
        if target_hash is None:
            return
        try:
            events = self._repository.events()
        except Exception:
            return
        for event in reversed(events):
            if event.event_hash == target_hash:
                self._project_event(
                    event,
                    actor=actor,
                    request_id=request_id,
                    decision_id=decision.decision_hash,
                )
                return

    def _project_event(
        self,
        event: KillSwitchAuditEvent,
        *,
        actor: KillSwitchActor,
        request_id: str,
        decision_id: str,
    ) -> None:
        sink = self._audit_sink
        if sink is None or actor.actor_hash != event.actor_hash:
            return
        with self._projection_lock:
            if event.event_hash in self._projected_event_hashes:
                return
            try:
                projection = AuditEvent(
                    event_type="KILL_SWITCH",
                    actor_id=actor.actor_id,
                    action=event.event_type.value,
                    result=(
                        "BLOCKED" if event.event_type.value == "ORDER_BLOCKED" else "COMMITTED"
                    ),
                    request_id=request_id,
                    decision_id=decision_id,
                    occurred_at=event.occurred_at,
                    metadata={
                        "scope": event.scope.value,
                        "account_id": event.account_id,
                        "state_revision": event.state_revision,
                        "state_after_hash": event.state_after_hash,
                        "operation_hash": event.operation_hash,
                        "actor_hash": event.actor_hash,
                        "incident_id": event.incident_id,
                        "reason_codes": list(event.reason_codes),
                        "evidence_hashes": list(event.evidence_hashes),
                        "previous_event_hash": event.previous_event_hash,
                        "event_hash": event.event_hash,
                    },
                )
                sink.append(projection)
            except Exception:
                # The repository event is authoritative; a projection is never a
                # transaction participant and may be retried by an exact replay.
                return
            self._projected_event_hashes.add(event.event_hash)


__all__ = [
    "KillSwitchOrderBlocked",
    "KillSwitchRecoveryAuthorizer",
    "KillSwitchService",
]
