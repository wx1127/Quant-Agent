"""Integration and adversarial tests for the transactional kill-switch service."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from threading import Event as ThreadEvent
from threading import Thread

import pytest

from quant_agent.observability.audit import AuditEvent
from quant_agent.reconciliation import ReconciliationEngine, ReconciliationResult
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk.kill_switch import (
    InMemoryKillSwitchRepository,
    KillSwitchActivationRequest,
    KillSwitchActor,
    KillSwitchActorKind,
    KillSwitchActorRole,
    KillSwitchAuthorizationError,
    KillSwitchConcurrentUpdate,
    KillSwitchEventType,
    KillSwitchGateDecision,
    KillSwitchGateReason,
    KillSwitchGateRequest,
    KillSwitchIdempotencyConflict,
    KillSwitchIncident,
    KillSwitchIncidentConflict,
    KillSwitchOrderBlocked,
    KillSwitchRecoveryApproval,
    KillSwitchRecoveryAuthorizer,
    KillSwitchRecoveryRequest,
    KillSwitchRepositoryConflict,
    KillSwitchScope,
    KillSwitchService,
    KillSwitchStateNotFound,
    KillSwitchStatus,
    KillSwitchTransition,
    KillSwitchTriggerSource,
)
from quant_agent.risk.kill_switch.adapters import activation_request_from_reconciliation

from .test_reconciliation_engine import _execute_full, _request

T0 = datetime(2026, 9, 2, 1, 0, tzinfo=UTC)
ACCOUNT_A = "paper-account-a"
ACCOUNT_B = "paper-account-b"


def _digest(label: str) -> str:
    return stable_hash({"kill-switch-service-fixture": label})


def _actor(
    *,
    actor_id: str,
    kind: KillSwitchActorKind,
    role: KillSwitchActorRole,
) -> KillSwitchActor:
    return KillSwitchActor.build(actor_id=actor_id, kind=kind, role=role)


def _system_actor() -> KillSwitchActor:
    return _actor(
        actor_id="kill-switch-controller",
        kind=KillSwitchActorKind.SYSTEM,
        role=KillSwitchActorRole.SYSTEM,
    )


def _risk_admin() -> KillSwitchActor:
    return _actor(
        actor_id="risk-admin-1",
        kind=KillSwitchActorKind.HUMAN,
        role=KillSwitchActorRole.RISK_ADMIN,
    )


def _agent_actor() -> KillSwitchActor:
    return _actor(
        actor_id="quant-agent",
        kind=KillSwitchActorKind.AGENT,
        role=KillSwitchActorRole.AGENT,
    )


def _incident(
    *,
    label: str,
    scope: KillSwitchScope = KillSwitchScope.ACCOUNT,
    account_id: str | None = ACCOUNT_A,
    triggered_at: datetime = T0 + timedelta(minutes=1),
    summary: str | None = None,
) -> KillSwitchIncident:
    return KillSwitchIncident.build(
        scope=scope,
        account_id=account_id,
        trigger_source=KillSwitchTriggerSource.RISK,
        reason_codes=(f"{label.upper()}_LIMIT",),
        evidence_hashes=(_digest(f"evidence:{label}"),),
        source_reference_hash=_digest(f"source:{label}"),
        summary=summary or f"{label} safety threshold crossed",
        triggered_at=triggered_at,
    )


def _activation_request(
    incident: KillSwitchIncident,
    *,
    key: str,
    actor: KillSwitchActor | None = None,
    requested_at: datetime = T0 + timedelta(minutes=2),
) -> KillSwitchActivationRequest:
    return KillSwitchActivationRequest.build(
        request_id=f"activate:{key}",
        idempotency_key=key,
        incident=incident,
        actor=actor or _system_actor(),
        requested_at=requested_at,
    )


def _gate_request(
    *,
    account_id: str = ACCOUNT_A,
    label: str,
    checked_at: datetime = T0 + timedelta(minutes=20),
) -> KillSwitchGateRequest:
    return KillSwitchGateRequest.build(
        request_id=f"gate:{label}",
        account_id=account_id,
        idempotency_key=f"gate-key:{label}",
        batch_hash=_digest(f"batch:{label}"),
        source_request_hash=_digest(f"source-request:{label}"),
        checked_at=checked_at,
    )


def _service(
    repository: InMemoryKillSwitchRepository,
    *,
    audit_sink: _FailingAuditSink | None = None,
    recovery_authorizer: KillSwitchRecoveryAuthorizer | None = None,
) -> KillSwitchService:
    return KillSwitchService(
        repository=repository,
        audit_sink=audit_sink,
        recovery_authorizer=recovery_authorizer,
    )


def _initialize(
    service: KillSwitchService,
    *,
    scope: KillSwitchScope,
    account_id: str | None,
    label: str,
    changed_at: datetime = T0,
) -> KillSwitchTransition:
    return service.initialize(
        scope=scope,
        account_id=account_id,
        actor=_system_actor(),
        changed_at=changed_at,
        idempotency_key=f"initialize:{label}",
    )


def _initialize_gate_pair(
    service: KillSwitchService,
    account_id: str = ACCOUNT_A,
) -> None:
    _initialize(
        service,
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        label="global",
    )
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=account_id,
        label=f"account:{account_id}",
    )


def _blocked(
    service: KillSwitchService,
    request: KillSwitchGateRequest,
    *,
    actor: KillSwitchActor | None = None,
) -> KillSwitchOrderBlocked:
    with (
        pytest.raises(KillSwitchOrderBlocked) as captured,
        service.guard_new_order(request, actor or _system_actor()),
    ):
        pass
    return captured.value


class _FailingAuditSink:
    def __init__(self) -> None:
        self.calls: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.calls.append(event)
        raise OSError("audit projection unavailable")


class _RecoveryAuthorizer:
    def __init__(self, result: bool | BaseException = True) -> None:
        self.result = result
        self.requests: list[KillSwitchRecoveryRequest] = []

    def authorize_recovery(self, request: KillSwitchRecoveryRequest) -> bool:
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _ExplodingGuardRepository(InMemoryKillSwitchRepository):
    def guard_new_order(
        self,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> AbstractContextManager[KillSwitchGateDecision]:
        del request, actor
        raise OSError("authoritative repository unavailable")


class _ActivationProbeRepository(InMemoryKillSwitchRepository):
    def __init__(self) -> None:
        super().__init__()
        self.activation_attempted = ThreadEvent()

    def activate(self, request: KillSwitchActivationRequest) -> KillSwitchTransition:
        self.activation_attempted.set()
        return super().activate(request)


def _reconciliation_results() -> tuple[ReconciliationResult, ReconciliationResult]:
    fixture, receipt = _execute_full()
    clean_request = _request(fixture, receipt)
    clean = ReconciliationEngine().reconcile(clean_request)
    required_request = _request(
        fixture,
        receipt,
        order_reports=clean_request.observed.order_reports[1:],
    )
    required = ReconciliationEngine().reconcile(required_request)
    assert not clean.stop_signal.required
    assert required.stop_signal.required
    return clean, required


def test_scopes_must_be_explicitly_initialized_before_new_orders_are_allowed() -> None:
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)

    with pytest.raises(KillSwitchStateNotFound, match="not initialized"):
        service.activate(
            _activation_request(_incident(label="before-initialize"), key="before-initialize")
        )
    absent = _blocked(service, _gate_request(label="neither-initialized"))
    assert absent.decision.reason_codes == (KillSwitchGateReason.STATE_UNAVAILABLE,)
    assert absent.cause is None

    account_transition = _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    account_replay = _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    assert account_replay == account_transition
    with pytest.raises(KillSwitchAuthorizationError, match="SYSTEM or human RISK_ADMIN"):
        service.initialize(
            scope=KillSwitchScope.GLOBAL,
            account_id=None,
            actor=_agent_actor(),
            changed_at=T0,
            idempotency_key="unauthorized-global-initialize",
        )
    one_missing = _blocked(service, _gate_request(label="global-missing"))
    assert one_missing.decision.reason_codes == (KillSwitchGateReason.STATE_UNAVAILABLE,)

    global_transition = _initialize(
        service,
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        label="global",
    )
    with service.guard_new_order(_gate_request(label="both-initialized"), _system_actor()) as gate:
        assert gate.allowed

    assert account_transition.state_before is None
    assert global_transition.state_before is None
    assert account_transition.state_after.status is KillSwitchStatus.INACTIVE
    assert global_transition.state_after.status is KillSwitchStatus.INACTIVE
    assert [event.event_type for event in repository.events()] == [
        KillSwitchEventType.INITIALIZED,
        KillSwitchEventType.INITIALIZED,
    ]


def test_account_and_global_scopes_are_isolated_and_compose_at_the_gate() -> None:
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)
    _initialize_gate_pair(service, ACCOUNT_A)
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_B,
        label="account-b",
    )

    account_incident = _incident(label="account-a")
    service.activate(_activation_request(account_incident, key="activate-account-a"))

    account_a_block = _blocked(
        service,
        _gate_request(account_id=ACCOUNT_A, label="account-a-active"),
    )
    assert account_a_block.decision.reason_codes == (KillSwitchGateReason.ACCOUNT_ACTIVE,)
    with service.guard_new_order(
        _gate_request(account_id=ACCOUNT_B, label="account-b-isolated"),
        _system_actor(),
    ) as account_b_gate:
        assert account_b_gate.allowed

    global_incident = _incident(
        label="global",
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        triggered_at=T0 + timedelta(minutes=3),
    )
    service.activate(
        _activation_request(
            global_incident,
            key="activate-global",
            requested_at=T0 + timedelta(minutes=4),
        )
    )

    both_block = _blocked(
        service,
        _gate_request(account_id=ACCOUNT_A, label="both-active"),
    )
    global_block = _blocked(
        service,
        _gate_request(account_id=ACCOUNT_B, label="global-active"),
    )
    assert both_block.decision.reason_codes == (
        KillSwitchGateReason.ACCOUNT_ACTIVE,
        KillSwitchGateReason.GLOBAL_ACTIVE,
    )
    assert set(both_block.decision.incident_ids) == {
        account_incident.incident_id,
        global_incident.incident_id,
    }
    assert global_block.decision.reason_codes == (KillSwitchGateReason.GLOBAL_ACTIVE,)
    assert global_block.decision.incident_ids == (global_incident.incident_id,)
    assert (
        repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_B).status is KillSwitchStatus.INACTIVE
    )


def test_activation_is_idempotent_rejects_key_reuse_and_appends_active_incidents() -> None:
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    first_incident = _incident(label="first")
    first_request = _activation_request(first_incident, key="activation-key")

    first = service.activate(first_request)
    replay = service.activate(first_request)

    assert replay == first
    assert len(repository.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A)) == 2
    conflicting_request = _activation_request(
        _incident(label="same-key-different-content"),
        key=first_request.idempotency_key,
    )
    with pytest.raises(KillSwitchIdempotencyConflict, match="different operation content"):
        service.activate(conflicting_request)
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == first.state_after

    second_incident = _incident(
        label="second",
        triggered_at=T0 + timedelta(minutes=3),
    )
    second = service.activate(
        _activation_request(
            second_incident,
            key="second-activation-key",
            requested_at=T0 + timedelta(minutes=4),
        )
    )

    assert second.audit_event.event_type is KillSwitchEventType.TRIGGER_ADDED
    assert second.state_before == first.state_after
    assert second.state_after.status is KillSwitchStatus.ACTIVE
    assert second.state_after.revision == 2
    assert second.state_after.active_since == first_incident.triggered_at
    assert second.state_after.active_incident_ids == tuple(
        sorted((first_incident.incident_id, second_incident.incident_id))
    )

    same_identity_changed_content = _incident(
        label="second",
        triggered_at=second_incident.triggered_at,
        summary="same source identity with conflicting immutable content",
    )
    assert same_identity_changed_content.incident_id == second_incident.incident_id
    assert same_identity_changed_content.incident_hash != second_incident.incident_hash
    with pytest.raises(KillSwitchIncidentConflict, match="different content"):
        service.activate(
            _activation_request(
                same_identity_changed_content,
                key="conflicting-incident-content",
                requested_at=T0 + timedelta(minutes=5),
            )
        )


def test_global_and_each_account_keep_independent_scope_local_event_chains() -> None:
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    _initialize(
        service,
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        label="global",
    )
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_B,
        label="account-b",
    )
    service.activate(_activation_request(_incident(label="account-a"), key="account-a-active"))
    service.activate(
        _activation_request(
            _incident(
                label="global",
                scope=KillSwitchScope.GLOBAL,
                account_id=None,
                triggered_at=T0 + timedelta(minutes=3),
            ),
            key="global-active",
            requested_at=T0 + timedelta(minutes=4),
        )
    )
    _blocked(service, _gate_request(account_id=ACCOUNT_A, label="scope-chain"))

    chains = (
        repository.events(scope=KillSwitchScope.GLOBAL),
        repository.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A),
        repository.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_B),
    )
    for chain in chains:
        previous_hash: str | None = None
        for event in chain:
            assert event.previous_event_hash == previous_hash
            previous_hash = event.event_hash

    assert chains[0][0].previous_event_hash is None
    assert chains[1][0].previous_event_hash is None
    assert chains[2][0].previous_event_hash is None
    assert chains[0][-1].event_type is KillSwitchEventType.ORDER_BLOCKED
    assert [event.event_type for event in chains[1]] == [
        KillSwitchEventType.INITIALIZED,
        KillSwitchEventType.ACTIVATED,
    ]
    assert [event.event_type for event in chains[2]] == [KillSwitchEventType.INITIALIZED]


def test_recovery_is_cas_idempotent_and_rejects_stale_or_replayed_approval() -> None:
    repository = InMemoryKillSwitchRepository(clock=lambda: T0 + timedelta(minutes=5))
    authorizer = _RecoveryAuthorizer()
    service = _service(
        repository,
        recovery_authorizer=authorizer,
    )
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    active = service.activate(
        _activation_request(_incident(label="recoverable"), key="recoverable")
    ).state_after
    approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )
    request = KillSwitchRecoveryRequest.build(
        request_id="recover:account-a",
        idempotency_key="recover-key",
        approval=approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=4),
    )

    recovered = service.recover(request)
    exact_replay = service.recover(request)

    assert exact_replay == recovered
    assert recovered.audit_event.event_type is KillSwitchEventType.RECOVERED
    assert recovered.state_after.status is KillSwitchStatus.INACTIVE
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == recovered.state_after
    distinct_replay = KillSwitchRecoveryRequest.build(
        request_id="recover:account-a:replay",
        idempotency_key="recover-key:replay",
        approval=approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=5),
    )
    with pytest.raises(KillSwitchConcurrentUpdate, match="no longer matches"):
        service.recover(distinct_replay)

    stale_repository = InMemoryKillSwitchRepository(clock=lambda: T0 + timedelta(minutes=7))
    stale_service = _service(
        stale_repository,
        recovery_authorizer=_RecoveryAuthorizer(),
    )
    _initialize(
        stale_service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="stale-account-a",
    )
    first_active = stale_service.activate(
        _activation_request(_incident(label="stale-first"), key="stale-first")
    ).state_after
    stale_approval = KillSwitchRecoveryApproval.build(
        state=first_active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )
    stale_service.activate(
        _activation_request(
            _incident(label="stale-second", triggered_at=T0 + timedelta(minutes=4)),
            key="stale-second",
            requested_at=T0 + timedelta(minutes=5),
        )
    )
    stale_request = KillSwitchRecoveryRequest.build(
        request_id="recover:stale",
        idempotency_key="recover-stale-key",
        approval=stale_approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=6),
    )
    with pytest.raises(KillSwitchConcurrentUpdate, match="no longer matches"):
        stale_service.recover(stale_request)


def test_recovery_rejects_expired_approval_at_creation_and_again_at_processing() -> None:
    repository = InMemoryKillSwitchRepository(clock=lambda: T0 + timedelta(minutes=10))
    service = _service(
        repository,
        recovery_authorizer=_RecoveryAuthorizer(),
    )
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    active = service.activate(
        _activation_request(_incident(label="expiry"), key="expiry")
    ).state_after
    approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )

    with pytest.raises(ValueError, match="currently valid"):
        KillSwitchRecoveryRequest.build(
            request_id="recover:already-expired",
            idempotency_key="already-expired",
            approval=approval,
            actor=_risk_admin(),
            requested_at=approval.expires_at,
        )

    queued_while_valid = KillSwitchRecoveryRequest.build(
        request_id="recover:expired-in-queue",
        idempotency_key="expired-in-queue",
        approval=approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=9),
    )
    with pytest.raises(KillSwitchAuthorizationError, match="expired before atomic commit"):
        service.recover(queued_while_valid)
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active
    assert repository.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A)[
        -1
    ].event_type is (KillSwitchEventType.ACTIVATED)


def test_agent_has_no_path_to_approve_or_request_kill_switch_recovery() -> None:
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    active = service.activate(
        _activation_request(_incident(label="agent-cannot-close"), key="agent-cannot-close")
    ).state_after
    agent = _agent_actor()

    with pytest.raises(ValueError, match="human RISK_ADMIN"):
        KillSwitchRecoveryApproval.build(
            state=active,
            approver=agent,
            approved_at=T0 + timedelta(minutes=3),
            expires_at=T0 + timedelta(minutes=10),
        )

    valid_approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )
    with pytest.raises(ValueError, match="only a human RISK_ADMIN"):
        KillSwitchRecoveryRequest.build(
            request_id="agent-recovery",
            idempotency_key="agent-recovery-key",
            approval=valid_approval,
            actor=agent,
            requested_at=T0 + timedelta(minutes=4),
        )
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active


def test_recovery_without_external_authorization_or_with_provider_failure_fails_closed() -> None:
    repository = InMemoryKillSwitchRepository()
    setup_service = _service(repository)
    _initialize(
        setup_service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        label="account-a",
    )
    active = setup_service.activate(
        _activation_request(_incident(label="external-auth"), key="external-auth")
    ).state_after
    approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )
    request = KillSwitchRecoveryRequest.build(
        request_id="recover:external-auth",
        idempotency_key="recover:external-auth",
        approval=approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=4),
    )
    events_before = repository.events()

    without_authorizer = _service(repository)
    with pytest.raises(KillSwitchAuthorizationError, match="authenticated external"):
        without_authorizer.recover(request)

    denied_authorizer = _RecoveryAuthorizer(False)
    denied = _service(
        repository,
        recovery_authorizer=denied_authorizer,
    )
    with pytest.raises(KillSwitchAuthorizationError, match="authenticated external"):
        denied.recover(request)
    assert denied_authorizer.requests == [request]

    broken_authorizer = _RecoveryAuthorizer(ConnectionError("identity provider offline"))
    broken = _service(
        repository,
        recovery_authorizer=broken_authorizer,
    )
    with pytest.raises(KillSwitchAuthorizationError, match="provider failed closed") as captured:
        broken.recover(request)
    assert isinstance(captured.value.__cause__, ConnectionError)
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active
    assert repository.events() == events_before


def test_missing_or_failing_authoritative_repository_fails_closed() -> None:
    missing_repository = InMemoryKillSwitchRepository()
    missing_service = _service(missing_repository)

    missing = _blocked(missing_service, _gate_request(label="missing-state"))
    assert missing.decision.reason_codes == (KillSwitchGateReason.STATE_UNAVAILABLE,)
    assert missing.decision.global_state_hash is None
    assert missing.decision.account_state_hash is None
    assert missing.cause is None

    failure_service = KillSwitchService(
        repository=_ExplodingGuardRepository(),
    )
    failure = _blocked(failure_service, _gate_request(label="repository-error"))
    assert failure.decision.reason_codes == (KillSwitchGateReason.REPOSITORY_FAILURE,)
    assert failure.decision.global_state_hash is None
    assert failure.decision.account_state_hash is None
    assert isinstance(failure.cause, OSError)
    assert str(failure.cause) == "authoritative repository unavailable"


def test_audit_sink_failure_never_rolls_back_authoritative_transitions_or_blocks() -> None:
    repository = InMemoryKillSwitchRepository(clock=lambda: T0 + timedelta(minutes=7))
    sink = _FailingAuditSink()
    service = _service(
        repository,
        audit_sink=sink,
        recovery_authorizer=_RecoveryAuthorizer(),
    )
    _initialize_gate_pair(service)
    active = service.activate(
        _activation_request(_incident(label="audit-failure"), key="audit-failure")
    ).state_after

    blocked = _blocked(service, _gate_request(label="audit-failure-block"))
    assert blocked.decision.reason_codes == (KillSwitchGateReason.ACCOUNT_ACTIVE,)
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active
    assert repository.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A)[
        -1
    ].event_type is (KillSwitchEventType.ORDER_BLOCKED)

    approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=5),
        expires_at=T0 + timedelta(minutes=10),
    )
    recovery = KillSwitchRecoveryRequest.build(
        request_id="recover-after-audit-failure",
        idempotency_key="recover-after-audit-failure",
        approval=approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=6),
    )
    recovered = service.recover(recovery)

    assert recovered.state_after.status is KillSwitchStatus.INACTIVE
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == recovered.state_after
    assert repository.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A)[
        -1
    ].event_type is (KillSwitchEventType.RECOVERED)
    assert [event.action for event in sink.calls] == [
        "INITIALIZED",
        "INITIALIZED",
        "ACTIVATED",
        "ORDER_BLOCKED",
        "RECOVERED",
    ]


def test_guard_holds_repository_lock_and_linearizes_concurrent_activation_after_submit() -> None:
    repository = _ActivationProbeRepository()
    service = _service(repository)
    _initialize_gate_pair(service)
    activation = _activation_request(
        _incident(label="concurrent"),
        key="concurrent-activation",
    )
    completed = ThreadEvent()
    errors: list[BaseException] = []

    def activate() -> None:
        try:
            service.activate(activation)
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            completed.set()

    worker = Thread(target=activate, name="kill-switch-activation", daemon=True)
    with service.guard_new_order(
        _gate_request(label="concurrent-allowed"),
        _system_actor(),
    ) as decision:
        assert decision.allowed
        worker.start()
        assert repository.activation_attempted.wait(timeout=2)
        assert not completed.wait(timeout=0.1)
        assert (
            repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A).status
            is KillSwitchStatus.INACTIVE
        )

    assert completed.wait(timeout=2)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert not errors
    assert (
        repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A).status is KillSwitchStatus.ACTIVE
    )
    after_activation = _blocked(
        service,
        _gate_request(label="concurrent-after-activation"),
    )
    assert after_activation.decision.reason_codes == (KillSwitchGateReason.ACCOUNT_ACTIVE,)


def test_complete_required_reconciliation_result_automatically_activates_account_scope() -> None:
    _, required = _reconciliation_results()
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)
    _initialize(
        service,
        scope=KillSwitchScope.ACCOUNT,
        account_id=required.account_id,
        label="reconciliation-account",
        changed_at=required.reconciled_at - timedelta(seconds=1),
    )
    actor = _system_actor()

    transition = service.activate_from_reconciliation(
        result=required,
        actor=actor,
        request_id="activate-from-reconciliation",
        idempotency_key="activate-from-reconciliation-key",
        requested_at=required.reconciled_at + timedelta(seconds=1),
    )

    assert transition.incident is not None
    assert transition.incident.scope is KillSwitchScope.ACCOUNT
    assert transition.incident.account_id == required.account_id
    assert transition.incident.trigger_source is KillSwitchTriggerSource.RECONCILIATION
    assert transition.incident.source_reference_hash == required.result_hash
    assert transition.incident.evidence_hashes == required.stop_signal.trigger_hashes
    assert set(transition.incident.reason_codes) == {
        code.value for code in required.stop_signal.reason_codes
    }
    assert transition.state_after.status is KillSwitchStatus.ACTIVE
    assert (
        repository.get_state(KillSwitchScope.ACCOUNT, required.account_id) == transition.state_after
    )


@pytest.mark.parametrize("authorization_result", ["deny", 1, object()])
def test_recovery_rejects_truthy_nonboolean_authorization(authorization_result: object) -> None:
    repository = InMemoryKillSwitchRepository(clock=lambda: T0 + timedelta(minutes=5))
    service = _service(repository)
    _initialize(service, scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A, label="auth-type")
    active = service.activate(
        _activation_request(_incident(label="auth-type"), key="auth-type")
    ).state_after
    approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )
    request = KillSwitchRecoveryRequest.build(
        request_id="auth-type-recover",
        idempotency_key="auth-type-recover",
        approval=approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=4),
    )

    class InvalidAuthorizer:
        def authorize_recovery(self, request: KillSwitchRecoveryRequest) -> bool:
            return authorization_result  # type: ignore[return-value]

    invalid = KillSwitchService(repository=repository, recovery_authorizer=InvalidAuthorizer())
    with pytest.raises(KillSwitchAuthorizationError, match="authenticated external"):
        invalid.recover(request)
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active


def test_recovery_checks_expiry_after_waiting_for_order_lock() -> None:
    now = [T0 + timedelta(minutes=5)]
    clock_called = ThreadEvent()

    def clock() -> datetime:
        clock_called.set()
        return now[0]

    repository = InMemoryKillSwitchRepository(clock=clock)
    service = _service(repository, recovery_authorizer=_RecoveryAuthorizer())
    _initialize_gate_pair(service, ACCOUNT_B)
    _initialize(service, scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A, label="expiry-lock")
    active = service.activate(
        _activation_request(_incident(label="expiry-lock"), key="expiry-lock")
    ).state_after
    approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )
    request = KillSwitchRecoveryRequest.build(
        request_id="expiry-lock-recover",
        idempotency_key="expiry-lock-recover",
        approval=approval,
        actor=_risk_admin(),
        requested_at=T0 + timedelta(minutes=4),
    )
    attempted = ThreadEvent()
    errors: list[Exception] = []

    def recover() -> None:
        attempted.set()
        try:
            service.recover(request)
        except Exception as error:
            errors.append(error)

    worker = Thread(target=recover, daemon=True)
    with service.guard_new_order(
        _gate_request(account_id=ACCOUNT_B, label="hold-expiry-lock"), _system_actor()
    ):
        worker.start()
        assert attempted.wait(timeout=2)
        assert not clock_called.wait(timeout=0.05)
        now[0] = approval.expires_at
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KillSwitchAuthorizationError)
    assert "expired before atomic commit" in str(errors[0])
    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active
    assert all(
        event.event_type is not KillSwitchEventType.RECOVERED for event in repository.events()
    )


def test_same_thread_activation_cannot_reenter_allowed_order_guard() -> None:
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)
    _initialize_gate_pair(service)
    request = _activation_request(_incident(label="reentrant"), key="reentrant")
    with service.guard_new_order(_gate_request(label="reentrant-guard"), _system_actor()):
        with pytest.raises(KillSwitchRepositoryConflict, match="cannot reenter"):
            service.activate(request)
        assert (
            repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A).status
            is KillSwitchStatus.INACTIVE
        )
    assert service.activate(request).state_after.status is KillSwitchStatus.ACTIVE


def test_failed_audit_projection_cannot_reattribute_a_cached_block_to_retry_actor() -> None:
    repository = InMemoryKillSwitchRepository()
    sink = _FailingAuditSink()
    service = _service(repository, audit_sink=sink)
    _initialize_gate_pair(service)
    service.activate(_activation_request(_incident(label="audit-actor"), key="audit-actor"))
    request = _gate_request(label="audit-actor")
    first = _blocked(service, request, actor=_system_actor())
    calls_before = tuple(sink.calls)
    replay = _blocked(service, request, actor=_risk_admin())
    assert replay.decision == first.decision
    assert tuple(sink.calls) == calls_before
    _blocked(service, request, actor=_system_actor())
    assert len(sink.calls) == len(calls_before) + 1
    assert sink.calls[-1].metadata["actor_hash"] == _system_actor().actor_hash


def test_gate_retry_ignores_only_time_and_rejects_changed_source_order() -> None:
    repository = InMemoryKillSwitchRepository()
    service = _service(repository)
    _initialize_gate_pair(service)
    request = _gate_request(label="stable-source")
    later = _gate_request(
        label="stable-source", checked_at=request.checked_at + timedelta(seconds=1)
    )
    assert later.operation_hash == request.operation_hash
    assert later.request_hash != request.request_hash
    with service.guard_new_order(request, _system_actor()):
        pass
    with service.guard_new_order(later, _system_actor()) as decision:
        assert decision.request_hash == later.request_hash
    different = KillSwitchGateRequest.build(
        request_id=later.request_id,
        account_id=later.account_id,
        idempotency_key=later.idempotency_key,
        batch_hash=later.batch_hash,
        source_request_hash=_digest("different-source-order"),
        checked_at=later.checked_at,
    )
    assert different.operation_hash != request.operation_hash
    error = _blocked(service, different)
    assert isinstance(error.cause, KillSwitchIdempotencyConflict)


def test_new_incident_queued_before_recovery_reactivates_after_recovery_commits() -> None:
    repository = InMemoryKillSwitchRepository(clock=lambda: T0 + timedelta(minutes=7))
    service = _service(repository, recovery_authorizer=_RecoveryAuthorizer())
    _initialize_gate_pair(service)
    active = service.activate(
        _activation_request(_incident(label="first-incident"), key="first-incident")
    ).state_after
    approval = KillSwitchRecoveryApproval.build(
        state=active,
        approver=_risk_admin(),
        approved_at=T0 + timedelta(minutes=3),
        expires_at=T0 + timedelta(minutes=10),
    )
    # This independent incident exists before recovery commits, but its state
    # transaction has not acquired the lock yet.
    queued = _activation_request(
        _incident(label="queued-incident", triggered_at=T0 + timedelta(minutes=4)),
        key="queued-incident",
        requested_at=T0 + timedelta(minutes=5),
    )
    recovery = service.recover(
        KillSwitchRecoveryRequest.build(
            request_id="recover-before-queued",
            idempotency_key="recover-before-queued",
            approval=approval,
            actor=_risk_admin(),
            requested_at=T0 + timedelta(minutes=6),
        )
    )
    assert recovery.state_after.status is KillSwitchStatus.INACTIVE
    reactivated = service.activate(queued)
    assert reactivated.state_after.status is KillSwitchStatus.ACTIVE
    assert reactivated.state_after.revision == recovery.state_after.revision + 1
    assert reactivated.state_after.changed_at >= recovery.state_after.changed_at
    assert reactivated.incident == queued.incident
    assert _blocked(
        service, _gate_request(label="after-queued-incident")
    ).decision.reason_codes == (KillSwitchGateReason.ACCOUNT_ACTIVE,)


def test_reconciliation_adapter_rejects_nonrequired_wrong_actor_and_detached_signal() -> None:
    clean, required = _reconciliation_results()
    requested_at = required.reconciled_at + timedelta(seconds=1)

    with pytest.raises(ValueError, match="does not require"):
        activation_request_from_reconciliation(
            result=clean,
            actor=_system_actor(),
            request_id="clean-reconciliation",
            idempotency_key="clean-reconciliation-key",
            requested_at=clean.reconciled_at + timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="SYSTEM actor"):
        activation_request_from_reconciliation(
            result=required,
            actor=_risk_admin(),
            request_id="human-reconciliation",
            idempotency_key="human-reconciliation-key",
            requested_at=requested_at,
        )
    with pytest.raises((TypeError, ValueError), match=r"ReconciliationResult|complete"):
        activation_request_from_reconciliation(
            result=required.stop_signal,  # type: ignore[arg-type]
            actor=_system_actor(),
            request_id="detached-stop-signal",
            idempotency_key="detached-stop-signal-key",
            requested_at=requested_at,
        )
