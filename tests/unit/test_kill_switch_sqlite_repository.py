"""Durability and concurrency tests for the SQLite kill-switch repository."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock

import pytest

from quant_agent.regime.contracts import stable_hash
from quant_agent.risk.kill_switch import (
    KillSwitchActivationRequest,
    KillSwitchActor,
    KillSwitchActorKind,
    KillSwitchActorRole,
    KillSwitchAuditEvent,
    KillSwitchAuthorizationError,
    KillSwitchConcurrentUpdate,
    KillSwitchEventType,
    KillSwitchGateReason,
    KillSwitchGateRequest,
    KillSwitchGateStatus,
    KillSwitchIdempotencyConflict,
    KillSwitchIncident,
    KillSwitchIncidentNotFound,
    KillSwitchRecoveryApproval,
    KillSwitchRecoveryRequest,
    KillSwitchRepositoryError,
    KillSwitchScope,
    KillSwitchState,
    KillSwitchStateNotFound,
    KillSwitchStatus,
    KillSwitchTransition,
    KillSwitchTriggerSource,
)
from quant_agent.risk.kill_switch.sqlite_repository import SQLiteKillSwitchRepository

T0 = datetime(2026, 9, 7, 1, 0, tzinfo=UTC)
ACCOUNT_A = "sqlite-account-a"
ACCOUNT_B = "sqlite-account-b"

SYSTEM = KillSwitchActor.build(
    actor_id="sqlite-kill-switch-controller",
    kind=KillSwitchActorKind.SYSTEM,
    role=KillSwitchActorRole.SYSTEM,
)
RISK_ADMIN = KillSwitchActor.build(
    actor_id="sqlite-risk-admin",
    kind=KillSwitchActorKind.HUMAN,
    role=KillSwitchActorRole.RISK_ADMIN,
)


class _MutableClock:
    def __init__(self, current: datetime) -> None:
        self._current = current
        self._lock = Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._current

    def set(self, current: datetime) -> None:
        with self._lock:
            self._current = current


def _digest(label: str) -> str:
    return stable_hash({"sqlite-kill-switch-fixture": label})


def _repository(
    path: Path,
    *,
    clock_at: datetime = T0 + timedelta(minutes=30),
) -> SQLiteKillSwitchRepository:
    return SQLiteKillSwitchRepository(path, clock=lambda: clock_at)


def _initialize(
    repository: SQLiteKillSwitchRepository,
    *,
    scope: KillSwitchScope,
    account_id: str | None,
    key: str,
) -> KillSwitchTransition:
    return repository.initialize(
        scope=scope,
        account_id=account_id,
        actor=SYSTEM,
        changed_at=T0,
        idempotency_key=key,
    )


def _initialize_gate_pair(
    repository: SQLiteKillSwitchRepository,
    account_id: str = ACCOUNT_A,
) -> None:
    _initialize(
        repository,
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        key="initialize:global",
    )
    _initialize(
        repository,
        scope=KillSwitchScope.ACCOUNT,
        account_id=account_id,
        key=f"initialize:{account_id}",
    )


def _incident(
    label: str,
    *,
    scope: KillSwitchScope = KillSwitchScope.ACCOUNT,
    account_id: str | None = ACCOUNT_A,
    triggered_at: datetime = T0 + timedelta(minutes=1),
) -> KillSwitchIncident:
    evidence_hash = _digest(f"evidence:{label}")
    return KillSwitchIncident.build(
        scope=scope,
        account_id=account_id,
        trigger_source=KillSwitchTriggerSource.RISK,
        reason_codes=(f"{label.upper()}_LIMIT",),
        evidence_hashes=(evidence_hash,),
        source_reference_hash=_digest(f"source:{label}"),
        summary=f"SQLite safety incident {label}",
        triggered_at=triggered_at,
    )


def _activation(
    incident: KillSwitchIncident,
    *,
    key: str,
    requested_at: datetime = T0 + timedelta(minutes=2),
) -> KillSwitchActivationRequest:
    return KillSwitchActivationRequest.build(
        request_id=f"activation:{key}",
        idempotency_key=key,
        incident=incident,
        actor=SYSTEM,
        requested_at=requested_at,
    )


def _recovery(
    state: KillSwitchState,
    *,
    key: str = "recover:account-a",
    approved_at: datetime = T0 + timedelta(minutes=3),
    expires_at: datetime = T0 + timedelta(minutes=10),
    requested_at: datetime = T0 + timedelta(minutes=4),
) -> KillSwitchRecoveryRequest:
    approval = KillSwitchRecoveryApproval.build(
        state=state,
        approver=RISK_ADMIN,
        approved_at=approved_at,
        expires_at=expires_at,
    )
    return KillSwitchRecoveryRequest.build(
        request_id=f"recovery:{key}",
        idempotency_key=key,
        approval=approval,
        actor=RISK_ADMIN,
        requested_at=requested_at,
    )


def _gate(
    label: str,
    *,
    account_id: str = ACCOUNT_A,
    idempotency_key: str | None = None,
    checked_at: datetime = T0 + timedelta(minutes=5),
    source_label: str | None = None,
) -> KillSwitchGateRequest:
    source = source_label or label
    return KillSwitchGateRequest.build(
        request_id=f"gate:{label}",
        account_id=account_id,
        idempotency_key=idempotency_key or f"gate-key:{label}",
        batch_hash=_digest(f"batch:{source}"),
        source_request_hash=_digest(f"source-request:{source}"),
        checked_at=checked_at,
    )


def _assert_local_chain(events: tuple[KillSwitchAuditEvent, ...]) -> None:
    previous_hash: str | None = None
    for event in events:
        assert event.previous_event_hash == previous_hash
        previous_hash = event.event_hash


def test_reopen_preserves_active_state_incidents_events_and_operation_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable-active.db"
    repository = _repository(path)
    _initialize_gate_pair(repository)
    first_incident = _incident("gross-exposure")
    second_incident = _incident(
        "drawdown",
        triggered_at=T0 + timedelta(minutes=3),
    )
    first_request = _activation(first_incident, key="activate:gross-exposure")
    second_request = _activation(
        second_incident,
        key="activate:drawdown",
        requested_at=T0 + timedelta(minutes=4),
    )
    first = repository.activate(first_request)
    second = repository.activate(second_request)

    reopened = _repository(path)

    assert reopened.path == path.resolve()
    assert reopened.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == second.state_after
    assert reopened.get_incident(first_incident.incident_id) == first_incident
    assert reopened.get_incident(second_incident.incident_id) == second_incident
    assert reopened.activate(first_request) == first
    assert reopened.activate(second_request) == second
    account_events = reopened.events(
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
    )
    assert tuple(event.event_type for event in account_events) == (
        KillSwitchEventType.INITIALIZED,
        KillSwitchEventType.ACTIVATED,
        KillSwitchEventType.TRIGGER_ADDED,
    )
    _assert_local_chain(account_events)


def test_scope_identity_isolates_shared_idempotency_keys_and_event_chains(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scope-isolation.db"
    repository = _repository(path)
    for scope, account_id in (
        (KillSwitchScope.GLOBAL, None),
        (KillSwitchScope.ACCOUNT, ACCOUNT_A),
        (KillSwitchScope.ACCOUNT, ACCOUNT_B),
    ):
        _initialize(
            repository,
            scope=scope,
            account_id=account_id,
            key="shared-initialize-key",
        )

    global_request = _activation(
        _incident("global", scope=KillSwitchScope.GLOBAL, account_id=None),
        key="shared-activation-key",
    )
    account_request = _activation(
        _incident("account-a"),
        key="shared-activation-key",
    )
    global_transition = repository.activate(global_request)
    account_transition = repository.activate(account_request)

    reopened = _repository(path)
    assert reopened.get_state(KillSwitchScope.GLOBAL) == global_transition.state_after
    assert reopened.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == account_transition.state_after
    assert (
        reopened.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_B).status is KillSwitchStatus.INACTIVE
    )
    global_events = reopened.events(scope=KillSwitchScope.GLOBAL)
    account_a_events = reopened.events(
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
    )
    account_b_events = reopened.events(
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_B,
    )
    assert len(global_events) == 2
    assert len(account_a_events) == 2
    assert len(account_b_events) == 1
    _assert_local_chain(global_events)
    _assert_local_chain(account_a_events)
    _assert_local_chain(account_b_events)


def test_recovery_cas_is_durable_and_exactly_replayed_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "durable-recovery.db"
    repository = _repository(path, clock_at=T0 + timedelta(minutes=5))
    _initialize_gate_pair(repository)
    active = repository.activate(_activation(_incident("recovery"), key="activate:recovery"))
    request = _recovery(active.state_after)

    recovered = repository.recover(request)
    reopened = _repository(path, clock_at=T0 + timedelta(days=1))

    assert recovered.state_after.status is KillSwitchStatus.INACTIVE
    assert reopened.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == recovered.state_after
    # A committed operation remains replayable after its approval expires.
    assert reopened.recover(request) == recovered
    assert tuple(
        event.event_type
        for event in reopened.events(
            scope=KillSwitchScope.ACCOUNT,
            account_id=ACCOUNT_A,
        )
    ) == (
        KillSwitchEventType.INITIALIZED,
        KillSwitchEventType.ACTIVATED,
        KillSwitchEventType.RECOVERED,
    )


def test_recovery_rejects_approval_for_state_superseded_by_another_trigger(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "stale-recovery.db", clock_at=T0 + timedelta(minutes=7))
    _initialize_gate_pair(repository)
    first = repository.activate(_activation(_incident("first"), key="activate:first"))
    stale_request = _recovery(
        first.state_after,
        approved_at=T0 + timedelta(minutes=3),
        requested_at=T0 + timedelta(minutes=4),
        expires_at=T0 + timedelta(minutes=9),
    )
    current = repository.activate(
        _activation(
            _incident("second", triggered_at=T0 + timedelta(minutes=5)),
            key="activate:second",
            requested_at=T0 + timedelta(minutes=6),
        )
    )

    with pytest.raises(KillSwitchConcurrentUpdate):
        repository.recover(stale_request)

    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == current.state_after


@pytest.mark.parametrize(
    "processed_at",
    (
        T0 + timedelta(minutes=3, seconds=59),
        T0 + timedelta(minutes=10),
        T0 + timedelta(minutes=11),
    ),
)
def test_recovery_runtime_clock_rejects_pre_request_or_expired_approval(
    tmp_path: Path,
    processed_at: datetime,
) -> None:
    repository = _repository(
        tmp_path / f"invalid-clock-{processed_at.minute}.db",
        clock_at=processed_at,
    )
    _initialize_gate_pair(repository)
    active = repository.activate(_activation(_incident("clock"), key="activate:clock"))
    request = _recovery(active.state_after)

    with pytest.raises(KillSwitchAuthorizationError):
        repository.recover(request)

    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active.state_after


def test_missing_scope_fails_closed_and_replays_first_decision_across_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-scope.db"
    repository = _repository(path)
    _initialize(
        repository,
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        key="initialize:global",
    )
    first_request = _gate("missing-account")

    with repository.guard_new_order(first_request, SYSTEM) as first:
        assert first.status is KillSwitchGateStatus.BLOCKED
        assert first.reason_codes == (KillSwitchGateReason.STATE_UNAVAILABLE,)
        assert first.audit_event_hash is None

    retry = _gate(
        "missing-account",
        checked_at=first_request.checked_at + timedelta(minutes=1),
    )
    reopened = _repository(path)
    with reopened.guard_new_order(retry, SYSTEM) as replayed:
        assert replayed == first
        assert replayed.request_hash == first_request.request_hash
    assert reopened.events(scope=KillSwitchScope.GLOBAL)[-1].event_type is (
        KillSwitchEventType.INITIALIZED
    )


def test_active_block_and_audit_event_remain_sticky_after_recovery_and_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sticky-block.db"
    repository = _repository(path, clock_at=T0 + timedelta(minutes=8))
    _initialize_gate_pair(repository)
    active = repository.activate(_activation(_incident("block"), key="activate:block"))
    first_request = _gate("sticky-block", checked_at=T0 + timedelta(minutes=3))

    with repository.guard_new_order(first_request, SYSTEM) as blocked:
        assert blocked.status is KillSwitchGateStatus.BLOCKED
        assert blocked.reason_codes == (KillSwitchGateReason.ACCOUNT_ACTIVE,)
        assert blocked.blocking_state_hashes == (active.state_after.state_hash,)
        assert blocked.incident_ids == (active.incident.incident_id,)
        assert blocked.audit_event_hash is not None

    repository.recover(
        _recovery(
            active.state_after,
            approved_at=T0 + timedelta(minutes=4),
            requested_at=T0 + timedelta(minutes=5),
            expires_at=T0 + timedelta(minutes=9),
        )
    )
    events_before_replay = repository.events(
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
    )
    assert tuple(event.event_type for event in events_before_replay) == (
        KillSwitchEventType.INITIALIZED,
        KillSwitchEventType.ACTIVATED,
        KillSwitchEventType.ORDER_BLOCKED,
        KillSwitchEventType.RECOVERED,
    )

    reopened = _repository(path)
    retry = _gate(
        "sticky-block",
        checked_at=first_request.checked_at + timedelta(minutes=20),
    )
    with reopened.guard_new_order(retry, SYSTEM) as replayed:
        assert replayed == blocked
    assert (
        reopened.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A) == events_before_replay
    )


def test_gate_same_key_different_order_content_conflicts_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "gate-conflict.db"
    repository = _repository(path)
    first_request = _gate("reserved", idempotency_key="shared-gate-key")
    with repository.guard_new_order(first_request, SYSTEM) as decision:
        assert not decision.allowed

    conflicting = _gate(
        "conflicting",
        idempotency_key="shared-gate-key",
        source_label="different-order",
    )
    reopened = _repository(path)
    with (
        pytest.raises(KillSwitchIdempotencyConflict),
        reopened.guard_new_order(conflicting, SYSTEM),
    ):
        pass


def test_allowed_guard_reservation_survives_caller_failure(tmp_path: Path) -> None:
    path = tmp_path / "allowed-reservation.db"
    repository = _repository(path)
    _initialize_gate_pair(repository)
    first_request = _gate("allowed-error", idempotency_key="allowed-reservation")

    with (
        pytest.raises(RuntimeError, match="downstream commit failed"),
        repository.guard_new_order(first_request, SYSTEM) as decision,
    ):
        assert decision.allowed
        raise RuntimeError("downstream commit failed")

    reopened = _repository(path)
    same_operation_retry = _gate(
        "allowed-error",
        idempotency_key="allowed-reservation",
        checked_at=first_request.checked_at + timedelta(minutes=1),
    )
    with reopened.guard_new_order(same_operation_retry, SYSTEM) as retried:
        assert retried.allowed
    competing = _gate(
        "allowed-competitor",
        idempotency_key="allowed-reservation",
        source_label="competing-order",
    )
    with (
        pytest.raises(KillSwitchIdempotencyConflict),
        reopened.guard_new_order(competing, SYSTEM),
    ):
        pass


def test_allowed_failure_then_activation_persists_later_block_for_same_operation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "allowed-then-sticky-block.db"
    repository = _repository(path)
    _initialize_gate_pair(repository)
    first_request = _gate(
        "allowed-then-blocked",
        checked_at=T0 + timedelta(minutes=3),
    )
    with (
        pytest.raises(RuntimeError, match="paper commit failed"),
        repository.guard_new_order(first_request, SYSTEM) as first_decision,
    ):
        assert first_decision.allowed
        raise RuntimeError("paper commit failed")

    repository.activate(
        _activation(
            _incident("after-allowed-failure", triggered_at=T0 + timedelta(minutes=4)),
            key="activate:after-allowed-failure",
            requested_at=T0 + timedelta(minutes=4),
        )
    )
    second_request = _gate(
        "allowed-then-blocked",
        checked_at=T0 + timedelta(minutes=5),
    )
    assert second_request.operation_hash == first_request.operation_hash
    assert second_request.request_hash != first_request.request_hash
    with repository.guard_new_order(second_request, SYSTEM) as blocked:
        assert blocked.status is KillSwitchGateStatus.BLOCKED
        assert blocked.request_hash == second_request.request_hash
        assert blocked.reason_codes == (KillSwitchGateReason.ACCOUNT_ACTIVE,)
    events_after_block = repository.events(
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
    )

    reopened = _repository(path)
    third_request = _gate(
        "allowed-then-blocked",
        checked_at=T0 + timedelta(minutes=6),
    )
    with reopened.guard_new_order(third_request, SYSTEM) as replayed:
        assert replayed == blocked
    assert (
        reopened.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A) == events_after_block
    )


def test_activation_idempotency_conflict_does_not_mutate_committed_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "activation-conflict.db"
    repository = _repository(path)
    _initialize_gate_pair(repository)
    first = repository.activate(_activation(_incident("identity-one"), key="shared-activation"))
    conflicting = _activation(
        _incident("identity-two"),
        key="shared-activation",
        requested_at=T0 + timedelta(minutes=3),
    )

    reopened = _repository(path)
    with pytest.raises(KillSwitchIdempotencyConflict):
        reopened.activate(conflicting)

    assert reopened.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == first.state_after
    assert len(reopened.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A)) == 2


def test_activation_queued_before_recovery_reactivates_with_monotonic_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queued-activation-after-recovery.db"
    repository = _repository(path, clock_at=T0 + timedelta(minutes=7))
    _initialize_gate_pair(repository)
    active = repository.activate(
        _activation(_incident("initial"), key="activate:initial")
    ).state_after
    queued = _activation(
        _incident("queued", triggered_at=T0 + timedelta(minutes=4)),
        key="activate:queued",
        requested_at=T0 + timedelta(minutes=5),
    )
    recovered = repository.recover(
        _recovery(
            active,
            key="recover:before-queued",
            approved_at=T0 + timedelta(minutes=3),
            requested_at=T0 + timedelta(minutes=6),
            expires_at=T0 + timedelta(minutes=10),
        )
    )

    reactivated = repository.activate(queued)

    assert reactivated.state_after.status is KillSwitchStatus.ACTIVE
    assert reactivated.state_after.revision == recovered.state_after.revision + 1
    assert reactivated.state_after.changed_at == recovered.state_after.changed_at
    assert reactivated.incident == queued.incident
    assert reactivated.audit_event.occurred_at == recovered.state_after.changed_at
    reopened = _repository(path)
    assert reopened.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == reactivated.state_after


def test_cross_connection_allowed_guard_linearizes_before_activation(tmp_path: Path) -> None:
    path = tmp_path / "guard-activation-linearization.db"
    gate_repository = _repository(path)
    _initialize_gate_pair(gate_repository)
    activation_repository = _repository(path)
    request = _activation(_incident("concurrent"), key="activate:concurrent")
    gate_request = _gate("concurrent", checked_at=T0 + timedelta(minutes=3))
    activation_started = Event()

    def activate() -> KillSwitchTransition:
        activation_started.set()
        return activation_repository.activate(request)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with gate_repository.guard_new_order(gate_request, SYSTEM) as decision:
            assert decision.allowed
            future = pool.submit(activate)
            assert activation_started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
            assert (
                gate_repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A).status
                is KillSwitchStatus.INACTIVE
            )
        activated = future.result(timeout=5)

    assert activated.state_after.status is KillSwitchStatus.ACTIVE
    assert gate_repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == activated.state_after


def test_recovery_rechecks_expiry_after_waiting_for_another_connection_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recovery-expiry-linearization.db"
    clock = _MutableClock(T0 + timedelta(minutes=6))
    gate_repository = SQLiteKillSwitchRepository(path, clock=clock)
    _initialize_gate_pair(gate_repository, ACCOUNT_B)
    _initialize(
        gate_repository,
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
        key="initialize:account-a",
    )
    active = gate_repository.activate(
        _activation(_incident("expiry-race"), key="activate:expiry-race")
    )
    recovery_request = _recovery(
        active.state_after,
        approved_at=T0 + timedelta(minutes=3),
        requested_at=T0 + timedelta(minutes=5),
        expires_at=T0 + timedelta(minutes=7),
    )
    recovery_repository = SQLiteKillSwitchRepository(path, clock=clock)
    gate_request = _gate(
        "hold-writer-lock",
        account_id=ACCOUNT_B,
        checked_at=T0 + timedelta(minutes=6),
    )
    recovery_started = Event()

    def recover() -> KillSwitchTransition:
        recovery_started.set()
        return recovery_repository.recover(recovery_request)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with gate_repository.guard_new_order(gate_request, SYSTEM) as decision:
            assert decision.allowed
            future = pool.submit(recover)
            assert recovery_started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
            clock.set(recovery_request.approval.expires_at)
        with pytest.raises(KillSwitchAuthorizationError):
            future.result(timeout=5)

    assert gate_repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == active.state_after
    assert KillSwitchEventType.RECOVERED not in {
        event.event_type
        for event in gate_repository.events(
            scope=KillSwitchScope.ACCOUNT,
            account_id=ACCOUNT_A,
        )
    }


def test_missing_state_and_incident_queries_fail_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "missing-entities.db")

    with pytest.raises(KillSwitchStateNotFound):
        repository.get_state(KillSwitchScope.GLOBAL)
    with pytest.raises(KillSwitchStateNotFound):
        repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A)
    with pytest.raises(KillSwitchIncidentNotFound):
        repository.get_incident("incident:missing")


@pytest.mark.parametrize("schema_kind", ("future-version", "missing-columns", "missing-table"))
def test_repository_rejects_unknown_or_malformed_schema(
    tmp_path: Path,
    schema_kind: str,
) -> None:
    path = tmp_path / f"bad-schema-{schema_kind}.db"
    if schema_kind == "future-version":
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA user_version = 99")
    elif schema_kind == "missing-columns":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE kill_switch_states (scope TEXT)")
            connection.execute("PRAGMA user_version = 1")
    else:
        repository = _repository(path)
        del repository
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TABLE kill_switch_gate_requests")

    with pytest.raises(KillSwitchRepositoryError):
        _repository(path)


@pytest.mark.parametrize(
    ("target", "column", "replacement"),
    (
        ("state", "state_json", b"{"),
        ("state", "state_hash", "0" * 64),
        ("incident", "incident_json", b"{"),
        ("incident", "incident_hash", "1" * 64),
        ("event", "event_json", b"{"),
        ("event", "previous_event_hash", "2" * 64),
        ("operation", "transition_json", b"{"),
    ),
)
def test_corrupt_json_hash_or_row_binding_never_returns_authoritative_data(
    tmp_path: Path,
    target: str,
    column: str,
    replacement: bytes | str,
) -> None:
    path = tmp_path / f"corrupt-{target}-{column}.db"
    repository = _repository(path)
    _initialize_gate_pair(repository)
    incident = _incident(f"corrupt-{target}-{column}")
    request = _activation(incident, key=f"activate:{target}:{column}")
    repository.activate(request)
    table = {
        "state": "kill_switch_states",
        "incident": "kill_switch_incidents",
        "event": "kill_switch_events",
        "operation": "kill_switch_operations",
    }[target]
    where = (
        "scope = 'ACCOUNT' AND account_id = ?"
        if target == "state"
        else (
            "incident_id = ?"
            if target == "incident"
            else (
                "event_seq = (SELECT max(event_seq) FROM kill_switch_events)"
                if target == "event"
                else "idempotency_key = ?"
            )
        )
    )
    parameter = (
        ACCOUNT_A
        if target == "state"
        else incident.incident_id
        if target == "incident"
        else request.idempotency_key
    )
    with sqlite3.connect(path) as connection:
        parameters = (replacement,) if target == "event" else (replacement, parameter)
        connection.execute(f"UPDATE {table} SET {column} = ? WHERE {where}", parameters)

    with pytest.raises(KillSwitchRepositoryError):
        reopened = _repository(path)
        if target == "state":
            reopened.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A)
        elif target == "incident":
            reopened.get_incident(incident.incident_id)
        elif target == "event":
            reopened.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A)
        else:
            reopened.activate(request)


def test_corrupt_persisted_block_decision_fails_closed_instead_of_recomputing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corrupt-block-decision.db"
    repository = _repository(path)
    _initialize_gate_pair(repository)
    repository.activate(_activation(_incident("corrupt-block"), key="activate:block"))
    request = _gate("corrupt-block", checked_at=T0 + timedelta(minutes=3))
    with repository.guard_new_order(request, SYSTEM) as decision:
        assert not decision.allowed
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE kill_switch_gate_requests SET decision_json = ?",
            (b"{",),
        )

    with pytest.raises(KillSwitchRepositoryError):
        reopened = _repository(path)
        with reopened.guard_new_order(request, SYSTEM):
            pass


def test_sqlite_failure_rolls_back_incident_event_state_and_operation(tmp_path: Path) -> None:
    path = tmp_path / "atomic-rollback.db"
    repository = _repository(path)
    _initialize_gate_pair(repository)
    state_before = repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A)
    events_before = repository.events(
        scope=KillSwitchScope.ACCOUNT,
        account_id=ACCOUNT_A,
    )
    incident = _incident("forced-rollback")
    request = _activation(incident, key="activate:forced-rollback")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_kill_event BEFORE INSERT ON kill_switch_events "
            "BEGIN SELECT RAISE(ABORT, 'forced kill-switch event failure'); END"
        )

    with pytest.raises(KillSwitchRepositoryError):
        repository.activate(request)

    assert repository.get_state(KillSwitchScope.ACCOUNT, ACCOUNT_A) == state_before
    assert repository.events(scope=KillSwitchScope.ACCOUNT, account_id=ACCOUNT_A) == events_before
    with pytest.raises(KillSwitchIncidentNotFound):
        repository.get_incident(incident.incident_id)
