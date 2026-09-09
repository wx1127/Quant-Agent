"""Durable SQLite repository for the global and account kill switch."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import local

from pydantic import TypeAdapter, ValidationError

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
from .repository import (
    KillSwitchAuthorizationError,
    KillSwitchConcurrentUpdate,
    KillSwitchIdempotencyConflict,
    KillSwitchIncidentConflict,
    KillSwitchIncidentNotFound,
    KillSwitchRepositoryConflict,
    KillSwitchRepositoryError,
    KillSwitchStateNotFound,
)

_SCHEMA_VERSION = 1
_INITIALIZE = "INITIALIZE"
_ACTIVATE = "ACTIVATE"
_RECOVER = "RECOVER"

_ACTOR_ADAPTER: TypeAdapter[KillSwitchActor] = TypeAdapter(KillSwitchActor)
_ACTIVATION_REQUEST_ADAPTER: TypeAdapter[KillSwitchActivationRequest] = TypeAdapter(
    KillSwitchActivationRequest
)
_RECOVERY_REQUEST_ADAPTER: TypeAdapter[KillSwitchRecoveryRequest] = TypeAdapter(
    KillSwitchRecoveryRequest
)
_GATE_REQUEST_ADAPTER: TypeAdapter[KillSwitchGateRequest] = TypeAdapter(KillSwitchGateRequest)
_STATE_ADAPTER: TypeAdapter[KillSwitchState] = TypeAdapter(KillSwitchState)
_INCIDENT_ADAPTER: TypeAdapter[KillSwitchIncident] = TypeAdapter(KillSwitchIncident)
_EVENT_ADAPTER: TypeAdapter[KillSwitchAuditEvent] = TypeAdapter(KillSwitchAuditEvent)
_TRANSITION_ADAPTER: TypeAdapter[KillSwitchTransition] = TypeAdapter(KillSwitchTransition)
_DECISION_ADAPTER: TypeAdapter[KillSwitchGateDecision] = TypeAdapter(KillSwitchGateDecision)

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS kill_switch_states (
        scope TEXT NOT NULL,
        account_id TEXT NOT NULL,
        state_id TEXT NOT NULL UNIQUE,
        state_hash TEXT NOT NULL UNIQUE,
        revision INTEGER NOT NULL,
        state_json BLOB NOT NULL,
        PRIMARY KEY (scope, account_id),
        CHECK (
            (scope = 'GLOBAL' AND account_id = '')
            OR
            (scope = 'ACCOUNT' AND account_id = trim(account_id) AND length(account_id) > 0)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kill_switch_incidents (
        incident_id TEXT PRIMARY KEY,
        incident_hash TEXT NOT NULL UNIQUE,
        scope TEXT NOT NULL,
        account_id TEXT NOT NULL,
        incident_json BLOB NOT NULL,
        CHECK (
            (scope = 'GLOBAL' AND account_id = '')
            OR
            (scope = 'ACCOUNT' AND account_id = trim(account_id) AND length(account_id) > 0)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kill_switch_events (
        event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        event_hash TEXT NOT NULL UNIQUE,
        scope TEXT NOT NULL,
        account_id TEXT NOT NULL,
        previous_event_hash TEXT,
        state_after_hash TEXT NOT NULL,
        state_revision INTEGER NOT NULL,
        event_json BLOB NOT NULL,
        FOREIGN KEY (previous_event_hash) REFERENCES kill_switch_events(event_hash),
        CHECK (
            (scope = 'GLOBAL' AND account_id = '')
            OR
            (scope = 'ACCOUNT' AND account_id = trim(account_id) AND length(account_id) > 0)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kill_switch_operations (
        scope TEXT NOT NULL,
        account_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        operation_kind TEXT NOT NULL,
        operation_hash TEXT NOT NULL,
        transition_hash TEXT NOT NULL UNIQUE,
        event_hash TEXT NOT NULL UNIQUE,
        transition_json BLOB NOT NULL,
        PRIMARY KEY (scope, account_id, idempotency_key),
        FOREIGN KEY (event_hash) REFERENCES kill_switch_events(event_hash),
        CHECK (operation_kind IN ('INITIALIZE', 'ACTIVATE', 'RECOVER')),
        CHECK (
            (scope = 'GLOBAL' AND account_id = '')
            OR
            (scope = 'ACCOUNT' AND account_id = trim(account_id) AND length(account_id) > 0)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kill_switch_gate_requests (
        account_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        request_json BLOB NOT NULL,
        decision_hash TEXT UNIQUE,
        audit_event_hash TEXT,
        decision_json BLOB,
        PRIMARY KEY (account_id, idempotency_key),
        FOREIGN KEY (audit_event_hash) REFERENCES kill_switch_events(event_hash),
        CHECK (account_id = trim(account_id) AND length(account_id) > 0),
        CHECK (
            (decision_hash IS NULL AND decision_json IS NULL AND audit_event_hash IS NULL)
            OR
            (decision_hash IS NOT NULL AND decision_json IS NOT NULL)
        )
    )
    """,
)

_EXPECTED_COLUMNS = {
    "kill_switch_states": (
        ("scope", "TEXT", 1, 1),
        ("account_id", "TEXT", 1, 2),
        ("state_id", "TEXT", 1, 0),
        ("state_hash", "TEXT", 1, 0),
        ("revision", "INTEGER", 1, 0),
        ("state_json", "BLOB", 1, 0),
    ),
    "kill_switch_incidents": (
        ("incident_id", "TEXT", 0, 1),
        ("incident_hash", "TEXT", 1, 0),
        ("scope", "TEXT", 1, 0),
        ("account_id", "TEXT", 1, 0),
        ("incident_json", "BLOB", 1, 0),
    ),
    "kill_switch_events": (
        ("event_seq", "INTEGER", 0, 1),
        ("event_id", "TEXT", 1, 0),
        ("event_hash", "TEXT", 1, 0),
        ("scope", "TEXT", 1, 0),
        ("account_id", "TEXT", 1, 0),
        ("previous_event_hash", "TEXT", 0, 0),
        ("state_after_hash", "TEXT", 1, 0),
        ("state_revision", "INTEGER", 1, 0),
        ("event_json", "BLOB", 1, 0),
    ),
    "kill_switch_operations": (
        ("scope", "TEXT", 1, 1),
        ("account_id", "TEXT", 1, 2),
        ("idempotency_key", "TEXT", 1, 3),
        ("operation_kind", "TEXT", 1, 0),
        ("operation_hash", "TEXT", 1, 0),
        ("transition_hash", "TEXT", 1, 0),
        ("event_hash", "TEXT", 1, 0),
        ("transition_json", "BLOB", 1, 0),
    ),
    "kill_switch_gate_requests": (
        ("account_id", "TEXT", 1, 1),
        ("idempotency_key", "TEXT", 1, 2),
        ("request_hash", "TEXT", 1, 0),
        ("request_json", "BLOB", 1, 0),
        ("decision_hash", "TEXT", 0, 0),
        ("audit_event_hash", "TEXT", 0, 0),
        ("decision_json", "BLOB", 0, 0),
    ),
}

_REQUIRED_UNIQUE_KEYS: dict[str, set[tuple[str, ...]]] = {
    "kill_switch_states": {
        ("scope", "account_id"),
        ("state_id",),
        ("state_hash",),
    },
    "kill_switch_incidents": {("incident_id",), ("incident_hash",)},
    "kill_switch_events": {("event_id",), ("event_hash",)},
    "kill_switch_operations": {
        ("scope", "account_id", "idempotency_key"),
        ("transition_hash",),
        ("event_hash",),
    },
    "kill_switch_gate_requests": {
        ("account_id", "idempotency_key"),
        ("decision_hash",),
    },
}

_EXPECTED_FOREIGN_KEYS: dict[
    str,
    set[tuple[str, str, str, str, str]],
] = {
    "kill_switch_states": set(),
    "kill_switch_incidents": set(),
    "kill_switch_events": {
        (
            "kill_switch_events",
            "previous_event_hash",
            "event_hash",
            "NO ACTION",
            "NO ACTION",
        )
    },
    "kill_switch_operations": {
        (
            "kill_switch_events",
            "event_hash",
            "event_hash",
            "NO ACTION",
            "NO ACTION",
        )
    },
    "kill_switch_gate_requests": {
        (
            "kill_switch_events",
            "audit_event_hash",
            "event_hash",
            "NO ACTION",
            "NO ACTION",
        )
    },
}

_EXPECTED_TABLE_SQL = {
    table: "".join(statement.lower().split()).replace(
        "createtableifnotexists",
        "createtable",
        1,
    )
    for table, statement in zip(_EXPECTED_COLUMNS, _DDL, strict=True)
}

_StateKey = tuple[KillSwitchScope, str | None]


class SQLiteKillSwitchRepository:
    """File-backed kill switch with durable CAS and cross-process order gating."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] = shanghai_now,
    ) -> None:
        if isinstance(path, str) and not path.strip():
            raise ValueError("kill-switch repository path must be non-empty")
        self._path = Path(path).expanduser().resolve()
        self._clock = clock
        self._guard_local = local()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction(immediate=True) as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
            if (
                row is None
                or len(row) != 1
                or not isinstance(row[0], int)
                or isinstance(row[0], bool)
            ):
                raise KillSwitchRepositoryError("stored kill-switch schema version is invalid")
            schema_version = row[0]
            if schema_version not in {0, _SCHEMA_VERSION}:
                raise KillSwitchRepositoryError(
                    f"unsupported kill-switch repository schema version: {schema_version}"
                )
            if schema_version == 0:
                for statement in _DDL:
                    connection.execute(statement)
                self._verify_schema(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            else:
                self._verify_schema(connection)

    @property
    def path(self) -> Path:
        return self._path

    def initialize(
        self,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        actor: KillSwitchActor,
        changed_at: datetime,
        idempotency_key: str | None = None,
    ) -> KillSwitchTransition:
        """Atomically initialize one inactive state and its first event."""

        self._ensure_not_guarding()
        actor = _round_trip(_ACTOR_ADAPTER, actor, "kill-switch actor")
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
            _non_empty(idempotency_key, "idempotency_key")
            if idempotency_key is not None
            else f"initialize:{state_after.state_id}"
        )
        operation_hash = stable_hash(
            {
                "operation": _INITIALIZE,
                "idempotency_key": effective_key,
                "state_hash": state_after.state_hash,
            }
        )
        key = _state_key(state_after.scope, state_after.account_id)
        with self._transaction(immediate=True) as connection:
            committed = self._committed_operation(
                connection,
                key=key,
                idempotency_key=effective_key,
                operation_kind=_INITIALIZE,
                operation_hash=operation_hash,
            )
            if committed is not None:
                return committed
            if self._state_or_none(connection, *key) is not None:
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
            transition = _round_trip(
                _TRANSITION_ADAPTER,
                KillSwitchTransition.build(
                    operation_hash=operation_hash,
                    state_before=None,
                    state_after=state_after,
                    incident=None,
                    audit_event=event,
                ),
                "kill-switch transition",
            )
            self._insert_state(connection, state_after)
            self._append_event(connection, event)
            self._insert_operation(
                connection,
                key=key,
                idempotency_key=effective_key,
                operation_kind=_INITIALIZE,
                transition=transition,
            )
            return transition

    def activate(self, request: KillSwitchActivationRequest) -> KillSwitchTransition:
        """Atomically persist an incident, active state, event, and operation."""

        self._ensure_not_guarding()
        request = _round_trip(
            _ACTIVATION_REQUEST_ADAPTER,
            request,
            "kill-switch activation request",
        )
        incident = request.incident
        key = _state_key(incident.scope, incident.account_id)
        with self._transaction(immediate=True) as connection:
            committed = self._committed_operation(
                connection,
                key=key,
                idempotency_key=request.idempotency_key,
                operation_kind=_ACTIVATE,
                operation_hash=request.request_hash,
            )
            if committed is not None:
                return committed
            state_before = self._state_or_none(connection, *key)
            if state_before is None:
                raise KillSwitchStateNotFound(
                    f"kill switch is not initialized: {incident.scope.value}:{incident.account_id}"
                )
            self._ensure_incident_compatible(connection, incident)
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
                previous_event_hash=self._latest_event_hash(connection, *key),
            )
            transition = _round_trip(
                _TRANSITION_ADAPTER,
                KillSwitchTransition.build(
                    operation_hash=request.request_hash,
                    state_before=state_before,
                    state_after=state_after,
                    incident=incident,
                    audit_event=event,
                ),
                "kill-switch transition",
            )
            self._insert_incident_if_absent(connection, incident)
            self._append_event(connection, event)
            self._update_state(connection, state_before, state_after)
            self._insert_operation(
                connection,
                key=key,
                idempotency_key=request.idempotency_key,
                operation_kind=_ACTIVATE,
                transition=transition,
            )
            return transition

    def recover(self, request: KillSwitchRecoveryRequest) -> KillSwitchTransition:
        """CAS-recover an active state and persist its event and operation."""

        self._ensure_not_guarding()
        request = _round_trip(
            _RECOVERY_REQUEST_ADAPTER,
            request,
            "kill-switch recovery request",
        )
        approval = request.approval
        key = _state_key(approval.scope, approval.account_id)
        with self._transaction(immediate=True) as connection:
            committed = self._committed_operation(
                connection,
                key=key,
                idempotency_key=request.idempotency_key,
                operation_kind=_RECOVER,
                operation_hash=request.request_hash,
            )
            if committed is not None:
                return committed
            processed_at = ensure_aware(self._clock())
            if processed_at < request.requested_at:
                raise KillSwitchAuthorizationError(
                    "recovery processing time cannot precede its request"
                )
            if processed_at >= approval.expires_at:
                raise KillSwitchAuthorizationError("recovery approval expired before atomic commit")
            state_before = self._state_or_none(connection, *key)
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
                previous_event_hash=self._latest_event_hash(connection, *key),
            )
            transition = _round_trip(
                _TRANSITION_ADAPTER,
                KillSwitchTransition.build(
                    operation_hash=request.request_hash,
                    state_before=state_before,
                    state_after=state_after,
                    incident=None,
                    audit_event=event,
                ),
                "kill-switch transition",
            )
            self._append_event(connection, event)
            self._update_state(connection, state_before, state_after)
            self._insert_operation(
                connection,
                key=key,
                idempotency_key=request.idempotency_key,
                operation_kind=_RECOVER,
                transition=transition,
            )
            return transition

    def get_state(
        self,
        scope: KillSwitchScope,
        account_id: str | None = None,
    ) -> KillSwitchState:
        """Return the current durable state after validating its event chain."""

        key = _state_key(scope, account_id)
        with self._transaction() as connection:
            state = self._state_or_none(connection, *key)
        if state is None:
            raise KillSwitchStateNotFound(f"kill switch is not initialized: {scope.value}:{key[1]}")
        return state

    def get_incident(self, incident_id: str) -> KillSwitchIncident:
        """Return an immutable incident after strict row/JSON validation."""

        normalized_id = _non_empty(incident_id, "incident_id")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT incident_id, incident_hash, scope, account_id, incident_json "
                "FROM kill_switch_incidents WHERE incident_id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                raise KillSwitchIncidentNotFound(f"unknown kill-switch incident: {normalized_id}")
            return _load_bound_incident(row)

    def events(
        self,
        *,
        scope: KillSwitchScope | None = None,
        account_id: str | None = None,
    ) -> tuple[KillSwitchAuditEvent, ...]:
        """Return append-ordered events with every selected local chain verified."""

        normalized_account = _event_filter(scope, account_id)
        query = (
            "SELECT event_seq, event_id, event_hash, scope, account_id, "
            "previous_event_hash, state_after_hash, state_revision, event_json "
            "FROM kill_switch_events"
        )
        parameters: tuple[object, ...] = ()
        if scope is not None:
            query += " WHERE scope = ?"
            parameters = (scope.value,)
            if normalized_account is not None:
                query += " AND account_id = ?"
                parameters = (scope.value, normalized_account)
        query += " ORDER BY event_seq"
        with self._transaction() as connection:
            result = _load_event_chain(connection.execute(query, parameters).fetchall())
            self._validate_event_projection(
                connection,
                result,
                scope=scope,
                encoded_account_id=normalized_account,
            )
            return result

    @contextmanager
    def guard_new_order(
        self,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> Iterator[KillSwitchGateDecision]:
        """Serialize one order admission decision with concurrent state changes."""

        self._ensure_not_guarding()
        request = _round_trip(
            _GATE_REQUEST_ADAPTER,
            request,
            "kill-switch gate request",
        )
        actor = _round_trip(_ACTOR_ADAPTER, actor, "kill-switch actor")
        connection: sqlite3.Connection | None = None
        self._guard_local.active = True
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            stored = self._gate_record(
                connection,
                account_id=request.account_id,
                idempotency_key=request.idempotency_key,
            )
            if stored is not None:
                stored_request, stored_decision = stored
                if stored_request.operation_hash != request.operation_hash:
                    raise KillSwitchIdempotencyConflict(
                        "gate idempotency key belongs to a different new-order request"
                    )
                if stored_decision is not None:
                    connection.commit()
                    yield stored_decision
                    return
            else:
                self._insert_gate_reservation(connection, request)

            decision, event = self._evaluate_gate(connection, request, actor)
            if not decision.allowed:
                if event is not None:
                    self._append_event(connection, event)
                self._store_gate_decision(connection, request, decision)
                connection.commit()
                yield decision
                return

            try:
                yield decision
            except BaseException:
                # The order identity remains reserved even if downstream commit fails.
                connection.commit()
                raise
            else:
                connection.commit()
        except sqlite3.Error as error:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise KillSwitchRepositoryError("SQLite kill-switch gate operation failed") from error
        except BaseException:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise
        finally:
            self._guard_local.active = False
            if connection is not None:
                connection.close()

    def _ensure_not_guarding(self) -> None:
        if bool(getattr(self._guard_local, "active", False)):
            raise KillSwitchRepositoryConflict(
                "kill-switch mutation or nested guard is forbidden inside an order guard"
            )

    def _gate_record(
        self,
        connection: sqlite3.Connection,
        *,
        account_id: str,
        idempotency_key: str,
    ) -> tuple[KillSwitchGateRequest, KillSwitchGateDecision | None] | None:
        row = connection.execute(
            "SELECT account_id, idempotency_key, request_hash, request_json, "
            "decision_hash, audit_event_hash, decision_json "
            "FROM kill_switch_gate_requests "
            "WHERE account_id = ? AND idempotency_key = ?",
            (
                _non_empty(account_id, "account_id"),
                _non_empty(idempotency_key, "idempotency_key"),
            ),
        ).fetchone()
        if row is None:
            return None
        stored_request, decision = _load_bound_gate_record(row)
        if decision is not None:
            self._validate_gate_decision_links(
                connection,
                stored_request,
                decision,
            )
        return stored_request, decision

    @staticmethod
    def _insert_gate_reservation(
        connection: sqlite3.Connection,
        request: KillSwitchGateRequest,
    ) -> None:
        connection.execute(
            "INSERT INTO kill_switch_gate_requests("
            "account_id, idempotency_key, request_hash, request_json"
            ") VALUES (?, ?, ?, ?)",
            (
                request.account_id,
                request.idempotency_key,
                request.request_hash,
                _dump_json(_GATE_REQUEST_ADAPTER, request),
            ),
        )

    @staticmethod
    def _store_gate_decision(
        connection: sqlite3.Connection,
        request: KillSwitchGateRequest,
        decision: KillSwitchGateDecision,
    ) -> None:
        decision = _round_trip(
            _DECISION_ADAPTER,
            decision,
            "kill-switch gate decision",
        )
        if decision.allowed:
            raise KillSwitchRepositoryConflict("allowed gate decisions must not be cached")
        cursor = connection.execute(
            "UPDATE kill_switch_gate_requests "
            "SET request_hash = ?, request_json = ?, decision_hash = ?, "
            "audit_event_hash = ?, decision_json = ? "
            "WHERE account_id = ? AND idempotency_key = ? AND decision_hash IS NULL",
            (
                request.request_hash,
                _dump_json(_GATE_REQUEST_ADAPTER, request),
                decision.decision_hash,
                decision.audit_event_hash,
                _dump_json(_DECISION_ADAPTER, decision),
                request.account_id,
                request.idempotency_key,
            ),
        )
        if cursor.rowcount != 1:
            raise KillSwitchRepositoryConflict(
                "gate decision could not be attached to its reservation"
            )

    def _evaluate_gate(
        self,
        connection: sqlite3.Connection,
        request: KillSwitchGateRequest,
        actor: KillSwitchActor,
    ) -> tuple[KillSwitchGateDecision, KillSwitchAuditEvent | None]:
        global_state = self._state_or_none(connection, KillSwitchScope.GLOBAL, None)
        account_state = self._state_or_none(
            connection,
            KillSwitchScope.ACCOUNT,
            request.account_id,
        )
        if global_state is None or account_state is None:
            return (
                KillSwitchGateDecision.build(
                    request=request,
                    status=KillSwitchGateStatus.BLOCKED,
                    global_state_hash=(
                        global_state.state_hash if global_state is not None else None
                    ),
                    account_state_hash=(
                        account_state.state_hash if account_state is not None else None
                    ),
                    reason_codes=(KillSwitchGateReason.STATE_UNAVAILABLE,),
                ),
                None,
            )
        if (
            request.checked_at < global_state.changed_at
            or request.checked_at < account_state.changed_at
        ):
            return (
                KillSwitchGateDecision.build(
                    request=request,
                    status=KillSwitchGateStatus.BLOCKED,
                    global_state_hash=global_state.state_hash,
                    account_state_hash=account_state.state_hash,
                    reason_codes=(KillSwitchGateReason.STATE_UNAVAILABLE,),
                ),
                None,
            )

        blocking_states = tuple(
            state
            for state in (global_state, account_state)
            if state.status is KillSwitchStatus.ACTIVE
        )
        if not blocking_states:
            return (
                KillSwitchGateDecision.build(
                    request=request,
                    status=KillSwitchGateStatus.ALLOWED,
                    global_state_hash=global_state.state_hash,
                    account_state_hash=account_state.state_hash,
                ),
                None,
            )

        reasons: list[KillSwitchGateReason] = []
        if global_state.status is KillSwitchStatus.ACTIVE:
            reasons.append(KillSwitchGateReason.GLOBAL_ACTIVE)
        if account_state.status is KillSwitchStatus.ACTIVE:
            reasons.append(KillSwitchGateReason.ACCOUNT_ACTIVE)
        primary_state = blocking_states[0]
        previous_event_hash = self._latest_event_hash(
            connection,
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
        return decision, event

    def _state_or_none(
        self,
        connection: sqlite3.Connection,
        scope: KillSwitchScope,
        account_id: str | None,
    ) -> KillSwitchState | None:
        key = _state_key(scope, account_id)
        encoded_account = _encoded_account_id(key[1])
        row = connection.execute(
            "SELECT scope, account_id, state_id, state_hash, revision, state_json "
            "FROM kill_switch_states WHERE scope = ? AND account_id = ?",
            (key[0].value, encoded_account),
        ).fetchone()
        event_rows = connection.execute(
            "SELECT event_seq, event_id, event_hash, scope, account_id, "
            "previous_event_hash, state_after_hash, state_revision, event_json "
            "FROM kill_switch_events WHERE scope = ? AND account_id = ? "
            "ORDER BY event_seq",
            (key[0].value, encoded_account),
        ).fetchall()
        events = _load_event_chain(event_rows)
        if row is None:
            if events:
                raise KillSwitchRepositoryError("kill-switch events exist without a current state")
            return None
        state = _load_bound_state(row)
        self._validate_scope_projection(connection, state, events)
        return state

    @staticmethod
    def _insert_state(
        connection: sqlite3.Connection,
        state: KillSwitchState,
    ) -> None:
        state = _round_trip(_STATE_ADAPTER, state, "kill-switch state")
        connection.execute(
            "INSERT INTO kill_switch_states("
            "scope, account_id, state_id, state_hash, revision, state_json"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                state.scope.value,
                _encoded_account_id(state.account_id),
                state.state_id,
                state.state_hash,
                state.revision,
                _dump_json(_STATE_ADAPTER, state),
            ),
        )

    @staticmethod
    def _update_state(
        connection: sqlite3.Connection,
        state_before: KillSwitchState,
        state_after: KillSwitchState,
    ) -> None:
        state_after = _round_trip(
            _STATE_ADAPTER,
            state_after,
            "kill-switch state",
        )
        if (
            (state_before.scope, state_before.account_id)
            != (state_after.scope, state_after.account_id)
            or state_after.previous_state_hash != state_before.state_hash
            or state_after.revision != state_before.revision + 1
        ):
            raise KillSwitchRepositoryConflict(
                "kill-switch state update does not extend the current state"
            )
        cursor = connection.execute(
            "UPDATE kill_switch_states "
            "SET state_id = ?, state_hash = ?, revision = ?, state_json = ? "
            "WHERE scope = ? AND account_id = ? AND state_hash = ? AND revision = ?",
            (
                state_after.state_id,
                state_after.state_hash,
                state_after.revision,
                _dump_json(_STATE_ADAPTER, state_after),
                state_before.scope.value,
                _encoded_account_id(state_before.account_id),
                state_before.state_hash,
                state_before.revision,
            ),
        )
        if cursor.rowcount != 1:
            raise KillSwitchConcurrentUpdate(
                "kill-switch state changed before the durable CAS update"
            )

    @staticmethod
    def _ensure_incident_compatible(
        connection: sqlite3.Connection,
        incident: KillSwitchIncident,
    ) -> None:
        row = connection.execute(
            "SELECT incident_id, incident_hash, scope, account_id, incident_json "
            "FROM kill_switch_incidents "
            "WHERE incident_id = ? OR incident_hash = ?",
            (incident.incident_id, incident.incident_hash),
        ).fetchone()
        if row is None:
            return
        stored = _load_bound_incident(row)
        if stored != incident:
            raise KillSwitchIncidentConflict(
                "incident identity or hash is stored with different content"
            )

    @classmethod
    def _insert_incident_if_absent(
        cls,
        connection: sqlite3.Connection,
        incident: KillSwitchIncident,
    ) -> None:
        incident = _round_trip(
            _INCIDENT_ADAPTER,
            incident,
            "kill-switch incident",
        )
        cls._ensure_incident_compatible(connection, incident)
        existing = connection.execute(
            "SELECT 1 FROM kill_switch_incidents WHERE incident_id = ?",
            (incident.incident_id,),
        ).fetchone()
        if existing is not None:
            return
        connection.execute(
            "INSERT INTO kill_switch_incidents("
            "incident_id, incident_hash, scope, account_id, incident_json"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                incident.incident_id,
                incident.incident_hash,
                incident.scope.value,
                _encoded_account_id(incident.account_id),
                _dump_json(_INCIDENT_ADAPTER, incident),
            ),
        )

    @staticmethod
    def _latest_event_hash(
        connection: sqlite3.Connection,
        scope: KillSwitchScope,
        account_id: str | None,
    ) -> str | None:
        key = _state_key(scope, account_id)
        rows = connection.execute(
            "SELECT event_seq, event_id, event_hash, scope, account_id, "
            "previous_event_hash, state_after_hash, state_revision, event_json "
            "FROM kill_switch_events WHERE scope = ? AND account_id = ? "
            "ORDER BY event_seq",
            (key[0].value, _encoded_account_id(key[1])),
        ).fetchall()
        events = _load_event_chain(rows)
        return events[-1].event_hash if events else None

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        event: KillSwitchAuditEvent,
    ) -> None:
        event = _round_trip(_EVENT_ADAPTER, event, "kill-switch event")
        collision = connection.execute(
            "SELECT event_seq, event_id, event_hash, scope, account_id, "
            "previous_event_hash, state_after_hash, state_revision, event_json "
            "FROM kill_switch_events WHERE event_id = ? OR event_hash = ?",
            (event.event_id, event.event_hash),
        ).fetchone()
        if collision is not None:
            _load_bound_event(collision)
            raise KillSwitchRepositoryConflict(
                "kill-switch event identity or hash is already stored"
            )
        key = _state_key(event.scope, event.account_id)
        rows = connection.execute(
            "SELECT event_seq, event_id, event_hash, scope, account_id, "
            "previous_event_hash, state_after_hash, state_revision, event_json "
            "FROM kill_switch_events WHERE scope = ? AND account_id = ? "
            "ORDER BY event_seq",
            (key[0].value, _encoded_account_id(key[1])),
        ).fetchall()
        existing = _load_event_chain(rows)
        _assert_event_follows(existing[-1] if existing else None, event)
        connection.execute(
            "INSERT INTO kill_switch_events("
            "event_id, event_hash, scope, account_id, previous_event_hash, "
            "state_after_hash, state_revision, event_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.event_hash,
                event.scope.value,
                _encoded_account_id(event.account_id),
                event.previous_event_hash,
                event.state_after_hash,
                event.state_revision,
                _dump_json(_EVENT_ADAPTER, event),
            ),
        )

    def _committed_operation(
        self,
        connection: sqlite3.Connection,
        *,
        key: _StateKey,
        idempotency_key: str,
        operation_kind: str,
        operation_hash: str,
    ) -> KillSwitchTransition | None:
        normalized_key = _state_key(*key)
        normalized_idempotency = _non_empty(idempotency_key, "idempotency_key")
        row = connection.execute(
            "SELECT scope, account_id, idempotency_key, operation_kind, "
            "operation_hash, transition_hash, event_hash, transition_json "
            "FROM kill_switch_operations "
            "WHERE scope = ? AND account_id = ? AND idempotency_key = ?",
            (
                normalized_key[0].value,
                _encoded_account_id(normalized_key[1]),
                normalized_idempotency,
            ),
        ).fetchone()
        if row is None:
            return None
        stored_kind, stored_idempotency, transition = _load_bound_operation(row)
        self._validate_operation_links(connection, transition, stored_kind)
        if (
            stored_idempotency != normalized_idempotency
            or (transition.state_after.scope, transition.state_after.account_id) != normalized_key
        ):
            raise KillSwitchRepositoryError(
                "stored kill-switch operation has the wrong scope identity"
            )
        if stored_kind != operation_kind or transition.operation_hash != operation_hash:
            raise KillSwitchIdempotencyConflict(
                "kill-switch idempotency key belongs to different operation content"
            )
        self._state_or_none(connection, *normalized_key)
        return transition

    def _insert_operation(
        self,
        connection: sqlite3.Connection,
        *,
        key: _StateKey,
        idempotency_key: str,
        operation_kind: str,
        transition: KillSwitchTransition,
    ) -> None:
        normalized_key = _state_key(*key)
        normalized_idempotency = _non_empty(idempotency_key, "idempotency_key")
        transition = _round_trip(
            _TRANSITION_ADAPTER,
            transition,
            "kill-switch transition",
        )
        if (
            transition.state_after.scope,
            transition.state_after.account_id,
        ) != normalized_key or not _operation_kind_matches(
            operation_kind, transition.audit_event.event_type
        ):
            raise KillSwitchRepositoryConflict(
                "kill-switch operation does not match its scope or event type"
            )
        self._validate_operation_links(connection, transition, operation_kind)
        connection.execute(
            "INSERT INTO kill_switch_operations("
            "scope, account_id, idempotency_key, operation_kind, operation_hash, "
            "transition_hash, event_hash, transition_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                normalized_key[0].value,
                _encoded_account_id(normalized_key[1]),
                normalized_idempotency,
                operation_kind,
                transition.operation_hash,
                transition.transition_hash,
                transition.audit_event.event_hash,
                _dump_json(_TRANSITION_ADAPTER, transition),
            ),
        )

    @staticmethod
    def _validate_operation_links(
        connection: sqlite3.Connection,
        transition: KillSwitchTransition,
        operation_kind: str,
    ) -> None:
        if not _operation_kind_matches(operation_kind, transition.audit_event.event_type):
            raise KillSwitchRepositoryError(
                "stored kill-switch operation kind does not match its event"
            )
        event_row = connection.execute(
            "SELECT event_seq, event_id, event_hash, scope, account_id, "
            "previous_event_hash, state_after_hash, state_revision, event_json "
            "FROM kill_switch_events WHERE event_hash = ?",
            (transition.audit_event.event_hash,),
        ).fetchone()
        if event_row is None or _load_bound_event(event_row) != transition.audit_event:
            raise KillSwitchRepositoryError(
                "stored kill-switch operation does not bind its audit event"
            )
        if transition.incident is not None:
            incident_row = connection.execute(
                "SELECT incident_id, incident_hash, scope, account_id, incident_json "
                "FROM kill_switch_incidents WHERE incident_id = ?",
                (transition.incident.incident_id,),
            ).fetchone()
            if incident_row is None or _load_bound_incident(incident_row) != transition.incident:
                raise KillSwitchRepositoryError(
                    "stored kill-switch operation does not bind its incident"
                )

    def _validate_event_projection(
        self,
        connection: sqlite3.Connection,
        events: tuple[KillSwitchAuditEvent, ...],
        *,
        scope: KillSwitchScope | None,
        encoded_account_id: str | None,
    ) -> None:
        query = (
            "SELECT scope, account_id, state_id, state_hash, revision, state_json "
            "FROM kill_switch_states"
        )
        parameters: tuple[object, ...] = ()
        if scope is not None:
            query += " WHERE scope = ?"
            parameters = (scope.value,)
            if encoded_account_id is not None:
                query += " AND account_id = ?"
                parameters = (scope.value, encoded_account_id)
        rows = connection.execute(query, parameters).fetchall()
        states: dict[_StateKey, KillSwitchState] = {}
        for row in rows:
            state = _load_bound_state(row)
            key = _state_key(state.scope, state.account_id)
            if key in states:
                raise KillSwitchRepositoryError(
                    "stored kill-switch states contain a duplicate scope identity"
                )
            states[key] = state

        grouped: dict[_StateKey, list[KillSwitchAuditEvent]] = {}
        for event in events:
            grouped.setdefault(_state_key(event.scope, event.account_id), []).append(event)
        if set(states) != set(grouped):
            raise KillSwitchRepositoryError(
                "stored kill-switch states do not match their event projections"
            )
        for key, state in states.items():
            self._validate_scope_projection(connection, state, tuple(grouped[key]))

    def _validate_scope_projection(
        self,
        connection: sqlite3.Connection,
        state: KillSwitchState,
        events: tuple[KillSwitchAuditEvent, ...],
    ) -> None:
        if not events:
            raise KillSwitchRepositoryError("stored kill-switch state has no event history")
        tail = events[-1]
        if tail.state_after_hash != state.state_hash or tail.state_revision != state.revision:
            raise KillSwitchRepositoryError(
                "stored kill-switch state does not match its event-chain tail"
            )
        state_event = next(
            (
                event
                for event in reversed(events)
                if event.event_type is not KillSwitchEventType.ORDER_BLOCKED
            ),
            None,
        )
        if (
            state_event is None
            or state_event.state_after_hash != state.state_hash
            or state_event.state_revision != state.revision
            or state_event.state_before_hash != state.previous_state_hash
        ):
            raise KillSwitchRepositoryError(
                "stored kill-switch state does not match its latest transition"
            )

        for event in events:
            if event.event_type is KillSwitchEventType.ORDER_BLOCKED:
                self._validate_blocked_event(connection, event)
                continue
            operation_row = connection.execute(
                "SELECT scope, account_id, idempotency_key, operation_kind, "
                "operation_hash, transition_hash, event_hash, transition_json "
                "FROM kill_switch_operations WHERE event_hash = ?",
                (event.event_hash,),
            ).fetchone()
            if operation_row is None:
                raise KillSwitchRepositoryError(
                    "kill-switch transition event has no durable operation"
                )
            operation_kind, _, transition = _load_bound_operation(operation_row)
            self._validate_operation_links(connection, transition, operation_kind)
            if transition.audit_event != event:
                raise KillSwitchRepositoryError(
                    "kill-switch operation transition does not match its event"
                )
            if event.event_type in {
                KillSwitchEventType.ACTIVATED,
                KillSwitchEventType.TRIGGER_ADDED,
            }:
                self._validate_activation_incident(connection, event)

        stored_incidents: list[KillSwitchIncident] = []
        for incident_id in state.active_incident_ids:
            row = connection.execute(
                "SELECT incident_id, incident_hash, scope, account_id, incident_json "
                "FROM kill_switch_incidents WHERE incident_id = ?",
                (incident_id,),
            ).fetchone()
            if row is None:
                raise KillSwitchRepositoryError(
                    "active kill-switch state references a missing incident"
                )
            incident = _load_bound_incident(row)
            if (incident.scope, incident.account_id) != (state.scope, state.account_id):
                raise KillSwitchRepositoryError(
                    "active kill-switch state references another scope's incident"
                )
            stored_incidents.append(incident)
        if state.latest_incident_hash is not None:
            row = connection.execute(
                "SELECT incident_id, incident_hash, scope, account_id, incident_json "
                "FROM kill_switch_incidents WHERE incident_hash = ?",
                (state.latest_incident_hash,),
            ).fetchone()
            if row is None:
                raise KillSwitchRepositoryError(
                    "kill-switch state references a missing latest incident"
                )
            latest = _load_bound_incident(row)
            if (latest.scope, latest.account_id) != (state.scope, state.account_id):
                raise KillSwitchRepositoryError(
                    "kill-switch latest incident belongs to another scope"
                )
            if state.status is KillSwitchStatus.ACTIVE and latest not in stored_incidents:
                raise KillSwitchRepositoryError(
                    "active kill-switch latest incident is not in its incident set"
                )

    @staticmethod
    def _validate_activation_incident(
        connection: sqlite3.Connection,
        event: KillSwitchAuditEvent,
    ) -> None:
        if event.incident_id is None:
            raise KillSwitchRepositoryError("activation event has no durable incident identity")
        row = connection.execute(
            "SELECT incident_id, incident_hash, scope, account_id, incident_json "
            "FROM kill_switch_incidents WHERE incident_id = ?",
            (event.incident_id,),
        ).fetchone()
        if row is None:
            raise KillSwitchRepositoryError("activation event references a missing incident")
        incident = _load_bound_incident(row)
        if (
            (incident.scope, incident.account_id) != (event.scope, event.account_id)
            or incident.reason_codes != event.reason_codes
            or incident.evidence_hashes != event.evidence_hashes
        ):
            raise KillSwitchRepositoryError("activation event does not bind its incident evidence")

    def _validate_gate_decision_links(
        self,
        connection: sqlite3.Connection,
        stored_request: KillSwitchGateRequest,
        decision: KillSwitchGateDecision,
        *,
        expected_event: KillSwitchAuditEvent | None = None,
    ) -> None:
        if decision.audit_event_hash is None:
            if decision.reason_codes != (KillSwitchGateReason.STATE_UNAVAILABLE,):
                raise KillSwitchRepositoryError(
                    "stored unaudited gate block is not a state-unavailable decision"
                )
            return
        if expected_event is None:
            row = connection.execute(
                "SELECT event_seq, event_id, event_hash, scope, account_id, "
                "previous_event_hash, state_after_hash, state_revision, event_json "
                "FROM kill_switch_events WHERE event_hash = ?",
                (decision.audit_event_hash,),
            ).fetchone()
            if row is None:
                raise KillSwitchRepositoryError(
                    "stored gate decision references a missing audit event"
                )
            event = _load_bound_event(row)
        else:
            event = expected_event
        expected_operation_hash = stable_hash(
            {
                "operation": "ORDER_BLOCKED",
                "request_hash": decision.request_hash,
                "global_state_hash": decision.global_state_hash,
                "account_state_hash": decision.account_state_hash,
                "previous_event_hash": event.previous_event_hash,
            }
        )
        if (
            event.event_hash != decision.audit_event_hash
            or event.event_type is not KillSwitchEventType.ORDER_BLOCKED
            or event.operation_hash != expected_operation_hash
            or event.state_before_hash != event.state_after_hash
            or event.state_after_hash not in decision.blocking_state_hashes
            or event.reason_codes != tuple(reason.value for reason in decision.reason_codes)
            or event.evidence_hashes != decision.blocking_state_hashes
            or (
                event.scope is KillSwitchScope.ACCOUNT
                and event.account_id != stored_request.account_id
            )
        ):
            raise KillSwitchRepositoryError(
                "stored gate decision does not bind its ORDER_BLOCKED event"
            )
        for incident_id in decision.incident_ids:
            row = connection.execute(
                "SELECT incident_id, incident_hash, scope, account_id, incident_json "
                "FROM kill_switch_incidents WHERE incident_id = ?",
                (incident_id,),
            ).fetchone()
            if row is None:
                raise KillSwitchRepositoryError(
                    "stored gate decision references a missing incident"
                )
            incident = _load_bound_incident(row)
            if incident.scope is KillSwitchScope.ACCOUNT and (
                incident.account_id != stored_request.account_id
            ):
                raise KillSwitchRepositoryError(
                    "stored gate decision references another account's incident"
                )

    def _validate_blocked_event(
        self,
        connection: sqlite3.Connection,
        event: KillSwitchAuditEvent,
    ) -> None:
        rows = connection.execute(
            "SELECT account_id, idempotency_key, request_hash, request_json, "
            "decision_hash, audit_event_hash, decision_json "
            "FROM kill_switch_gate_requests WHERE audit_event_hash = ?",
            (event.event_hash,),
        ).fetchall()
        if len(rows) != 1:
            raise KillSwitchRepositoryError(
                "ORDER_BLOCKED event must have exactly one durable gate decision"
            )
        stored_request, decision = _load_bound_gate_record(rows[0])
        if decision is None:
            raise KillSwitchRepositoryError("ORDER_BLOCKED event gate row has no decision")
        self._validate_gate_decision_links(
            connection,
            stored_request,
            decision,
            expected_event=event,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        if foreign_keys != (1,):
            connection.close()
            raise KillSwitchRepositoryError("SQLite foreign-key enforcement could not be enabled")
        return connection

    @contextmanager
    def _transaction(
        self,
        *,
        immediate: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except sqlite3.Error as error:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise KillSwitchRepositoryError(
                "SQLite kill-switch repository operation failed"
            ) from error
        except BaseException:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        quick_check = connection.execute("PRAGMA quick_check").fetchall()
        if quick_check != [("ok",)]:
            raise KillSwitchRepositoryError("stored kill-switch database failed SQLite quick_check")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise KillSwitchRepositoryError(
                "stored kill-switch database contains foreign-key violations"
            )

        for table, expected_columns in _EXPECTED_COLUMNS.items():
            quoted_table = _quote_identifier(table)
            columns = connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            actual_columns: list[tuple[str, str, int, int]] = []
            for row in columns:
                if (
                    len(row) < 6
                    or not isinstance(row[1], str)
                    or not isinstance(row[2], str)
                    or not isinstance(row[3], int)
                    or isinstance(row[3], bool)
                    or not isinstance(row[5], int)
                    or isinstance(row[5], bool)
                ):
                    raise KillSwitchRepositoryError(
                        f"stored kill-switch table metadata is invalid: {table}"
                    )
                actual_columns.append((row[1], row[2].upper(), row[3], row[5]))
            if tuple(actual_columns) != expected_columns:
                raise KillSwitchRepositoryError(
                    f"stored kill-switch table schema is invalid: {table}"
                )

            unique_keys = _unique_keys(connection, table, columns)
            if not _REQUIRED_UNIQUE_KEYS[table].issubset(unique_keys):
                raise KillSwitchRepositoryError(
                    f"stored kill-switch unique constraints are invalid: {table}"
                )

            sql_row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if (
                sql_row is None
                or len(sql_row) != 1
                or not isinstance(sql_row[0], str)
                or not _has_required_checks(table, sql_row[0])
            ):
                raise KillSwitchRepositoryError(
                    f"stored kill-switch table constraints are invalid: {table}"
                )

            actual_foreign_keys = _foreign_keys(connection, table)
            if actual_foreign_keys != _EXPECTED_FOREIGN_KEYS[table]:
                raise KillSwitchRepositoryError(
                    f"stored kill-switch foreign keys are invalid: {table}"
                )

        protected_tables = tuple(_EXPECTED_COLUMNS)
        placeholders = ", ".join("?" for _ in protected_tables)
        triggers = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            f"AND tbl_name IN ({placeholders})",
            protected_tables,
        ).fetchall()
        if triggers:
            raise KillSwitchRepositoryError(
                "stored kill-switch schema contains unexpected triggers"
            )


def _quote_identifier(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz_" for character in value):
        raise KillSwitchRepositoryError("unsafe SQLite schema identifier")
    return f'"{value}"'


def _unique_keys(
    connection: sqlite3.Connection,
    table: str,
    columns: list[tuple[object, ...]],
) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    primary = tuple(
        name
        for _, name in sorted(
            (
                (int(row[5]), row[1])
                for row in columns
                if isinstance(row[5], int)
                and not isinstance(row[5], bool)
                and row[5] > 0
                and isinstance(row[1], str)
            ),
            key=lambda item: item[0],
        )
    )
    if primary:
        keys.add(primary)

    quoted_table = _quote_identifier(table)
    for row in connection.execute(f"PRAGMA index_list({quoted_table})").fetchall():
        if len(row) < 5 or not isinstance(row[1], str) or row[2] != 1 or row[4] != 0:
            continue
        quoted_index = '"' + row[1].replace('"', '""') + '"'
        index_rows = connection.execute(f"PRAGMA index_info({quoted_index})").fetchall()
        names: list[str] = []
        valid = bool(index_rows)
        for index_row in index_rows:
            if (
                len(index_row) < 3
                or not isinstance(index_row[1], int)
                or isinstance(index_row[1], bool)
                or index_row[1] < 0
                or not isinstance(index_row[2], str)
            ):
                valid = False
                break
            names.append(index_row[2])
        if valid:
            keys.add(tuple(names))
    return keys


def _foreign_keys(
    connection: sqlite3.Connection,
    table: str,
) -> set[tuple[str, str, str, str, str]]:
    quoted_table = _quote_identifier(table)
    result: set[tuple[str, str, str, str, str]] = set()
    for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall():
        if len(row) < 7 or not all(isinstance(row[index], str) for index in range(2, 7)):
            raise KillSwitchRepositoryError(
                f"stored kill-switch foreign-key metadata is invalid: {table}"
            )
        result.add((row[2], row[3], row[4], row[5], row[6]))
    return result


def _has_required_checks(table: str, sql: str) -> bool:
    normalized = "".join(sql.lower().split())
    return normalized == _EXPECTED_TABLE_SQL[table]


def _dump_json[T](adapter: TypeAdapter[T], value: T) -> bytes:
    try:
        return adapter.dump_json(value)
    except (TypeError, ValueError) as error:
        raise KillSwitchRepositoryError("kill-switch value cannot be serialized") from error


def _load_json[T](
    adapter: TypeAdapter[T],
    value: object,
    label: str,
) -> T:
    if not isinstance(value, bytes | str):
        raise KillSwitchRepositoryError(f"stored {label} JSON has an invalid type")
    try:
        return adapter.validate_json(value)
    except (ValidationError, TypeError, ValueError) as error:
        raise KillSwitchRepositoryError(f"stored {label} JSON is invalid") from error


def _round_trip[T](
    adapter: TypeAdapter[T],
    value: T,
    label: str,
) -> T:
    return _load_json(adapter, _dump_json(adapter, value), label)


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _state_key(
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
    return (scope, _non_empty(account_id, "account_id"))


def _event_filter(
    scope: KillSwitchScope | None,
    account_id: str | None,
) -> str | None:
    if scope is None:
        if account_id is not None:
            raise ValueError("account_id filtering requires ACCOUNT scope")
        return None
    if not isinstance(scope, KillSwitchScope):
        raise ValueError("kill-switch scope is invalid")
    if scope is KillSwitchScope.GLOBAL:
        if account_id is not None:
            raise ValueError("GLOBAL event filtering cannot carry account_id")
        return None
    return None if account_id is None else _non_empty(account_id, "account_id")


def _encoded_account_id(account_id: str | None) -> str:
    return "" if account_id is None else account_id


def _load_bound_state(row: tuple[object, ...]) -> KillSwitchState:
    if (
        len(row) != 6
        or not all(isinstance(row[index], str) for index in range(4))
        or not isinstance(row[4], int)
        or isinstance(row[4], bool)
        or not isinstance(row[5], bytes | str)
    ):
        raise KillSwitchRepositoryError("stored kill-switch state row is invalid")
    state = _load_json(_STATE_ADAPTER, row[5], "kill-switch state")
    if (
        state.scope.value != row[0]
        or _encoded_account_id(state.account_id) != row[1]
        or state.state_id != row[2]
        or state.state_hash != row[3]
        or state.revision != row[4]
    ):
        raise KillSwitchRepositoryError("stored kill-switch state row does not bind its JSON")
    return state


def _load_bound_incident(row: tuple[object, ...]) -> KillSwitchIncident:
    if (
        len(row) != 5
        or not all(isinstance(row[index], str) for index in range(4))
        or not isinstance(row[4], bytes | str)
    ):
        raise KillSwitchRepositoryError("stored kill-switch incident row is invalid")
    incident = _load_json(_INCIDENT_ADAPTER, row[4], "kill-switch incident")
    if (
        incident.incident_id != row[0]
        or incident.incident_hash != row[1]
        or incident.scope.value != row[2]
        or _encoded_account_id(incident.account_id) != row[3]
    ):
        raise KillSwitchRepositoryError("stored kill-switch incident row does not bind its JSON")
    return incident


def _load_bound_event(row: tuple[object, ...]) -> KillSwitchAuditEvent:
    if (
        len(row) != 9
        or not isinstance(row[0], int)
        or isinstance(row[0], bool)
        or row[0] < 1
        or not all(isinstance(row[index], str) for index in range(1, 5))
        or (row[5] is not None and not isinstance(row[5], str))
        or not isinstance(row[6], str)
        or not isinstance(row[7], int)
        or isinstance(row[7], bool)
        or not isinstance(row[8], bytes | str)
    ):
        raise KillSwitchRepositoryError("stored kill-switch event row is invalid")
    event = _load_json(_EVENT_ADAPTER, row[8], "kill-switch event")
    if (
        event.event_id != row[1]
        or event.event_hash != row[2]
        or event.scope.value != row[3]
        or _encoded_account_id(event.account_id) != row[4]
        or event.previous_event_hash != row[5]
        or event.state_after_hash != row[6]
        or event.state_revision != row[7]
    ):
        raise KillSwitchRepositoryError("stored kill-switch event row does not bind its JSON")
    return event


def _load_event_chain(
    rows: list[tuple[object, ...]],
) -> tuple[KillSwitchAuditEvent, ...]:
    events: list[KillSwitchAuditEvent] = []
    tails: dict[_StateKey, KillSwitchAuditEvent] = {}
    previous_sequence = 0
    for row in rows:
        if (
            not row
            or not isinstance(row[0], int)
            or isinstance(row[0], bool)
            or row[0] <= previous_sequence
        ):
            raise KillSwitchRepositoryError("stored kill-switch event sequence is invalid")
        previous_sequence = row[0]
        event = _load_bound_event(row)
        key = _state_key(event.scope, event.account_id)
        _assert_event_follows(tails.get(key), event)
        tails[key] = event
        events.append(event)
    return tuple(events)


def _assert_event_follows(
    previous: KillSwitchAuditEvent | None,
    event: KillSwitchAuditEvent,
) -> None:
    if previous is None:
        if (
            event.event_type is not KillSwitchEventType.INITIALIZED
            or event.previous_event_hash is not None
            or event.state_before_hash is not None
            or event.state_revision != 0
        ):
            raise KillSwitchRepositoryError(
                "kill-switch event chain does not begin with initialization"
            )
        return
    if (
        event.event_type is KillSwitchEventType.INITIALIZED
        or event.previous_event_hash != previous.event_hash
        or event.state_before_hash != previous.state_after_hash
    ):
        raise KillSwitchRepositoryError("kill-switch event does not extend its scope-local chain")
    if event.event_type is KillSwitchEventType.ORDER_BLOCKED:
        if (
            event.state_after_hash != previous.state_after_hash
            or event.state_revision != previous.state_revision
        ):
            raise KillSwitchRepositoryError("ORDER_BLOCKED event must preserve the current state")
    elif event.state_revision != previous.state_revision + 1:
        raise KillSwitchRepositoryError(
            "kill-switch state-changing event revision is not contiguous"
        )


def _load_bound_operation(
    row: tuple[object, ...],
) -> tuple[str, str, KillSwitchTransition]:
    if (
        len(row) != 8
        or not all(isinstance(row[index], str) for index in range(7))
        or not isinstance(row[7], bytes | str)
    ):
        raise KillSwitchRepositoryError("stored kill-switch operation row is invalid")
    transition = _load_json(
        _TRANSITION_ADAPTER,
        row[7],
        "kill-switch transition",
    )
    scope_value = row[0]
    account_value = row[1]
    idempotency_key = row[2]
    operation_kind = row[3]
    operation_hash = row[4]
    transition_hash = row[5]
    event_hash = row[6]
    assert isinstance(scope_value, str)
    assert isinstance(account_value, str)
    assert isinstance(idempotency_key, str)
    assert isinstance(operation_kind, str)
    assert isinstance(operation_hash, str)
    assert isinstance(transition_hash, str)
    assert isinstance(event_hash, str)
    if (
        transition.state_after.scope.value != scope_value
        or _encoded_account_id(transition.state_after.account_id) != account_value
        or transition.operation_hash != operation_hash
        or transition.transition_hash != transition_hash
        or transition.audit_event.event_hash != event_hash
        or operation_kind not in {_INITIALIZE, _ACTIVATE, _RECOVER}
        or not _operation_kind_matches(
            operation_kind,
            transition.audit_event.event_type,
        )
    ):
        raise KillSwitchRepositoryError(
            "stored kill-switch operation row does not bind its transition"
        )
    try:
        _non_empty(idempotency_key, "stored idempotency_key")
    except ValueError as error:
        raise KillSwitchRepositoryError(
            "stored kill-switch operation idempotency key is invalid"
        ) from error
    return operation_kind, idempotency_key, transition


def _operation_kind_matches(
    operation_kind: str,
    event_type: KillSwitchEventType,
) -> bool:
    if operation_kind == _INITIALIZE:
        return event_type is KillSwitchEventType.INITIALIZED
    if operation_kind == _ACTIVATE:
        return event_type in {
            KillSwitchEventType.ACTIVATED,
            KillSwitchEventType.TRIGGER_ADDED,
        }
    if operation_kind == _RECOVER:
        return event_type is KillSwitchEventType.RECOVERED
    return False


def _load_bound_gate_record(
    row: tuple[object, ...],
) -> tuple[KillSwitchGateRequest, KillSwitchGateDecision | None]:
    if (
        len(row) != 7
        or not all(isinstance(row[index], str) for index in range(3))
        or not isinstance(row[3], bytes | str)
        or (row[4] is not None and not isinstance(row[4], str))
        or (row[5] is not None and not isinstance(row[5], str))
        or (row[6] is not None and not isinstance(row[6], bytes | str))
    ):
        raise KillSwitchRepositoryError("stored kill-switch gate row is invalid")
    request = _load_json(
        _GATE_REQUEST_ADAPTER,
        row[3],
        "kill-switch gate request",
    )
    if (
        request.account_id != row[0]
        or request.idempotency_key != row[1]
        or request.request_hash != row[2]
    ):
        raise KillSwitchRepositoryError(
            "stored kill-switch gate row does not bind its request JSON"
        )
    if row[4] is None and row[5] is None and row[6] is None:
        return request, None
    if row[4] is None or row[6] is None:
        raise KillSwitchRepositoryError("stored kill-switch gate decision columns are incomplete")
    decision = _load_json(
        _DECISION_ADAPTER,
        row[6],
        "kill-switch gate decision",
    )
    if (
        decision.decision_hash != row[4]
        or decision.audit_event_hash != row[5]
        or decision.status is not KillSwitchGateStatus.BLOCKED
    ):
        raise KillSwitchRepositoryError(
            "stored kill-switch gate row does not bind its decision JSON"
        )
    return request, decision
