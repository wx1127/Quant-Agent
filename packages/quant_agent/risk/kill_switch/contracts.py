"""Immutable contracts for the global and account-level new-order kill switch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from quant_agent.core.time import ensure_aware
from quant_agent.regime.contracts import stable_hash

KILL_SWITCH_ENGINE_VERSION = "kill-switch-engine-v1"


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


def _optional_sha256(value: str | None, field_name: str) -> None:
    if value is not None:
        _sha256(value, field_name)


def _content(value: Any, hash_field: str) -> dict[str, Any]:
    return {key: item for key, item in asdict(value).items() if key != hash_field}


class KillSwitchScope(StrEnum):
    GLOBAL = "GLOBAL"
    ACCOUNT = "ACCOUNT"


class KillSwitchStatus(StrEnum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"


class KillSwitchTriggerSource(StrEnum):
    MANUAL = "MANUAL"
    RECONCILIATION = "RECONCILIATION"
    RISK = "RISK"
    DATA = "DATA"
    EXECUTION = "EXECUTION"
    STRATEGY = "STRATEGY"


class KillSwitchActorKind(StrEnum):
    HUMAN = "HUMAN"
    SYSTEM = "SYSTEM"
    AGENT = "AGENT"


class KillSwitchActorRole(StrEnum):
    RISK_ADMIN = "RISK_ADMIN"
    OPERATOR = "OPERATOR"
    SYSTEM = "SYSTEM"
    AGENT = "AGENT"


class KillSwitchEventType(StrEnum):
    INITIALIZED = "INITIALIZED"
    ACTIVATED = "ACTIVATED"
    TRIGGER_ADDED = "TRIGGER_ADDED"
    RECOVERED = "RECOVERED"
    ORDER_BLOCKED = "ORDER_BLOCKED"


class KillSwitchGateStatus(StrEnum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"


class KillSwitchGateReason(StrEnum):
    GLOBAL_ACTIVE = "GLOBAL_ACTIVE"
    ACCOUNT_ACTIVE = "ACCOUNT_ACTIVE"
    STATE_UNAVAILABLE = "STATE_UNAVAILABLE"
    REPOSITORY_FAILURE = "REPOSITORY_FAILURE"


def _scope_account(scope: KillSwitchScope, account_id: str | None) -> str | None:
    if not isinstance(scope, KillSwitchScope):
        raise ValueError("kill-switch scope is invalid")
    if scope is KillSwitchScope.GLOBAL:
        if account_id is not None:
            raise ValueError("GLOBAL kill switch cannot carry account_id")
        return None
    if account_id is None:
        raise ValueError("ACCOUNT kill switch requires account_id")
    return _non_empty(account_id, "account_id")


def _state_identity(scope: KillSwitchScope, account_id: str | None) -> str:
    return f"kill-switch:{stable_hash({'scope': scope, 'account_id': account_id})}"


@dataclass(frozen=True, slots=True)
class KillSwitchActor:
    actor_id: str
    kind: KillSwitchActorKind
    role: KillSwitchActorRole
    actor_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _non_empty(self.actor_id, "actor_id"))
        if not isinstance(self.kind, KillSwitchActorKind) or not isinstance(
            self.role, KillSwitchActorRole
        ):
            raise ValueError("kill-switch actor kind or role is invalid")
        valid_pairs = {
            (KillSwitchActorKind.HUMAN, KillSwitchActorRole.RISK_ADMIN),
            (KillSwitchActorKind.HUMAN, KillSwitchActorRole.OPERATOR),
            (KillSwitchActorKind.SYSTEM, KillSwitchActorRole.SYSTEM),
            (KillSwitchActorKind.AGENT, KillSwitchActorRole.AGENT),
        }
        if (self.kind, self.role) not in valid_pairs:
            raise ValueError("kill-switch actor kind and role are inconsistent")
        _sha256(self.actor_hash, "actor_hash")
        if self.actor_hash != stable_hash(self._content_payload()):
            raise ValueError("actor_hash does not match actor identity")

    @classmethod
    def build(
        cls,
        *,
        actor_id: str,
        kind: KillSwitchActorKind,
        role: KillSwitchActorRole,
    ) -> KillSwitchActor:
        values = {"actor_id": actor_id, "kind": kind, "role": role}
        return cls(
            actor_id=actor_id,
            kind=kind,
            role=role,
            actor_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, object]:
        return {"actor_id": self.actor_id, "kind": self.kind, "role": self.role}


@dataclass(frozen=True, slots=True)
class KillSwitchIncident:
    schema_version: str
    incident_id: str
    scope: KillSwitchScope
    account_id: str | None
    trigger_source: KillSwitchTriggerSource
    reason_codes: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    source_reference_hash: str
    summary: str
    triggered_at: datetime
    incident_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("kill-switch incident schema_version must be '1'")
        object.__setattr__(self, "account_id", _scope_account(self.scope, self.account_id))
        if not isinstance(self.trigger_source, KillSwitchTriggerSource):
            raise ValueError("kill-switch trigger_source is invalid")
        object.__setattr__(self, "summary", _non_empty(self.summary, "incident summary"))
        canonical_reasons = tuple(
            sorted({_non_empty(item, "reason_code") for item in self.reason_codes})
        )
        if not canonical_reasons or self.reason_codes != canonical_reasons:
            raise ValueError("incident reason_codes must be non-empty, unique, and sorted")
        if not self.evidence_hashes or self.evidence_hashes != tuple(
            sorted(set(self.evidence_hashes))
        ):
            raise ValueError("incident evidence_hashes must be non-empty, unique, and sorted")
        for digest in self.evidence_hashes:
            _sha256(digest, "incident evidence_hash")
        _sha256(self.source_reference_hash, "source_reference_hash")
        ensure_aware(self.triggered_at)
        expected_id = f"incident:{stable_hash(self._identity_payload())}"
        if self.incident_id != expected_id:
            raise ValueError("incident_id does not match source identity")
        _sha256(self.incident_hash, "incident_hash")
        if self.incident_hash != stable_hash(self._content_payload()):
            raise ValueError("incident_hash does not match incident content")

    @classmethod
    def build(
        cls,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        trigger_source: KillSwitchTriggerSource,
        reason_codes: tuple[str, ...],
        evidence_hashes: tuple[str, ...],
        source_reference_hash: str,
        summary: str,
        triggered_at: datetime,
    ) -> KillSwitchIncident:
        normalized_account = _scope_account(scope, account_id)
        ordered_reasons = tuple(sorted({_non_empty(item, "reason_code") for item in reason_codes}))
        ordered_evidence = tuple(sorted(set(evidence_hashes)))
        identity = {
            "scope": scope,
            "account_id": normalized_account,
            "trigger_source": trigger_source,
            "source_reference_hash": source_reference_hash,
        }
        values: dict[str, object] = {
            "schema_version": "1",
            "incident_id": f"incident:{stable_hash(identity)}",
            "scope": scope,
            "account_id": normalized_account,
            "trigger_source": trigger_source,
            "reason_codes": ordered_reasons,
            "evidence_hashes": ordered_evidence,
            "source_reference_hash": source_reference_hash,
            "summary": summary,
            "triggered_at": triggered_at,
        }
        return cls(**values, incident_hash=stable_hash(values))  # type: ignore[arg-type]

    def _identity_payload(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "account_id": self.account_id,
            "trigger_source": self.trigger_source,
            "source_reference_hash": self.source_reference_hash,
        }

    def _content_payload(self) -> dict[str, Any]:
        return _content(self, "incident_hash")


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    schema_version: str
    state_id: str
    scope: KillSwitchScope
    account_id: str | None
    status: KillSwitchStatus
    revision: int
    active_since: datetime | None
    active_incident_ids: tuple[str, ...]
    latest_incident_hash: str | None
    changed_at: datetime
    changed_by_actor_hash: str
    previous_state_hash: str | None
    state_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("kill-switch state schema_version must be '1'")
        object.__setattr__(self, "account_id", _scope_account(self.scope, self.account_id))
        if not isinstance(self.status, KillSwitchStatus):
            raise ValueError("kill-switch status is invalid")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 0
        ):
            raise ValueError("kill-switch revision must be a non-negative integer")
        ensure_aware(self.changed_at)
        if self.active_since is not None:
            ensure_aware(self.active_since)
            if self.active_since > self.changed_at:
                raise ValueError("active_since cannot follow changed_at")
        _sha256(self.changed_by_actor_hash, "changed_by_actor_hash")
        _optional_sha256(self.latest_incident_hash, "latest_incident_hash")
        _optional_sha256(self.previous_state_hash, "previous_state_hash")
        if self.active_incident_ids != tuple(sorted(set(self.active_incident_ids))):
            raise ValueError("active incident IDs must be unique and sorted")
        for incident_id in self.active_incident_ids:
            _non_empty(incident_id, "active incident_id")
        if self.revision == 0:
            if self.previous_state_hash is not None or self.status is not KillSwitchStatus.INACTIVE:
                raise ValueError("initial kill-switch state must be inactive without a predecessor")
            if self.latest_incident_hash is not None:
                raise ValueError("initial kill-switch state cannot reference an incident")
        elif self.previous_state_hash is None:
            raise ValueError("non-initial kill-switch state requires previous_state_hash")
        if self.status is KillSwitchStatus.ACTIVE:
            if (
                self.active_since is None
                or not self.active_incident_ids
                or self.latest_incident_hash is None
            ):
                raise ValueError("active kill switch requires time and incident evidence")
        elif self.active_since is not None or self.active_incident_ids:
            raise ValueError("inactive kill switch cannot retain active incident state")
        expected_id = _state_identity(self.scope, self.account_id)
        if self.state_id != expected_id:
            raise ValueError("kill-switch state_id does not match its scope")
        _sha256(self.state_hash, "state_hash")
        if self.state_hash != stable_hash(self._content_payload()):
            raise ValueError("kill-switch state_hash does not match content")

    @classmethod
    def initial(
        cls,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        actor: KillSwitchActor,
        changed_at: datetime,
    ) -> KillSwitchState:
        normalized_account = _scope_account(scope, account_id)
        values: dict[str, object] = {
            "schema_version": "1",
            "state_id": _state_identity(scope, normalized_account),
            "scope": scope,
            "account_id": normalized_account,
            "status": KillSwitchStatus.INACTIVE,
            "revision": 0,
            "active_since": None,
            "active_incident_ids": (),
            "latest_incident_hash": None,
            "changed_at": changed_at,
            "changed_by_actor_hash": actor.actor_hash,
            "previous_state_hash": None,
        }
        return cls(**values, state_hash=stable_hash(values))  # type: ignore[arg-type]

    @classmethod
    def activate(
        cls,
        *,
        previous: KillSwitchState,
        incident: KillSwitchIncident,
        actor: KillSwitchActor,
        changed_at: datetime,
    ) -> KillSwitchState:
        if (previous.scope, previous.account_id) != (incident.scope, incident.account_id):
            raise ValueError("incident scope does not match kill-switch state")
        ensure_aware(changed_at)
        if changed_at < previous.changed_at or changed_at < incident.triggered_at:
            raise ValueError("activation cannot precede state or incident time")
        incident_ids = tuple(sorted({*previous.active_incident_ids, incident.incident_id}))
        values: dict[str, object] = {
            "schema_version": "1",
            "state_id": previous.state_id,
            "scope": previous.scope,
            "account_id": previous.account_id,
            "status": KillSwitchStatus.ACTIVE,
            "revision": previous.revision + 1,
            "active_since": (
                previous.active_since
                if previous.status is KillSwitchStatus.ACTIVE
                else incident.triggered_at
            ),
            "active_incident_ids": incident_ids,
            "latest_incident_hash": incident.incident_hash,
            "changed_at": changed_at,
            "changed_by_actor_hash": actor.actor_hash,
            "previous_state_hash": previous.state_hash,
        }
        return cls(**values, state_hash=stable_hash(values))  # type: ignore[arg-type]

    @classmethod
    def recover(
        cls,
        *,
        previous: KillSwitchState,
        actor: KillSwitchActor,
        changed_at: datetime,
    ) -> KillSwitchState:
        if previous.status is not KillSwitchStatus.ACTIVE:
            raise ValueError("only an active kill switch can recover")
        ensure_aware(changed_at)
        if changed_at < previous.changed_at:
            raise ValueError("recovery cannot precede current state")
        values: dict[str, object] = {
            "schema_version": "1",
            "state_id": previous.state_id,
            "scope": previous.scope,
            "account_id": previous.account_id,
            "status": KillSwitchStatus.INACTIVE,
            "revision": previous.revision + 1,
            "active_since": None,
            "active_incident_ids": (),
            "latest_incident_hash": previous.latest_incident_hash,
            "changed_at": changed_at,
            "changed_by_actor_hash": actor.actor_hash,
            "previous_state_hash": previous.state_hash,
        }
        return cls(**values, state_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, Any]:
        return _content(self, "state_hash")


@dataclass(frozen=True, slots=True)
class KillSwitchActivationRequest:
    schema_version: str
    request_id: str
    idempotency_key: str
    incident: KillSwitchIncident
    actor: KillSwitchActor
    requested_at: datetime
    request_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("activation request schema_version must be '1'")
        for field_name in ("request_id", "idempotency_key"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if not isinstance(self.incident, KillSwitchIncident) or not isinstance(
            self.actor, KillSwitchActor
        ):
            raise ValueError("activation request incident or actor is invalid")
        ensure_aware(self.requested_at)
        if self.requested_at < self.incident.triggered_at:
            raise ValueError("activation request cannot precede its incident")
        _sha256(self.request_hash, "activation request_hash")
        if self.request_hash != stable_hash(self._content_payload()):
            raise ValueError("activation request_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        idempotency_key: str,
        incident: KillSwitchIncident,
        actor: KillSwitchActor,
        requested_at: datetime,
    ) -> KillSwitchActivationRequest:
        values = {
            "schema_version": "1",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "incident": asdict(incident),
            "actor": asdict(actor),
            "requested_at": requested_at,
        }
        return cls(
            schema_version="1",
            request_id=request_id,
            idempotency_key=idempotency_key,
            incident=incident,
            actor=actor,
            requested_at=requested_at,
            request_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "incident": asdict(self.incident),
            "actor": asdict(self.actor),
            "requested_at": self.requested_at,
        }


@dataclass(frozen=True, slots=True)
class KillSwitchRecoveryApproval:
    schema_version: str
    approval_id: str
    scope: KillSwitchScope
    account_id: str | None
    expected_state_hash: str
    expected_revision: int
    expected_latest_incident_hash: str
    approver: KillSwitchActor
    approved_at: datetime
    expires_at: datetime
    approval_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("recovery approval schema_version must be '1'")
        object.__setattr__(self, "account_id", _scope_account(self.scope, self.account_id))
        if not isinstance(self.approver, KillSwitchActor):
            raise ValueError("recovery approver is invalid")
        if (
            self.approver.kind is not KillSwitchActorKind.HUMAN
            or self.approver.role is not KillSwitchActorRole.RISK_ADMIN
        ):
            raise ValueError("recovery approval requires a human RISK_ADMIN")
        if (
            not isinstance(self.expected_revision, int)
            or isinstance(self.expected_revision, bool)
            or self.expected_revision < 1
        ):
            raise ValueError("recovery approval expected_revision must be positive")
        _sha256(self.expected_state_hash, "approval expected_state_hash")
        _sha256(self.expected_latest_incident_hash, "approval expected_latest_incident_hash")
        ensure_aware(self.approved_at)
        ensure_aware(self.expires_at)
        if self.expires_at <= self.approved_at:
            raise ValueError("recovery approval expiry must follow approval time")
        expected_id = f"approval:{stable_hash(self._identity_payload())}"
        if self.approval_id != expected_id:
            raise ValueError("approval_id does not match approval identity")
        _sha256(self.approval_hash, "approval_hash")
        if self.approval_hash != stable_hash(self._content_payload()):
            raise ValueError("approval_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        state: KillSwitchState,
        approver: KillSwitchActor,
        approved_at: datetime,
        expires_at: datetime,
    ) -> KillSwitchRecoveryApproval:
        if state.status is not KillSwitchStatus.ACTIVE or state.latest_incident_hash is None:
            raise ValueError("recovery approval requires an active state")
        ensure_aware(approved_at)
        if approved_at < state.changed_at:
            raise ValueError("recovery approval cannot precede the active state")
        identity = {
            "state_hash": state.state_hash,
            "revision": state.revision,
            "approver_hash": approver.actor_hash,
            "approved_at": approved_at,
        }
        values: dict[str, object] = {
            "schema_version": "1",
            "approval_id": f"approval:{stable_hash(identity)}",
            "scope": state.scope,
            "account_id": state.account_id,
            "expected_state_hash": state.state_hash,
            "expected_revision": state.revision,
            "expected_latest_incident_hash": state.latest_incident_hash,
            "approver": asdict(approver),
            "approved_at": approved_at,
            "expires_at": expires_at,
        }
        return cls(
            schema_version="1",
            approval_id=values["approval_id"],  # type: ignore[arg-type]
            scope=state.scope,
            account_id=state.account_id,
            expected_state_hash=state.state_hash,
            expected_revision=state.revision,
            expected_latest_incident_hash=state.latest_incident_hash,
            approver=approver,
            approved_at=approved_at,
            expires_at=expires_at,
            approval_hash=stable_hash(values),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "state_hash": self.expected_state_hash,
            "revision": self.expected_revision,
            "approver_hash": self.approver.actor_hash,
            "approved_at": self.approved_at,
        }

    def _content_payload(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in _content(self, "approval_hash").items()
                if key != "approver"
            },
            "approver": asdict(self.approver),
        }


@dataclass(frozen=True, slots=True)
class KillSwitchRecoveryRequest:
    schema_version: str
    request_id: str
    idempotency_key: str
    approval: KillSwitchRecoveryApproval
    actor: KillSwitchActor
    requested_at: datetime
    request_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("recovery request schema_version must be '1'")
        for field_name in ("request_id", "idempotency_key"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if not isinstance(self.approval, KillSwitchRecoveryApproval) or not isinstance(
            self.actor, KillSwitchActor
        ):
            raise ValueError("recovery request approval or actor is invalid")
        if (
            self.actor.kind is not KillSwitchActorKind.HUMAN
            or self.actor.role is not KillSwitchActorRole.RISK_ADMIN
        ):
            raise ValueError("only a human RISK_ADMIN may request recovery")
        ensure_aware(self.requested_at)
        if not (self.approval.approved_at <= self.requested_at < self.approval.expires_at):
            raise ValueError("recovery request must use a currently valid approval")
        _sha256(self.request_hash, "recovery request_hash")
        if self.request_hash != stable_hash(self._content_payload()):
            raise ValueError("recovery request_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        idempotency_key: str,
        approval: KillSwitchRecoveryApproval,
        actor: KillSwitchActor,
        requested_at: datetime,
    ) -> KillSwitchRecoveryRequest:
        values = {
            "schema_version": "1",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "approval": asdict(approval),
            "actor": asdict(actor),
            "requested_at": requested_at,
        }
        return cls(
            schema_version="1",
            request_id=request_id,
            idempotency_key=idempotency_key,
            approval=approval,
            actor=actor,
            requested_at=requested_at,
            request_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "approval": asdict(self.approval),
            "actor": asdict(self.actor),
            "requested_at": self.requested_at,
        }


@dataclass(frozen=True, slots=True)
class KillSwitchAuditEvent:
    schema_version: str
    event_id: str
    event_type: KillSwitchEventType
    scope: KillSwitchScope
    account_id: str | None
    state_before_hash: str | None
    state_after_hash: str
    state_revision: int
    actor_hash: str
    operation_hash: str
    incident_id: str | None
    reason_codes: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    occurred_at: datetime
    previous_event_hash: str | None
    event_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("kill-switch event schema_version must be '1'")
        object.__setattr__(self, "account_id", _scope_account(self.scope, self.account_id))
        if not isinstance(self.event_type, KillSwitchEventType):
            raise ValueError("kill-switch event_type is invalid")
        if (
            not isinstance(self.state_revision, int)
            or isinstance(self.state_revision, bool)
            or self.state_revision < 0
        ):
            raise ValueError("event state_revision must be non-negative")
        for value, field_name in (
            (self.state_after_hash, "event state_after_hash"),
            (self.actor_hash, "event actor_hash"),
            (self.operation_hash, "event operation_hash"),
        ):
            _sha256(value, field_name)
        _optional_sha256(self.state_before_hash, "event state_before_hash")
        _optional_sha256(self.previous_event_hash, "previous_event_hash")
        if self.event_type is KillSwitchEventType.INITIALIZED:
            if self.state_before_hash is not None or self.previous_event_hash is not None:
                raise ValueError("initialization event cannot have prior state or event")
        elif self.state_before_hash is None or self.previous_event_hash is None:
            raise ValueError("non-initial event requires prior state and event hashes")
        if self.incident_id is not None:
            object.__setattr__(self, "incident_id", _non_empty(self.incident_id, "incident_id"))
        if (
            self.event_type
            in {
                KillSwitchEventType.ACTIVATED,
                KillSwitchEventType.TRIGGER_ADDED,
            }
            and self.incident_id is None
        ):
            raise ValueError("activation events require an incident_id")
        canonical_reasons = tuple(
            sorted({_non_empty(item, "reason_code") for item in self.reason_codes})
        )
        if self.reason_codes != canonical_reasons:
            raise ValueError("event reason_codes must be unique and sorted")
        if self.evidence_hashes != tuple(sorted(set(self.evidence_hashes))):
            raise ValueError("event evidence_hashes must be unique and sorted")
        for digest in self.evidence_hashes:
            _sha256(digest, "event evidence_hash")
        ensure_aware(self.occurred_at)
        expected_id = f"kill-event:{stable_hash(self._identity_payload())}"
        if self.event_id != expected_id:
            raise ValueError("event_id does not match event identity")
        _sha256(self.event_hash, "event_hash")
        if self.event_hash != stable_hash(self._content_payload()):
            raise ValueError("event_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        event_type: KillSwitchEventType,
        state_before: KillSwitchState | None,
        state_after: KillSwitchState,
        actor: KillSwitchActor,
        operation_hash: str,
        incident: KillSwitchIncident | None,
        reason_codes: tuple[str, ...],
        evidence_hashes: tuple[str, ...],
        occurred_at: datetime,
        previous_event_hash: str | None,
    ) -> KillSwitchAuditEvent:
        ordered_reasons = tuple(sorted({_non_empty(item, "reason_code") for item in reason_codes}))
        ordered_evidence = tuple(sorted(set(evidence_hashes)))
        identity = {
            "event_type": event_type,
            "operation_hash": operation_hash,
            "state_after_hash": state_after.state_hash,
        }
        values: dict[str, object] = {
            "schema_version": "1",
            "event_id": f"kill-event:{stable_hash(identity)}",
            "event_type": event_type,
            "scope": state_after.scope,
            "account_id": state_after.account_id,
            "state_before_hash": state_before.state_hash if state_before else None,
            "state_after_hash": state_after.state_hash,
            "state_revision": state_after.revision,
            "actor_hash": actor.actor_hash,
            "operation_hash": operation_hash,
            "incident_id": incident.incident_id if incident else None,
            "reason_codes": ordered_reasons,
            "evidence_hashes": ordered_evidence,
            "occurred_at": occurred_at,
            "previous_event_hash": previous_event_hash,
        }
        return cls(**values, event_hash=stable_hash(values))  # type: ignore[arg-type]

    def _identity_payload(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "operation_hash": self.operation_hash,
            "state_after_hash": self.state_after_hash,
        }

    def _content_payload(self) -> dict[str, Any]:
        return _content(self, "event_hash")


@dataclass(frozen=True, slots=True)
class KillSwitchTransition:
    engine_version: str
    operation_hash: str
    state_before: KillSwitchState | None
    state_after: KillSwitchState
    incident: KillSwitchIncident | None
    audit_event: KillSwitchAuditEvent
    transition_hash: str

    def __post_init__(self) -> None:
        if self.engine_version != KILL_SWITCH_ENGINE_VERSION:
            raise ValueError("unknown kill-switch engine version")
        if self.state_before is not None and not isinstance(self.state_before, KillSwitchState):
            raise ValueError("transition state_before is invalid")
        if not isinstance(self.state_after, KillSwitchState) or not isinstance(
            self.audit_event, KillSwitchAuditEvent
        ):
            raise ValueError("transition state_after or audit_event is invalid")
        if self.incident is not None and not isinstance(self.incident, KillSwitchIncident):
            raise ValueError("transition incident is invalid")
        _sha256(self.operation_hash, "transition operation_hash")
        if self.audit_event.operation_hash != self.operation_hash:
            raise ValueError("transition event does not bind its operation")
        if self.audit_event.state_after_hash != self.state_after.state_hash:
            raise ValueError("transition event does not bind state_after")
        if (
            (self.audit_event.scope, self.audit_event.account_id)
            != (self.state_after.scope, self.state_after.account_id)
            or self.audit_event.actor_hash != self.state_after.changed_by_actor_hash
            or self.audit_event.occurred_at != self.state_after.changed_at
        ):
            raise ValueError("transition event identity, actor, or time does not bind state_after")
        expected_before = self.state_before.state_hash if self.state_before else None
        if self.audit_event.state_before_hash != expected_before:
            raise ValueError("transition event does not bind state_before")
        if self.audit_event.state_revision != self.state_after.revision:
            raise ValueError("transition event revision does not bind state_after")
        if self.state_before is None:
            if self.state_after.revision != 0:
                raise ValueError("initial transition must create revision zero")
            expected_event_type = KillSwitchEventType.INITIALIZED
        elif (
            (self.state_before.scope, self.state_before.account_id)
            != (self.state_after.scope, self.state_after.account_id)
            or self.state_after.revision != self.state_before.revision + 1
            or self.state_after.previous_state_hash != self.state_before.state_hash
        ):
            raise ValueError("transition states do not form one scope-local revision chain")
        elif (
            self.state_before.status is KillSwitchStatus.INACTIVE
            and self.state_after.status is KillSwitchStatus.ACTIVE
            and self.incident is not None
        ):
            expected_event_type = KillSwitchEventType.ACTIVATED
        elif (
            self.state_before.status is KillSwitchStatus.ACTIVE
            and self.state_after.status is KillSwitchStatus.ACTIVE
            and self.incident is not None
        ):
            expected_event_type = KillSwitchEventType.TRIGGER_ADDED
        elif (
            self.state_before.status is KillSwitchStatus.ACTIVE
            and self.state_after.status is KillSwitchStatus.INACTIVE
            and self.incident is None
        ):
            expected_event_type = KillSwitchEventType.RECOVERED
        else:
            raise ValueError("transition status and incident combination is invalid")
        if self.audit_event.event_type is not expected_event_type:
            raise ValueError("transition event_type does not match its state change")
        if self.incident is not None and (
            self.audit_event.incident_id != self.incident.incident_id
            or (self.incident.scope, self.incident.account_id)
            != (self.state_after.scope, self.state_after.account_id)
            or self.audit_event.reason_codes != self.incident.reason_codes
            or self.audit_event.evidence_hashes != self.incident.evidence_hashes
            or self.incident.incident_id not in self.state_after.active_incident_ids
            or self.state_after.latest_incident_hash != self.incident.incident_hash
        ):
            raise ValueError("transition event and state do not bind their incident")
        _sha256(self.transition_hash, "transition_hash")
        if self.transition_hash != stable_hash(self._content_payload()):
            raise ValueError("transition_hash does not match transition content")

    @classmethod
    def build(
        cls,
        *,
        operation_hash: str,
        state_before: KillSwitchState | None,
        state_after: KillSwitchState,
        incident: KillSwitchIncident | None,
        audit_event: KillSwitchAuditEvent,
    ) -> KillSwitchTransition:
        values = {
            "engine_version": KILL_SWITCH_ENGINE_VERSION,
            "operation_hash": operation_hash,
            "state_before": asdict(state_before) if state_before else None,
            "state_after": asdict(state_after),
            "incident": asdict(incident) if incident else None,
            "audit_event": asdict(audit_event),
        }
        return cls(
            engine_version=KILL_SWITCH_ENGINE_VERSION,
            operation_hash=operation_hash,
            state_before=state_before,
            state_after=state_after,
            incident=incident,
            audit_event=audit_event,
            transition_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "operation_hash": self.operation_hash,
            "state_before": asdict(self.state_before) if self.state_before else None,
            "state_after": asdict(self.state_after),
            "incident": asdict(self.incident) if self.incident else None,
            "audit_event": asdict(self.audit_event),
        }


@dataclass(frozen=True, slots=True)
class KillSwitchGateRequest:
    schema_version: str
    request_id: str
    account_id: str
    idempotency_key: str
    batch_hash: str
    source_request_hash: str
    checked_at: datetime
    request_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("gate request schema_version must be '1'")
        for field_name in ("request_id", "account_id", "idempotency_key"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        _sha256(self.batch_hash, "gate batch_hash")
        _sha256(self.source_request_hash, "gate source_request_hash")
        ensure_aware(self.checked_at)
        _sha256(self.request_hash, "gate request_hash")
        if self.request_hash != stable_hash(self._content_payload()):
            raise ValueError("gate request_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        account_id: str,
        idempotency_key: str,
        batch_hash: str,
        source_request_hash: str,
        checked_at: datetime,
    ) -> KillSwitchGateRequest:
        values = {
            "schema_version": "1",
            "request_id": request_id,
            "account_id": account_id,
            "idempotency_key": idempotency_key,
            "batch_hash": batch_hash,
            "source_request_hash": source_request_hash,
            "checked_at": checked_at,
        }
        return cls(
            schema_version="1",
            request_id=request_id,
            account_id=account_id,
            idempotency_key=idempotency_key,
            batch_hash=batch_hash,
            source_request_hash=source_request_hash,
            checked_at=checked_at,
            request_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return _content(self, "request_hash")

    @property
    def operation_hash(self) -> str:
        """Identify the order independently of a particular gate evaluation time."""

        return stable_hash(
            {key: value for key, value in self._content_payload().items() if key != "checked_at"}
        )


@dataclass(frozen=True, slots=True)
class KillSwitchGateDecision:
    engine_version: str
    request_hash: str
    status: KillSwitchGateStatus
    checked_at: datetime
    global_state_hash: str | None
    account_state_hash: str | None
    blocking_state_hashes: tuple[str, ...]
    reason_codes: tuple[KillSwitchGateReason, ...]
    incident_ids: tuple[str, ...]
    audit_event_hash: str | None
    decision_hash: str

    def __post_init__(self) -> None:
        if self.engine_version != KILL_SWITCH_ENGINE_VERSION:
            raise ValueError("unknown gate engine version")
        if not isinstance(self.status, KillSwitchGateStatus):
            raise ValueError("gate status is invalid")
        _sha256(self.request_hash, "gate request_hash")
        ensure_aware(self.checked_at)
        _optional_sha256(self.global_state_hash, "global_state_hash")
        _optional_sha256(self.account_state_hash, "account_state_hash")
        _optional_sha256(self.audit_event_hash, "gate audit_event_hash")
        if self.blocking_state_hashes != tuple(sorted(set(self.blocking_state_hashes))):
            raise ValueError("blocking state hashes must be unique and sorted")
        for digest in self.blocking_state_hashes:
            _sha256(digest, "blocking state_hash")
        if any(not isinstance(item, KillSwitchGateReason) for item in self.reason_codes):
            raise ValueError("gate reason code is invalid")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("gate reason_codes must be unique and sorted")
        canonical_incident_ids = tuple(
            sorted({_non_empty(item, "gate incident_id") for item in self.incident_ids})
        )
        if self.incident_ids != canonical_incident_ids:
            raise ValueError("gate incident_ids must be non-empty, unique, and sorted")
        if self.status is KillSwitchGateStatus.ALLOWED:
            if (
                self.global_state_hash is None
                or self.account_state_hash is None
                or self.blocking_state_hashes
                or self.reason_codes
                or self.incident_ids
                or self.audit_event_hash is not None
            ):
                raise ValueError("allowed gate decision must bind two inactive states only")
        elif self.status is KillSwitchGateStatus.BLOCKED:
            if not self.reason_codes:
                raise ValueError("blocked gate decision requires a reason")
            active_reasons = {
                KillSwitchGateReason.GLOBAL_ACTIVE,
                KillSwitchGateReason.ACCOUNT_ACTIVE,
            }
            reports_active_state = bool(set(self.reason_codes) & active_reasons)
            if reports_active_state and (
                not self.blocking_state_hashes
                or not self.incident_ids
                or self.audit_event_hash is None
            ):
                raise ValueError("active-state block requires state, incident, and audit evidence")
            if not reports_active_state and (
                self.blocking_state_hashes or self.incident_ids or self.audit_event_hash is not None
            ):
                raise ValueError("fail-closed state-unavailable block cannot claim active evidence")
        else:
            raise ValueError("gate status is invalid")
        _sha256(self.decision_hash, "gate decision_hash")
        if self.decision_hash != stable_hash(self._content_payload()):
            raise ValueError("gate decision_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        request: KillSwitchGateRequest,
        status: KillSwitchGateStatus,
        global_state_hash: str | None,
        account_state_hash: str | None,
        blocking_state_hashes: tuple[str, ...] = (),
        reason_codes: tuple[KillSwitchGateReason, ...] = (),
        incident_ids: tuple[str, ...] = (),
        audit_event_hash: str | None = None,
    ) -> KillSwitchGateDecision:
        values: dict[str, object] = {
            "engine_version": KILL_SWITCH_ENGINE_VERSION,
            "request_hash": request.request_hash,
            "status": status,
            "checked_at": request.checked_at,
            "global_state_hash": global_state_hash,
            "account_state_hash": account_state_hash,
            "blocking_state_hashes": tuple(sorted(set(blocking_state_hashes))),
            "reason_codes": tuple(sorted(set(reason_codes), key=lambda item: item.value)),
            "incident_ids": tuple(sorted(set(incident_ids))),
            "audit_event_hash": audit_event_hash,
        }
        return cls(**values, decision_hash=stable_hash(values))  # type: ignore[arg-type]

    @property
    def allowed(self) -> bool:
        return self.status is KillSwitchGateStatus.ALLOWED

    def _content_payload(self) -> dict[str, Any]:
        return _content(self, "decision_hash")


__all__ = [
    "KILL_SWITCH_ENGINE_VERSION",
    "KillSwitchActivationRequest",
    "KillSwitchActor",
    "KillSwitchActorKind",
    "KillSwitchActorRole",
    "KillSwitchAuditEvent",
    "KillSwitchEventType",
    "KillSwitchGateDecision",
    "KillSwitchGateReason",
    "KillSwitchGateRequest",
    "KillSwitchGateStatus",
    "KillSwitchIncident",
    "KillSwitchRecoveryApproval",
    "KillSwitchRecoveryRequest",
    "KillSwitchScope",
    "KillSwitchState",
    "KillSwitchStatus",
    "KillSwitchTransition",
    "KillSwitchTriggerSource",
]
