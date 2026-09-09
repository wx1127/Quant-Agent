"""Adversarial tests for immutable, fail-closed kill-switch contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from quant_agent.regime.contracts import stable_hash
from quant_agent.risk.kill_switch.contracts import (
    KILL_SWITCH_ENGINE_VERSION,
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
    KillSwitchRecoveryApproval,
    KillSwitchRecoveryRequest,
    KillSwitchScope,
    KillSwitchState,
    KillSwitchStatus,
    KillSwitchTransition,
    KillSwitchTriggerSource,
)

T0 = datetime(2026, 9, 2, 1, 0, tzinfo=UTC)


def _digest(label: str) -> str:
    return stable_hash({"fixture": label})


def _actor(
    *,
    actor_id: str = "risk-admin-1",
    kind: KillSwitchActorKind = KillSwitchActorKind.HUMAN,
    role: KillSwitchActorRole = KillSwitchActorRole.RISK_ADMIN,
) -> KillSwitchActor:
    return KillSwitchActor.build(actor_id=actor_id, kind=kind, role=role)


def _system_actor() -> KillSwitchActor:
    return _actor(
        actor_id="kill-switch-service",
        kind=KillSwitchActorKind.SYSTEM,
        role=KillSwitchActorRole.SYSTEM,
    )


def _agent_actor() -> KillSwitchActor:
    return _actor(
        actor_id="quant-agent",
        kind=KillSwitchActorKind.AGENT,
        role=KillSwitchActorRole.AGENT,
    )


def _incident(
    *,
    scope: KillSwitchScope = KillSwitchScope.ACCOUNT,
    account_id: str | None = "paper-account-1",
    trigger_source: KillSwitchTriggerSource = KillSwitchTriggerSource.RECONCILIATION,
    source_label: str = "reconciliation-result-1",
    reason_codes: tuple[str, ...] = ("POSITION_TOTAL_MISMATCH", "CASH_TOTAL_MISMATCH"),
    evidence_hashes: tuple[str, ...] | None = None,
    summary: str = "authoritative account reconciliation failed",
    triggered_at: datetime = T0 + timedelta(minutes=1),
) -> KillSwitchIncident:
    evidence = evidence_hashes or (_digest("finding-b"), _digest("finding-a"))
    return KillSwitchIncident.build(
        scope=scope,
        account_id=account_id,
        trigger_source=trigger_source,
        reason_codes=reason_codes,
        evidence_hashes=evidence,
        source_reference_hash=_digest(source_label),
        summary=summary,
        triggered_at=triggered_at,
    )


def _state_chain() -> tuple[
    KillSwitchActor,
    KillSwitchIncident,
    KillSwitchState,
    KillSwitchState,
]:
    actor = _system_actor()
    incident = _incident()
    initial = KillSwitchState.initial(
        scope=KillSwitchScope.ACCOUNT,
        account_id="paper-account-1",
        actor=actor,
        changed_at=T0,
    )
    active = KillSwitchState.activate(
        previous=initial,
        incident=incident,
        actor=actor,
        changed_at=T0 + timedelta(minutes=2),
    )
    return actor, incident, initial, active


def _approval(
    *,
    state: KillSwitchState | None = None,
    approver: KillSwitchActor | None = None,
    approved_at: datetime = T0 + timedelta(minutes=3),
    expires_at: datetime = T0 + timedelta(minutes=8),
) -> KillSwitchRecoveryApproval:
    if state is None:
        state = _state_chain()[3]
    return KillSwitchRecoveryApproval.build(
        state=state,
        approver=approver or _actor(),
        approved_at=approved_at,
        expires_at=expires_at,
    )


def _gate_request() -> KillSwitchGateRequest:
    return KillSwitchGateRequest.build(
        request_id="gate-request-1",
        account_id="paper-account-1",
        idempotency_key="gate-key-1",
        source_request_hash=_digest("paper-execution-request-1"),
        batch_hash=_digest("order-draft-batch-1"),
        checked_at=T0 + timedelta(minutes=10),
    )


def _rehash(value: Any, hash_field: str, **changes: object) -> Any:
    """Rebuild a contract with a valid content hash after an adversarial edit."""

    values = {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name != hash_field
    }
    values.update(changes)
    payload = {
        key: asdict(item) if is_dataclass(item) and not isinstance(item, type) else item
        for key, item in values.items()
    }
    return type(value)(**values, **{hash_field: stable_hash(payload)})


def _audit_chain() -> tuple[
    KillSwitchActor,
    KillSwitchIncident,
    KillSwitchState,
    KillSwitchState,
    KillSwitchAuditEvent,
    KillSwitchAuditEvent,
]:
    actor, incident, initial, active = _state_chain()
    initialized = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.INITIALIZED,
        state_before=None,
        state_after=initial,
        actor=actor,
        operation_hash=_digest("initialize-operation"),
        incident=None,
        reason_codes=(),
        evidence_hashes=(),
        occurred_at=initial.changed_at,
        previous_event_hash=None,
    )
    activated = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.ACTIVATED,
        state_before=initial,
        state_after=active,
        actor=actor,
        operation_hash=_digest("activation-operation"),
        incident=incident,
        reason_codes=incident.reason_codes,
        evidence_hashes=incident.evidence_hashes,
        occurred_at=active.changed_at,
        previous_event_hash=initialized.event_hash,
    )
    return actor, incident, initial, active, initialized, activated


@pytest.mark.parametrize(
    ("scope", "account_id", "expected_account"),
    (
        (KillSwitchScope.GLOBAL, None, None),
        (KillSwitchScope.ACCOUNT, "paper-account-1", "paper-account-1"),
        (KillSwitchScope.ACCOUNT, "  paper-account-1  ", "paper-account-1"),
    ),
)
def test_scope_account_combinations_have_stable_distinct_identities(
    scope: KillSwitchScope,
    account_id: str | None,
    expected_account: str | None,
) -> None:
    actor = _system_actor()
    state = KillSwitchState.initial(
        scope=scope,
        account_id=account_id,
        actor=actor,
        changed_at=T0,
    )
    incident = _incident(scope=scope, account_id=account_id)

    assert state.account_id == expected_account
    assert incident.account_id == expected_account
    assert state.state_id.startswith("kill-switch:")
    assert incident.incident_id.startswith("incident:")


def test_global_and_account_state_identities_cannot_collide() -> None:
    actor = _system_actor()
    global_state = KillSwitchState.initial(
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        actor=actor,
        changed_at=T0,
    )
    account_state = KillSwitchState.initial(
        scope=KillSwitchScope.ACCOUNT,
        account_id="paper-account-1",
        actor=actor,
        changed_at=T0,
    )

    assert global_state.state_id != account_state.state_id


@pytest.mark.parametrize(
    ("scope", "account_id", "message"),
    (
        (KillSwitchScope.GLOBAL, "paper-account-1", "GLOBAL"),
        (KillSwitchScope.GLOBAL, "", "GLOBAL"),
        (KillSwitchScope.ACCOUNT, None, "requires account_id"),
        (KillSwitchScope.ACCOUNT, "   ", "account_id must be non-empty"),
        ("ACCOUNT", "paper-account-1", "scope is invalid"),
    ),
)
def test_scope_account_rejects_ambiguous_or_mistyped_combinations(
    scope: Any,
    account_id: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        KillSwitchState.initial(
            scope=scope,
            account_id=account_id,
            actor=_system_actor(),
            changed_at=T0,
        )


@pytest.mark.parametrize(
    ("kind", "role"),
    (
        (KillSwitchActorKind.HUMAN, KillSwitchActorRole.RISK_ADMIN),
        (KillSwitchActorKind.HUMAN, KillSwitchActorRole.OPERATOR),
        (KillSwitchActorKind.SYSTEM, KillSwitchActorRole.SYSTEM),
        (KillSwitchActorKind.AGENT, KillSwitchActorRole.AGENT),
    ),
)
def test_actor_accepts_only_the_four_explicit_kind_role_pairs(
    kind: KillSwitchActorKind,
    role: KillSwitchActorRole,
) -> None:
    actor = _actor(actor_id=f"{kind.value.lower()}-{role.value.lower()}", kind=kind, role=role)

    assert actor.kind is kind
    assert actor.role is role
    assert len(actor.actor_hash) == 64


@pytest.mark.parametrize(
    ("kind", "role"),
    (
        (KillSwitchActorKind.HUMAN, KillSwitchActorRole.SYSTEM),
        (KillSwitchActorKind.HUMAN, KillSwitchActorRole.AGENT),
        (KillSwitchActorKind.SYSTEM, KillSwitchActorRole.RISK_ADMIN),
        (KillSwitchActorKind.SYSTEM, KillSwitchActorRole.OPERATOR),
        (KillSwitchActorKind.AGENT, KillSwitchActorRole.RISK_ADMIN),
        (KillSwitchActorKind.AGENT, KillSwitchActorRole.OPERATOR),
    ),
)
def test_actor_rejects_inconsistent_kind_role_pairs(
    kind: KillSwitchActorKind,
    role: KillSwitchActorRole,
) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        _actor(kind=kind, role=role)


@pytest.mark.parametrize(
    ("kind", "role"),
    (
        ("HUMAN", KillSwitchActorRole.RISK_ADMIN),
        (KillSwitchActorKind.HUMAN, "RISK_ADMIN"),
        ("HUMAN", "RISK_ADMIN"),
    ),
)
def test_actor_rejects_plain_strings_that_compare_equal_to_str_enums(
    kind: Any,
    role: Any,
) -> None:
    with pytest.raises(ValueError, match="kind or role is invalid"):
        KillSwitchActor.build(actor_id="typed-actor", kind=kind, role=role)


def test_actor_is_frozen_and_content_addressed() -> None:
    actor = _actor()
    same = _actor()
    changed = _actor(actor_id="risk-admin-2")

    assert actor == same
    assert actor.actor_hash != changed.actor_hash
    with pytest.raises(FrozenInstanceError):
        actor.actor_id = "forged"  # type: ignore[misc]
    with pytest.raises(ValueError, match="actor_hash"):
        replace(actor, actor_hash=_digest("forged-actor"))
    with pytest.raises(ValueError, match="actor_id"):
        KillSwitchActor.build(
            actor_id="  ",
            kind=KillSwitchActorKind.HUMAN,
            role=KillSwitchActorRole.RISK_ADMIN,
        )


def test_incident_builder_canonicalizes_reason_and_evidence_order() -> None:
    evidence_a = _digest("evidence-a")
    evidence_b = _digest("evidence-b")
    incident = _incident(
        reason_codes=("Z_REASON", "A_REASON", "Z_REASON"),
        evidence_hashes=(evidence_b, evidence_a, evidence_b),
    )
    reordered = _incident(
        reason_codes=("A_REASON", "Z_REASON"),
        evidence_hashes=(evidence_a, evidence_b),
    )

    assert incident.reason_codes == ("A_REASON", "Z_REASON")
    assert incident.evidence_hashes == tuple(sorted((evidence_a, evidence_b)))
    assert incident == reordered


def test_incident_id_is_source_identity_while_hash_binds_full_content() -> None:
    incident = _incident()
    revised = _incident(
        summary="same authoritative source with a clarified human summary",
        reason_codes=("CASH_TOTAL_MISMATCH",),
    )

    assert incident.incident_id == revised.incident_id
    assert incident.incident_hash != revised.incident_hash


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema_version": "2"}, "schema_version"),
        ({"trigger_source": "RISK"}, "trigger_source is invalid"),
        ({"summary": "  "}, "summary"),
        ({"reason_codes": ()}, "reason_codes"),
        ({"reason_codes": ("B", "A")}, "unique, and sorted"),
        ({"reason_codes": ("A", "A")}, "unique, and sorted"),
        ({"evidence_hashes": ()}, "evidence_hashes"),
        ({"evidence_hashes": ("f" * 64, "0" * 64)}, "unique, and sorted"),
        ({"evidence_hashes": (_digest("a"), _digest("a"))}, "unique, and sorted"),
        ({"source_reference_hash": "bad"}, "source_reference_hash"),
        ({"triggered_at": datetime(2026, 9, 2, 1, 0)}, "timezone information"),
        ({"incident_id": "incident:forged"}, "incident_id"),
        ({"incident_hash": _digest("forged-incident")}, "incident_hash"),
    ),
)
def test_incident_rejects_noncanonical_invalid_or_tampered_content(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_incident(), **changes)


def test_incident_rejects_bad_evidence_digest_even_when_ordered() -> None:
    with pytest.raises(ValueError, match="evidence_hash"):
        _incident(evidence_hashes=("0" * 63,))


def test_incident_hash_is_timezone_canonical_for_the_same_instant() -> None:
    utc_incident = _incident(triggered_at=T0)
    offset_incident = _incident(
        triggered_at=T0.astimezone(timezone(timedelta(hours=8))),
    )

    assert utc_incident.triggered_at == offset_incident.triggered_at
    assert utc_incident.incident_id == offset_incident.incident_id
    assert utc_incident.incident_hash == offset_incident.incident_hash


def test_state_initial_activate_add_trigger_and_recover_form_a_hash_chain() -> None:
    actor, incident, initial, active = _state_chain()
    second_incident = _incident(
        trigger_source=KillSwitchTriggerSource.RISK,
        source_label="intraday-drawdown-limit",
        reason_codes=("DRAWDOWN_LIMIT_BREACH",),
        summary="hard intraday drawdown threshold breached",
        triggered_at=T0 + timedelta(minutes=3),
    )
    second_active = KillSwitchState.activate(
        previous=active,
        incident=second_incident,
        actor=actor,
        changed_at=T0 + timedelta(minutes=4),
    )
    recovered = KillSwitchState.recover(
        previous=second_active,
        actor=_actor(),
        changed_at=T0 + timedelta(minutes=5),
    )

    assert initial.status is KillSwitchStatus.INACTIVE
    assert initial.revision == 0
    assert initial.previous_state_hash is None
    assert initial.active_incident_ids == ()
    assert active.status is KillSwitchStatus.ACTIVE
    assert active.revision == 1
    assert active.previous_state_hash == initial.state_hash
    assert active.active_since == incident.triggered_at
    assert active.active_incident_ids == (incident.incident_id,)
    assert second_active.revision == 2
    assert second_active.previous_state_hash == active.state_hash
    assert second_active.active_since == incident.triggered_at
    assert second_active.active_incident_ids == tuple(
        sorted((incident.incident_id, second_incident.incident_id))
    )
    assert second_active.latest_incident_hash == second_incident.incident_hash
    assert recovered.status is KillSwitchStatus.INACTIVE
    assert recovered.revision == 3
    assert recovered.previous_state_hash == second_active.state_hash
    assert recovered.active_since is None
    assert recovered.active_incident_ids == ()
    assert recovered.latest_incident_hash == second_incident.incident_hash
    assert (
        len({initial.state_hash, active.state_hash, second_active.state_hash, recovered.state_hash})
        == 4
    )


def test_state_activation_rejects_scope_and_time_mismatches() -> None:
    actor, incident, initial, active = _state_chain()
    global_incident = _incident(scope=KillSwitchScope.GLOBAL, account_id=None)

    with pytest.raises(ValueError, match="scope does not match"):
        KillSwitchState.activate(
            previous=initial,
            incident=global_incident,
            actor=actor,
            changed_at=T0 + timedelta(minutes=2),
        )
    with pytest.raises(ValueError, match="cannot precede"):
        KillSwitchState.activate(
            previous=initial,
            incident=incident,
            actor=actor,
            changed_at=incident.triggered_at - timedelta(microseconds=1),
        )
    with pytest.raises(ValueError, match="timezone information"):
        KillSwitchState.activate(
            previous=initial,
            incident=incident,
            actor=actor,
            changed_at=datetime(2026, 9, 2, 1, 2),
        )
    with pytest.raises(ValueError, match="only an active"):
        KillSwitchState.recover(
            previous=initial,
            actor=_actor(),
            changed_at=T0 + timedelta(minutes=3),
        )
    with pytest.raises(ValueError, match="cannot precede"):
        KillSwitchState.recover(
            previous=active,
            actor=_actor(),
            changed_at=active.changed_at - timedelta(microseconds=1),
        )


def test_state_rejects_invalid_shape_identity_chain_and_hash() -> None:
    _, _, initial, active = _state_chain()
    cases: tuple[tuple[KillSwitchState, dict[str, object], str], ...] = (
        (initial, {"schema_version": "2"}, "schema_version"),
        (initial, {"status": "INACTIVE"}, "status is invalid"),
        (initial, {"revision": True}, "non-negative integer"),
        (initial, {"revision": -1}, "non-negative integer"),
        (initial, {"previous_state_hash": _digest("prior")}, "initial"),
        (initial, {"latest_incident_hash": _digest("incident")}, "cannot reference"),
        (active, {"previous_state_hash": None}, "requires previous_state_hash"),
        (active, {"active_since": None}, "active kill switch requires"),
        (active, {"active_incident_ids": ()}, "active kill switch requires"),
        (active, {"latest_incident_hash": None}, "active kill switch requires"),
        (active, {"active_since": active.changed_at + timedelta(seconds=1)}, "cannot follow"),
        (active, {"changed_at": datetime(2026, 9, 2, 1, 2)}, "timezone information"),
        (active, {"changed_by_actor_hash": "bad"}, "changed_by_actor_hash"),
        (active, {"previous_state_hash": "bad"}, "previous_state_hash"),
        (active, {"state_id": "kill-switch:forged"}, "state_id"),
        (active, {"state_hash": _digest("forged-state")}, "state_hash"),
    )

    for state, changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(state, **changes)

    recovered = KillSwitchState.recover(
        previous=active,
        actor=_actor(),
        changed_at=active.changed_at + timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="inactive kill switch"):
        _rehash(recovered, "state_hash", active_since=active.active_since)


def test_state_rejects_unsorted_or_duplicate_active_incident_ids() -> None:
    actor, _, _, active = _state_chain()
    second_incident = _incident(
        trigger_source=KillSwitchTriggerSource.RISK,
        source_label="second-state-incident",
        reason_codes=("RISK_LIMIT",),
        triggered_at=T0 + timedelta(minutes=3),
    )
    second_active = KillSwitchState.activate(
        previous=active,
        incident=second_incident,
        actor=actor,
        changed_at=T0 + timedelta(minutes=4),
    )
    reversed_ids = tuple(reversed(second_active.active_incident_ids))
    assert reversed_ids != second_active.active_incident_ids

    with pytest.raises(ValueError, match="unique and sorted"):
        replace(second_active, active_incident_ids=reversed_ids)
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(
            second_active,
            active_incident_ids=(
                second_active.active_incident_ids[0],
                second_active.active_incident_ids[0],
            ),
        )


def test_state_hash_is_timezone_canonical_for_the_same_instant() -> None:
    actor = _system_actor()
    utc_state = KillSwitchState.initial(
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        actor=actor,
        changed_at=T0,
    )
    offset_state = KillSwitchState.initial(
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        actor=actor,
        changed_at=T0.astimezone(timezone(timedelta(hours=8))),
    )

    assert utc_state.state_hash == offset_state.state_hash


def test_activation_request_binds_incident_actor_and_time() -> None:
    incident = _incident()
    actor = _system_actor()
    request = KillSwitchActivationRequest.build(
        request_id="activation-request-1",
        idempotency_key="activation-key-1",
        incident=incident,
        actor=actor,
        requested_at=incident.triggered_at,
    )
    changed_actor = KillSwitchActivationRequest.build(
        request_id=request.request_id,
        idempotency_key=request.idempotency_key,
        incident=incident,
        actor=_agent_actor(),
        requested_at=request.requested_at,
    )

    assert request.incident == incident
    assert request.actor == actor
    assert request.request_hash != changed_actor.request_hash
    with pytest.raises(ValueError, match="cannot precede"):
        KillSwitchActivationRequest.build(
            request_id="activation-request-too-early",
            idempotency_key="activation-key-too-early",
            incident=incident,
            actor=actor,
            requested_at=incident.triggered_at - timedelta(microseconds=1),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema_version": "2"}, "schema_version"),
        ({"request_id": "  "}, "request_id"),
        ({"idempotency_key": ""}, "idempotency_key"),
        ({"requested_at": datetime(2026, 9, 2, 1, 2)}, "timezone information"),
        ({"request_hash": _digest("forged-activation-request")}, "request_hash"),
    ),
)
def test_activation_request_rejects_invalid_or_tampered_content(
    changes: dict[str, object],
    message: str,
) -> None:
    incident = _incident()
    request = KillSwitchActivationRequest.build(
        request_id="activation-request-1",
        idempotency_key="activation-key-1",
        incident=incident,
        actor=_system_actor(),
        requested_at=incident.triggered_at,
    )

    with pytest.raises(ValueError, match=message):
        replace(request, **changes)


def test_activation_request_rejects_mistyped_nested_contracts() -> None:
    incident = _incident()
    request = KillSwitchActivationRequest.build(
        request_id="activation-request-1",
        idempotency_key="activation-key-1",
        incident=incident,
        actor=_system_actor(),
        requested_at=incident.triggered_at,
    )

    with pytest.raises(ValueError, match="incident or actor is invalid"):
        replace(request, incident=object())
    with pytest.raises(ValueError, match="incident or actor is invalid"):
        replace(request, actor=object())


def test_recovery_approval_requires_current_active_state_and_human_risk_admin() -> None:
    _, _, initial, active = _state_chain()
    admin = _actor()
    approval = _approval(state=active, approver=admin)

    assert approval.scope is active.scope
    assert approval.account_id == active.account_id
    assert approval.expected_state_hash == active.state_hash
    assert approval.expected_revision == active.revision
    assert approval.expected_latest_incident_hash == active.latest_incident_hash
    assert approval.approver == admin

    with pytest.raises(ValueError, match="active state"):
        _approval(state=initial, approver=admin)
    for unauthorized in (
        _actor(
            actor_id="operator-1",
            kind=KillSwitchActorKind.HUMAN,
            role=KillSwitchActorRole.OPERATOR,
        ),
        _system_actor(),
        _agent_actor(),
    ):
        with pytest.raises(ValueError, match="human RISK_ADMIN"):
            _approval(state=active, approver=unauthorized)


def test_recovery_approval_cannot_predate_the_state_it_approves() -> None:
    active = _state_chain()[3]

    with pytest.raises(ValueError, match=r"cannot precede|state time"):
        _approval(
            state=active,
            approved_at=active.changed_at - timedelta(microseconds=1),
            expires_at=active.changed_at + timedelta(minutes=5),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema_version": "2"}, "schema_version"),
        ({"expected_revision": 0}, "positive"),
        ({"expected_revision": True}, "positive"),
        ({"expected_state_hash": "bad"}, "expected_state_hash"),
        ({"expected_latest_incident_hash": "bad"}, "expected_latest_incident_hash"),
        ({"approved_at": datetime(2026, 9, 2, 1, 3)}, "timezone information"),
        ({"expires_at": datetime(2026, 9, 2, 1, 8)}, "timezone information"),
        ({"approval_id": "approval:forged"}, "approval_id"),
        ({"approval_hash": _digest("forged-approval")}, "approval_hash"),
    ),
)
def test_recovery_approval_rejects_invalid_or_tampered_content(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_approval(), **changes)


def test_recovery_approval_rejects_mistyped_approver() -> None:
    with pytest.raises(ValueError, match="approver is invalid"):
        replace(_approval(), approver=object())


@pytest.mark.parametrize(
    ("approved_at", "expires_at"),
    (
        (T0 + timedelta(minutes=3), T0 + timedelta(minutes=3)),
        (T0 + timedelta(minutes=3), T0 + timedelta(minutes=2, seconds=59)),
    ),
)
def test_recovery_approval_requires_strictly_future_expiry(
    approved_at: datetime,
    expires_at: datetime,
) -> None:
    with pytest.raises(ValueError, match="expiry must follow"):
        _approval(approved_at=approved_at, expires_at=expires_at)


def test_recovery_request_validity_window_is_closed_open() -> None:
    approval = _approval()
    actor = approval.approver
    at_start = KillSwitchRecoveryRequest.build(
        request_id="recovery-request-at-start",
        idempotency_key="recovery-key-at-start",
        approval=approval,
        actor=actor,
        requested_at=approval.approved_at,
    )
    just_before_expiry = KillSwitchRecoveryRequest.build(
        request_id="recovery-request-before-expiry",
        idempotency_key="recovery-key-before-expiry",
        approval=approval,
        actor=actor,
        requested_at=approval.expires_at - timedelta(microseconds=1),
    )

    assert at_start.request_hash != just_before_expiry.request_hash
    for request_time in (
        approval.approved_at - timedelta(microseconds=1),
        approval.expires_at,
        approval.expires_at + timedelta(microseconds=1),
    ):
        with pytest.raises(ValueError, match="currently valid approval"):
            KillSwitchRecoveryRequest.build(
                request_id=f"recovery-request-{request_time.timestamp()}",
                idempotency_key=f"recovery-key-{request_time.timestamp()}",
                approval=approval,
                actor=actor,
                requested_at=request_time,
            )


def test_agent_and_system_cannot_request_recovery() -> None:
    approval = _approval()

    for unauthorized in (_agent_actor(), _system_actor()):
        with pytest.raises(ValueError, match="only a human RISK_ADMIN"):
            KillSwitchRecoveryRequest.build(
                request_id=f"recovery-request-{unauthorized.actor_id}",
                idempotency_key=f"recovery-key-{unauthorized.actor_id}",
                approval=approval,
                actor=unauthorized,
                requested_at=approval.approved_at,
            )


def test_recovery_request_hash_binds_approval_actor_and_idempotency() -> None:
    approval = _approval()
    request = KillSwitchRecoveryRequest.build(
        request_id="recovery-request-1",
        idempotency_key="recovery-key-1",
        approval=approval,
        actor=approval.approver,
        requested_at=approval.approved_at,
    )

    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"schema_version": "2"}, "schema_version"),
        ({"request_id": ""}, "request_id"),
        ({"idempotency_key": "  "}, "idempotency_key"),
        ({"requested_at": datetime(2026, 9, 2, 1, 3)}, "timezone information"),
        ({"request_hash": _digest("forged-recovery-request")}, "request_hash"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(request, **changes)

    with pytest.raises(ValueError, match="approval or actor is invalid"):
        replace(request, approval=object())
    with pytest.raises(ValueError, match="approval or actor is invalid"):
        replace(request, actor=object())


def test_audit_events_form_a_scope_local_hash_chain() -> None:
    actor, incident, initial, active, initialized, activated = _audit_chain()
    recovered = KillSwitchState.recover(
        previous=active,
        actor=_actor(),
        changed_at=T0 + timedelta(minutes=4),
    )
    recovery_event = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.RECOVERED,
        state_before=active,
        state_after=recovered,
        actor=_actor(),
        operation_hash=_digest("recovery-operation"),
        incident=None,
        reason_codes=("RISK_ADMIN_APPROVED",),
        evidence_hashes=(_approval(state=active).approval_hash,),
        occurred_at=recovered.changed_at,
        previous_event_hash=activated.event_hash,
    )

    assert initialized.state_before_hash is None
    assert initialized.previous_event_hash is None
    assert initialized.state_after_hash == initial.state_hash
    assert activated.previous_event_hash == initialized.event_hash
    assert activated.state_before_hash == initial.state_hash
    assert activated.state_after_hash == active.state_hash
    assert activated.state_revision == active.revision
    assert activated.actor_hash == actor.actor_hash
    assert activated.incident_id == incident.incident_id
    assert activated.reason_codes == incident.reason_codes
    assert activated.evidence_hashes == incident.evidence_hashes
    assert recovery_event.previous_event_hash == activated.event_hash
    assert recovery_event.incident_id is None
    assert len({initialized.event_hash, activated.event_hash, recovery_event.event_hash}) == 3


def test_audit_chain_requires_no_predecessor_for_initial_and_one_afterward() -> None:
    actor, incident, initial, active, _, _ = _audit_chain()

    with pytest.raises(ValueError, match="cannot have prior state or event"):
        KillSwitchAuditEvent.build(
            event_type=KillSwitchEventType.INITIALIZED,
            state_before=None,
            state_after=initial,
            actor=actor,
            operation_hash=_digest("bad-initial-chain"),
            incident=None,
            reason_codes=(),
            evidence_hashes=(),
            occurred_at=initial.changed_at,
            previous_event_hash=_digest("impossible-prior-event"),
        )
    with pytest.raises(ValueError, match="requires prior state and event hashes"):
        KillSwitchAuditEvent.build(
            event_type=KillSwitchEventType.ACTIVATED,
            state_before=initial,
            state_after=active,
            actor=actor,
            operation_hash=_digest("missing-event-chain"),
            incident=incident,
            reason_codes=incident.reason_codes,
            evidence_hashes=incident.evidence_hashes,
            occurred_at=active.changed_at,
            previous_event_hash=None,
        )


def test_activation_audit_event_requires_incident_and_canonical_collections() -> None:
    actor, incident, initial, active, _, activated = _audit_chain()

    with pytest.raises(ValueError, match="require an incident_id"):
        KillSwitchAuditEvent.build(
            event_type=KillSwitchEventType.ACTIVATED,
            state_before=initial,
            state_after=active,
            actor=actor,
            operation_hash=_digest("activation-without-incident"),
            incident=None,
            reason_codes=incident.reason_codes,
            evidence_hashes=incident.evidence_hashes,
            occurred_at=active.changed_at,
            previous_event_hash=_digest("previous-event"),
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(activated, reason_codes=tuple(reversed(activated.reason_codes)))
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(activated, evidence_hashes=("f" * 64, "0" * 64))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema_version": "2"}, "schema_version"),
        ({"event_type": "ACTIVATED"}, "event_type is invalid"),
        ({"state_revision": True}, "state_revision"),
        ({"state_revision": -1}, "state_revision"),
        ({"state_after_hash": "bad"}, "state_after_hash"),
        ({"actor_hash": "bad"}, "actor_hash"),
        ({"operation_hash": "bad"}, "operation_hash"),
        ({"state_before_hash": "bad"}, "state_before_hash"),
        ({"previous_event_hash": "bad"}, "previous_event_hash"),
        ({"incident_id": "  "}, "incident_id"),
        ({"occurred_at": datetime(2026, 9, 2, 1, 2)}, "timezone information"),
        ({"event_id": "kill-event:forged"}, "event_id"),
        ({"event_hash": _digest("forged-event")}, "event_hash"),
    ),
)
def test_audit_event_rejects_invalid_or_tampered_content(
    changes: dict[str, object],
    message: str,
) -> None:
    event = _audit_chain()[5]

    with pytest.raises(ValueError, match=message):
        replace(event, **changes)


def test_transition_binds_operation_states_incident_event_and_version() -> None:
    _, incident, initial, active, _, activated = _audit_chain()
    transition = KillSwitchTransition.build(
        operation_hash=activated.operation_hash,
        state_before=initial,
        state_after=active,
        incident=incident,
        audit_event=activated,
    )

    assert transition.engine_version == KILL_SWITCH_ENGINE_VERSION
    assert transition.state_before == initial
    assert transition.state_after == active
    assert transition.incident == incident
    assert transition.audit_event == activated

    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"engine_version": "kill-switch-engine-v2"}, "engine version"),
        ({"operation_hash": _digest("other-operation")}, "operation"),
        ({"state_before": None}, "state_before"),
        ({"incident": _incident(source_label="other-incident")}, "incident"),
        ({"transition_hash": _digest("forged-transition")}, "transition_hash"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(transition, **changes)


def test_transition_accepts_initial_trigger_added_and_recovery_semantics() -> None:
    actor, first_incident, initial, active, initialized, activated = _audit_chain()
    initial_transition = KillSwitchTransition.build(
        operation_hash=initialized.operation_hash,
        state_before=None,
        state_after=initial,
        incident=None,
        audit_event=initialized,
    )
    second_incident = _incident(
        trigger_source=KillSwitchTriggerSource.RISK,
        source_label="second-transition-incident",
        reason_codes=("SECOND_HARD_LIMIT",),
        triggered_at=T0 + timedelta(minutes=3),
    )
    second_active = KillSwitchState.activate(
        previous=active,
        incident=second_incident,
        actor=actor,
        changed_at=T0 + timedelta(minutes=4),
    )
    trigger_event = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.TRIGGER_ADDED,
        state_before=active,
        state_after=second_active,
        actor=actor,
        operation_hash=_digest("second-trigger-operation"),
        incident=second_incident,
        reason_codes=second_incident.reason_codes,
        evidence_hashes=second_incident.evidence_hashes,
        occurred_at=second_active.changed_at,
        previous_event_hash=activated.event_hash,
    )
    trigger_transition = KillSwitchTransition.build(
        operation_hash=trigger_event.operation_hash,
        state_before=active,
        state_after=second_active,
        incident=second_incident,
        audit_event=trigger_event,
    )
    admin = _actor()
    recovered = KillSwitchState.recover(
        previous=second_active,
        actor=admin,
        changed_at=T0 + timedelta(minutes=5),
    )
    recovery_event = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.RECOVERED,
        state_before=second_active,
        state_after=recovered,
        actor=admin,
        operation_hash=_digest("recovery-transition-operation"),
        incident=None,
        reason_codes=("RISK_ADMIN_APPROVED",),
        evidence_hashes=(
            _approval(
                state=second_active,
                approved_at=T0 + timedelta(minutes=4),
                expires_at=T0 + timedelta(minutes=8),
            ).approval_hash,
        ),
        occurred_at=recovered.changed_at,
        previous_event_hash=trigger_event.event_hash,
    )
    recovery_transition = KillSwitchTransition.build(
        operation_hash=recovery_event.operation_hash,
        state_before=second_active,
        state_after=recovered,
        incident=None,
        audit_event=recovery_event,
    )

    assert initial_transition.audit_event.event_type is KillSwitchEventType.INITIALIZED
    assert trigger_transition.audit_event.event_type is KillSwitchEventType.TRIGGER_ADDED
    assert trigger_transition.incident != first_incident
    assert recovery_transition.audit_event.event_type is KillSwitchEventType.RECOVERED
    assert recovery_transition.incident is None


def test_transition_rejects_mistyped_nested_contracts_and_wrong_after_state() -> None:
    _, incident, initial, active, _, activated = _audit_chain()
    transition = KillSwitchTransition.build(
        operation_hash=activated.operation_hash,
        state_before=initial,
        state_after=active,
        incident=incident,
        audit_event=activated,
    )

    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"state_before": object()}, "state_before is invalid"),
        ({"state_after": object()}, "state_after or audit_event is invalid"),
        ({"audit_event": object()}, "state_after or audit_event is invalid"),
        ({"incident": object()}, "incident is invalid"),
        ({"state_after": initial}, "does not bind state_after"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(transition, **changes)


def test_initial_transition_rejects_nonzero_state_revision() -> None:
    actor, _, _, active, _, _ = _audit_chain()
    impossible_initial_event = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.INITIALIZED,
        state_before=None,
        state_after=active,
        actor=actor,
        operation_hash=_digest("initial-with-active-state"),
        incident=None,
        reason_codes=(),
        evidence_hashes=(),
        occurred_at=active.changed_at,
        previous_event_hash=None,
    )

    with pytest.raises(ValueError, match="revision zero"):
        KillSwitchTransition.build(
            operation_hash=impossible_initial_event.operation_hash,
            state_before=None,
            state_after=active,
            incident=None,
            audit_event=impossible_initial_event,
        )


def test_transition_rejects_active_to_active_without_new_incident() -> None:
    actor, _, _, active, _, activated = _audit_chain()
    later_active = _rehash(
        active,
        "state_hash",
        revision=active.revision + 1,
        changed_at=active.changed_at + timedelta(minutes=1),
        changed_by_actor_hash=actor.actor_hash,
        previous_state_hash=active.state_hash,
    )
    event = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.RECOVERED,
        state_before=active,
        state_after=later_active,
        actor=actor,
        operation_hash=_digest("active-without-trigger"),
        incident=None,
        reason_codes=("NO_INCIDENT",),
        evidence_hashes=(),
        occurred_at=later_active.changed_at,
        previous_event_hash=activated.event_hash,
    )

    with pytest.raises(ValueError, match="status and incident combination"):
        KillSwitchTransition.build(
            operation_hash=event.operation_hash,
            state_before=active,
            state_after=later_active,
            incident=None,
            audit_event=event,
        )


def test_transition_rejects_event_revision_actor_scope_and_time_forgery() -> None:
    actor, incident, initial, active, _, activated = _audit_chain()
    other_actor = _actor(actor_id="different-system", kind=actor.kind, role=actor.role)
    for forged_event, message in (
        (_rehash(activated, "event_hash", state_revision=active.revision + 1), "revision"),
        (_rehash(activated, "event_hash", actor_hash=other_actor.actor_hash), "actor"),
        (
            _rehash(
                activated,
                "event_hash",
                scope=KillSwitchScope.GLOBAL,
                account_id=None,
            ),
            "identity|scope",
        ),
        (
            _rehash(
                activated,
                "event_hash",
                occurred_at=active.changed_at - timedelta(microseconds=1),
            ),
            "time|occurred",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            KillSwitchTransition.build(
                operation_hash=activated.operation_hash,
                state_before=initial,
                state_after=active,
                incident=incident,
                audit_event=forged_event,
            )


def test_transition_rejects_impossible_state_revision_chain() -> None:
    actor, _, initial, _, initialized, _ = _audit_chain()
    unrelated_initial = KillSwitchState.initial(
        scope=initial.scope,
        account_id=initial.account_id,
        actor=actor,
        changed_at=initial.changed_at + timedelta(minutes=20),
    )
    impossible_event = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.RECOVERED,
        state_before=initial,
        state_after=unrelated_initial,
        actor=actor,
        operation_hash=_digest("impossible-recovery"),
        incident=None,
        reason_codes=("IMPOSSIBLE",),
        evidence_hashes=(),
        occurred_at=unrelated_initial.changed_at,
        previous_event_hash=initialized.event_hash,
    )

    with pytest.raises(ValueError, match="revision chain"):
        KillSwitchTransition.build(
            operation_hash=impossible_event.operation_hash,
            state_before=initial,
            state_after=unrelated_initial,
            incident=None,
            audit_event=impossible_event,
        )


def test_transition_rejects_event_type_that_contradicts_state_change() -> None:
    actor, incident, initial, active, initialized, _ = _audit_chain()
    mislabeled = KillSwitchAuditEvent.build(
        event_type=KillSwitchEventType.RECOVERED,
        state_before=initial,
        state_after=active,
        actor=actor,
        operation_hash=_digest("mislabeled-activation"),
        incident=incident,
        reason_codes=incident.reason_codes,
        evidence_hashes=incident.evidence_hashes,
        occurred_at=active.changed_at,
        previous_event_hash=initialized.event_hash,
    )

    with pytest.raises(ValueError, match=r"event_type|state change"):
        KillSwitchTransition.build(
            operation_hash=mislabeled.operation_hash,
            state_before=initial,
            state_after=active,
            incident=incident,
            audit_event=mislabeled,
        )


def test_gate_request_binds_original_request_batch_account_and_time() -> None:
    request = _gate_request()
    changed_source = KillSwitchGateRequest.build(
        request_id=request.request_id,
        account_id=request.account_id,
        idempotency_key=request.idempotency_key,
        batch_hash=request.batch_hash,
        source_request_hash=_digest("different-paper-request"),
        checked_at=request.checked_at,
    )
    changed_batch = KillSwitchGateRequest.build(
        request_id=request.request_id,
        account_id=request.account_id,
        idempotency_key=request.idempotency_key,
        batch_hash=_digest("different-order-batch"),
        source_request_hash=request.source_request_hash,
        checked_at=request.checked_at,
    )

    assert request.request_hash != changed_source.request_hash
    assert request.request_hash != changed_batch.request_hash


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema_version": "2"}, "schema_version"),
        ({"request_id": ""}, "request_id"),
        ({"account_id": "  "}, "account_id"),
        ({"idempotency_key": ""}, "idempotency_key"),
        ({"batch_hash": "bad"}, "batch_hash"),
        ({"source_request_hash": "bad"}, "source_request_hash"),
        ({"checked_at": datetime(2026, 9, 2, 1, 10)}, "timezone information"),
        ({"request_hash": _digest("forged-gate-request")}, "request_hash"),
    ),
)
def test_gate_request_rejects_invalid_or_tampered_content(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_gate_request(), **changes)


def test_gate_allowed_requires_both_inactive_state_hashes_and_no_block_metadata() -> None:
    actor = _system_actor()
    global_state = KillSwitchState.initial(
        scope=KillSwitchScope.GLOBAL,
        account_id=None,
        actor=actor,
        changed_at=T0,
    )
    account_state = KillSwitchState.initial(
        scope=KillSwitchScope.ACCOUNT,
        account_id="paper-account-1",
        actor=actor,
        changed_at=T0,
    )
    decision = KillSwitchGateDecision.build(
        request=_gate_request(),
        status=KillSwitchGateStatus.ALLOWED,
        global_state_hash=global_state.state_hash,
        account_state_hash=account_state.state_hash,
    )

    assert decision.allowed
    assert decision.status is KillSwitchGateStatus.ALLOWED
    assert decision.blocking_state_hashes == ()
    assert decision.reason_codes == ()
    assert decision.incident_ids == ()
    assert decision.audit_event_hash is None

    forbidden_values: tuple[dict[str, object], ...] = (
        {"global_state_hash": None},
        {"account_state_hash": None},
        {"blocking_state_hashes": (global_state.state_hash,)},
        {"reason_codes": (KillSwitchGateReason.GLOBAL_ACTIVE,)},
        {"incident_ids": ("incident:unexpected",)},
        {"audit_event_hash": _digest("unexpected-audit")},
    )
    for changes in forbidden_values:
        with pytest.raises(ValueError, match="inactive states only"):
            _rehash(decision, "decision_hash", **changes)


@pytest.mark.parametrize(
    ("reason", "state_slot"),
    (
        (KillSwitchGateReason.GLOBAL_ACTIVE, "global"),
        (KillSwitchGateReason.ACCOUNT_ACTIVE, "account"),
    ),
)
def test_gate_blocked_binds_active_state_incident_reason_and_audit(
    reason: KillSwitchGateReason,
    state_slot: str,
) -> None:
    _, incident, _, active, _, activated = _audit_chain()
    global_hash = active.state_hash if state_slot == "global" else _digest("inactive-global")
    account_hash = active.state_hash if state_slot == "account" else _digest("inactive-account")
    decision = KillSwitchGateDecision.build(
        request=_gate_request(),
        status=KillSwitchGateStatus.BLOCKED,
        global_state_hash=global_hash,
        account_state_hash=account_hash,
        blocking_state_hashes=(active.state_hash,),
        reason_codes=(reason,),
        incident_ids=(incident.incident_id,),
        audit_event_hash=activated.event_hash,
    )

    assert not decision.allowed
    assert decision.status is KillSwitchGateStatus.BLOCKED
    assert decision.blocking_state_hashes == (active.state_hash,)
    assert decision.reason_codes == (reason,)
    assert decision.incident_ids == (incident.incident_id,)
    assert decision.audit_event_hash == activated.event_hash


def test_gate_blocked_fail_closed_when_state_or_repository_is_unavailable() -> None:
    request = _gate_request()
    for reason in (
        KillSwitchGateReason.STATE_UNAVAILABLE,
        KillSwitchGateReason.REPOSITORY_FAILURE,
    ):
        decision = KillSwitchGateDecision.build(
            request=request,
            status=KillSwitchGateStatus.BLOCKED,
            global_state_hash=None,
            account_state_hash=None,
            reason_codes=(reason,),
        )
        assert not decision.allowed
        assert decision.reason_codes == (reason,)


def test_gate_builder_canonicalizes_blockers_reasons_and_incidents() -> None:
    hash_a, hash_b = sorted((_digest("blocking-a"), _digest("blocking-b")))
    decision = KillSwitchGateDecision.build(
        request=_gate_request(),
        status=KillSwitchGateStatus.BLOCKED,
        global_state_hash=hash_a,
        account_state_hash=hash_b,
        blocking_state_hashes=(hash_b, hash_a, hash_b),
        reason_codes=(
            KillSwitchGateReason.REPOSITORY_FAILURE,
            KillSwitchGateReason.ACCOUNT_ACTIVE,
            KillSwitchGateReason.REPOSITORY_FAILURE,
        ),
        incident_ids=("incident:z", "incident:a", "incident:z"),
        audit_event_hash=_digest("blocked-audit"),
    )

    assert decision.blocking_state_hashes == (hash_a, hash_b)
    assert decision.reason_codes == (
        KillSwitchGateReason.ACCOUNT_ACTIVE,
        KillSwitchGateReason.REPOSITORY_FAILURE,
    )
    assert decision.incident_ids == ("incident:a", "incident:z")


def test_gate_decision_rejects_invalid_status_collections_version_and_hash() -> None:
    request = _gate_request()
    blocked = KillSwitchGateDecision.build(
        request=request,
        status=KillSwitchGateStatus.BLOCKED,
        global_state_hash=None,
        account_state_hash=None,
        reason_codes=(KillSwitchGateReason.STATE_UNAVAILABLE,),
    )
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"engine_version": "kill-switch-engine-v2"}, "engine version"),
        ({"request_hash": "bad"}, "request_hash"),
        ({"status": "BLOCKED"}, "status is invalid"),
        ({"checked_at": datetime(2026, 9, 2, 1, 10)}, "timezone information"),
        ({"global_state_hash": "bad"}, "global_state_hash"),
        ({"account_state_hash": "bad"}, "account_state_hash"),
        ({"audit_event_hash": "bad"}, "audit_event_hash"),
        ({"blocking_state_hashes": ("f" * 64, "0" * 64)}, "unique and sorted"),
        ({"reason_codes": ("STATE_UNAVAILABLE",)}, "reason code is invalid"),
        (
            {
                "reason_codes": (
                    KillSwitchGateReason.STATE_UNAVAILABLE,
                    KillSwitchGateReason.STATE_UNAVAILABLE,
                )
            },
            "unique and sorted",
        ),
        ({"incident_ids": ("incident:z", "incident:a")}, r"unique.*sorted"),
        ({"decision_hash": _digest("forged-gate-decision")}, "decision_hash"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(blocked, **changes)

    with pytest.raises(ValueError, match="requires a reason"):
        KillSwitchGateDecision.build(
            request=request,
            status=KillSwitchGateStatus.BLOCKED,
            global_state_hash=None,
            account_state_hash=None,
        )


def test_gate_decision_rejects_empty_incident_identifier() -> None:
    blocked = KillSwitchGateDecision.build(
        request=_gate_request(),
        status=KillSwitchGateStatus.BLOCKED,
        global_state_hash=None,
        account_state_hash=None,
        reason_codes=(KillSwitchGateReason.STATE_UNAVAILABLE,),
    )

    with pytest.raises(ValueError, match="incident"):
        _rehash(blocked, "decision_hash", incident_ids=("",))
