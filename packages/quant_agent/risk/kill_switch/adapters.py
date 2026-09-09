"""Explicit adapters from upstream safety evidence into kill-switch commands."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from quant_agent.reconciliation import ReconciliationResult, ReconciliationStopAction

from .contracts import (
    KillSwitchActivationRequest,
    KillSwitchActor,
    KillSwitchActorKind,
    KillSwitchActorRole,
    KillSwitchIncident,
    KillSwitchScope,
    KillSwitchTriggerSource,
)


def activation_request_from_reconciliation(
    *,
    result: ReconciliationResult,
    actor: KillSwitchActor,
    request_id: str,
    idempotency_key: str,
    requested_at: datetime,
) -> KillSwitchActivationRequest:
    """Convert a complete validated result; a detached stop signal is never accepted."""

    if not isinstance(result, ReconciliationResult):
        raise TypeError("result must be a complete ReconciliationResult")
    result = replace(result)
    if actor.kind is not KillSwitchActorKind.SYSTEM or actor.role is not KillSwitchActorRole.SYSTEM:
        raise ValueError("automatic reconciliation activation requires a SYSTEM actor")
    signal = result.stop_signal
    if not signal.required or signal.action is not ReconciliationStopAction.STOP_NEW_ORDERS:
        raise ValueError("reconciliation result does not require kill-switch activation")
    incident = KillSwitchIncident.build(
        scope=KillSwitchScope.ACCOUNT,
        account_id=result.account_id,
        trigger_source=KillSwitchTriggerSource.RECONCILIATION,
        reason_codes=tuple(item.value for item in signal.reason_codes),
        evidence_hashes=signal.trigger_hashes,
        source_reference_hash=result.result_hash,
        summary=(
            f"reconciliation {result.reconciliation_id} requires new-order stop "
            f"at severity {result.max_severity.value}"
        ),
        triggered_at=result.reconciled_at,
    )
    return KillSwitchActivationRequest.build(
        request_id=request_id,
        idempotency_key=idempotency_key,
        incident=incident,
        actor=actor,
        requested_at=requested_at,
    )


__all__ = ["activation_request_from_reconciliation"]
