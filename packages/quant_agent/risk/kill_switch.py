"""Audited global/account Kill Switch with privileged recovery."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware


class KillSwitchScope(StrEnum):
    GLOBAL = "GLOBAL"
    ACCOUNT = "ACCOUNT"


@dataclass(frozen=True, slots=True)
class KillSwitchAudit:
    sequence: int
    action: str
    scope: KillSwitchScope
    account_id: str | None
    reason: str
    actor_id: str
    actor_role: str
    occurred_at: datetime
    incident_snapshot_hash: str


class KillSwitch:
    def __init__(self) -> None:
        self._global_active = False
        self._accounts: set[str] = set()
        self._audit: list[KillSwitchAudit] = []

    def trigger(
        self,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        reason: str,
        actor_id: str,
        actor_role: str,
        occurred_at: datetime,
        incident_snapshot_hash: str,
    ) -> KillSwitchAudit:
        ensure_aware(occurred_at)
        if not reason or not incident_snapshot_hash:
            raise ValueError("trigger reason and incident snapshot hash are required")
        if scope is KillSwitchScope.ACCOUNT and not account_id:
            raise ValueError("account scope requires account_id")
        if scope is KillSwitchScope.GLOBAL:
            self._global_active = True
        else:
            assert account_id is not None
            self._accounts.add(account_id)
        record = KillSwitchAudit(
            sequence=len(self._audit) + 1,
            action="TRIGGER",
            scope=scope,
            account_id=account_id,
            reason=reason,
            actor_id=actor_id,
            actor_role=actor_role,
            occurred_at=occurred_at,
            incident_snapshot_hash=incident_snapshot_hash,
        )
        self._audit.append(record)
        return record

    def trigger_from_reconciliation(
        self,
        account_id: str,
        *,
        trigger_stop: bool,
        occurred_at: datetime,
        incident_snapshot_hash: str,
    ) -> KillSwitchAudit | None:
        if not trigger_stop:
            return None
        return self.trigger(
            scope=KillSwitchScope.ACCOUNT,
            account_id=account_id,
            reason="critical reconciliation difference",
            actor_id="system",
            actor_role="SYSTEM",
            occurred_at=occurred_at,
            incident_snapshot_hash=incident_snapshot_hash,
        )

    def recover(
        self,
        *,
        scope: KillSwitchScope,
        account_id: str | None,
        reason: str,
        actor_id: str,
        actor_role: str,
        occurred_at: datetime,
        reviewed_snapshot_hash: str,
    ) -> KillSwitchAudit:
        ensure_aware(occurred_at)
        if actor_role != "RISK_ADMIN":
            raise PermissionError("only RISK_ADMIN can recover Kill Switch")
        if not reason or not reviewed_snapshot_hash:
            raise ValueError("recovery reason and reviewed snapshot hash are required")
        if scope is KillSwitchScope.GLOBAL:
            self._global_active = False
        elif account_id:
            self._accounts.discard(account_id)
        else:
            raise ValueError("account scope requires account_id")
        record = KillSwitchAudit(
            sequence=len(self._audit) + 1,
            action="RECOVER",
            scope=scope,
            account_id=account_id,
            reason=reason,
            actor_id=actor_id,
            actor_role=actor_role,
            occurred_at=occurred_at,
            incident_snapshot_hash=reviewed_snapshot_hash,
        )
        self._audit.append(record)
        return record

    def order_allowed(self, account_id: str) -> bool:
        return not self._global_active and account_id not in self._accounts

    @property
    def audit_log(self) -> tuple[KillSwitchAudit, ...]:
        return tuple(self._audit)
