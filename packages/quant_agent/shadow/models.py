"""Immutable shadow-session, daily evidence and acceptance contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware


class IncidentSeverity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class ShadowAcceptanceStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    PASSED = "PASSED"


@dataclass(frozen=True, slots=True)
class ShadowSessionConfig:
    session_id: str
    started_on: date
    required_trading_days: int
    provider: str
    universe_version: str
    strategy_versions: tuple[str, ...]
    minimum_data_success_rate: float = 0.98
    minimum_report_success_rate: float = 0.98

    def __post_init__(self) -> None:
        if not self.session_id.startswith("shadow-"):
            raise ValueError("shadow session id must start with shadow-")
        if self.required_trading_days < 20:
            raise ValueError("shadow acceptance requires at least 20 trading days")
        if not self.provider or not self.universe_version or not self.strategy_versions:
            raise ValueError("provider, universe and strategy versions are required")
        for value in (
            self.minimum_data_success_rate,
            self.minimum_report_success_rate,
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError("shadow success thresholds must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ShadowIncident:
    incident_id: str
    severity: IncidentSeverity
    component: str
    summary: str

    def __post_init__(self) -> None:
        if not self.incident_id.startswith("inc-"):
            raise ValueError("incident id must start with inc-")
        if not self.component or not self.summary:
            raise ValueError("incident component and summary are required")
        forbidden = ("token=", "password=", "secret=", "authorization:")
        normalized = self.summary.casefold()
        if any(value in normalized for value in forbidden):
            raise ValueError("incident summary must not contain credentials")


@dataclass(frozen=True, slots=True)
class ShadowDayEvidence:
    trading_date: date
    observed_at: datetime
    market_data_as_of: datetime
    input_snapshot_hash: str
    virtual_account_snapshot_hash: str
    data_version: str
    regime: str
    mainline_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    data_complete: bool
    report_generated: bool
    pipeline_succeeded: bool
    reconciliation_matched: bool
    future_data_violations: int
    executable_orders_emitted: int
    opened_incidents: tuple[ShadowIncident, ...] = ()
    resolved_incident_ids: tuple[str, ...] = ()
    manual_intervention_minutes: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        ensure_aware(self.observed_at)
        ensure_aware(self.market_data_as_of)
        if self.trading_date > self.observed_at.date():
            raise ValueError("trading date cannot be after observation date")
        if self.market_data_as_of > self.observed_at and self.future_data_violations == 0:
            raise ValueError("future-dated input must be recorded as a violation")
        for snapshot_hash in (
            self.input_snapshot_hash,
            self.virtual_account_snapshot_hash,
        ):
            if len(snapshot_hash) != 64 or any(
                value not in "0123456789abcdef" for value in snapshot_hash
            ):
                raise ValueError("snapshot hashes must be lowercase SHA-256")
        if not self.data_version or not self.regime:
            raise ValueError("data version and regime are required")
        if len(set(self.mainline_ids)) != len(self.mainline_ids):
            raise ValueError("mainline ids must be unique")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate ids must be unique")
        if self.future_data_violations < 0 or self.executable_orders_emitted < 0:
            raise ValueError("violation and order counts cannot be negative")
        if self.manual_intervention_minutes < 0:
            raise ValueError("manual intervention cannot be negative")
        opened_ids = [incident.incident_id for incident in self.opened_incidents]
        if len(set(opened_ids)) != len(opened_ids):
            raise ValueError("opened incident ids must be unique")
        if set(opened_ids) & set(self.resolved_incident_ids):
            raise ValueError("an incident cannot open and resolve in the same evidence record")
        if not self.reconciliation_matched and not any(
            incident.severity in {IncidentSeverity.P0, IncidentSeverity.P1}
            for incident in self.opened_incidents
        ):
            raise ValueError("reconciliation mismatch requires a P0/P1 incident")


@dataclass(frozen=True, slots=True)
class ShadowAcceptance:
    status: ShadowAcceptanceStatus
    observed_trading_days: int
    consecutive_trading_days: int
    required_trading_days: int
    missing_trading_days: tuple[date, ...]
    data_success_rate: float
    report_success_rate: float
    pipeline_success_rate: float
    average_mainline_stability: float
    average_candidate_stability: float
    future_data_violations: int
    executable_orders_emitted: int
    unreconciled_trading_days: tuple[date, ...]
    open_high_incident_ids: tuple[str, ...]
    reasons: tuple[str, ...]
