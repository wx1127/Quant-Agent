"""Deterministic alert evaluation for API and execution safety signals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AlertSeverity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


@dataclass(frozen=True, slots=True)
class Alert:
    code: str
    severity: AlertSeverity
    message: str


def evaluate_alerts(
    *,
    requests: int,
    errors: int,
    average_seconds: float,
    reconciliation_failures: int = 0,
    kill_switch_active: bool = False,
) -> tuple[Alert, ...]:
    alerts: list[Alert] = []
    if reconciliation_failures > 0:
        alerts.append(
            Alert("RECONCILIATION_FAILURE", AlertSeverity.P0, "account reconciliation mismatch")
        )
    if kill_switch_active:
        alerts.append(Alert("KILL_SWITCH_ACTIVE", AlertSeverity.P0, "kill switch is active"))
    if requests and errors / requests >= 0.05:
        alerts.append(Alert("API_ERROR_RATE", AlertSeverity.P1, "API error rate exceeds 5%"))
    if average_seconds >= 1.0:
        alerts.append(
            Alert("API_LATENCY", AlertSeverity.P2, "API average latency exceeds 1 second")
        )
    return tuple(alerts)
