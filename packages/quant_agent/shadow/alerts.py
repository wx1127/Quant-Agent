"""Bridge shadow-run safety evidence into the shared P0-P3 alert pipeline."""

from __future__ import annotations

from quant_agent.observability.alerting import (
    AlertEvent,
    AlertSeverity,
    create_alert_event,
)
from quant_agent.shadow.models import ShadowAcceptance, ShadowDayEvidence


def shadow_alerts(
    evidence: ShadowDayEvidence,
    acceptance: ShadowAcceptance,
) -> tuple[AlertEvent, ...]:
    events: list[AlertEvent] = []
    if evidence.executable_orders_emitted:
        events.append(
            create_alert_event(
                "shadow-executable-order",
                AlertSeverity.P0,
                "Shadow mode emitted executable orders",
                "Shadow execution boundary was violated; keep Kill Switch active.",
                "shadow-run",
                "docs/08-shadow-runbook.md#p0-executable-order",
                {
                    "count": str(evidence.executable_orders_emitted),
                    "trading_date": evidence.trading_date.isoformat(),
                },
                evidence.observed_at,
            )
        )
    if evidence.future_data_violations:
        events.append(
            create_alert_event(
                "shadow-future-data",
                AlertSeverity.P0,
                "Shadow run detected future data",
                "Point-in-time evidence failed; invalidate the affected run.",
                "shadow-run",
                "docs/08-shadow-runbook.md#p0-future-data",
                {
                    "count": str(evidence.future_data_violations),
                    "trading_date": evidence.trading_date.isoformat(),
                },
                evidence.observed_at,
            )
        )
    if not evidence.pipeline_succeeded:
        events.append(
            create_alert_event(
                "shadow-pipeline-failed",
                AlertSeverity.P1,
                "Shadow daily pipeline failed",
                "Daily evidence is incomplete and must be recovered before acceptance.",
                "shadow-run",
                "docs/08-shadow-runbook.md#p1-pipeline-failure",
                {"trading_date": evidence.trading_date.isoformat()},
                evidence.observed_at,
            )
        )
    if not evidence.data_complete or not evidence.report_generated:
        events.append(
            create_alert_event(
                "shadow-daily-incomplete",
                AlertSeverity.P2,
                "Shadow daily outputs are incomplete",
                "Review data arrival and report generation for the affected session.",
                "shadow-run",
                "docs/08-shadow-runbook.md#p2-incomplete-day",
                {
                    "data_complete": str(evidence.data_complete).lower(),
                    "report_generated": str(evidence.report_generated).lower(),
                    "trading_date": evidence.trading_date.isoformat(),
                },
                evidence.observed_at,
            )
        )
    for incident in evidence.opened_incidents:
        events.append(
            create_alert_event(
                f"shadow-incident:{incident.incident_id}",
                AlertSeverity(incident.severity),
                f"Shadow incident {incident.incident_id}",
                incident.summary,
                incident.component,
                "docs/08-shadow-runbook.md#incident-handling",
                {
                    "incident_id": incident.incident_id,
                    "trading_date": evidence.trading_date.isoformat(),
                },
                evidence.observed_at,
            )
        )
    if acceptance.missing_trading_days:
        events.append(
            create_alert_event(
                "shadow-missing-trading-day",
                AlertSeverity.P1,
                "Shadow evidence has a trading-day gap",
                "Recover the missing day; later records do not restore continuity.",
                "shadow-run",
                "docs/08-shadow-runbook.md#p1-missing-day",
                {
                    "missing_count": str(len(acceptance.missing_trading_days)),
                    "trading_date": evidence.trading_date.isoformat(),
                },
                evidence.observed_at,
            )
        )
    return tuple(events)
