"""Twenty-trading-day shadow-run evaluation and stability metrics."""

from __future__ import annotations

from datetime import date
from itertools import pairwise

from quant_agent.shadow.ledger import ShadowLedgerRecord
from quant_agent.shadow.models import (
    IncidentSeverity,
    ShadowAcceptance,
    ShadowAcceptanceStatus,
    ShadowDayEvidence,
    ShadowSessionConfig,
)


class ShadowRunEvaluator:
    def evaluate(
        self,
        config: ShadowSessionConfig,
        records: tuple[ShadowLedgerRecord, ...],
        *,
        trading_calendar: tuple[date, ...],
    ) -> ShadowAcceptance:
        calendar = tuple(day for day in sorted(set(trading_calendar)) if day >= config.started_on)
        if len(calendar) < config.required_trading_days:
            raise ValueError("calendar must cover the full shadow acceptance window")
        evidence = tuple(record.evidence for record in records)
        observed_dates = tuple(item.trading_date for item in evidence)
        if any(day not in calendar for day in observed_dates):
            raise ValueError("shadow evidence contains a non-trading date")

        expected_through_latest: tuple[date, ...] = ()
        if observed_dates:
            latest_index = calendar.index(observed_dates[-1])
            expected_through_latest = calendar[: latest_index + 1]
        missing = tuple(day for day in expected_through_latest if day not in observed_dates)
        consecutive = _consecutive_days(observed_dates, calendar)
        data_rate = _rate(evidence, "data_complete")
        report_rate = _rate(evidence, "report_generated")
        pipeline_rate = _rate(evidence, "pipeline_succeeded")
        future_violations = sum(item.future_data_violations for item in evidence)
        executable_orders = sum(item.executable_orders_emitted for item in evidence)
        unreconciled = tuple(
            item.trading_date for item in evidence if not item.reconciliation_matched
        )
        open_high = _open_high_incidents(evidence)
        reasons: list[str] = []
        if consecutive < config.required_trading_days:
            reasons.append(
                f"requires {config.required_trading_days} consecutive trading days; "
                f"current={consecutive}"
            )
        if missing:
            reasons.append("missing expected trading-day evidence")
        if data_rate < config.minimum_data_success_rate:
            reasons.append("data success rate is below threshold")
        if report_rate < config.minimum_report_success_rate:
            reasons.append("report success rate is below threshold")
        if pipeline_rate < config.minimum_data_success_rate:
            reasons.append("pipeline success rate is below threshold")
        if future_violations:
            reasons.append("future-data violations must be zero")
        if executable_orders:
            reasons.append("shadow mode emitted executable orders")
        if open_high:
            reasons.append("P0/P1 incidents remain open")

        if not evidence:
            status = ShadowAcceptanceStatus.NOT_STARTED
        elif missing or future_violations or executable_orders or open_high:
            status = ShadowAcceptanceStatus.BLOCKED
        elif consecutive < config.required_trading_days:
            status = ShadowAcceptanceStatus.RUNNING
        elif reasons:
            status = ShadowAcceptanceStatus.BLOCKED
        else:
            status = ShadowAcceptanceStatus.PASSED
        return ShadowAcceptance(
            status=status,
            observed_trading_days=len(evidence),
            consecutive_trading_days=consecutive,
            required_trading_days=config.required_trading_days,
            missing_trading_days=missing,
            data_success_rate=data_rate,
            report_success_rate=report_rate,
            pipeline_success_rate=pipeline_rate,
            average_mainline_stability=_average_stability(
                tuple(item.mainline_ids for item in evidence)
            ),
            average_candidate_stability=_average_stability(
                tuple(item.candidate_ids for item in evidence)
            ),
            future_data_violations=future_violations,
            executable_orders_emitted=executable_orders,
            unreconciled_trading_days=unreconciled,
            open_high_incident_ids=open_high,
            reasons=tuple(reasons),
        )


def _rate(evidence: tuple[ShadowDayEvidence, ...], field: str) -> float:
    if not evidence:
        return 0.0
    return sum(bool(getattr(item, field)) for item in evidence) / len(evidence)


def _consecutive_days(observed: tuple[date, ...], calendar: tuple[date, ...]) -> int:
    count = 0
    for expected, actual in zip(calendar, observed, strict=False):
        if actual != expected:
            break
        count += 1
    return count


def _average_stability(values: tuple[tuple[str, ...], ...]) -> float:
    if len(values) < 2:
        return 1.0 if values else 0.0
    scores: list[float] = []
    for previous, current in pairwise(values):
        union = set(previous) | set(current)
        scores.append(len(set(previous) & set(current)) / len(union) if union else 1.0)
    return sum(scores) / len(scores)


def _open_high_incidents(evidence: tuple[ShadowDayEvidence, ...]) -> tuple[str, ...]:
    active: dict[str, IncidentSeverity] = {}
    for item in evidence:
        for incident in item.opened_incidents:
            active[incident.incident_id] = incident.severity
        for incident_id in item.resolved_incident_ids:
            active.pop(incident_id, None)
    return tuple(
        sorted(
            incident_id
            for incident_id, severity in active.items()
            if severity in {IncidentSeverity.P0, IncidentSeverity.P1}
        )
    )
