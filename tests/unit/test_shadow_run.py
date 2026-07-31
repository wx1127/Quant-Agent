from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from quant_agent.observability.alerting import AlertSeverity
from quant_agent.shadow.alerts import shadow_alerts
from quant_agent.shadow.io import (
    load_shadow_config,
    load_trading_calendar,
    shadow_day_to_mapping,
)
from quant_agent.shadow.ledger import ShadowEvidenceLedger
from quant_agent.shadow.models import (
    IncidentSeverity,
    ShadowAcceptanceStatus,
    ShadowDayEvidence,
    ShadowIncident,
)
from quant_agent.shadow.reporting import write_shadow_report
from quant_agent.shadow.session import ShadowRunEvaluator

ROOT = Path(__file__).resolve().parents[2]
START = date(2026, 8, 3)


def _calendar() -> tuple[date, ...]:
    days: list[date] = []
    current = START
    while len(days) < 20:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _evidence(day: date, *, index: int = 0) -> ShadowDayEvidence:
    observed_at = datetime.fromisoformat(f"{day}T15:45:00+08:00")
    return ShadowDayEvidence(
        trading_date=day,
        observed_at=observed_at,
        market_data_as_of=observed_at - timedelta(minutes=15),
        input_snapshot_hash=f"{index + 1:064x}",
        virtual_account_snapshot_hash=f"{index + 101:064x}",
        data_version=f"market-{day}",
        regime="UPTREND",
        mainline_ids=("theme-ai", "theme-broker"),
        candidate_ids=("CN.SSE.600000", "CN.SZSE.000001"),
        data_complete=True,
        report_generated=True,
        pipeline_succeeded=True,
        reconciliation_matched=True,
        future_data_violations=0,
        executable_orders_emitted=0,
    )


def test_hash_chained_ledger_is_idempotent_and_detects_tampering(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "shadow.jsonl"
    ledger = ShadowEvidenceLedger(ledger_path)
    first = ledger.append(_evidence(START))
    duplicate = ledger.append(_evidence(START))
    assert duplicate == first
    with pytest.raises(ValueError, match="already exists"):
        ledger.append(replace(_evidence(START), regime="DOWNTREND"))

    ledger.append(_evidence(_calendar()[1], index=1))
    records = ledger.read_all()
    assert len(records) == 2
    assert records[1].previous_record_hash == records[0].record_hash

    tampered = ledger_path.read_text(encoding="utf-8").replace(
        '"regime": "UPTREND"',
        '"regime": "DOWNTREND"',
        1,
    )
    ledger_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="record hash"):
        ledger.read_all()


def test_twenty_day_gate_passes_only_complete_safe_continuous_evidence(
    tmp_path: Path,
) -> None:
    config = load_shadow_config(ROOT / "configs" / "shadow" / "shadow_20260803_v1.toml")
    calendar = _calendar()
    ledger = ShadowEvidenceLedger(tmp_path / "shadow.jsonl")
    evaluator = ShadowRunEvaluator()
    empty = evaluator.evaluate(config, (), trading_calendar=calendar)
    assert empty.status is ShadowAcceptanceStatus.NOT_STARTED
    assert empty.consecutive_trading_days == 0

    for index, day in enumerate(calendar):
        ledger.append(_evidence(day, index=index))
        current = evaluator.evaluate(
            config,
            ledger.read_all(),
            trading_calendar=calendar,
        )
        if index < 19:
            assert current.status is ShadowAcceptanceStatus.RUNNING
    assert current.status is ShadowAcceptanceStatus.PASSED
    assert current.consecutive_trading_days == 20
    assert current.data_success_rate == current.report_success_rate == 1.0
    assert current.future_data_violations == 0
    assert current.executable_orders_emitted == 0
    assert current.average_mainline_stability == 1.0

    json_path, markdown_path = write_shadow_report(
        tmp_path / "reports",
        config,
        current,
        generated_on=calendar[-1],
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["acceptance"]["status"] == "PASSED"
    assert "20/20" in markdown_path.read_text(encoding="utf-8")


def test_gap_safety_violation_and_open_incident_fail_closed(tmp_path: Path) -> None:
    config = load_shadow_config(ROOT / "configs" / "shadow" / "shadow_20260803_v1.toml")
    calendar = _calendar()
    gap_ledger = ShadowEvidenceLedger(tmp_path / "gap.jsonl")
    gap_ledger.append(_evidence(calendar[0]))
    gap_ledger.append(_evidence(calendar[2], index=2))
    gap = ShadowRunEvaluator().evaluate(
        config,
        gap_ledger.read_all(),
        trading_calendar=calendar,
    )
    assert gap.status is ShadowAcceptanceStatus.BLOCKED
    assert gap.missing_trading_days == (calendar[1],)

    incident = ShadowIncident(
        "inc-shadow-001",
        IncidentSeverity.P0,
        "shadow-run",
        "executable order boundary violation",
    )
    unsafe = replace(
        _evidence(calendar[0]),
        executable_orders_emitted=1,
        reconciliation_matched=False,
        opened_incidents=(incident,),
    )
    unsafe_ledger = ShadowEvidenceLedger(tmp_path / "unsafe.jsonl")
    unsafe_ledger.append(unsafe)
    acceptance = ShadowRunEvaluator().evaluate(
        config,
        unsafe_ledger.read_all(),
        trading_calendar=calendar,
    )
    assert acceptance.status is ShadowAcceptanceStatus.BLOCKED
    alerts = shadow_alerts(unsafe, acceptance)
    assert any(
        event.policy_id == "shadow-executable-order" and event.severity is AlertSeverity.P0
        for event in alerts
    )
    assert any(event.policy_id == "shadow-incident:inc-shadow-001" for event in alerts)


def test_config_calendar_and_evidence_validation() -> None:
    planned = load_trading_calendar(ROOT / "configs" / "shadow" / "calendar_20260803_20d.json")
    assert planned == _calendar()
    with pytest.raises(ValueError, match="future-dated"):
        replace(
            _evidence(START),
            market_data_as_of=datetime.fromisoformat("2026-08-03T16:00:00+08:00"),
        )
    with pytest.raises(ValueError, match="P0/P1"):
        replace(_evidence(START), reconciliation_matched=False)
    mapping = shadow_day_to_mapping(_evidence(START))
    assert mapping["executable_orders_emitted"] == 0
    assert mapping["reconciliation_matched"] is True
