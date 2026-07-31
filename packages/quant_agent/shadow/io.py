"""Strict JSON/TOML input adapters for shadow-running tools."""

from __future__ import annotations

import json
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Any

from quant_agent.shadow.models import (
    IncidentSeverity,
    ShadowDayEvidence,
    ShadowIncident,
    ShadowRunMode,
    ShadowSessionConfig,
)


def load_shadow_config(path: str | Path) -> ShadowSessionConfig:
    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)["session"]
    return ShadowSessionConfig(
        session_id=payload["session_id"],
        started_on=date.fromisoformat(payload["started_on"]),
        required_trading_days=payload["required_trading_days"],
        provider=payload["provider"],
        universe_version=payload["universe_version"],
        strategy_versions=tuple(payload["strategy_versions"]),
        minimum_data_success_rate=payload["minimum_data_success_rate"],
        minimum_report_success_rate=payload["minimum_report_success_rate"],
        run_mode=ShadowRunMode(payload.get("run_mode", "REALTIME")),
    )


def load_trading_calendar(path: str | Path) -> tuple[date, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    days = tuple(date.fromisoformat(value) for value in payload["trading_days"])
    if tuple(sorted(set(days))) != days:
        raise ValueError("trading calendar must be unique and ascending")
    return days


def load_shadow_day(path: str | Path) -> ShadowDayEvidence:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return shadow_day_from_mapping(payload)


def shadow_day_to_mapping(evidence: ShadowDayEvidence) -> dict[str, Any]:
    return {
        "trading_date": evidence.trading_date.isoformat(),
        "observed_at": evidence.observed_at.isoformat(),
        "market_data_as_of": evidence.market_data_as_of.isoformat(),
        "input_snapshot_hash": evidence.input_snapshot_hash,
        "virtual_account_snapshot_hash": evidence.virtual_account_snapshot_hash,
        "data_version": evidence.data_version,
        "regime": evidence.regime,
        "mainline_ids": list(evidence.mainline_ids),
        "candidate_ids": list(evidence.candidate_ids),
        "data_complete": evidence.data_complete,
        "report_generated": evidence.report_generated,
        "pipeline_succeeded": evidence.pipeline_succeeded,
        "reconciliation_matched": evidence.reconciliation_matched,
        "future_data_violations": evidence.future_data_violations,
        "executable_orders_emitted": evidence.executable_orders_emitted,
        "opened_incidents": [
            {
                "incident_id": incident.incident_id,
                "severity": incident.severity,
                "component": incident.component,
                "summary": incident.summary,
            }
            for incident in evidence.opened_incidents
        ],
        "resolved_incident_ids": list(evidence.resolved_incident_ids),
        "manual_intervention_minutes": evidence.manual_intervention_minutes,
        "notes": evidence.notes,
        "run_mode": evidence.run_mode,
    }


def shadow_day_from_mapping(payload: dict[str, Any]) -> ShadowDayEvidence:
    incidents = tuple(
        ShadowIncident(
            incident_id=item["incident_id"],
            severity=IncidentSeverity(item["severity"]),
            component=item["component"],
            summary=item["summary"],
        )
        for item in payload.get("opened_incidents", [])
    )
    return ShadowDayEvidence(
        trading_date=date.fromisoformat(payload["trading_date"]),
        observed_at=datetime.fromisoformat(payload["observed_at"]),
        market_data_as_of=datetime.fromisoformat(payload["market_data_as_of"]),
        input_snapshot_hash=payload["input_snapshot_hash"],
        virtual_account_snapshot_hash=payload["virtual_account_snapshot_hash"],
        data_version=payload["data_version"],
        regime=payload["regime"],
        mainline_ids=tuple(payload["mainline_ids"]),
        candidate_ids=tuple(payload["candidate_ids"]),
        data_complete=payload["data_complete"],
        report_generated=payload["report_generated"],
        pipeline_succeeded=payload["pipeline_succeeded"],
        reconciliation_matched=payload["reconciliation_matched"],
        future_data_violations=payload["future_data_violations"],
        executable_orders_emitted=payload["executable_orders_emitted"],
        opened_incidents=incidents,
        resolved_incident_ids=tuple(payload.get("resolved_incident_ids", [])),
        manual_intervention_minutes=payload.get("manual_intervention_minutes", 0),
        notes=payload.get("notes", ""),
        run_mode=ShadowRunMode(payload.get("run_mode", "REALTIME")),
    )
