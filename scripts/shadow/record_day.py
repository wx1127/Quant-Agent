"""Append one real shadow day, evaluate the session and write status reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, timedelta

from quant_agent.observability.alerting import AlertDispatcher, JsonLinesAlertSink
from quant_agent.shadow.alerts import shadow_alerts
from quant_agent.shadow.io import (
    load_shadow_config,
    load_shadow_day,
    load_trading_calendar,
)
from quant_agent.shadow.ledger import ShadowEvidenceLedger
from quant_agent.shadow.reporting import write_shadow_report
from quant_agent.shadow.session import ShadowRunEvaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--alert-log", required=True)
    arguments = parser.parse_args()

    config = load_shadow_config(arguments.config)
    calendar = load_trading_calendar(arguments.calendar)
    evidence = load_shadow_day(arguments.evidence)
    ledger = ShadowEvidenceLedger(arguments.ledger)
    record = ledger.append(evidence)
    acceptance = ShadowRunEvaluator().evaluate(
        config,
        ledger.read_all(),
        trading_calendar=calendar,
    )
    write_shadow_report(
        arguments.report_dir,
        config,
        acceptance,
        generated_on=date.today(),
    )
    deliveries = AlertDispatcher(
        JsonLinesAlertSink(arguments.alert_log),
        cooldown=timedelta(0),
    ).dispatch(
        shadow_alerts(evidence, acceptance),
        delivered_at=evidence.observed_at,
    )
    print(
        json.dumps(
            {
                "sequence": record.sequence,
                "trading_date": evidence.trading_date,
                "record_hash": record.record_hash,
                "acceptance": asdict(acceptance),
                "alerts_delivered": len(deliveries),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
