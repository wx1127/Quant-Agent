"""Validate the complete evidence chain and report current shadow status."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date

from quant_agent.shadow.io import load_shadow_config, load_trading_calendar
from quant_agent.shadow.ledger import ShadowEvidenceLedger
from quant_agent.shadow.reporting import write_shadow_report
from quant_agent.shadow.session import ShadowRunEvaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--report-dir", required=True)
    arguments = parser.parse_args()

    config = load_shadow_config(arguments.config)
    records = ShadowEvidenceLedger(arguments.ledger).read_all()
    acceptance = ShadowRunEvaluator().evaluate(
        config,
        records,
        trading_calendar=load_trading_calendar(arguments.calendar),
    )
    write_shadow_report(
        arguments.report_dir,
        config,
        acceptance,
        generated_on=date.today(),
    )
    print(json.dumps(asdict(acceptance), ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
