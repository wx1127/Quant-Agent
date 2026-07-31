"""Collect one real Tushare market snapshot without logging its credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

from quant_agent.core.time import shanghai_now
from quant_agent.data.providers.tushare import TushareHttpProvider


def _market_data_token() -> str:
    token = os.getenv("MARKET_DATA_TOKEN")
    token_file = os.getenv("MARKET_DATA_TOKEN_FILE")
    if token and token_file:
        raise SystemExit("configure only one of MARKET_DATA_TOKEN or MARKET_DATA_TOKEN_FILE")
    if token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(
            "MARKET_DATA_TOKEN or MARKET_DATA_TOKEN_FILE is required; no shadow day was recorded"
        )
    return token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trading-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--market", default="SSE")
    arguments = parser.parse_args()

    token = _market_data_token()
    trading_date = date.fromisoformat(arguments.trading_date)
    with TushareHttpProvider(token) as provider:
        calendar = provider.fetch_trading_calendar(
            arguments.market,
            trading_date,
            trading_date,
        )
        if not any(item.trade_date == trading_date and item.is_open for item in calendar.records):
            raise SystemExit("requested date is not an open trading day")
        bars = provider.fetch_daily_bars(trading_date)
    if not bars.records:
        raise SystemExit("provider returned no daily bars; no shadow day was recorded")
    observed_at = shanghai_now()
    if trading_date == observed_at.date() and observed_at < bars.available_at:
        raise SystemExit("daily bars are not point-in-time available yet")

    payload = {
        "provider": bars.provider,
        "endpoint": bars.endpoint,
        "trading_date": trading_date,
        "available_at": bars.available_at,
        "observed_at": observed_at,
        "record_count": len(bars.records),
        "records": [asdict(item) for item in bars.records],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    snapshot_hash = hashlib.sha256(canonical.encode()).hexdigest()
    output = Path(arguments.output_dir) / f"market-{trading_date}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {**payload, "snapshot_hash": snapshot_hash},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "record_count": len(bars.records),
                "snapshot_hash": snapshot_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
