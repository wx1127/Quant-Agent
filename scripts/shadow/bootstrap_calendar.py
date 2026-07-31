"""Fetch and freeze the real provider calendar for a shadow session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

from quant_agent.data.providers.tushare import TushareHttpProvider


def _market_data_token() -> str:
    token = os.getenv("MARKET_DATA_TOKEN")
    token_file = os.getenv("MARKET_DATA_TOKEN_FILE")
    if token and token_file:
        raise SystemExit("configure only one of MARKET_DATA_TOKEN or MARKET_DATA_TOKEN_FILE")
    if token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("MARKET_DATA_TOKEN or MARKET_DATA_TOKEN_FILE is required")
    return token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--required-days", type=int, default=20)
    parser.add_argument("--market", default="SSE")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.required_days < 20:
        raise SystemExit("shadow calendar requires at least 20 trading days")
    token = _market_data_token()

    start = date.fromisoformat(arguments.start)
    with TushareHttpProvider(token) as provider:
        batch = provider.fetch_trading_calendar(
            arguments.market,
            start,
            start + timedelta(days=60),
        )
    days = sorted(
        item.trade_date for item in batch.records if item.is_open and item.trade_date >= start
    )[: arguments.required_days]
    if len(days) != arguments.required_days:
        raise SystemExit("provider calendar does not cover the required window")
    canonical = json.dumps([item.isoformat() for item in days], separators=(",", ":"))
    payload = {
        "provider": batch.provider,
        "market": arguments.market,
        "available_at": batch.available_at,
        "trading_days": days,
        "calendar_hash": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
