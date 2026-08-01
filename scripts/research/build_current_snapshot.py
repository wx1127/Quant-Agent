"""Build the local current-industry research snapshot from Tushare data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

from quant_agent.core.time import shanghai_now
from quant_agent.data.domain import DailyBar
from quant_agent.data.providers.tushare import TushareHttpProvider
from quant_agent.research.current import CurrentIndustryResearchEngine


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


def _canonical_hash(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _read_snapshot(path: Path) -> tuple[DailyBar, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = tuple(DailyBar.model_validate(item) for item in payload["records"])
    if payload["snapshot_hash"] != _canonical_hash(payload["records"]):
        raise ValueError(f"cached market snapshot hash mismatch: {path}")
    return records


def _load_or_fetch(
    provider: TushareHttpProvider,
    trading_date: date,
    snapshot_dir: Path,
    source_cache: Path | None,
) -> tuple[DailyBar, ...]:
    output = snapshot_dir / f"market-{trading_date}.json"
    source = source_cache / output.name if source_cache else None
    if output.exists():
        return _read_snapshot(output)
    if source is not None and source.exists():
        return _read_snapshot(source)
    batch = provider.fetch_daily_bars(trading_date)
    if not batch.records:
        raise ValueError(f"provider returned no daily bars for {trading_date}")
    records = [item.model_dump(mode="json") for item in batch.records]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "provider": batch.provider,
                "endpoint": batch.endpoint,
                "trading_date": trading_date.isoformat(),
                "available_at": batch.available_at.isoformat(),
                "record_count": len(records),
                "records": records,
                "snapshot_hash": _canonical_hash(records),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return batch.records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-date", required=True)
    parser.add_argument("--market", default="SSE")
    parser.add_argument("--history-days", type=int, default=25)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-cache")
    arguments = parser.parse_args()
    if arguments.history_days < 21:
        raise SystemExit("current research requires at least 21 trading days")
    market_date = date.fromisoformat(arguments.market_date)
    generated_at = shanghai_now()
    if market_date > generated_at.date():
        raise SystemExit("market date cannot be in the future")
    token = _market_data_token()
    output_root = Path(arguments.output_root)
    source_cache = Path(arguments.source_cache) if arguments.source_cache else None
    with TushareHttpProvider(token) as provider:
        calendar = provider.fetch_trading_calendar(
            arguments.market,
            market_date - timedelta(days=60),
            market_date,
        )
        open_days = sorted(item.trade_date for item in calendar.records if item.is_open)
        if market_date not in open_days:
            raise SystemExit("market date is not an open Tushare trading day")
        selected_days = open_days[-arguments.history_days :]
        if len(selected_days) < arguments.history_days:
            raise SystemExit("provider calendar does not cover the requested history")
        bars = tuple(
            bar
            for trading_date in selected_days
            for bar in _load_or_fetch(
                provider,
                trading_date,
                output_root / "snapshots",
                source_cache,
            )
        )
        profiles = provider.fetch_stock_profiles(generated_at).records
    snapshot = CurrentIndustryResearchEngine().build(
        market_date=market_date,
        generated_at=generated_at,
        bars=bars,
        profiles=profiles,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "research-current.json"
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "market_as_of": snapshot["market_as_of"],
                "themes": len(snapshot["themes"]),
                "theme_members": len(snapshot["theme_members"]),
                "leaders": len(snapshot["leaders"]),
                "candidates": len(snapshot["candidates"]),
                "data_version": snapshot["data_version"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
