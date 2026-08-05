"""Run one append-only P8-T08 PAPER validation day or record a missed day."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_agent.core.time import shanghai_now
from quant_agent.data.domain import DailyBar
from quant_agent.data.providers.tushare import TushareHttpProvider
from quant_agent.paper_validation.daily import PaperDailyEngine
from quant_agent.portfolio.snapshots import AccountSnapshot

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _token() -> str:
    value = os.getenv("MARKET_DATA_TOKEN")
    token_file = os.getenv("MARKET_DATA_TOKEN_FILE")
    if value and token_file:
        raise ValueError("configure only one market-data credential source")
    if token_file:
        value = Path(token_file).read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("MARKET_DATA_TOKEN_FILE is required")
    return value


def _canonical_hash(payload: object) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def _read_cached(path: Path) -> tuple[DailyBar, ...]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    records_hash = _canonical_hash(records)
    without_hash = {key: value for key, value in payload.items() if key != "snapshot_hash"}
    payload_hash = _canonical_hash(without_hash)
    if payload.get("snapshot_hash") not in {records_hash, payload_hash}:
        raise ValueError(f"market snapshot hash mismatch: {path}")
    return tuple(DailyBar.model_validate(item) for item in records)


def _load_or_fetch(
    provider: TushareHttpProvider,
    trading_date: date,
    output: Path,
    caches: list[Path],
) -> tuple[DailyBar, ...]:
    destination = output / "snapshots" / f"market-{trading_date}.json"
    candidates = [destination, *(path / destination.name for path in caches)]
    source = next((path for path in candidates if path.exists()), None)
    if source is not None:
        return _read_cached(source)
    batch = provider.fetch_daily_bars(trading_date)
    if not batch.records:
        raise ValueError(f"provider returned no bars for {trading_date}")
    records = [item.model_dump(mode="json") for item in batch.records]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "provider": batch.provider,
                "trading_date": trading_date.isoformat(),
                "available_at": batch.available_at.isoformat(),
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


def _initial_account(observed_at: datetime) -> AccountSnapshot:
    return AccountSnapshot(
        "paper-initial",
        "p8-paper-1",
        observed_at - timedelta(days=1),
        1_000_000.0,
        0.0,
        (),
        "paper",
        "p8-paper-account-v1",
    )


def _latest_account(root: Path, observed_at: datetime) -> AccountSnapshot:
    files = sorted((root / "accounts").glob("*.json"))
    if not files:
        return _initial_account(observed_at)
    return AccountSnapshot.from_json(files[-1].read_text(encoding="utf-8"))


def _pending(root: Path, trading_date: date) -> dict[str, Any] | None:
    matches = []
    for path in sorted((root / "pending").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["execute_on"] == trading_date.isoformat():
            matches.append(payload)
    if len(matches) > 1:
        raise ValueError("multiple pending batches target the same trading day")
    return matches[0] if matches else None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"append-only artifact already exists: {path}")
        return
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _record_failure(root: Path, trading_date: date, reason: str, observed_at: datetime) -> None:
    payload = {
        "schema_version": "p8-paper-failure-v1",
        "trading_date": trading_date.isoformat(),
        "observed_at": observed_at.isoformat(),
        "status": "FAILED",
        "reason": reason,
        "backfill_as_success_allowed": False,
    }
    _write_json(root / "failures" / f"{trading_date}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trading-date", default=date.today().isoformat())
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-cache", action="append", default=[])
    parser.add_argument("--market", default="SSE")
    parser.add_argument("--record-missed", action="store_true")
    parser.add_argument("--retry-after-failure", action="store_true")
    parser.add_argument("--reason")
    arguments = parser.parse_args()
    trading_date = date.fromisoformat(arguments.trading_date)
    output = Path(arguments.output_root)
    observed_at = shanghai_now()
    if arguments.record_missed:
        if not arguments.reason:
            raise SystemExit("--reason is required with --record-missed")
        _record_failure(output, trading_date, arguments.reason, observed_at)
        return 2
    success_path = output / "days" / f"{trading_date}.json"
    failure_path = output / "failures" / f"{trading_date}.json"
    prior_failure: dict[str, Any] | None = None
    if failure_path.exists() and not arguments.retry_after_failure:
        raise SystemExit("day already has a final failure record; success backfill is forbidden")
    if failure_path.exists():
        prior_failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if success_path.exists():
        print(success_path.read_text(encoding="utf-8"))
        return 0
    if trading_date != observed_at.date():
        raise SystemExit("successful PAPER days must run on their actual calendar date")
    if observed_at.time() < time(16, 5):
        raise SystemExit("market close data is not available yet")
    try:
        with TushareHttpProvider(_token()) as provider:
            calendar = provider.fetch_trading_calendar(
                arguments.market,
                trading_date - timedelta(days=60),
                trading_date + timedelta(days=14),
            )
            days = sorted(item.trade_date for item in calendar.records if item.is_open)
            if trading_date not in days:
                print("requested day is not a Tushare trading day")
                return 0
            prior = [item for item in days if item <= trading_date][-25:]
            following = next(item for item in days if item > trading_date)
            if len(prior) < 25:
                raise ValueError("trading calendar does not cover 25 history days")
            bars = tuple(
                bar
                for day in prior
                for bar in _load_or_fetch(
                    provider,
                    day,
                    output,
                    [Path(item) for item in arguments.source_cache],
                )
            )
        record = PaperDailyEngine().run(
            trading_date=trading_date,
            next_trading_date=following,
            observed_at=observed_at,
            bars=bars,
            account=_latest_account(output, observed_at),
            pending=_pending(output, trading_date),
        )
        if prior_failure is not None:
            record["recovery"] = {
                "status": "RECOVERED_AFTER_FAILURE",
                "original_failure": prior_failure,
                "failure_record_preserved": True,
                "counts_as_clean_scheduler_success": False,
            }
        _write_json(
            output / "accounts" / f"{trading_date}.json",
            record["account"],
        )
        _write_json(
            output / "pending" / f"{trading_date}.json",
            record["next_day_order_draft"],
        )
        _write_json(
            output / "reports" / f"{trading_date}.json",
            {
                "trading_date": record["trading_date"],
                "pipeline_succeeded": True,
                "reconciliation": record["reconciliation"],
                "orders": len(record["execution"]["orders"]),
                "fills": len(record["execution"]["fills"]),
                "fees": record["execution"]["fees"],
                "slippage": record["execution"]["slippage"],
                "alerts": record["alerts"],
                "manual_interventions": record["manual_interventions"],
                "recovery": record.get("recovery"),
            },
        )
        _write_json(success_path, record)
    except Exception as exc:
        if prior_failure is None:
            _record_failure(output, trading_date, str(exc), observed_at)
        else:
            _write_json(
                output
                / "recovery_failures"
                / f"{trading_date}-{observed_at.strftime('%H%M%S')}.json",
                {
                    "trading_date": trading_date.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    "status": "RECOVERY_FAILED",
                    "reason": str(exc),
                    "original_failure_record": str(failure_path),
                },
            )
        raise
    print(json.dumps({"status": "SUCCEEDED", "output": str(success_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
