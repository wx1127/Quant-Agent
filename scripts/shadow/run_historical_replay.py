"""Run a complete Tushare-backed point-in-time historical shadow session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

from quant_agent.data.domain import DailyBar
from quant_agent.data.providers.tushare import TushareHttpProvider
from quant_agent.shadow.historical import HistoricalShadowEngine
from quant_agent.shadow.io import load_shadow_config, shadow_day_to_mapping
from quant_agent.shadow.ledger import ShadowEvidenceLedger
from quant_agent.shadow.models import ShadowRunMode
from quant_agent.shadow.reporting import write_shadow_report
from quant_agent.shadow.session import ShadowRunEvaluator


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
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _load_or_fetch_day(
    provider: TushareHttpProvider,
    trading_date: date,
    snapshot_dir: Path,
    *,
    allow_empty: bool = False,
) -> tuple[DailyBar, ...]:
    path = snapshot_dir / f"market-{trading_date}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = tuple(DailyBar.model_validate(item) for item in payload["records"])
        if payload["snapshot_hash"] != _canonical_hash(payload["records"]):
            raise ValueError(f"cached market snapshot hash mismatch: {path}")
        return records
    batch = provider.fetch_daily_bars(trading_date)
    if not batch.records:
        if allow_empty:
            return ()
        raise ValueError(f"provider returned no daily bars for {trading_date}")
    record_payloads = [item.model_dump(mode="json") for item in batch.records]
    payload = {
        "provider": batch.provider,
        "endpoint": batch.endpoint,
        "trading_date": trading_date.isoformat(),
        "available_at": batch.available_at.isoformat(),
        "record_count": len(record_payloads),
        "records": record_payloads,
        "snapshot_hash": _canonical_hash(record_payloads),
    }
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return batch.records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--market", default="SSE")
    parser.add_argument("--history-days", type=int, default=25)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()

    config = load_shadow_config(arguments.config)
    if config.run_mode is not ShadowRunMode.HISTORICAL_POINT_IN_TIME:
        raise SystemExit("historical replay requires HISTORICAL_POINT_IN_TIME config")
    if arguments.history_days < 21:
        raise SystemExit("historical replay requires at least 21 prior trading days")
    start = date.fromisoformat(arguments.start)
    if start != config.started_on:
        raise SystemExit("replay start must match session started_on")
    root = Path(arguments.output_root)
    calendar_input = json.loads(Path(arguments.calendar).read_text(encoding="utf-8"))
    target_days = [date.fromisoformat(item) for item in calendar_input["trading_days"]]
    if len(target_days) != config.required_trading_days:
        raise SystemExit("frozen calendar does not contain the required target days")
    if target_days[0] != start or target_days != sorted(set(target_days)):
        raise SystemExit("frozen calendar must start on session date and be unique/ascending")
    token = _market_data_token()
    with TushareHttpProvider(token) as provider:
        prior_days: list[date] = []
        prior_bars: dict[date, tuple[DailyBar, ...]] = {}
        candidate = start - timedelta(days=1)
        while len(prior_days) < arguments.history_days:
            if candidate.weekday() < 5:
                records = _load_or_fetch_day(
                    provider,
                    candidate,
                    root / "snapshots",
                    allow_empty=True,
                )
                if records:
                    prior_days.append(candidate)
                    prior_bars[candidate] = records
            candidate -= timedelta(days=1)
            if candidate < start - timedelta(days=120):
                raise SystemExit("provider daily bars lack required replay history")
        prior_days.sort()
        all_bars: list[DailyBar] = []
        for trading_date in prior_days:
            all_bars.extend(prior_bars[trading_date])
        for trading_date in target_days:
            all_bars.extend(_load_or_fetch_day(provider, trading_date, root / "snapshots"))

    calendar_values = [item.isoformat() for item in target_days]
    calendar_payload = {
        "provider": "tushare",
        "market": arguments.market,
        "trading_days": calendar_values,
        "calendar_hash": _canonical_hash(calendar_values),
        "mode": ShadowRunMode.HISTORICAL_POINT_IN_TIME,
        "source": calendar_input.get("source", "frozen calendar input"),
    }
    root.mkdir(parents=True, exist_ok=True)
    calendar_path = root / "calendar.json"
    calendar_path.write_text(
        json.dumps(calendar_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    ledger = ShadowEvidenceLedger(root / "evidence.jsonl")
    engine = HistoricalShadowEngine()
    for trading_date in target_days:
        result = engine.run_day(trading_date, tuple(all_bars))
        detail_path = root / "days" / f"{trading_date}.analysis.json"
        evidence_path = root / "days" / f"{trading_date}.evidence.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(
            json.dumps(result.details, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        evidence_path.write_text(
            json.dumps(
                shadow_day_to_mapping(result.evidence),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        ledger.append(result.evidence)

    acceptance = ShadowRunEvaluator().evaluate(
        config,
        ledger.read_all(),
        trading_calendar=tuple(target_days),
    )
    report_paths = write_shadow_report(
        root / "reports",
        config,
        acceptance,
        generated_on=date.today(),
    )
    print(
        json.dumps(
            {
                "session_id": config.session_id,
                "mode": config.run_mode,
                "calendar": str(calendar_path),
                "calendar_hash": calendar_payload["calendar_hash"],
                "snapshots": len(prior_days) + len(target_days),
                "evidence_days": len(ledger.read_all()),
                "status": acceptance.status,
                "future_data_violations": acceptance.future_data_violations,
                "executable_orders_emitted": acceptance.executable_orders_emitted,
                "reports": [str(item) for item in report_paths],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
