"""Enrich an existing current-research snapshot with cached daily stock bars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from quant_agent.data.domain import DailyBar
from quant_agent.research.current import stock_details_from_members


def _canonical_hash(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cached_history(snapshot_dirs: list[Path]) -> dict[str, list[DailyBar]]:
    history: dict[str, list[DailyBar]] = {}
    files = sorted(
        path for snapshot_dir in snapshot_dirs for path in snapshot_dir.glob("market-*.json")
    )
    if not files:
        raise ValueError("no cached market snapshots found")
    seen: set[tuple[str, object]] = set()
    for path in files:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("snapshot_hash") != _canonical_hash(payload.get("records", [])):
            raise ValueError(f"cached market snapshot hash mismatch: {path}")
        for item in payload["records"]:
            bar = DailyBar.model_validate(item)
            key = (bar.instrument_id, bar.trade_date)
            if key in seen:
                continue
            seen.add(key)
            history.setdefault(bar.instrument_id, []).append(bar)
    for series in history.values():
        series.sort(key=lambda item: item.trade_date)
    return history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot-dir", required=True, action="append")
    arguments = parser.parse_args()
    path = Path(arguments.snapshot)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    history = _cached_history([Path(item) for item in arguments.snapshot_dir])
    payload["stock_details"] = stock_details_from_members(payload["theme_members"], history)
    if not payload["stock_details"]:
        raise ValueError("cached bars did not match any scored stocks")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "snapshot": str(path),
                "stock_details": len(payload["stock_details"]),
                "bar_count": sum(len(item["bars"]) for item in payload["stock_details"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
