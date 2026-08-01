from __future__ import annotations

import json
from pathlib import Path

from apps.api.core.local_bootstrap import research_catalog_from_file, research_catalog_from_shadow


def _analysis(day: str, *, selected: tuple[str, str]) -> dict[str, object]:
    segments = [*selected, "BOARD.STAR"]
    return {
        "trading_date": day,
        "virtual_clock": f"{day}T16:05:00+08:00",
        "input_snapshot_hash": "a" * 64,
        "regime": {
            "name": "RANGE_STRONG",
            "score": 61.5,
            "advancing_ratio": 0.55,
            "above_ma20_ratio": 0.48,
        },
        "mainlines": [
            {
                "segment": segment,
                "median_return_5d": 0.02 - index * 0.01,
                "turnover": 1_000_000 - index,
                "members": 100,
            }
            for index, segment in enumerate(segments)
        ],
        "candidates": [
            {
                "instrument_id": "CN.SH.600001",
                "segment": selected[0],
                "return_5d": 0.05,
                "return_20d": 0.12,
                "turnover": 2_000_000,
                "score": 0.1,
            }
        ],
    }


def test_local_catalog_loads_latest_shadow_analysis(tmp_path: Path) -> None:
    first = tmp_path / "2026-07-27.analysis.json"
    latest = tmp_path / "2026-07-28.analysis.json"
    first.write_text(
        json.dumps(_analysis("2026-07-27", selected=("BOARD.SZ_MAIN", "BOARD.SH_MAIN"))),
        encoding="utf-8",
    )
    latest.write_text(
        json.dumps(_analysis("2026-07-28", selected=("BOARD.SZ_MAIN", "BOARD.SH_MAIN"))),
        encoding="utf-8",
    )

    catalog = research_catalog_from_shadow(latest)
    regime = catalog.regime()
    assert regime.payload["regime"] == "RANGE_STRONG"
    assert regime.payload["support_evidence"] == ["上涨家数占比高于一半"]
    themes, total = catalog.list("themes", offset=0, limit=10, filters={})
    assert total == 3
    assert themes[0].payload["persistence_days"] == 2
    candidates, total = catalog.list("candidates", offset=0, limit=10, filters={})
    assert total == 1
    assert candidates[0].payload["tier"] == "A"
    evidence = catalog.get("evidence", "CN.SH.600001")
    assert "20日收益 12.00%" in evidence.payload["support_evidence"]


def test_local_catalog_loads_current_industry_snapshot(tmp_path: Path) -> None:
    snapshot = {
        "schema_version": "current-industry-research-v1",
        "as_of": "2026-08-01T10:00:00+08:00",
        "market_as_of": "2026-07-31",
        "data_version": "current-v1",
        "classification": "TUSHARE_STOCK_BASIC_INDUSTRY",
        "regime": {
            "name": "RANGE_STRONG",
            "score": 58,
            "support_evidence": ["上涨家数占比高于一半"],
            "counter_evidence": [],
        },
        "themes": [{"id": "industry-a", "name": "机器人", "state": "CONFIRMED", "score": 81}],
        "theme_members": [
            {
                "instrument_id": "CN.SH.600001",
                "theme_id": "industry-a",
                "name": "示例股份",
                "rank": 1,
                "score": 92,
            }
        ],
        "leaders": [
            {
                "instrument_id": "CN.SH.600001",
                "theme_id": "industry-a",
                "name": "示例股份",
                "rank": 1,
                "score": 92,
            }
        ],
        "candidates": [
            {
                "instrument_id": "CN.SH.600001",
                "theme_id": "industry-a",
                "name": "示例股份",
                "score": 92,
            }
        ],
        "evidence": [
            {
                "instrument_id": "CN.SH.600001",
                "name": "示例股份",
                "support_evidence": ["行业内第1名"],
                "counter_evidence": [],
                "invalidations": ["行业退潮"],
            }
        ],
        "stock_details": [
            {
                "instrument_id": "CN.SH.600001",
                "name": "示例股份",
                "symbol": "600001",
                "theme_id": "industry-a",
                "theme_name": "机器人",
                "trade_date": "2026-07-31",
                "latest_price": 12.3,
                "previous_close": 12.0,
                "change": 0.3,
                "change_pct": 0.025,
                "bars": [
                    {
                        "trade_date": "2026-07-31",
                        "open": 12.0,
                        "high": 12.5,
                        "low": 11.9,
                        "close": 12.3,
                        "volume": 100000,
                        "turnover": 1230000,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "research-current.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    catalog = research_catalog_from_file(path)

    assert catalog.regime().payload["market_as_of"] == "2026-07-31"
    members, total = catalog.list(
        "theme_members",
        offset=0,
        limit=10,
        filters={"theme_id": "industry-a"},
    )
    assert total == 1
    assert members[0].payload["instrument_id"] == "CN.SH.600001"
    detail = catalog.get("stock_details", "CN.SH.600001")
    assert detail.payload["latest_price"] == 12.3
    assert detail.payload["bars"][0]["trade_date"] == "2026-07-31"
