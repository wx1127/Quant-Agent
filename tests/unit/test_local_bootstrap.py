from __future__ import annotations

import json
from pathlib import Path

from apps.api.core.local_bootstrap import research_catalog_from_shadow


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
