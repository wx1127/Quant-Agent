"""Local-only adapters for displaying verified shadow analysis in the Web UI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from apps.api.core.services import ResearchCatalog, ResearchRecord


def research_catalog_from_file(path: str | Path) -> ResearchCatalog:
    """Load a supported local research snapshot without weakening production defaults."""

    source = Path(path)
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "current-industry-research-v1":
        return _current_industry_catalog(payload)
    return research_catalog_from_shadow(source)


def research_catalog_from_shadow(path: str | Path) -> ResearchCatalog:
    """Build an immutable API read model from one historical-shadow analysis file."""

    source = Path(path)
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(payload["virtual_clock"])
    data_version = f"shadow-{payload['trading_date']}-{payload['input_snapshot_hash'][:12]}"
    catalog = ResearchCatalog()

    regime = payload["regime"]
    support: list[str] = []
    counter: list[str] = []
    _evidence_ratio(
        support,
        counter,
        regime["advancing_ratio"],
        "上涨家数占比高于一半",
        "上涨家数占比不足一半",
    )
    _evidence_ratio(
        support,
        counter,
        regime["above_ma20_ratio"],
        "多数股票位于20日均线上方",
        "多数股票仍位于20日均线下方",
    )
    catalog.publish_regime(
        ResearchRecord(
            "CN_A",
            as_of,
            data_version,
            {
                "regime": regime["name"],
                "score": round(float(regime["score"]), 2),
                "support_evidence": support,
                "counter_evidence": counter,
            },
        )
    )

    mainlines = payload["mainlines"]
    theme_records: list[ResearchRecord] = []
    for index, item in enumerate(mainlines):
        score = max(0.0, min(100.0, 50 + float(item["median_return_5d"]) * 500))
        theme_records.append(
            ResearchRecord(
                item["segment"],
                as_of,
                data_version,
                {
                    "name": _segment_name(item["segment"]),
                    "state": "CONFIRMED" if index < 2 else "WATCH",
                    "score": round(score, 2),
                    "persistence_days": _persistence_days(source.parent, item["segment"]),
                    "members": item["members"],
                    "turnover": item["turnover"],
                },
            )
        )
    catalog.publish("themes", theme_records)

    candidate_records: list[ResearchRecord] = []
    evidence_records: list[ResearchRecord] = []
    for index, item in enumerate(payload["candidates"]):
        instrument_id = item["instrument_id"]
        tier = "A" if index < 3 else "B" if index < 7 else "C"
        candidate_records.append(
            ResearchRecord(
                instrument_id,
                as_of,
                data_version,
                {
                    "name": instrument_id,
                    "theme_id": item["segment"],
                    "tier": tier,
                    "score": round(float(item["score"]) * 100, 2),
                    "tradable": True,
                },
            )
        )
        candidate_support = [
            f"20日收益 {float(item['return_20d']):.2%}",
            f"5日收益 {float(item['return_5d']):.2%}",
            f"当日成交额 {float(item['turnover']):,.0f} 元",
        ]
        candidate_counter = []
        if float(item["return_5d"]) < 0:
            candidate_counter.append("5日动量已经转弱")
        if not candidate_counter:
            candidate_counter.append("历史影子排名不代表未来收益概率")
        evidence_records.append(
            ResearchRecord(
                instrument_id,
                as_of,
                data_version,
                {
                    "name": instrument_id,
                    "support_evidence": candidate_support,
                    "counter_evidence": candidate_counter,
                    "invalidations": [
                        "所属领先板块退出前两名",
                        "5日或20日动量跌破策略阈值",
                        "停牌、涨跌停或数据完整性门禁失败",
                    ],
                },
            )
        )
    catalog.publish("candidates", candidate_records)
    catalog.publish("leaders", candidate_records[:3])
    catalog.publish("evidence", evidence_records)
    return catalog


def _current_industry_catalog(payload: dict[str, Any]) -> ResearchCatalog:
    as_of = datetime.fromisoformat(payload["as_of"])
    data_version = str(payload["data_version"])
    catalog = ResearchCatalog()
    regime = payload["regime"]
    catalog.publish_regime(
        ResearchRecord(
            "CN_A",
            as_of,
            data_version,
            {
                "regime": regime["name"],
                "score": regime["score"],
                "market_as_of": payload["market_as_of"],
                "classification": payload["classification"],
                "support_evidence": regime["support_evidence"],
                "counter_evidence": regime["counter_evidence"],
            },
        )
    )
    catalog.publish(
        "themes",
        [
            ResearchRecord(
                str(item["id"]),
                as_of,
                data_version,
                {key: value for key, value in item.items() if key != "id"},
            )
            for item in payload["themes"]
        ],
    )
    catalog.publish(
        "theme_members",
        [
            ResearchRecord(
                f"{item['theme_id']}::{item['instrument_id']}",
                as_of,
                data_version,
                dict(item),
            )
            for item in payload["theme_members"]
        ],
    )
    catalog.publish(
        "leaders",
        [
            ResearchRecord(
                f"{item['theme_id']}::{item['instrument_id']}",
                as_of,
                data_version,
                dict(item),
            )
            for item in payload["leaders"]
        ],
    )
    catalog.publish(
        "candidates",
        [
            ResearchRecord(
                str(item["instrument_id"]),
                as_of,
                data_version,
                {
                    **item,
                    "tier": "A" if index < 5 else "B" if index < 12 else "C",
                },
            )
            for index, item in enumerate(payload["candidates"])
        ],
    )
    catalog.publish(
        "evidence",
        [
            ResearchRecord(
                str(item["instrument_id"]),
                as_of,
                data_version,
                dict(item),
            )
            for item in payload["evidence"]
        ],
    )
    return catalog


def _evidence_ratio(
    support: list[str],
    counter: list[str],
    value: float,
    positive: str,
    negative: str,
) -> None:
    (support if value >= 0.5 else counter).append(positive if value >= 0.5 else negative)


def _segment_name(segment: str) -> str:
    return {
        "BOARD.SH_MAIN": "沪市主板",
        "BOARD.SZ_MAIN": "深市主板",
        "BOARD.STAR": "科创板",
        "BOARD.CHINEXT": "创业板",
        "BOARD.BJ": "北交所",
    }.get(segment, segment)


def _persistence_days(days_dir: Path, segment: str) -> int:
    count = 0
    for path in reversed(sorted(days_dir.glob("*.analysis.json"))):
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = {item["segment"] for item in payload["mainlines"][:2]}
        if segment not in selected:
            break
        count += 1
    return count
