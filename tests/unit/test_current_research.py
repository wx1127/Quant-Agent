from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.domain import DailyBar
from quant_agent.research.current import CurrentIndustryResearchEngine, StockProfile

TZ = ZoneInfo("Asia/Shanghai")


def _bar(instrument_id: str, day: date, close: float, turnover: int) -> DailyBar:
    price = Decimal(str(close))
    return DailyBar(
        instrument_id=instrument_id,
        trade_date=day,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=Decimal("100000"),
        turnover=Decimal(turnover),
        source="fixture",
        available_at=datetime.combine(day, datetime.min.time(), tzinfo=TZ).replace(hour=16),
        version=day.isoformat(),
    )


def test_current_research_ranks_industries_and_stocks() -> None:
    start = date(2026, 7, 1)
    generated_at = datetime(2026, 7, 21, 18, tzinfo=TZ)
    definitions = (
        ("CN.SH.600001", "甲公司", "机器人", 0.030, 300_000_000),
        ("CN.SH.600002", "乙公司", "机器人", 0.020, 200_000_000),
        ("CN.SZ.000001", "丙公司", "银行", 0.002, 400_000_000),
        ("CN.SZ.000002", "丁公司", "银行", -0.001, 100_000_000),
    )
    bars = tuple(
        _bar(instrument_id, start + timedelta(days=index), 10 * (1 + growth) ** index, turnover)
        for instrument_id, _name, _industry, growth, turnover in definitions
        for index in range(21)
    )
    profiles = tuple(
        StockProfile(
            instrument_id,
            name,
            industry,
            date(2020, 1, 1),
            "fixture",
            generated_at,
            "profiles-v1",
        )
        for instrument_id, name, industry, _growth, _turnover in definitions
    )

    snapshot = CurrentIndustryResearchEngine(minimum_industry_members=2).build(
        market_date=start + timedelta(days=20),
        generated_at=generated_at,
        bars=bars,
        profiles=profiles,
    )

    assert snapshot["schema_version"] == "current-industry-research-v1"
    assert snapshot["themes"][0]["name"] == "机器人"
    assert snapshot["themes"][0]["kind"] == "INDUSTRY"
    assert snapshot["themes"][0]["scored_member_count"] == 2
    robot_members = [item for item in snapshot["theme_members"] if item["theme_name"] == "机器人"]
    assert [item["name"] for item in robot_members] == ["甲公司", "乙公司"]
    assert robot_members[0]["role"] == "LEADER"
    assert robot_members[0]["score"] > robot_members[1]["score"]
    assert snapshot["candidates"]
    assert all(item["instrument_id"].startswith("CN.") for item in snapshot["leaders"])


def test_current_research_rejects_unavailable_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        CurrentIndustryResearchEngine(minimum_industry_members=1)
