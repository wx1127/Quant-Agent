from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from quant_agent.features.core import InsufficientObservationsError
from quant_agent.features.fundamentals import (
    FundamentalFeatureEngine,
    FundamentalMetric,
)
from quant_agent.features.stocks import (
    ReferencePricePoint,
    StockPricePoint,
    StockTrendEngine,
)
from quant_agent.features.tradeability import (
    LimitStatus,
    TradeabilityConfig,
    TradeabilityEngine,
    TradeabilityInput,
)


def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 16, tzinfo=UTC)


def test_stock_trend_uses_only_available_tradable_history() -> None:
    start = date(2025, 1, 1)
    prices = []
    benchmark = []
    industry = []
    for i in range(70):
        day = start + timedelta(days=i)
        prices.append(StockPricePoint("S1", day, 100 * 1.01**i, True, at(day)))
        benchmark.append(ReferencePricePoint(day, 100 * 1.002**i, at(day)))
        industry.append(ReferencePricePoint(day, 100 * 1.004**i, at(day)))
    as_of = at(start + timedelta(days=69))
    result = StockTrendEngine().calculate(
        prices,
        benchmark,
        industry=industry,
        as_of=as_of,
        data_version="snap",
    )
    assert result.relative_industry_20d is not None
    assert result.relative_benchmark_20d > 0
    assert result.trend_quality_score > 50
    without_industry = StockTrendEngine().calculate(
        prices,
        benchmark,
        industry=None,
        as_of=as_of,
        data_version="snap",
    )
    assert without_industry.relative_industry_20d is None
    assert without_industry.warnings == ("industry reference missing",)
    with pytest.raises(InsufficientObservationsError):
        StockTrendEngine().calculate(
            prices[:20], benchmark, industry=None, as_of=as_of, data_version="snap"
        )


def test_tradeability_capacity_and_structured_reasons() -> None:
    as_of = at(date(2026, 1, 1))
    good = TradeabilityInput(
        "S1",
        as_of,
        date(2020, 1, 1),
        (100_000_000,) * 20,
        (0.02,) * 20,
        False,
        LimitStatus.NONE,
        False,
        False,
    )
    engine = TradeabilityEngine()
    result = engine.evaluate(
        good,
        requested_capital=1_000_000,
        market_turnover_samples=[10_000_000, 100_000_000],
        market_turnover_rate_samples=[0.01, 0.02],
        data_version="snap",
    )
    assert result.tradable
    assert result.estimated_capacity == 5_000_000
    bad = TradeabilityInput(
        "S2",
        as_of,
        date(2025, 12, 1),
        (1_000_000,),
        (0.001,),
        True,
        LimitStatus.LIMIT_UP,
        True,
        True,
        False,
    )
    rejected = engine.evaluate(
        bad,
        requested_capital=10_000_000,
        market_turnover_samples=[1_000_000],
        market_turnover_rate_samples=[0.001],
        data_version="snap",
    )
    assert not rejected.tradable
    assert "suspended" in rejected.reasons
    assert "requested capital exceeds capacity" in rejected.reasons
    with pytest.raises(ValueError):
        TradeabilityConfig(maximum_participation_rate=0)


def metric(name: str, value: str, available: datetime, revision: str = "1") -> FundamentalMetric:
    return FundamentalMetric(
        "S1",
        date(2025, 9, 30),
        name,
        Decimal(value),
        available - timedelta(hours=1),
        available,
        "exchange",
        revision,
    )


def test_fundamentals_are_point_in_time_and_missing_is_not_neutral() -> None:
    now = at(date(2026, 1, 1))
    metrics = [
        metric("roe", "0.12", now - timedelta(days=2)),
        metric("revenue_growth", "0.20", now - timedelta(days=2)),
        metric("operating_cash_flow_ratio", "1.0", now - timedelta(days=2)),
        metric("debt_ratio", "0.40", now - timedelta(days=2)),
        metric("profit_stability", "0.8", now - timedelta(days=2)),
        metric("roe", "-0.50", now + timedelta(days=1), "future"),
    ]
    result = FundamentalFeatureEngine().calculate(
        metrics, instrument_id="S1", as_of=now, data_version="snap"
    )
    assert result.quality_score is not None and result.quality_score > 50
    assert not result.missing_metrics
    assert all("future" not in source for source in result.source_records)
    missing = FundamentalFeatureEngine().calculate(
        metrics[:1], instrument_id="S1", as_of=now, data_version="snap"
    )
    assert missing.quality_score is None
    assert missing.risk_penalty > 0
