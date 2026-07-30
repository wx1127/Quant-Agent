from datetime import UTC, date, datetime, timedelta

import pytest

from quant_agent.features.core import InsufficientObservationsError
from quant_agent.features.market_breadth import MarketBreadthEngine, StockBar
from quant_agent.features.market_trend import IndexBar, MarketTrendEngine


def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 16, tzinfo=UTC)


def test_market_trend_ignores_future_and_degrades_incomplete_index() -> None:
    start = date(2025, 1, 1)
    bars = []
    for index_id, gain in (("CSI300", 0.001), ("CSI500", 0.0015)):
        bars.extend(
            IndexBar(
                index_id,
                start + timedelta(days=i),
                100 * (1 + gain) ** i,
                at(start + timedelta(days=i)),
            )
            for i in range(130)
        )
    bars.extend(
        IndexBar("SHORT", start + timedelta(days=i), 100 + i, at(start + timedelta(days=i)))
        for i in range(10)
    )
    as_of = at(start + timedelta(days=129))
    engine = MarketTrendEngine()
    base = engine.calculate(bars, as_of=as_of, data_version="snap")
    future = IndexBar("CSI300", as_of.date() + timedelta(days=1), 1, as_of)
    again = engine.calculate([*bars, future], as_of=as_of, data_version="snap")
    assert base.score == again.score
    assert base.score > 60
    assert len(base.per_index) == 2
    assert any("degraded" in warning for warning in base.warnings)
    with pytest.raises(InsufficientObservationsError):
        MarketTrendEngine(min_indices=3).calculate(bars, as_of=as_of, data_version="snap")
    with pytest.raises(ValueError):
        IndexBar("X", start, 0, at(start))


def test_market_breadth_excludes_suspended_stock_from_current_denominator() -> None:
    start = date(2025, 1, 1)
    bars = []
    for instrument, gain in (("UP", 0.01), ("DOWN", -0.002), ("HALT", 0.005)):
        for i in range(65):
            day = start + timedelta(days=i)
            bars.append(
                StockBar(
                    instrument,
                    day,
                    100 * (1 + gain) ** i,
                    1000 + i * 10,
                    not (instrument == "HALT" and i == 64),
                    at(day),
                )
            )
    as_of = at(start + timedelta(days=64))
    result = MarketBreadthEngine().calculate(bars, as_of=as_of, data_version="snap")
    assert result.eligible_count == 2
    assert (result.advancing_count, result.declining_count) == (1, 1)
    assert result.advancing_ratio == 0.5
    assert 0 <= result.risk_score <= 100
    assert result.turnover_percentile_20d == pytest.approx(0.05)
    with pytest.raises(InsufficientObservationsError):
        MarketBreadthEngine().calculate([], as_of=as_of, data_version="snap")
    with pytest.raises(ValueError):
        StockBar("X", start, -1, 0, True, at(start))
