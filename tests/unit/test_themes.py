from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from quant_agent.features.themes import (
    BenchmarkBar,
    IndustryStockBar,
    ThemeFeatureEngine,
    ThemeFeatureResult,
)
from quant_agent.themes.scoring import (
    MainlineConfig,
    ThemeScoringEngine,
    ThemeState,
    ThemeStateMachine,
)


class Resolver:
    def __init__(self) -> None:
        self.dates: list[date] = []

    def industry_for(self, instrument_id: str, as_of: date) -> str | None:
        self.dates.append(as_of)
        if instrument_id == "MOVED" and as_of < date(2025, 2, 1):
            return "OLD"
        return "TECH"


def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 16, tzinfo=UTC)


def feature(
    industry: str = "TECH", score: float = 70, as_of: datetime | None = None
) -> ThemeFeatureResult:
    return ThemeFeatureResult(
        as_of or at(date(2026, 1, 1)),
        industry,
        5,
        True,
        0.05,
        0.10,
        0.20,
        score,
        0.7,
        score,
        0.2,
        score,
        0.4,
        score,
        0.01,
        score,
        0.6,
        score,
        0.3,
        "snap",
        "theme-v1",
        (),
    )


def test_historical_industry_features_and_point_in_time_filter() -> None:
    resolver = Resolver()
    start = date(2025, 1, 1)
    bars = []
    benchmark = []
    for i in range(70):
        day = start + timedelta(days=i)
        benchmark.append(BenchmarkBar(day, 100 * 1.001**i, at(day)))
        for j, instrument in enumerate(("A", "B", "MOVED")):
            bars.append(
                IndustryStockBar(
                    instrument,
                    day,
                    100 * (1.003 + j * 0.0001) ** i,
                    1000 * (1.002**i),
                    True,
                    at(day),
                )
            )
    as_of = at(start + timedelta(days=69))
    bars.append(IndustryStockBar("A", as_of.date() + timedelta(days=1), 1, 1, True, as_of))
    results = ThemeFeatureEngine(resolver).calculate(
        bars, benchmark, as_of=as_of, data_version="snap"
    )
    tech = next(item for item in results if item.industry_id == "TECH")
    assert tech.eligible
    assert tech.relative_return_60d > 0
    assert tech.member_count == 3
    assert min(resolver.dates) < date(2025, 2, 1)
    assert max(resolver.dates) == as_of.date()
    with pytest.raises(ValueError):
        ThemeFeatureEngine(resolver, min_members=0)
    with pytest.raises(ValueError, match="require stock bars"):
        ThemeFeatureEngine(resolver).calculate([], benchmark, as_of=as_of, data_version="x")


def test_scoring_confirmation_crowding_and_fading() -> None:
    scorer = ThemeScoringEngine()
    with pytest.raises(ValueError):
        scorer.score(feature(), event_support_score=101)
    machine = ThemeStateMachine()
    states = []
    for offset in range(3):
        when = at(date(2026, 1, 1) + timedelta(days=offset))
        daily = scorer.score(replace(feature(as_of=when), as_of=when))
        states.append(machine.apply([daily])[0].state)
    assert states == [ThemeState.EMERGING, ThemeState.EMERGING, ThemeState.CONFIRMED]
    when = at(date(2026, 1, 4))
    crowded_feature = replace(feature(as_of=when), relative_return_5d=0.15, turnover_expansion=1.5)
    assert machine.apply([scorer.score(crowded_feature)])[0].state is ThemeState.CROWDED
    when = at(date(2026, 1, 5))
    weak = scorer.score(feature(score=10, as_of=when))
    assert machine.apply([weak])[0].state is ThemeState.FADING
    with pytest.raises(ValueError, match="increasing"):
        machine.apply([weak])
    assert ThemeStateMachine().apply([]) == ()
    with pytest.raises(ValueError):
        MainlineConfig(relative_strength_weight=0.9)
