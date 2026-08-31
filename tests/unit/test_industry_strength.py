"""Tests for historical-membership-aware industry relative strength."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.domain import IndustryMembership
from quant_agent.features.core import FeatureObservation
from quant_agent.features.industry_strength import (
    IndustryStrengthAnalyzer,
    IndustryStrengthConfig,
    IndustryStrengthStatus,
    InsufficientIndustryData,
    PointInTimeIndustryMembership,
)
from quant_agent.features.market_breadth import BreadthObservation

TZ = ZoneInfo("Asia/Shanghai")
START_DATE = date(2026, 8, 3)
START_TIME = datetime(2026, 8, 3, 15, 0, tzinfo=TZ)
INDUSTRY_UP = "SW2021:UP"
INDUSTRY_DOWN = "SW2021:DOWN"


def _config(*, minimum_members: int = 2) -> IndustryStrengthConfig:
    return IndustryStrengthConfig(
        version="industry-test-v1",
        horizons=(2, 3),
        horizon_weights=(Decimal("0.4"), Decimal("0.6")),
        minimum_members=minimum_members,
        new_high_window=3,
        turnover_lookback=2,
        downside_window=3,
    )


def _membership(
    instrument_id: str,
    industry_id: str,
    *,
    start_day: int = 0,
    end_day: int | None = None,
) -> PointInTimeIndustryMembership:
    return PointInTimeIndustryMembership(
        membership=IndustryMembership(
            instrument_id=instrument_id,
            industry_id=industry_id,
            effective_from=START_DATE + timedelta(days=start_day),
            effective_to=(START_DATE + timedelta(days=end_day) if end_day is not None else None),
            source="test",
            version="SW2021",
        ),
        level=3,
        available_at=START_TIME,
        revision="v1",
    )


def _stock_row(
    instrument_id: str,
    day: int,
    close: Decimal,
    *,
    turnover: Decimal = Decimal(100),
    suspended: bool = False,
) -> BreadthObservation:
    observed_at = START_TIME + timedelta(days=day)
    return BreadthObservation(
        instrument_id=instrument_id,
        trade_date=START_DATE + timedelta(days=day),
        observed_at=observed_at,
        available_at=observed_at + timedelta(minutes=30),
        close=close,
        turnover=turnover,
        is_active=True,
        is_suspended=suspended,
        revision="v1",
    )


def _benchmark(day: int, close: Decimal) -> FeatureObservation:
    observed_at = START_TIME + timedelta(days=day)
    return FeatureObservation(
        observation_key=f"benchmark-{day}",
        observed_at=observed_at,
        available_at=observed_at + timedelta(minutes=30),
        value=close,
        revision="v1",
    )


def _sample() -> tuple[
    list[PointInTimeIndustryMembership],
    list[BreadthObservation],
    list[FeatureObservation],
]:
    memberships = [
        _membership("UP-A", INDUSTRY_UP),
        _membership("UP-B", INDUSTRY_UP),
        _membership("DOWN-A", INDUSTRY_DOWN),
        _membership("DOWN-B", INDUSTRY_DOWN),
    ]
    stocks: list[BreadthObservation] = []
    for day in range(6):
        stocks.extend(
            [
                _stock_row("UP-A", day, Decimal(100 + 2 * day), turnover=Decimal(100 + day * 20)),
                _stock_row("UP-B", day, Decimal(100 + day), turnover=Decimal(100 + day * 10)),
                _stock_row("DOWN-A", day, Decimal(100 - 2 * day), turnover=Decimal(100 - day * 5)),
                _stock_row("DOWN-B", day, Decimal(100 - day), turnover=Decimal(100 - day * 5)),
            ]
        )
    benchmark = [_benchmark(day, Decimal(100 + day // 2)) for day in range(6)]
    return memberships, stocks, benchmark


def _analyze(
    memberships: list[PointInTimeIndustryMembership],
    stocks: list[BreadthObservation],
    benchmark: list[FeatureObservation],
):
    return IndustryStrengthAnalyzer(_config()).analyze(
        session_date=START_DATE + timedelta(days=5),
        as_of=START_TIME + timedelta(days=6),
        data_version="snapshot-v1",
        classification_version="SW2021",
        industry_level=3,
        industry_ids=(INDUSTRY_UP, INDUSTRY_DOWN),
        memberships=memberships,
        stock_observations=stocks,
        benchmark_observations=benchmark,
    )


def test_relative_returns_rank_strong_industry_first() -> None:
    snapshot = _analyze(*_sample())
    results = {item.industry_id: item for item in snapshot.industries}

    assert results[INDUSTRY_UP].status is IndustryStrengthStatus.READY
    assert results[INDUSTRY_UP].rank == 1
    assert results[INDUSTRY_DOWN].rank == 2
    assert all(row.relative_return > 0 for row in results[INDUSTRY_UP].horizon_returns)
    assert all(row.relative_return < 0 for row in results[INDUSTRY_DOWN].horizon_returns)


def test_member_contributions_are_reconcilable() -> None:
    result = next(
        item for item in _analyze(*_sample()).industries if item.industry_id == INDUSTRY_UP
    )

    assert len(result.contributions) == 2
    current_industry_return = sum(
        (item.return_contribution for item in result.contributions), Decimal(0)
    )
    expected_current_return = (
        (Decimal(110) / Decimal(108) - 1) + (Decimal(105) / Decimal(104) - 1)
    ) / 2
    assert current_industry_return == expected_current_return
    assert sum((item.return_contribution for item in result.contributions), Decimal(0)) == sum(
        (item.current_return for item in result.contributions), Decimal(0)
    ) / len(result.contributions)
    assert sum((item.turnover_share for item in result.contributions), Decimal(0)) == 1
    assert {item.instrument_id for item in result.contributions} == {"UP-A", "UP-B"}


def test_historical_membership_switch_assigns_returns_by_effective_date() -> None:
    memberships, stocks, benchmark = _sample()
    memberships = [item for item in memberships if item.membership.instrument_id != "UP-A"]
    memberships.extend(
        [
            _membership("UP-A", INDUSTRY_UP, end_day=2),
            _membership("UP-A", INDUSTRY_DOWN, start_day=3),
        ]
    )
    snapshot = IndustryStrengthAnalyzer(
        replace(
            _config(minimum_members=1),
            minimum_member_coverage=Decimal("0.5"),
        )
    ).analyze(
        session_date=START_DATE + timedelta(days=5),
        as_of=START_TIME + timedelta(days=6),
        data_version="snapshot-v1",
        classification_version="SW2021",
        industry_level=3,
        industry_ids=(INDUSTRY_UP, INDUSTRY_DOWN),
        memberships=memberships,
        stock_observations=stocks,
        benchmark_observations=benchmark,
    )
    results = {item.industry_id: item for item in snapshot.industries}

    assert "UP-A" not in {item.instrument_id for item in results[INDUSTRY_UP].contributions}
    assert "UP-A" in {item.instrument_id for item in results[INDUSTRY_DOWN].contributions}

    boundary = IndustryStrengthAnalyzer(
        replace(
            _config(minimum_members=1),
            minimum_member_coverage=Decimal("0.5"),
        )
    ).analyze(
        session_date=START_DATE + timedelta(days=3),
        as_of=START_TIME + timedelta(days=4),
        data_version="snapshot-v1",
        classification_version="SW2021",
        industry_level=3,
        industry_ids=(INDUSTRY_UP, INDUSTRY_DOWN),
        memberships=memberships,
        stock_observations=stocks,
        benchmark_observations=benchmark,
    )
    boundary_down = next(item for item in boundary.industries if item.industry_id == INDUSTRY_DOWN)
    assert boundary_down.status is IndustryStrengthStatus.READY
    assert "UP-A" not in {item.instrument_id for item in boundary_down.contributions}


def test_insufficient_members_are_reported_without_score_or_rank() -> None:
    memberships, stocks, benchmark = _sample()
    memberships = [item for item in memberships if item.membership.instrument_id != "UP-B"]
    snapshot = _analyze(memberships, stocks, benchmark)
    result = next(item for item in snapshot.industries if item.industry_id == INDUSTRY_UP)

    assert result.status is IndustryStrengthStatus.INSUFFICIENT_DATA
    assert result.score is None
    assert result.rank is None
    assert "member" in (result.reason or "")


def test_future_stock_record_does_not_change_historical_snapshot() -> None:
    memberships, stocks, benchmark = _sample()
    baseline = _analyze(memberships, stocks, benchmark)
    future = _stock_row("UP-A", 7, Decimal("999999"))
    repeated = _analyze(memberships, [*stocks, future], benchmark)

    assert repeated == baseline


def test_membership_unavailable_at_as_of_cannot_leak_into_snapshot() -> None:
    memberships, stocks, benchmark = _sample()
    leaked = replace(
        _membership("UP-A", INDUSTRY_DOWN),
        available_at=START_TIME + timedelta(days=10),
        revision="future-v1",
    )

    assert _analyze([*memberships, leaked], stocks, benchmark) == _analyze(
        memberships,
        stocks,
        benchmark,
    )


def test_low_relative_member_coverage_is_not_ranked() -> None:
    memberships, stocks, benchmark = _sample()
    memberships.extend(_membership(f"MISSING-{index}", INDUSTRY_UP) for index in range(10))
    result = next(
        item
        for item in _analyze(memberships, stocks, benchmark).industries
        if item.industry_id == INDUSTRY_UP
    )

    assert result.status is IndustryStrengthStatus.INSUFFICIENT_DATA
    assert result.current_member_count == 12
    assert result.comparable_return_count == 2
    assert result.current_member_coverage == Decimal(1) / 6
    assert "coverage" in (result.reason or "")


def test_missing_intermediate_benchmark_session_breaks_interval_alignment() -> None:
    memberships, stocks, benchmark = _sample()
    without_day_four = [
        item for item in benchmark if item.observed_at.date() != START_DATE + timedelta(days=4)
    ]
    snapshot = _analyze(memberships, stocks, without_day_four)

    assert all(
        item.status is IndustryStrengthStatus.INSUFFICIENT_DATA for item in snapshot.industries
    )


def test_input_order_is_reproducible_and_data_version_changes_cache_identity() -> None:
    memberships, stocks, benchmark = _sample()
    analyzer = IndustryStrengthAnalyzer(_config())
    first = _analyze(memberships, stocks, benchmark)
    repeated = _analyze(
        list(reversed(memberships)),
        list(reversed(stocks)),
        list(reversed(benchmark)),
    )
    changed = analyzer.analyze(
        session_date=START_DATE + timedelta(days=5),
        as_of=START_TIME + timedelta(days=6),
        data_version="snapshot-v2",
        classification_version="SW2021",
        industry_level=3,
        industry_ids=(INDUSTRY_UP, INDUSTRY_DOWN),
        memberships=memberships,
        stock_observations=stocks,
        benchmark_observations=benchmark,
    )

    assert first == repeated
    assert first.cache_key != changed.cache_key


def test_industry_id_set_order_and_decimal_scale_are_canonical() -> None:
    memberships, stocks, benchmark = _sample()
    analyzer = IndustryStrengthAnalyzer(_config())
    first = _analyze(memberships, stocks, benchmark)
    reordered = analyzer.analyze(
        session_date=START_DATE + timedelta(days=5),
        as_of=START_TIME + timedelta(days=6),
        data_version="snapshot-v1",
        classification_version="SW2021",
        industry_level=3,
        industry_ids=(INDUSTRY_DOWN, INDUSTRY_UP),
        memberships=memberships,
        stock_observations=stocks,
        benchmark_observations=benchmark,
    )

    assert reordered == first
    assert replace(_config(), score_scale=Decimal("200.0")).config_hash == _config().config_hash


def test_insufficient_benchmark_fails_closed() -> None:
    memberships, stocks, benchmark = _sample()
    with pytest.raises(InsufficientIndustryData, match="benchmark"):
        _analyze(memberships, stocks, benchmark[:3])


def test_overlapping_target_memberships_are_rejected() -> None:
    memberships, stocks, benchmark = _sample()
    memberships.append(_membership("UP-A", INDUSTRY_DOWN))
    with pytest.raises(ValueError, match="overlapping"):
        _analyze(memberships, stocks, benchmark)


def test_other_classification_level_does_not_create_false_overlap() -> None:
    memberships, stocks, benchmark = _sample()
    level_one = replace(
        _membership("UP-A", "SW2021:L1"),
        level=1,
    )

    snapshot = _analyze([*memberships, level_one], stocks, benchmark)
    assert any(item.status is IndustryStrengthStatus.READY for item in snapshot.industries)


def test_zero_historical_turnover_is_explicitly_insufficient() -> None:
    memberships, stocks, benchmark = _sample()
    zero_history = [
        replace(
            item,
            turnover=(
                Decimal(100) if item.trade_date == START_DATE + timedelta(days=5) else Decimal(0)
            ),
        )
        for item in stocks
    ]
    snapshot = _analyze(memberships, zero_history, benchmark)

    assert all(item.score is None for item in snapshot.industries)
    assert all("turnover" in (item.reason or "") for item in snapshot.industries)


def test_duplicate_benchmark_session_is_rejected() -> None:
    memberships, stocks, benchmark = _sample()
    duplicate = FeatureObservation(
        observation_key="different-key-same-session",
        observed_at=benchmark[-1].observed_at,
        available_at=benchmark[-1].available_at,
        value=benchmark[-1].value,
        revision="v1",
    )
    with pytest.raises(ValueError, match="multiple observations"):
        _analyze(memberships, stocks, [*benchmark, duplicate])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: IndustryStrengthConfig(horizons=()),
        lambda: IndustryStrengthConfig(horizons=(5, 5, 60)),
        lambda: IndustryStrengthConfig(horizon_weights=(Decimal(1),)),
        lambda: IndustryStrengthConfig(minimum_members=0),
        lambda: IndustryStrengthConfig(relative_weight=Decimal("0.4")),
        lambda: IndustryStrengthConfig(score_scale=Decimal(0)),
        lambda: IndustryStrengthConfig(score_scale=Decimal("Infinity")),
        lambda: IndustryStrengthConfig(minimum_member_coverage=Decimal("NaN")),
    ],
)
def test_invalid_configuration_is_rejected(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]
