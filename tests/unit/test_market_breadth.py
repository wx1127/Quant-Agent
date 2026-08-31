"""Tests for historical-universe-safe market breadth metrics."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.features.market_breadth import (
    BreadthObservation,
    InsufficientBreadthData,
    MarketBreadthAnalyzer,
    MarketBreadthConfig,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
START_DATE = date(2026, 8, 3)
START_TIME = datetime(2026, 8, 3, 15, 0, tzinfo=SHANGHAI)


def _row(
    instrument_id: str,
    day: int,
    close: str,
    turnover: str,
    *,
    active: bool = True,
    suspended: bool = False,
    available_delay: timedelta = timedelta(minutes=30),
    revision: str = "v1",
) -> BreadthObservation:
    observed_at = START_TIME + timedelta(days=day)
    return BreadthObservation(
        instrument_id=instrument_id,
        trade_date=START_DATE + timedelta(days=day),
        observed_at=observed_at,
        available_at=observed_at + available_delay,
        close=Decimal(close),
        turnover=Decimal(turnover),
        is_active=active,
        is_suspended=suspended,
        revision=revision,
    )


def _config(*, coverage: str = "1") -> MarketBreadthConfig:
    return MarketBreadthConfig(
        version="breadth-test-v1",
        moving_average_window=2,
        new_high_low_window=3,
        turnover_percentile_window=3,
        downside_volatility_window=3,
        min_downside_observations=2,
        minimum_current_coverage=Decimal(coverage),
        minimum_historical_coverage=Decimal(coverage),
        minimum_turnover_observations=2,
    )


def _fixed_sample() -> list[BreadthObservation]:
    values = {
        "A": (("10", "100"), ("11", "150"), ("12", "200")),
        "B": (("10", "100"), ("9", "150"), ("8", "200")),
        "C": (("10", "100"), ("10", "150"), ("10", "200")),
    }
    return [
        _row(instrument_id, day, close, turnover)
        for instrument_id, history in values.items()
        for day, (close, turnover) in enumerate(history)
    ]


def _analyze(
    observations: list[BreadthObservation],
    *,
    config: MarketBreadthConfig | None = None,
    expected: tuple[str, ...] = ("A", "B", "C"),
    expected_by_date: dict[date, tuple[str, ...]] | None = None,
    as_of: datetime | None = None,
):
    return MarketBreadthAnalyzer(config or _config()).analyze(
        session_date=START_DATE + timedelta(days=2),
        as_of=as_of or START_TIME + timedelta(days=3),
        data_version="snapshot-v1",
        expected_active_by_date=expected_by_date
        or {START_DATE + timedelta(days=day): expected for day in range(3)},
        observations=observations,
    )


def test_fixed_sample_metrics_and_denominators_match_manual_values() -> None:
    snapshot = _analyze(_fixed_sample())

    assert (snapshot.advancing_count, snapshot.declining_count) == (1, 1)
    assert snapshot.unchanged_count == 1
    assert snapshot.return_denominator == 3
    assert snapshot.advance_decline_ratio == Decimal(1) / 3
    assert snapshot.above_average_count == 1
    assert snapshot.moving_average_denominator == 3
    assert snapshot.above_average_ratio == Decimal(1) / 3
    assert (snapshot.new_high_count, snapshot.new_low_count) == (1, 1)
    assert snapshot.high_low_denominator == 3
    assert snapshot.total_turnover == Decimal(600)
    assert snapshot.turnover_percentile == 1
    assert snapshot.turnover_history_count == 3
    assert snapshot.downside_observation_count == 2
    assert snapshot.downside_volatility is not None
    assert snapshot.downside_volatility > 0
    assert snapshot.historical_session_count == 3
    assert snapshot.minimum_historical_coverage_observed == 1


def test_suspended_current_stock_is_excluded_from_every_metric_denominator() -> None:
    rows = [
        (
            _row("C", 2, "10", "200", suspended=True)
            if row.instrument_id == "C" and row.trade_date == START_DATE + timedelta(days=2)
            else row
        )
        for row in _fixed_sample()
    ]
    snapshot = _analyze(rows)

    assert snapshot.suspended_count == 1
    assert snapshot.return_denominator == 2
    assert snapshot.moving_average_denominator == 2
    assert snapshot.high_low_denominator == 2
    assert snapshot.total_turnover == Decimal(400)


def test_historical_inactive_stock_cannot_distort_turnover_or_returns() -> None:
    baseline = _analyze(_fixed_sample())
    inactive = [
        _row("DELISTED", day, str(1 + 1000 * day), "999999", active=False) for day in range(3)
    ]
    with_inactive = _analyze([*_fixed_sample(), *inactive])

    assert with_inactive.total_turnover == baseline.total_turnover
    assert with_inactive.turnover_percentile == baseline.turnover_percentile
    assert with_inactive.downside_volatility == baseline.downside_volatility
    assert with_inactive.cache_key != baseline.cache_key


def test_low_current_coverage_fails_closed() -> None:
    only_a = [row for row in _fixed_sample() if row.instrument_id == "A"]
    with pytest.raises(InsufficientBreadthData, match="coverage"):
        _analyze(only_a)


def test_degraded_coverage_preserves_missing_ids_and_denominators() -> None:
    only_a = [row for row in _fixed_sample() if row.instrument_id == "A"]
    snapshot = _analyze(only_a, config=_config(coverage="0"))

    assert snapshot.current_coverage == Decimal(1) / 3
    assert snapshot.missing_instruments == ("B", "C")
    assert snapshot.return_denominator == 1


def test_future_observation_and_revision_do_not_change_historical_result() -> None:
    as_of = START_TIME + timedelta(days=2, hours=1)
    baseline = _analyze(_fixed_sample(), as_of=as_of)
    original = next(
        row
        for row in _fixed_sample()
        if row.instrument_id == "A" and row.trade_date == START_DATE + timedelta(days=2)
    )
    future_revision = BreadthObservation(
        instrument_id=original.instrument_id,
        trade_date=original.trade_date,
        observed_at=original.observed_at,
        available_at=START_TIME + timedelta(days=4),
        close=Decimal("999999"),
        turnover=Decimal("999999"),
        is_active=True,
        is_suspended=False,
        revision="v2",
    )
    future_day = _row("A", 3, "999999", "999999")
    repeated = _analyze(
        [*_fixed_sample(), future_revision, future_day],
        as_of=as_of,
    )

    assert repeated.cache_key == baseline.cache_key
    assert repeated.total_turnover == baseline.total_turnover


def test_short_history_reports_zero_denominators_instead_of_inventing_ratios() -> None:
    rows = [_row("A", 2, "10", "100")]
    snapshot = _analyze(
        rows,
        config=_config(coverage="1"),
        expected=("A",),
        expected_by_date={START_DATE + timedelta(days=2): ("A",)},
    )

    assert snapshot.return_denominator == 0
    assert snapshot.advance_decline_ratio is None
    assert snapshot.moving_average_denominator == 0
    assert snapshot.above_average_ratio is None
    assert snapshot.high_low_denominator == 0
    assert snapshot.new_high_ratio is None
    assert snapshot.downside_volatility is None
    assert snapshot.turnover_percentile is None


def test_same_inputs_are_reproducible_and_version_changes_identity() -> None:
    first = _analyze(_fixed_sample())
    repeated = _analyze(list(reversed(_fixed_sample())))
    changed = MarketBreadthAnalyzer(_config()).analyze(
        session_date=START_DATE + timedelta(days=2),
        as_of=START_TIME + timedelta(days=3),
        data_version="snapshot-v2",
        expected_active_by_date={
            START_DATE + timedelta(days=day): ("A", "B", "C") for day in range(3)
        },
        observations=_fixed_sample(),
    )

    assert first == repeated
    assert first.cache_key != changed.cache_key


def test_unexpected_current_active_record_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside expected universe"):
        _analyze([*_fixed_sample(), _row("X", 2, "10", "10")])


def test_missing_historical_active_member_fails_before_metrics_can_shift() -> None:
    universes = {
        START_DATE: ("A", "B", "C", "DELISTED"),
        START_DATE + timedelta(days=1): ("A", "B", "C"),
        START_DATE + timedelta(days=2): ("A", "B", "C"),
    }

    with pytest.raises(InsufficientBreadthData, match=str(START_DATE)):
        _analyze(_fixed_sample(), expected_by_date=universes)

    complete = _analyze(
        [*_fixed_sample(), _row("DELISTED", 0, "100", "500")],
        expected_by_date=universes,
    )
    assert complete.minimum_historical_coverage_observed == 1


def test_universe_set_order_and_decimal_scale_do_not_change_identity() -> None:
    first = _analyze(_fixed_sample(), expected=("A", "B", "C"))
    reordered = _analyze(
        _fixed_sample(),
        expected=("C", "B", "A"),
        config=_config(coverage="1.0"),
    )
    assert reordered == first
    assert _config(coverage="1.0").config_hash == _config(coverage="1.00").config_hash


def test_ambiguous_revision_is_rejected() -> None:
    rows = _fixed_sample()
    original = rows[0]
    rows.append(
        BreadthObservation(
            instrument_id=original.instrument_id,
            trade_date=original.trade_date,
            observed_at=original.observed_at,
            available_at=original.available_at,
            close=Decimal("999"),
            turnover=original.turnover,
            is_active=True,
            is_suspended=False,
            revision="v2",
        )
    )
    with pytest.raises(ValueError, match="ambiguous breadth revisions"):
        _analyze(rows)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MarketBreadthConfig(moving_average_window=1),
        lambda: MarketBreadthConfig(min_downside_observations=21),
        lambda: MarketBreadthConfig(minimum_current_coverage=Decimal("1.1")),
        lambda: MarketBreadthConfig(minimum_historical_coverage=Decimal("NaN")),
        lambda: MarketBreadthConfig(minimum_turnover_observations=61),
        lambda: _row("A", 0, "0", "1"),
        lambda: _row("A", 0, "1", "-1"),
    ],
)
def test_invalid_contracts_are_rejected(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]
