"""Tests for point-in-time stock trend and relative-strength features."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.domain import DailyBar, IndustryMembership
from quant_agent.features.stock_strength import (
    HorizonStrengthMetrics,
    IndustryMembershipError,
    InsufficientHistoryPolicy,
    InsufficientStockData,
    MissingIndustryPolicy,
    PointInTimeStockIndustryMembership,
    ReferenceBarObservation,
    ScoreContribution,
    StockBarObservation,
    StockStrengthAnalyzer,
    StockStrengthConfig,
    StockStrengthStatus,
    SuspendedStockError,
    SuspensionPolicy,
    TrendQualityMetrics,
    VolumePriceConfirmation,
)

TZ = ZoneInfo("Asia/Shanghai")
START_DATE = date(2026, 6, 1)
START_TIME = datetime(2026, 6, 1, 15, 0, tzinfo=TZ)
SESSION_DAY = 69
SESSION_DATE = START_DATE + timedelta(days=SESSION_DAY)
AS_OF = START_TIME + timedelta(days=SESSION_DAY, hours=1)
STOCK_ID = "600000.SH"
BENCHMARK_ID = "000300.SH"
INDUSTRY_A = "SW2021:A"
INDUSTRY_B = "SW2021:B"


def _bar(
    day: int,
    close: Decimal,
    *,
    instrument_id: str = STOCK_ID,
    adjusted_close: Decimal | None = None,
    volume: Decimal = Decimal(100),
    suspended: bool = False,
    available_delay: timedelta = timedelta(minutes=30),
    revision: str = "v1",
) -> StockBarObservation:
    observed_at = START_TIME + timedelta(days=day)
    return StockBarObservation(
        bar=DailyBar(
            instrument_id=instrument_id,
            trade_date=START_DATE + timedelta(days=day),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=volume,
            turnover=volume * close,
            source="test",
            available_at=observed_at + available_delay,
            version="raw-v1",
        ),
        observed_at=observed_at,
        adjusted_close=adjusted_close if adjusted_close is not None else close,
        is_suspended=suspended,
        revision=revision,
    )


def _reference(
    reference_id: str,
    day: int,
    close: Decimal,
    *,
    available_delay: timedelta = timedelta(minutes=30),
    revision: str = "v1",
) -> ReferenceBarObservation:
    observed_at = START_TIME + timedelta(days=day)
    return ReferenceBarObservation(
        reference_id=reference_id,
        trade_date=START_DATE + timedelta(days=day),
        observed_at=observed_at,
        available_at=observed_at + available_delay,
        close=close,
        revision=revision,
    )


def _membership(
    industry_id: str,
    *,
    start_day: int = 0,
    end_day: int | None = None,
    level: int = 3,
    available_at: datetime = START_TIME - timedelta(hours=1),
    revision: str = "v1",
) -> PointInTimeStockIndustryMembership:
    return PointInTimeStockIndustryMembership(
        membership=IndustryMembership(
            instrument_id=STOCK_ID,
            industry_id=industry_id,
            effective_from=START_DATE + timedelta(days=start_day),
            effective_to=START_DATE + timedelta(days=end_day) if end_day is not None else None,
            source="test",
            version="SW2021",
        ),
        level=level,
        available_at=available_at,
        revision=revision,
    )


def _sample() -> tuple[
    list[StockBarObservation],
    list[ReferenceBarObservation],
    list[PointInTimeStockIndustryMembership],
]:
    stocks = [
        _bar(
            day,
            Decimal(100 + day),
            volume=Decimal(200) if day == SESSION_DAY else Decimal(100),
        )
        for day in range(SESSION_DAY + 1)
    ]
    references: list[ReferenceBarObservation] = []
    for day in range(SESSION_DAY + 1):
        references.extend(
            [
                _reference(BENCHMARK_ID, day, Decimal(100) + Decimal(day) / 2),
                _reference(INDUSTRY_A, day, Decimal(100) + Decimal(day) * Decimal("0.75")),
                _reference(INDUSTRY_B, day, Decimal(200) - Decimal(day) / 2),
            ]
        )
    return stocks, references, [_membership(INDUSTRY_A)]


def _analyze(
    stocks: list[StockBarObservation],
    references: list[ReferenceBarObservation],
    memberships: list[PointInTimeStockIndustryMembership],
    *,
    config: StockStrengthConfig | None = None,
    data_version: str = "snapshot-v1",
):
    return StockStrengthAnalyzer(config).analyze(
        instrument_id=STOCK_ID,
        benchmark_id=BENCHMARK_ID,
        session_date=SESSION_DATE,
        as_of=AS_OF,
        data_version=data_version,
        classification_version="SW2021",
        industry_level=3,
        memberships=memberships,
        stock_observations=stocks,
        reference_observations=references,
    )


def _compound(values: list[Decimal]) -> Decimal:
    result = Decimal(1)
    for value in values:
        result *= 1 + value
    return result - 1


def _returns(values: list[Decimal], horizon: int) -> Decimal:
    selected = values[-(horizon + 1) :]
    return _compound([current / prior - 1 for prior, current in pairwise(selected)])


def test_fixed_sample_matches_manual_horizon_trend_and_volume_calculations() -> None:
    stocks, references, memberships = _sample()
    snapshot = _analyze(stocks, references, memberships)
    rows = {item.horizon: item for item in snapshot.horizons}
    stock_closes = [Decimal(100 + day) for day in range(SESSION_DAY + 1)]
    benchmark_closes = [Decimal(100) + Decimal(day) / 2 for day in range(SESSION_DAY + 1)]
    industry_closes = [
        Decimal(100) + Decimal(day) * Decimal("0.75") for day in range(SESSION_DAY + 1)
    ]

    assert snapshot.status is StockStrengthStatus.READY
    assert snapshot.current_industry_id == INDUSTRY_A
    assert tuple(rows) == (5, 20, 60)
    for horizon in (5, 20, 60):
        assert rows[horizon].stock_return == _returns(stock_closes, horizon)
        assert rows[horizon].benchmark_return == _returns(benchmark_closes, horizon)
        assert rows[horizon].industry_return == _returns(industry_closes, horizon)
        assert rows[horizon].benchmark_relative_return == (
            rows[horizon].stock_return - rows[horizon].benchmark_return
        )
        assert rows[horizon].industry_relative_return == (
            rows[horizon].stock_return - rows[horizon].industry_return
        )

    trend = snapshot.trend_quality
    assert trend is not None
    expected_average = sum(stock_closes[-20:], Decimal(0)) / 20
    prior_average = sum(stock_closes[-25:-5], Decimal(0)) / 20
    assert trend.latest_close == Decimal(169)
    assert trend.moving_average == expected_average
    assert trend.moving_average_slope == expected_average / prior_average - 1
    assert trend.breakout_level == Decimal(166)
    assert trend.breakout_hold_ratio == 1
    assert trend.pullback_depth == 0
    assert trend.distance_to_high == 0

    volume = snapshot.volume_confirmation
    assert volume is not None
    assert volume.average_prior_volume == 100
    assert volume.volume_ratio == 2
    assert volume.score == 100
    assert snapshot.score == sum((item.contribution for item in snapshot.contributions), Decimal(0))
    assert sum((item.effective_weight for item in snapshot.contributions), Decimal(0)) == 1
    assert len(snapshot.input_hash) == len(snapshot.cache_key) == len(snapshot.result_hash) == 64
    assert snapshot.identity_payload()["result_hash"] == snapshot.result_hash


def test_historical_membership_is_applied_per_interval_not_backfilled_from_current() -> None:
    stocks, references, _ = _sample()
    memberships = [
        _membership(INDUSTRY_A, end_day=34),
        _membership(INDUSTRY_B, start_day=35),
    ]
    snapshot = _analyze(stocks, references, memberships)
    sixty = next(item for item in snapshot.horizons if item.horizon == 60)
    reference_map = {(item.reference_id, item.trade_date): item.close for item in references}
    canonical_dates = [START_DATE + timedelta(days=day) for day in range(9, 70)]
    expected_daily: list[Decimal] = []
    for prior, current in pairwise(canonical_dates):
        industry_id = INDUSTRY_A if current <= START_DATE + timedelta(days=34) else INDUSTRY_B
        expected_daily.append(
            reference_map[(industry_id, current)] / reference_map[(industry_id, prior)] - 1
        )
    all_current_industry = _returns(
        [Decimal(200) - Decimal(day) / 2 for day in range(SESSION_DAY + 1)],
        60,
    )

    assert snapshot.current_industry_id == INDUSTRY_B
    assert sixty.historical_industry_ids == (INDUSTRY_A, INDUSTRY_B)
    assert sixty.industry_return == _compound(expected_daily)
    assert sixty.industry_return != all_current_industry


def test_future_records_and_future_revisions_do_not_change_old_result() -> None:
    stocks, references, memberships = _sample()
    baseline = _analyze(stocks, references, memberships)
    future_stock = _bar(70, Decimal("99999"))
    future_references = [
        _reference(BENCHMARK_ID, 70, Decimal("1")),
        _reference(INDUSTRY_A, 70, Decimal("1")),
    ]
    future_membership = _membership(
        INDUSTRY_B,
        start_day=70,
        available_at=AS_OF + timedelta(days=1),
    )
    unavailable_revision = _bar(
        SESSION_DAY,
        Decimal(169),
        adjusted_close=Decimal("999"),
        available_delay=timedelta(days=2),
        revision="future-revision",
    )

    repeated = _analyze(
        [*stocks, future_stock, unavailable_revision],
        [*references, *future_references],
        [*memberships, future_membership],
    )
    reordered = _analyze(
        list(reversed(stocks)),
        list(reversed(references)),
        list(reversed(memberships)),
    )
    assert repeated == baseline
    assert reordered == baseline

    known_revision = _bar(
        SESSION_DAY,
        Decimal(170),
        adjusted_close=Decimal(170),
        available_delay=timedelta(minutes=45),
        revision="v2",
    )
    changed = _analyze([*stocks, known_revision], references, memberships)
    assert changed.input_hash != baseline.input_hash
    assert changed.result_hash != baseline.result_hash


def test_missing_industry_can_degrade_with_renormalized_auditable_weights() -> None:
    stocks, references, _ = _sample()
    degraded = _analyze(stocks, references, [])

    assert degraded.status is StockStrengthStatus.DEGRADED
    assert degraded.current_industry_id is None
    assert "membership missing" in (degraded.reason or "")
    assert all(item.industry_return is None for item in degraded.horizons)
    assert "industry_relative" not in {item.component for item in degraded.contributions}
    assert sum((item.effective_weight for item in degraded.contributions), Decimal(0)) == 1

    fail_config = replace(
        StockStrengthConfig(),
        missing_industry_policy=MissingIndustryPolicy.FAIL,
    )
    with pytest.raises(IndustryMembershipError, match="membership missing"):
        _analyze(stocks, references, [], config=fail_config)


def test_current_and_historical_suspension_policies_are_explicit() -> None:
    stocks, references, memberships = _sample()
    current_suspended = [*stocks[:-1], replace(stocks[-1], is_suspended=True)]
    returned = _analyze(current_suspended, references, memberships)
    assert returned.status is StockStrengthStatus.SUSPENDED
    assert returned.score is None

    fail_config = replace(
        StockStrengthConfig(),
        suspension_policy=SuspensionPolicy.FAIL,
    )
    with pytest.raises(SuspendedStockError):
        _analyze(current_suspended, references, memberships, config=fail_config)

    carried = list(stocks)
    carried[68] = replace(
        carried[68],
        adjusted_close=carried[67].adjusted_close,
        is_suspended=True,
    )
    active = _analyze(carried, references, memberships)
    assert active.status is StockStrengthStatus.READY
    assert active.volume_confirmation is not None
    assert active.volume_confirmation.baseline_observations == 20

    inconsistent = list(stocks)
    inconsistent[68] = replace(inconsistent[68], is_suspended=True)
    with pytest.raises(ValueError, match="carry"):
        _analyze(inconsistent, references, memberships)


def test_excess_suspension_and_short_windows_fail_or_return_status() -> None:
    stocks, references, memberships = _sample()
    too_many = list(stocks)
    carry_value = too_many[59].adjusted_close
    for index in range(60, 64):
        too_many[index] = replace(
            too_many[index],
            adjusted_close=carry_value,
            is_suspended=True,
        )
    with pytest.raises(InsufficientStockData, match="suspension count"):
        _analyze(too_many, references, memberships)

    status_config = replace(
        StockStrengthConfig(),
        insufficient_history_policy=InsufficientHistoryPolicy.RETURN_STATUS,
    )
    returned = _analyze(too_many, references, memberships, config=status_config)
    assert returned.status is StockStrengthStatus.INSUFFICIENT_DATA
    assert "suspension count" in (returned.reason or "")

    with pytest.raises(InsufficientStockData, match="requires"):
        _analyze(stocks[-30:], references, memberships)
    short = _analyze(stocks[-30:], references, memberships, config=status_config)
    assert short.status is StockStrengthStatus.INSUFFICIENT_DATA


def test_missing_stock_or_benchmark_session_is_detected() -> None:
    stocks, references, memberships = _sample()
    without_stock_session = [
        item for item in stocks if item.trade_date != START_DATE + timedelta(days=50)
    ]
    with pytest.raises(InsufficientStockData, match="aligned"):
        _analyze(without_stock_session, references, memberships)

    without_benchmark_session = [
        item
        for item in references
        if not (
            item.reference_id == BENCHMARK_ID and item.trade_date == START_DATE + timedelta(days=50)
        )
    ]
    with pytest.raises(InsufficientStockData, match="aligned"):
        _analyze(stocks, without_benchmark_session, memberships)

    without_current = stocks[:-1]
    with pytest.raises(InsufficientStockData, match="session_date"):
        _analyze(without_current, references, memberships)


def test_membership_level_overlap_and_revision_ambiguity_fail_closed() -> None:
    stocks, references, memberships = _sample()
    mixed_level = [*memberships, _membership("SW2021:L2", level=2)]
    with pytest.raises(IndustryMembershipError, match="exactly"):
        _analyze(stocks, references, mixed_level)

    overlapping = [
        _membership(INDUSTRY_A),
        _membership(INDUSTRY_B, start_day=35),
    ]
    with pytest.raises(IndustryMembershipError, match="overlap"):
        _analyze(stocks, references, overlapping)

    ambiguous_membership = replace(
        memberships[0],
        membership=memberships[0].membership.model_copy(update={"industry_id": INDUSTRY_B}),
    )
    with pytest.raises(IndustryMembershipError, match="ambiguous"):
        _analyze(stocks, references, [memberships[0], ambiguous_membership])

    ambiguous_stock = replace(stocks[-1], adjusted_close=Decimal(170), revision="other")
    with pytest.raises(ValueError, match="ambiguous stock"):
        _analyze([*stocks, ambiguous_stock], references, memberships)

    benchmark = next(
        item
        for item in references
        if item.reference_id == BENCHMARK_ID and item.trade_date == SESSION_DATE
    )
    ambiguous_reference = replace(benchmark, close=benchmark.close + 1, revision="other")
    with pytest.raises(ValueError, match="ambiguous reference"):
        _analyze(stocks, [*references, ambiguous_reference], memberships)


def test_zero_volume_baseline_and_mixed_stock_inputs_fail_closed() -> None:
    stocks, references, memberships = _sample()
    zero_volume = [
        replace(item, bar=item.bar.model_copy(update={"volume": Decimal(0)}))
        if item.trade_date < SESSION_DATE
        else item
        for item in stocks
    ]
    with pytest.raises(InsufficientStockData, match="average volume"):
        _analyze(zero_volume, references, memberships)

    mixed = [*stocks, _bar(10, Decimal(110), instrument_id="OTHER")]
    with pytest.raises(ValueError, match="mix instrument"):
        _analyze(mixed, references, memberships)


def test_versions_config_identity_and_frozen_outputs_are_stable() -> None:
    stocks, references, memberships = _sample()
    first = _analyze(stocks, references, memberships)
    changed_version = _analyze(
        stocks,
        references,
        memberships,
        data_version="snapshot-v2",
    )
    assert first.input_hash == changed_version.input_hash
    assert first.cache_key != changed_version.cache_key
    assert first.result_hash != changed_version.result_hash
    assert (
        replace(
            StockStrengthConfig(),
            return_score_scale=Decimal("400.0"),
        ).config_hash
        == StockStrengthConfig().config_hash
    )
    with pytest.raises(FrozenInstanceError):
        first.score = Decimal(0)  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StockStrengthConfig(version=" "),
        lambda: StockStrengthConfig(horizons=()),
        lambda: StockStrengthConfig(horizons=(5, 5, 60)),
        lambda: StockStrengthConfig(horizon_weights=(Decimal(1),)),
        lambda: StockStrengthConfig(moving_average_window=1),
        lambda: StockStrengthConfig(maximum_historical_suspensions=-1),
        lambda: StockStrengthConfig(absolute_return_weight=Decimal("0.2")),
        lambda: StockStrengthConfig(trend_position_weight=Decimal("0.2")),
        lambda: StockStrengthConfig(return_score_scale=Decimal(0)),
        lambda: StockStrengthConfig(relative_score_scale=Decimal("Infinity")),
    ],
)
def test_invalid_configurations_are_rejected(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_input_contracts_reject_non_finite_and_temporally_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _reference(BENCHMARK_ID, 0, Decimal("Infinity"))
    with pytest.raises(ValueError, match="precede"):
        _reference(
            BENCHMARK_ID,
            0,
            Decimal(100),
            available_delay=timedelta(minutes=-1),
        )
    valid = _bar(0, Decimal(100))
    with pytest.raises(ValueError, match="finite"):
        StockBarObservation(
            bar=valid.bar.model_copy(update={"volume": Decimal("Infinity")}),
            observed_at=valid.observed_at,
            adjusted_close=valid.adjusted_close,
            is_suspended=False,
            revision="v1",
        )


def test_input_contract_boundaries_are_validated() -> None:
    valid = _bar(0, Decimal(100))
    with pytest.raises(ValueError, match="trade_date"):
        replace(valid, observed_at=valid.observed_at + timedelta(days=1))
    with pytest.raises(ValueError, match="precede"):
        replace(
            valid,
            bar=valid.bar.model_copy(
                update={"available_at": valid.observed_at - timedelta(minutes=1)}
            ),
        )
    with pytest.raises(ValueError, match="adjusted_close"):
        replace(valid, adjusted_close=Decimal(0))
    with pytest.raises(ValueError, match="identifiers"):
        replace(valid, bar=valid.bar.model_copy(update={"instrument_id": " "}))
    reference = _reference(BENCHMARK_ID, 0, Decimal(100))
    with pytest.raises(ValueError, match="trade_date"):
        replace(reference, trade_date=reference.trade_date + timedelta(days=1))
    with pytest.raises(ValueError, match="level"):
        replace(_membership(INDUSTRY_A), level=4)
    with pytest.raises(ValueError, match="identifiers"):
        replace(
            _membership(INDUSTRY_A),
            membership=_membership(INDUSTRY_A).membership.model_copy(update={"industry_id": " "}),
        )
    with pytest.raises(ValueError, match="horizon weights"):
        StockStrengthConfig(horizon_weights=(Decimal("-0.1"), Decimal("0.3"), Decimal("0.8")))


def test_metric_contracts_reject_inconsistent_or_non_finite_values() -> None:
    horizon = HorizonStrengthMetrics(
        horizon=5,
        stock_return=Decimal("0.1"),
        benchmark_return=Decimal("0.05"),
        benchmark_relative_return=Decimal("0.05"),
        industry_return=Decimal("0.04"),
        industry_relative_return=Decimal("0.06"),
        historical_industry_ids=(INDUSTRY_A,),
        aligned_sessions=5,
    )
    invalid_horizons = (
        lambda: replace(horizon, aligned_sessions=4),
        lambda: replace(horizon, stock_return=Decimal("NaN")),
        lambda: replace(horizon, industry_relative_return=None),
        lambda: replace(horizon, industry_return=Decimal("Infinity")),
        lambda: replace(horizon, historical_industry_ids=()),
    )
    for factory in invalid_horizons:
        with pytest.raises(ValueError):
            factory()

    trend = TrendQualityMetrics(
        latest_close=Decimal(10),
        moving_average=Decimal(9),
        position_vs_average=Decimal("0.1"),
        moving_average_slope=Decimal("0.01"),
        breakout_level=Decimal(9),
        breakout_hold_ratio=Decimal(1),
        pullback_depth=Decimal(0),
        distance_to_high=Decimal(0),
        score=Decimal(50),
    )
    invalid_trends = (
        lambda: replace(trend, score=Decimal("NaN")),
        lambda: replace(trend, latest_close=Decimal(0)),
        lambda: replace(trend, breakout_hold_ratio=Decimal("1.1")),
        lambda: replace(trend, pullback_depth=Decimal("-0.1")),
        lambda: replace(trend, score=Decimal(101)),
    )
    for factory in invalid_trends:
        with pytest.raises(ValueError):
            factory()

    volume = VolumePriceConfirmation(
        current_return=Decimal("0.01"),
        current_volume=Decimal(100),
        average_prior_volume=Decimal(90),
        volume_ratio=Decimal(100) / 90,
        score=Decimal(10),
        baseline_observations=20,
    )
    invalid_volumes = (
        lambda: replace(volume, score=Decimal("NaN")),
        lambda: replace(volume, current_volume=Decimal(-1)),
        lambda: replace(volume, average_prior_volume=Decimal(0)),
        lambda: replace(volume, score=Decimal(101)),
    )
    for factory in invalid_volumes:
        with pytest.raises(ValueError):
            factory()

    contribution = ScoreContribution(
        component="trend",
        raw_value=Decimal(10),
        normalized_score=Decimal(10),
        configured_weight=Decimal("0.5"),
        effective_weight=Decimal("0.5"),
        contribution=Decimal(5),
    )
    invalid_contributions = (
        lambda: replace(contribution, raw_value=Decimal("NaN")),
        lambda: replace(contribution, normalized_score=Decimal(101)),
        lambda: replace(contribution, effective_weight=Decimal("1.1")),
        lambda: replace(contribution, contribution=Decimal(4)),
    )
    for factory in invalid_contributions:
        with pytest.raises(ValueError):
            factory()


def test_snapshot_contract_rejects_inconsistent_status_and_identity() -> None:
    snapshot = _analyze(*_sample())
    invalid_snapshots = (
        lambda: replace(snapshot, data_version=" "),
        lambda: replace(snapshot, industry_level=4),
        lambda: replace(snapshot, current_industry_id=" "),
        lambda: replace(snapshot, input_hash="bad"),
        lambda: replace(snapshot, selected_dates=(snapshot.selected_dates[-1],) * 2),
        lambda: replace(snapshot, horizons=()),
        lambda: replace(snapshot, score=Decimal(101)),
        lambda: replace(snapshot, contributions=snapshot.contributions[:-1]),
        lambda: replace(snapshot, score=snapshot.score + Decimal(1)),
        lambda: replace(snapshot, reason="unexpected"),
        lambda: replace(snapshot, status=StockStrengthStatus.DEGRADED, reason=None),
        lambda: replace(snapshot, status=StockStrengthStatus.SUSPENDED),
        lambda: replace(
            snapshot,
            status=StockStrengthStatus.SUSPENDED,
            score=None,
            contributions=(),
            reason=None,
        ),
    )
    for factory in invalid_snapshots:
        with pytest.raises(ValueError):
            factory()
