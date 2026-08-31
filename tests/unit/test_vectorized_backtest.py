"""Tests for the deterministic point-in-time vectorized research engine."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.backtest.contracts import Side, TradableInstrumentType
from quant_agent.backtest.rules import FeeRule, FeeRuleBook
from quant_agent.backtest.vectorized import (
    BacktestMetrics,
    BenchmarkBar,
    DailyBacktestPoint,
    FactorSignal,
    PortfolioWeight,
    ResearchPriceBar,
    VectorizedBacktestEngine,
    VectorizedBacktestError,
    VectorizedBacktestRequest,
    VectorTrade,
    WeightSignal,
    build_equal_weight_layer,
)

TZ = ZoneInfo("Asia/Shanghai")
START = date(2026, 8, 3)
CALENDAR = (START, START + timedelta(days=1), START + timedelta(days=2))


def _at(day: date, hour: int = 16) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=TZ)


def _request(
    *,
    calendar: tuple[date, ...] = CALENDAR,
    as_of: datetime | None = None,
    initial_nav: str = "1000",
) -> VectorizedBacktestRequest:
    return VectorizedBacktestRequest(
        run_id="run-vector-1",
        data_version="snapshot-v1",
        strategy_version="strategy-v1",
        as_of=as_of or _at(calendar[-1], 18),
        trading_calendar=calendar,
        initial_nav=Decimal(initial_nav),
    )


def _signal(
    instrument_id: str,
    signal_day: date,
    weight: str,
    *,
    instrument_type: TradableInstrumentType = TradableInstrumentType.STOCK,
    data_version: str = "snapshot-v1",
    strategy_version: str = "strategy-v1",
) -> WeightSignal:
    return WeightSignal(
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        signal_date=signal_day,
        as_of=_at(signal_day, 15),
        target_weight=Decimal(weight),
        data_version=data_version,
        strategy_version=strategy_version,
    )


def _bar(
    instrument_id: str,
    trading_day: date,
    open_price: str,
    close_price: str,
    *,
    available_at: datetime | None = None,
    data_version: str = "snapshot-v1",
    revision: str = "1",
) -> ResearchPriceBar:
    return ResearchPriceBar(
        instrument_id=instrument_id,
        instrument_type=TradableInstrumentType.STOCK,
        trading_day=trading_day,
        open_price=Decimal(open_price),
        close_price=Decimal(close_price),
        available_at=available_at or _at(trading_day),
        data_version=data_version,
        revision=revision,
    )


def _benchmark(
    calendar: tuple[date, ...] = CALENDAR,
    closes: tuple[str, ...] = ("100", "100", "100"),
    *,
    data_version: str = "snapshot-v1",
) -> tuple[BenchmarkBar, ...]:
    return tuple(
        BenchmarkBar(
            benchmark_id="CN.SH.000300",
            trading_day=trading_day,
            close_price=Decimal(close),
            available_at=_at(trading_day),
            data_version=data_version,
        )
        for trading_day, close in zip(calendar, closes, strict=True)
    )


def test_signal_executes_only_at_next_session_open_and_manual_nav_matches() -> None:
    result = VectorizedBacktestEngine().run(
        request=_request(),
        signals=[_signal("A", CALENDAR[0], "0.5")],
        prices=[
            # This extreme same-day close is intentionally unavailable at signal time.
            _bar("A", CALENDAR[0], "10", "100"),
            _bar("A", CALENDAR[1], "10", "11"),
            _bar("A", CALENDAR[2], "11", "12"),
        ],
        benchmark=_benchmark(closes=("100", "101", "102")),
    )

    assert [point.nav for point in result.points] == [
        Decimal("1000"),
        Decimal("1050.0"),
        Decimal("1100.0"),
    ]
    assert len(result.trades) == 1
    assert result.trades[0].signal_date == CALENDAR[0]
    assert result.trades[0].execution_date == CALENDAR[1]
    assert result.trades[0].execution_price == 10
    assert result.points[0].executed_signal_date is None
    assert result.points[1].executed_signal_date == CALENDAR[0]
    assert result.points[1].weights[0].weight == Decimal("550") / Decimal("1050")
    assert result.points[2].benchmark_nav == Decimal("1020")


def test_rebalance_books_buy_and_sell_fees_and_conserves_nav() -> None:
    fee_book = FeeRuleBook(
        [
            FeeRule(
                instrument_type=TradableInstrumentType.STOCK,
                side=Side.BUY,
                effective_from=CALENDAR[0],
                version="buy-v1",
                commission_rate="0.001",
            ),
            FeeRule(
                instrument_type=TradableInstrumentType.STOCK,
                side=Side.SELL,
                effective_from=CALENDAR[0],
                version="sell-v1",
                commission_rate="0.001",
                stamp_duty_rate="0.002",
            ),
        ]
    )
    result = VectorizedBacktestEngine(fee_rule_book=fee_book).run(
        request=_request(),
        signals=[
            _signal("A", CALENDAR[0], "1"),
            _signal("B", CALENDAR[1], "1"),
        ],
        prices=[
            _bar("A", CALENDAR[1], "10", "10"),
            _bar("A", CALENDAR[2], "10", "10"),
            _bar("B", CALENDAR[2], "1", "1"),
        ],
        benchmark=_benchmark(),
    )

    first_buy, sell, second_buy = result.trades
    assert (first_buy.side, first_buy.total_fee, first_buy.fee_rule_version) == (
        Side.BUY,
        Decimal("1.00"),
        "buy-v1",
    )
    assert (sell.side, sell.total_fee, sell.fee_rule_version) == (
        Side.SELL,
        Decimal("3.00"),
        "sell-v1",
    )
    assert second_buy.side is Side.BUY
    assert second_buy.total_fee == Decimal("1.00")
    assert result.points[1].nav == Decimal("999.00")
    assert result.points[1].cash >= 0
    assert result.points[2].buy_fees == Decimal("1.00")
    assert result.points[2].sell_fees == Decimal("3.00")
    assert result.points[2].nav == Decimal("995.00")
    assert result.points[2].cash >= 0
    assert result.points[2].nav == Decimal("999") - result.points[2].total_fees
    assert result.metrics.total_fees == Decimal("5.00")


def test_full_weight_with_minimum_fee_scales_buys_without_implicit_leverage() -> None:
    fee_book = FeeRuleBook(
        [
            FeeRule(
                instrument_type=TradableInstrumentType.STOCK,
                side=Side.BUY,
                effective_from=CALENDAR[0],
                version="minimum-v1",
                minimum_commission="5",
            )
        ]
    )

    result = VectorizedBacktestEngine(fee_rule_book=fee_book).run(
        request=_request(),
        signals=[_signal("A", CALENDAR[0], "1")],
        prices=[
            _bar("A", CALENDAR[1], "10", "10"),
            _bar("A", CALENDAR[2], "10", "10"),
        ],
        benchmark=_benchmark(),
    )

    assert result.trades[0].notional == Decimal("995")
    assert result.trades[0].total_fee == Decimal("5")
    assert result.points[1].cash == 0
    assert result.points[1].nav == Decimal("995")
    assert result.points[1].cash_weight == 0


def test_buy_fails_closed_when_minimum_fee_exceeds_available_cash() -> None:
    fee_book = FeeRuleBook(
        [
            FeeRule(
                instrument_type=TradableInstrumentType.STOCK,
                side=Side.BUY,
                effective_from=CALENDAR[0],
                version="unaffordable-v1",
                minimum_commission="5",
            )
        ]
    )

    with pytest.raises(VectorizedBacktestError, match="cannot cover the minimum buy fee"):
        VectorizedBacktestEngine(fee_rule_book=fee_book).run(
            request=_request(initial_nav="1"),
            signals=[_signal("A", CALENDAR[0], "1")],
            prices=[
                _bar("A", CALENDAR[1], "1", "1"),
                _bar("A", CALENDAR[2], "1", "1"),
            ],
            benchmark=_benchmark(),
        )


def test_daily_point_contract_rejects_negative_research_cash() -> None:
    with pytest.raises(ValueError, match="cash and cash_weight"):
        DailyBacktestPoint(
            trading_day=CALENDAR[0],
            nav=Decimal("1"),
            benchmark_nav=Decimal("1"),
            daily_return=Decimal(0),
            benchmark_return=Decimal(0),
            excess_return=Decimal(0),
            drawdown=Decimal(0),
            turnover=Decimal(0),
            buy_fees=Decimal(0),
            sell_fees=Decimal(0),
            total_fees=Decimal(0),
            cash=Decimal("-0.01"),
            cash_weight=Decimal("-0.01"),
            weights=(),
            executed_signal_date=None,
        )


def test_target_weights_are_long_only_and_cannot_exceed_one() -> None:
    with pytest.raises(ValueError, match="shorting is unsupported"):
        _signal("A", CALENDAR[0], "-0.01")

    with pytest.raises(VectorizedBacktestError, match="exceed one"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[
                _signal("A", CALENDAR[0], "0.6"),
                _signal("B", CALENDAR[0], "0.5"),
            ],
            prices=[],
            benchmark=_benchmark(),
        )


def test_empty_signal_run_is_cash_only_and_zero_volatility_sharpe_is_none() -> None:
    result = VectorizedBacktestEngine().run(
        request=_request(),
        signals=[],
        prices=[],
        benchmark=_benchmark(),
    )

    assert result.trades == ()
    assert all(point.nav == Decimal("1000") for point in result.points)
    assert all(point.cash_weight == 1 for point in result.points)
    assert result.metrics.total_return == 0
    assert result.metrics.annualized_return == 0
    assert result.metrics.annualized_volatility == 0
    assert result.metrics.sharpe_ratio is None
    assert result.metrics.total_turnover == 0
    assert result.metrics.max_drawdown == 0


def test_metrics_include_drawdown_volatility_sharpe_and_turnover() -> None:
    calendar = (*CALENDAR, START + timedelta(days=3))
    result = VectorizedBacktestEngine().run(
        request=_request(calendar=calendar),
        signals=[_signal("A", calendar[0], "1")],
        prices=[
            _bar("A", calendar[1], "100", "110"),
            _bar("A", calendar[2], "99", "99"),
            _bar("A", calendar[3], "108.9", "108.9"),
        ],
        benchmark=_benchmark(calendar, ("100", "100", "100", "100")),
    )

    assert [point.daily_return for point in result.points[1:]] == [
        Decimal("0.1"),
        Decimal("-0.1"),
        Decimal("0.1"),
    ]
    assert result.metrics.total_return == Decimal("0.089")
    assert result.metrics.max_drawdown == Decimal("0.1")
    assert result.metrics.annualized_volatility > 0
    assert result.metrics.annualized_return > 0
    assert result.metrics.sharpe_ratio is not None
    assert result.metrics.total_turnover == 1
    assert result.metrics.average_daily_turnover == Decimal("0.25")


def test_benchmark_requires_exact_calendar_alignment_without_future_fill() -> None:
    misaligned = (
        _benchmark()[0],
        _benchmark()[2],
        BenchmarkBar(
            benchmark_id="CN.SH.000300",
            trading_day=CALENDAR[-1] + timedelta(days=1),
            close_price=Decimal("100"),
            available_at=_at(CALENDAR[-1] + timedelta(days=1)),
            data_version="snapshot-v1",
        ),
    )

    with pytest.raises(VectorizedBacktestError, match="align exactly"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[],
            prices=[],
            benchmark=misaligned,
        )


def test_future_prices_and_input_order_cannot_change_frozen_result() -> None:
    request = _request()
    signals = [
        _signal("B", CALENDAR[0], "0.4"),
        _signal("A", CALENDAR[0], "0.6"),
    ]
    prices = [
        _bar("A", CALENDAR[1], "10", "11"),
        _bar("B", CALENDAR[1], "20", "18"),
        _bar("A", CALENDAR[2], "11", "12"),
        _bar("B", CALENDAR[2], "18", "19"),
    ]
    engine = VectorizedBacktestEngine()
    baseline = engine.run(
        request=request,
        signals=signals,
        prices=prices,
        benchmark=_benchmark(),
    )
    future_revision = _bar(
        "A",
        CALENDAR[1],
        "999",
        "999",
        available_at=_at(CALENDAR[-1] + timedelta(days=1)),
        revision="future",
    )
    future_day = _bar(
        "A",
        CALENDAR[-1] + timedelta(days=1),
        "999",
        "999",
    )
    repeated = engine.run(
        request=request,
        signals=list(reversed(signals)),
        prices=[future_revision, future_day, *reversed(prices)],
        benchmark=list(reversed(_benchmark())),
    )

    assert repeated == baseline
    assert repeated.input_hash == baseline.input_hash
    assert repeated.result_hash == baseline.result_hash
    assert repeated.trades == baseline.trades


@pytest.mark.parametrize(
    ("signals", "prices", "benchmark", "message"),
    [
        (
            [_signal("A", CALENDAR[0], "1", data_version="wrong")],
            [],
            _benchmark(),
            "signal data_version",
        ),
        (
            [_signal("A", CALENDAR[0], "1", strategy_version="wrong")],
            [],
            _benchmark(),
            "strategy_version",
        ),
        (
            [],
            [],
            _benchmark(data_version="wrong"),
            "benchmark data_version",
        ),
        (
            [_signal("A", CALENDAR[-1], "1")],
            [],
            _benchmark(),
            "no next session",
        ),
    ],
)
def test_version_and_date_mismatches_fail_closed(
    signals: list[WeightSignal],
    prices: list[ResearchPriceBar],
    benchmark: tuple[BenchmarkBar, ...],
    message: str,
) -> None:
    with pytest.raises(VectorizedBacktestError, match=message):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=signals,
            prices=prices,
            benchmark=benchmark,
        )


def test_missing_held_price_and_ambiguous_known_revision_fail_closed() -> None:
    with pytest.raises(VectorizedBacktestError, match="missing price"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[_signal("A", CALENDAR[0], "1")],
            prices=[_bar("A", CALENDAR[1], "10", "10")],
            benchmark=_benchmark(),
        )

    first = _bar("A", CALENDAR[1], "10", "10")
    conflict = _bar(
        "A",
        CALENDAR[1],
        "10",
        "11",
        available_at=first.available_at,
        revision="2",
    )
    with pytest.raises(VectorizedBacktestError, match="ambiguous price revisions"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[],
            prices=[first, conflict],
            benchmark=_benchmark(),
        )


def test_factor_layers_are_equal_weighted_and_ties_use_identifier_order() -> None:
    items = tuple(
        FactorSignal(
            instrument_id=instrument_id,
            instrument_type=TradableInstrumentType.ETF,
            signal_date=CALENDAR[0],
            as_of=_at(CALENDAR[0], 15),
            score=Decimal(score),
            data_version="snapshot-v1",
            strategy_version="strategy-v1",
        )
        for instrument_id, score in (("B", "2"), ("D", "0"), ("A", "2"), ("C", "1"))
    )

    top = build_equal_weight_layer(items, layer=0, layer_count=2)
    repeated = build_equal_weight_layer(reversed(items), layer=0, layer_count=2)
    bottom = build_equal_weight_layer(items, layer=1, layer_count=2)

    assert top == repeated
    assert [(item.instrument_id, item.target_weight) for item in top] == [
        ("A", Decimal("0.5")),
        ("B", Decimal("0.5")),
    ]
    assert [item.instrument_id for item in bottom] == ["C", "D"]
    assert all(item.instrument_type is TradableInstrumentType.ETF for item in top)


def test_contracts_are_frozen_and_reject_naive_or_non_finite_values() -> None:
    signal = _signal("A", CALENDAR[0], "1")
    with pytest.raises(FrozenInstanceError):
        signal.target_weight = Decimal("0")  # type: ignore[misc]
    with pytest.raises(ValueError, match="timezone"):
        WeightSignal(
            instrument_id="A",
            instrument_type=TradableInstrumentType.STOCK,
            signal_date=CALENDAR[0],
            as_of=datetime.combine(CALENDAR[0], time(15)),
            target_weight=Decimal("1"),
            data_version="snapshot-v1",
            strategy_version="strategy-v1",
        )
    with pytest.raises(ValueError, match="finite"):
        _signal("A", CALENDAR[0], "NaN")
    with pytest.raises(ValueError, match="finite"):
        _bar("A", CALENDAR[0], "Infinity", "1")
    with pytest.raises(ValueError, match="must be finite"):
        BacktestMetrics(
            total_return=Decimal("NaN"),
            benchmark_total_return=Decimal(0),
            excess_return=Decimal(0),
            annualized_return=Decimal(0),
            annualized_volatility=Decimal(0),
            sharpe_ratio=None,
            max_drawdown=Decimal(0),
            total_turnover=Decimal(0),
            average_daily_turnover=Decimal(0),
            total_fees=Decimal(0),
            observation_count=1,
        )


def test_hashes_are_sha256_and_change_with_economic_inputs() -> None:
    engine = VectorizedBacktestEngine()
    base = engine.run(
        request=_request(),
        signals=[_signal("A", CALENDAR[0], "1")],
        prices=[
            _bar("A", CALENDAR[1], "10", "10"),
            _bar("A", CALENDAR[2], "10", "10"),
        ],
        benchmark=_benchmark(),
    )
    changed = engine.run(
        request=_request(),
        signals=[_signal("A", CALENDAR[0], "0.5")],
        prices=[
            _bar("A", CALENDAR[1], "10", "10"),
            _bar("A", CALENDAR[2], "10", "10"),
        ],
        benchmark=_benchmark(),
    )

    assert len(base.input_hash) == len(base.result_hash) == 64
    assert base.input_hash != changed.input_hash
    assert base.result_hash != changed.result_hash


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trading_calendar": (CALENDAR[1], CALENDAR[0])}, "unique ascending"),
        ({"annualization_periods": 0}, "positive"),
        ({"initial_nav": Decimal(0)}, "initial_nav"),
        ({"annual_risk_free_rate": Decimal("NaN")}, "finite"),
    ],
)
def test_request_rejects_unsafe_boundaries(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "run_id": "run",
        "data_version": "snapshot-v1",
        "strategy_version": "strategy-v1",
        "as_of": _at(CALENDAR[-1], 18),
        "trading_calendar": CALENDAR,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        VectorizedBacktestRequest(**values)  # type: ignore[arg-type]


def test_layer_builder_rejects_ambiguous_and_impossible_batches() -> None:
    item = FactorSignal(
        instrument_id="A",
        instrument_type=TradableInstrumentType.STOCK,
        signal_date=CALENDAR[0],
        as_of=_at(CALENDAR[0], 15),
        score=Decimal("1"),
        data_version="snapshot-v1",
        strategy_version="strategy-v1",
    )
    with pytest.raises(VectorizedBacktestError, match="duplicate"):
        build_equal_weight_layer([item, item], layer=0, layer_count=1)
    with pytest.raises(VectorizedBacktestError, match="fewer instruments"):
        build_equal_weight_layer([item], layer=0, layer_count=2)
    with pytest.raises(ValueError, match="layer must"):
        build_equal_weight_layer([item], layer=2, layer_count=2)


def test_latest_price_and_benchmark_revisions_known_at_cutoff_are_selected() -> None:
    earlier = _at(CALENDAR[1], 16)
    later = _at(CALENDAR[1], 17)
    result = VectorizedBacktestEngine().run(
        request=_request(),
        signals=[_signal("A", CALENDAR[0], "1")],
        prices=[
            _bar("A", CALENDAR[1], "10", "10", available_at=earlier, revision="old"),
            _bar("A", CALENDAR[1], "20", "22", available_at=later, revision="new"),
            _bar("A", CALENDAR[2], "22", "22"),
        ],
        benchmark=[
            *_benchmark()[:1],
            BenchmarkBar(
                benchmark_id="CN.SH.000300",
                trading_day=CALENDAR[1],
                close_price=Decimal("90"),
                available_at=earlier,
                data_version="snapshot-v1",
                revision="old",
            ),
            BenchmarkBar(
                benchmark_id="CN.SH.000300",
                trading_day=CALENDAR[1],
                close_price=Decimal("100"),
                available_at=later,
                data_version="snapshot-v1",
                revision="new",
            ),
            _benchmark()[2],
        ],
    )

    assert result.trades[0].execution_price == 20
    assert result.points[1].nav == Decimal("1100")
    assert result.points[1].benchmark_return == 0


def test_ambiguous_benchmark_revision_fails_closed() -> None:
    original = _benchmark()[1]
    conflict = BenchmarkBar(
        benchmark_id=original.benchmark_id,
        trading_day=original.trading_day,
        close_price=Decimal("101"),
        available_at=original.available_at,
        data_version=original.data_version,
        revision="conflict",
    )
    with pytest.raises(VectorizedBacktestError, match="ambiguous benchmark revisions"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[],
            prices=[],
            benchmark=[_benchmark()[0], original, conflict, _benchmark()[2]],
        )


def test_duplicate_or_internally_misaligned_signal_batches_fail_closed() -> None:
    duplicate = _signal("A", CALENDAR[0], "0.5")
    with pytest.raises(VectorizedBacktestError, match="duplicate weight signal"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[duplicate, duplicate],
            prices=[],
            benchmark=_benchmark(),
        )

    different_time = WeightSignal(
        instrument_id="B",
        instrument_type=TradableInstrumentType.STOCK,
        signal_date=CALENDAR[0],
        as_of=_at(CALENDAR[0], 14),
        target_weight=Decimal("0.5"),
        data_version="snapshot-v1",
        strategy_version="strategy-v1",
    )
    with pytest.raises(VectorizedBacktestError, match="one as_of"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[duplicate, different_time],
            prices=[],
            benchmark=_benchmark(),
        )

    weekend = START + timedelta(days=6)
    with pytest.raises(VectorizedBacktestError, match="absent from the trading calendar"):
        VectorizedBacktestEngine().run(
            request=_request(as_of=_at(weekend, 18)),
            signals=[_signal("A", weekend, "1")],
            prices=[],
            benchmark=_benchmark(),
        )


def test_price_versions_and_instrument_types_fail_closed() -> None:
    with pytest.raises(VectorizedBacktestError, match="price data_version"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[],
            prices=[_bar("A", CALENDAR[1], "1", "1", data_version="wrong")],
            benchmark=_benchmark(),
        )

    changed_type = ResearchPriceBar(
        instrument_id="A",
        instrument_type=TradableInstrumentType.ETF,
        trading_day=CALENDAR[2],
        open_price=Decimal("1"),
        close_price=Decimal("1"),
        available_at=_at(CALENDAR[2]),
        data_version="snapshot-v1",
    )
    with pytest.raises(VectorizedBacktestError, match="type changes"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[],
            prices=[_bar("A", CALENDAR[1], "1", "1"), changed_type],
            benchmark=_benchmark(),
        )

    etf_signal = _signal(
        "A",
        CALENDAR[0],
        "1",
        instrument_type=TradableInstrumentType.ETF,
    )
    with pytest.raises(VectorizedBacktestError, match="signal and price"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[etf_signal],
            prices=[_bar("A", CALENDAR[1], "1", "1")],
            benchmark=_benchmark(),
        )


def test_benchmark_identifier_mixing_is_rejected() -> None:
    mixed = list(_benchmark())
    mixed[-1] = BenchmarkBar(
        benchmark_id="OTHER",
        trading_day=CALENDAR[-1],
        close_price=Decimal("100"),
        available_at=_at(CALENDAR[-1]),
        data_version="snapshot-v1",
    )
    with pytest.raises(VectorizedBacktestError, match="exactly one identifier"):
        VectorizedBacktestEngine().run(
            request=_request(),
            signals=[],
            prices=[],
            benchmark=mixed,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _request(calendar=(), as_of=_at(CALENDAR[-1])),
            "must not be empty",
        ),
        (
            lambda: VectorizedBacktestRequest(
                run_id="run",
                data_version="v1",
                strategy_version="s1",
                as_of=_at(CALENDAR[0]),
                trading_calendar=CALENDAR,
            ),
            "extend beyond",
        ),
        (
            lambda: VectorizedBacktestRequest(
                run_id="run",
                data_version="v1",
                strategy_version="s1",
                as_of=_at(CALENDAR[-1]),
                trading_calendar=CALENDAR,
                annual_risk_free_rate=Decimal("-1"),
            ),
            "greater than -1",
        ),
        (
            lambda: _bar(
                "A",
                CALENDAR[1],
                "1",
                "1",
                available_at=_at(CALENDAR[0]),
            ),
            "cannot precede",
        ),
        (
            lambda: BenchmarkBar(
                benchmark_id="B",
                trading_day=CALENDAR[1],
                close_price=Decimal("0"),
                available_at=_at(CALENDAR[1]),
                data_version="v1",
            ),
            "must be positive",
        ),
        (
            lambda: PortfolioWeight(instrument_id="", weight=Decimal(0)),
            "non-empty",
        ),
    ],
)
def test_additional_contract_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_result_value_contracts_reject_impossible_accounting_states() -> None:
    with pytest.raises(ValueError, match="must follow"):
        VectorTrade(
            signal_date=CALENDAR[1],
            execution_date=CALENDAR[1],
            instrument_id="A",
            instrument_type=TradableInstrumentType.STOCK,
            side=Side.BUY,
            execution_price=Decimal("1"),
            notional=Decimal("1"),
            commission=Decimal(0),
            stamp_duty=Decimal(0),
            transfer_fee=Decimal(0),
            other_fee=Decimal(0),
            total_fee=Decimal(0),
            fee_rule_version=None,
        )
    with pytest.raises(ValueError, match="zero-volatility"):
        BacktestMetrics(
            total_return=Decimal(0),
            benchmark_total_return=Decimal(0),
            excess_return=Decimal(0),
            annualized_return=Decimal(0),
            annualized_volatility=Decimal(0),
            sharpe_ratio=Decimal(1),
            max_drawdown=Decimal(0),
            total_turnover=Decimal(0),
            average_daily_turnover=Decimal(0),
            total_fees=Decimal(0),
            observation_count=1,
        )
    with pytest.raises(ValueError, match="daily total_fees"):
        DailyBacktestPoint(
            trading_day=CALENDAR[0],
            nav=Decimal(1),
            benchmark_nav=Decimal(1),
            daily_return=Decimal(0),
            benchmark_return=Decimal(0),
            excess_return=Decimal(0),
            drawdown=Decimal(0),
            turnover=Decimal(0),
            buy_fees=Decimal(1),
            sell_fees=Decimal(0),
            total_fees=Decimal(0),
            cash=Decimal(1),
            cash_weight=Decimal(1),
            weights=(),
            executed_signal_date=None,
        )


def test_layer_builder_rejects_metadata_mismatch_and_bad_gross_weight() -> None:
    first = FactorSignal(
        instrument_id="A",
        instrument_type=TradableInstrumentType.STOCK,
        signal_date=CALENDAR[0],
        as_of=_at(CALENDAR[0], 15),
        score=Decimal("1"),
        data_version="snapshot-v1",
        strategy_version="strategy-v1",
    )
    second = FactorSignal(
        instrument_id="B",
        instrument_type=TradableInstrumentType.STOCK,
        signal_date=CALENDAR[0],
        as_of=_at(CALENDAR[0], 14),
        score=Decimal("0"),
        data_version="snapshot-v1",
        strategy_version="strategy-v1",
    )
    with pytest.raises(VectorizedBacktestError, match="share as_of and versions"):
        build_equal_weight_layer([first, second], layer=0, layer_count=1)
    with pytest.raises(ValueError, match="layer_count"):
        build_equal_weight_layer([first], layer=0, layer_count=0)
    with pytest.raises(ValueError, match="gross_weight"):
        build_equal_weight_layer([first], layer=0, layer_count=1, gross_weight=Decimal("1.1"))
