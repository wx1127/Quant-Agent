"""Tests for point-in-time liquidity and practical tradeability features."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.domain import InstrumentStatus, InstrumentType
from quant_agent.features.tradeability import (
    CapacityEstimate,
    InstrumentTradeabilityRule,
    InstrumentTradeabilityState,
    InsufficientTradeabilityData,
    MarketTradeState,
    TradeabilityAnalyzer,
    TradeabilityBar,
    TradeabilityCalendarSession,
    TradeabilityConfig,
    TradeabilityContributionCode,
    TradeabilityEligibility,
    TradeabilityInputError,
    TradeabilityReason,
    TradeabilityReasonCode,
    TradeabilityRequest,
    TradeSide,
)

TZ = ZoneInfo("Asia/Shanghai")
DATES = (date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5))
SESSION = DATES[-1]


def _at(value: date, hour: int) -> datetime:
    return datetime.combine(value, time(hour), tzinfo=TZ)


def _config(**overrides: object) -> TradeabilityConfig:
    values: dict[str, object] = {
        "version": "tradeability-test-v1",
        "short_turnover_window": 2,
        "long_turnover_window": 3,
        "minimum_average_turnover_short": Decimal("100"),
        "minimum_average_turnover_long": Decimal("100"),
        "turnover_rate_percentile_window": 3,
        "minimum_turnover_rate_observations": 3,
        "minimum_turnover_rate_percentile": Decimal("0.5"),
        "minimum_listing_days": 5,
        "participation_rate": Decimal("0.1"),
        "minimum_capacity_ratio": Decimal("1"),
    }
    values.update(overrides)
    return TradeabilityConfig(**values)  # type: ignore[arg-type]


def _request(
    *,
    account_value: str = "100",
    target_weight: str = "0.1",
    side: TradeSide = TradeSide.BUY,
    instrument_type: InstrumentType = InstrumentType.STOCK,
    as_of: datetime | None = None,
) -> TradeabilityRequest:
    return TradeabilityRequest(
        instrument_id="CN.SH.TEST",
        instrument_type=instrument_type,
        market="SSE",
        session_date=SESSION,
        as_of=as_of or _at(SESSION, 18),
        data_version="snapshot-v1",
        side=side,
        account_value=Decimal(account_value),
        target_position_weight=Decimal(target_weight),
    )


def _master(
    *,
    instrument_type: InstrumentType = InstrumentType.STOCK,
    listed_on: date = date(2020, 1, 1),
    status: InstrumentStatus = InstrumentStatus.LISTED,
    available_at: datetime | None = None,
    revision: str = "m1",
    data_version: str = "snapshot-v1",
) -> InstrumentTradeabilityState:
    return InstrumentTradeabilityState(
        instrument_id="CN.SH.TEST",
        instrument_type=instrument_type,
        listed_on=listed_on,
        delisted_on=None,
        status=status,
        effective_from=listed_on,
        effective_to=None,
        available_at=available_at or _at(SESSION, 9),
        revision=revision,
        data_version=data_version,
    )


def _calendar(
    *,
    data_version: str = "snapshot-v1",
) -> tuple[TradeabilityCalendarSession, ...]:
    return tuple(
        TradeabilityCalendarSession(
            market="SSE",
            trade_date=value,
            is_open=True,
            available_at=_at(value, 8),
            revision="c1",
            data_version=data_version,
        )
        for value in DATES
    )


def _bar(
    day_index: int,
    *,
    previous_close: str = "100",
    open_price: str = "100",
    high: str = "101",
    low: str = "99",
    close: str = "100",
    volume: str = "10",
    amount: str | None = None,
    turnover_rate: str | None = None,
    suspended: bool = False,
    available_at: datetime | None = None,
    revision: str = "b1",
    data_version: str = "snapshot-v1",
) -> TradeabilityBar:
    trade_date = DATES[day_index]
    default_amounts = ("100", "200", "300")
    default_rates = ("1", "2", "3")
    return TradeabilityBar(
        instrument_id="CN.SH.TEST",
        trade_date=trade_date,
        observed_at=_at(trade_date, 15),
        available_at=available_at or _at(trade_date, 16),
        previous_close=Decimal(previous_close),
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        turnover_amount=Decimal(amount if amount is not None else default_amounts[day_index]),
        turnover_rate=(
            Decimal(turnover_rate if turnover_rate is not None else default_rates[day_index])
            if turnover_rate != "MISSING"
            else None
        ),
        is_suspended=suspended,
        revision=revision,
        data_version=data_version,
    )


def _bars() -> tuple[TradeabilityBar, ...]:
    return tuple(_bar(index) for index in range(3))


def _rule(
    *,
    instrument_type: InstrumentType = InstrumentType.STOCK,
    price_limit_ratio: Decimal | None = Decimal("0.10"),
    effective_from: date = date(2020, 1, 1),
) -> InstrumentTradeabilityRule:
    return InstrumentTradeabilityRule(
        rule_id=f"rule-{instrument_type.value.lower()}",
        version="rule-v1",
        instrument_type=instrument_type,
        effective_from=effective_from,
        price_limit_ratio=price_limit_ratio,
        price_tick=Decimal("0.01"),
    )


def _analyze(
    *,
    request: TradeabilityRequest | None = None,
    config: TradeabilityConfig | None = None,
    masters: tuple[InstrumentTradeabilityState, ...] | None = None,
    calendar: tuple[TradeabilityCalendarSession, ...] | None = None,
    bars: tuple[TradeabilityBar, ...] | None = None,
    rules: tuple[InstrumentTradeabilityRule, ...] | None = None,
):  # type: ignore[no-untyped-def]
    return TradeabilityAnalyzer(config or _config()).analyze(
        request=request or _request(),
        instrument_states=masters if masters is not None else (_master(),),
        calendar=calendar if calendar is not None else _calendar(),
        bars=bars if bars is not None else _bars(),
        rules=rules if rules is not None else (_rule(),),
    )


def test_manual_liquidity_percentile_capacity_and_contributions() -> None:
    result = _analyze()

    assert result.eligibility is TradeabilityEligibility.ELIGIBLE
    assert result.eligible
    assert result.market_state is MarketTradeState.NORMAL
    assert result.average_turnover_short == Decimal("250")
    assert result.average_turnover_long == Decimal("200")
    assert result.turnover_rate_percentile == 1
    assert result.turnover_rate_observation_count == 3
    assert result.capacity.short_window_capacity == Decimal("25.0")
    assert result.capacity.long_window_capacity == Decimal("20.0")
    assert result.capacity.daily_capacity == Decimal("20.0")
    assert result.capacity.target_notional == Decimal("10.0")
    assert result.capacity.capacity_ratio == Decimal("2")
    assert result.capacity.maximum_account_weight == Decimal("0.2")
    assert result.capacity.estimated_trade_days == Decimal("0.5")
    assert result.short_window_dates == DATES[-2:]
    assert result.long_window_dates == DATES
    assert result.reasons == ()
    assert {item.code for item in result.contributions} == {
        TradeabilityContributionCode.AVERAGE_TURNOVER_SHORT,
        TradeabilityContributionCode.AVERAGE_TURNOVER_LONG,
        TradeabilityContributionCode.TURNOVER_RATE_PERCENTILE,
        TradeabilityContributionCode.LISTING_DAYS,
        TradeabilityContributionCode.CAPACITY_RATIO,
    }
    assert len(result.input_hash) == len(result.result_hash) == 64
    assert result.cache_key == result.result_hash


def test_account_scale_changes_capacity_ratio_but_not_market_capacity() -> None:
    small = _analyze(request=_request(account_value="20", target_weight="0.5"))
    large = _analyze(request=_request(account_value="1000", target_weight="0.5"))

    assert small.capacity.daily_capacity == large.capacity.daily_capacity == Decimal("20")
    assert small.capacity.capacity_ratio == Decimal("2")
    assert small.capacity.maximum_account_weight == 1
    assert small.eligible
    assert large.capacity.capacity_ratio == Decimal("0.04")
    assert large.capacity.maximum_account_weight == Decimal("0.02")
    assert not large.eligible
    assert TradeabilityReasonCode.INSUFFICIENT_CAPACITY in {reason.code for reason in large.reasons}


def test_suspension_is_blocking_and_zero_amount_remains_in_windows() -> None:
    suspended = _bar(
        2,
        open_price="100",
        high="100",
        low="100",
        close="100",
        volume="0",
        amount="0",
        turnover_rate="0",
        suspended=True,
    )
    result = _analyze(bars=(*_bars()[:2], suspended))

    assert result.market_state is MarketTradeState.SUSPENDED
    assert not result.eligible
    assert result.average_turnover_short == Decimal("100")
    assert result.average_turnover_long == Decimal("100")
    suspension = next(
        reason for reason in result.reasons if reason.code is TradeabilityReasonCode.SUSPENDED
    )
    assert suspension.blocking


def test_one_price_limit_up_is_structured_and_side_specific() -> None:
    locked = _bar(
        2,
        open_price="110",
        high="110",
        low="110",
        close="110",
    )
    buy = _analyze(bars=(*_bars()[:2], locked))
    sell = _analyze(
        request=_request(side=TradeSide.SELL),
        bars=(*_bars()[:2], locked),
    )

    assert buy.market_state is MarketTradeState.ONE_PRICE_LIMIT_UP
    assert buy.upper_limit_price == Decimal("110.00")
    assert not buy.eligible
    buy_reason = next(
        item for item in buy.reasons if item.code is TradeabilityReasonCode.ONE_PRICE_LIMIT_UP
    )
    sell_reason = next(
        item for item in sell.reasons if item.code is TradeabilityReasonCode.ONE_PRICE_LIMIT_UP
    )
    assert buy_reason.blocking
    assert not sell_reason.blocking
    assert sell.eligible


def test_regular_limit_down_is_distinct_and_blocks_only_configured_side() -> None:
    regular_down = _bar(
        2,
        open_price="95",
        high="100",
        low="90",
        close="90",
    )
    sell = _analyze(
        request=_request(side=TradeSide.SELL),
        bars=(*_bars()[:2], regular_down),
    )
    buy = _analyze(bars=(*_bars()[:2], regular_down))

    assert sell.market_state is MarketTradeState.LIMIT_DOWN
    assert sell.lower_limit_price == Decimal("90.00")
    assert not sell.eligible
    assert buy.market_state is MarketTradeState.LIMIT_DOWN
    assert buy.eligible
    assert (
        next(
            reason for reason in buy.reasons if reason.code is TradeabilityReasonCode.LIMIT_DOWN
        ).blocking
        is False
    )


def test_limit_boundary_uses_rule_tick_rounding() -> None:
    rounded_upper = _bar(
        2,
        previous_close="10.03",
        open_price="11.03",
        high="11.03",
        low="11.03",
        close="11.03",
    )
    below = _bar(
        2,
        previous_close="10.03",
        open_price="10.50",
        high="11.02",
        low="10.40",
        close="11.02",
    )

    boundary = _analyze(bars=(*_bars()[:2], rounded_upper))
    normal = _analyze(bars=(*_bars()[:2], below))

    assert boundary.upper_limit_price == Decimal("11.03")
    assert boundary.lower_limit_price == Decimal("9.03")
    assert boundary.market_state is MarketTradeState.ONE_PRICE_LIMIT_UP
    assert normal.market_state is MarketTradeState.NORMAL


def test_etf_behavior_is_driven_by_supplied_rule_without_type_branching() -> None:
    etf_request = _request(instrument_type=InstrumentType.ETF)
    large_move = _bar(
        2,
        open_price="120",
        high="125",
        low="115",
        close="120",
    )
    result = _analyze(
        request=etf_request,
        masters=(_master(instrument_type=InstrumentType.ETF),),
        bars=(*_bars()[:2], large_move),
        rules=(_rule(instrument_type=InstrumentType.ETF, price_limit_ratio=None),),
    )

    assert result.market_state is MarketTradeState.NORMAL
    assert result.upper_limit_price is None
    assert result.lower_limit_price is None
    assert result.eligible
    assert result.rule_id == "rule-etf"


def test_recent_listing_is_ineligible_with_exact_calendar_age() -> None:
    result = _analyze(
        masters=(_master(listed_on=SESSION),),
        bars=(_bar(2),),
        config=_config(minimum_turnover_rate_percentile=None),
    )

    assert result.listing_days == 1
    assert result.average_turnover_short == Decimal("150")
    assert result.average_turnover_long == Decimal("100")
    assert not result.eligible
    reason = next(
        item for item in result.reasons if item.code is TradeabilityReasonCode.NEWLY_LISTED
    )
    assert reason.value == Decimal(1)
    assert reason.threshold == Decimal(5)


def test_turnover_percentile_boundary_and_optional_mode() -> None:
    bars = (
        _bar(0, turnover_rate="1"),
        _bar(1, turnover_rate="3"),
        _bar(2, turnover_rate="2"),
    )
    exact = _analyze(
        bars=bars,
        config=_config(minimum_turnover_rate_percentile=Decimal(2) / Decimal(3)),
    )
    failed = _analyze(
        bars=bars,
        config=_config(minimum_turnover_rate_percentile=Decimal("0.67")),
    )
    optional = _analyze(
        bars=bars,
        config=_config(minimum_turnover_rate_percentile=None),
    )

    assert exact.turnover_rate_percentile == Decimal(2) / Decimal(3)
    assert exact.eligible
    assert not failed.eligible
    optional_contribution = next(
        item
        for item in optional.contributions
        if item.code is TradeabilityContributionCode.TURNOVER_RATE_PERCENTILE
    )
    assert optional_contribution.threshold is None
    assert optional_contribution.passed is None
    assert not optional_contribution.blocking


def test_future_rows_and_input_order_cannot_change_frozen_result() -> None:
    baseline = _analyze()
    tomorrow = SESSION + timedelta(days=1)
    future_bar = TradeabilityBar(
        instrument_id="CN.SH.TEST",
        trade_date=tomorrow,
        observed_at=_at(tomorrow, 15),
        available_at=_at(tomorrow, 16),
        previous_close=Decimal("1"),
        open=Decimal("999"),
        high=Decimal("999"),
        low=Decimal("999"),
        close=Decimal("999"),
        volume=Decimal("999"),
        turnover_amount=Decimal("999999"),
        turnover_rate=Decimal("999"),
        is_suspended=False,
        revision="future-day",
        data_version="snapshot-v1",
    )
    future_revision = _bar(
        2,
        open_price="110",
        high="110",
        low="110",
        close="110",
        available_at=_at(tomorrow, 16),
        revision="future-revision",
    )
    future_master = _master(
        status=InstrumentStatus.SUSPENDED,
        available_at=_at(tomorrow, 9),
        revision="future-master",
    )
    future_calendar = TradeabilityCalendarSession(
        market="SSE",
        trade_date=SESSION,
        is_open=False,
        available_at=_at(tomorrow, 8),
        revision="future-calendar",
        data_version="snapshot-v1",
    )
    repeated = _analyze(
        masters=(future_master, _master()),
        calendar=(future_calendar, *reversed(_calendar())),
        bars=(future_bar, future_revision, *reversed(_bars())),
        rules=(
            _rule(effective_from=tomorrow),
            _rule(),
        ),
    )

    assert repeated == baseline
    assert repeated.input_hash == baseline.input_hash
    assert repeated.result_hash == baseline.result_hash


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"masters": ()}, "master/status"),
        ({"calendar": _calendar()[1:]}, "open calendar sessions"),
        ({"bars": _bars()[:2]}, "window is incomplete"),
        ({"rules": ()}, "no instrument tradeability rule"),
    ],
)
def test_missing_authoritative_inputs_fail_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(InsufficientTradeabilityData, match=message):
        _analyze(**kwargs)  # type: ignore[arg-type]


def test_configured_turnover_percentile_fails_if_optional_data_is_incomplete() -> None:
    incomplete = (_bar(0, turnover_rate="MISSING"), _bar(1), _bar(2))
    with pytest.raises(InsufficientTradeabilityData, match="turnover-rate percentile"):
        _analyze(bars=incomplete)

    optional = _analyze(
        bars=incomplete,
        config=_config(minimum_turnover_rate_percentile=None),
    )
    assert optional.turnover_rate_percentile is None
    assert optional.turnover_rate_observation_count == 2


def test_ambiguous_revisions_and_mismatched_versions_fail_closed() -> None:
    current = _bar(2)
    conflict = _bar(
        2,
        close="100.5",
        high="101",
        available_at=current.available_at,
        revision="conflict",
    )
    with pytest.raises(TradeabilityInputError, match="ambiguous bar revisions"):
        _analyze(bars=(*_bars()[:2], current, conflict))

    with pytest.raises(TradeabilityInputError, match="bar data_version"):
        _analyze(bars=(*_bars()[:2], _bar(2, data_version="wrong")))
    with pytest.raises(TradeabilityInputError, match="calendar data_version"):
        _analyze(calendar=_calendar(data_version="wrong"))
    with pytest.raises(TradeabilityInputError, match="master data_version"):
        _analyze(masters=(_master(data_version="wrong"),))


def test_closed_session_and_off_calendar_bar_fail_closed() -> None:
    closed = list(_calendar())
    closed[-1] = TradeabilityCalendarSession(
        market="SSE",
        trade_date=SESSION,
        is_open=False,
        available_at=_at(SESSION, 8),
        revision="closed",
        data_version="snapshot-v1",
    )
    with pytest.raises(InsufficientTradeabilityData, match="not an open"):
        _analyze(calendar=tuple(closed))

    off_calendar_date = date(2026, 8, 2)
    off_calendar = TradeabilityBar(
        instrument_id="CN.SH.TEST",
        trade_date=off_calendar_date,
        observed_at=_at(off_calendar_date, 15),
        available_at=_at(off_calendar_date, 16),
        previous_close=Decimal("100"),
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume=Decimal("1"),
        turnover_amount=Decimal("1"),
        turnover_rate=Decimal("1"),
        is_suspended=False,
        revision="off-calendar",
        data_version="snapshot-v1",
    )
    # It precedes the selected window and is safely irrelevant.
    assert _analyze(bars=(off_calendar, *_bars())).input_hash == _analyze().input_hash


def test_rule_boundary_rejects_prices_outside_supplied_limits() -> None:
    impossible = _bar(
        2,
        open_price="100",
        high="111",
        low="99",
        close="100",
    )
    with pytest.raises(TradeabilityInputError, match="exceed"):
        _analyze(bars=(*_bars()[:2], impossible))


def test_thresholds_and_inputs_are_frozen_and_non_finite_values_are_rejected() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.participation_rate = Decimal("0.2")  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        _config(participation_rate=Decimal("NaN"))
    with pytest.raises(ValueError, match="finite"):
        _request(account_value="Infinity")
    with pytest.raises(ValueError, match="finite"):
        _bar(0, amount="NaN")
    with pytest.raises(ValueError, match="timezone"):
        TradeabilityRequest(
            instrument_id="A",
            instrument_type=InstrumentType.STOCK,
            market="SSE",
            session_date=SESSION,
            as_of=datetime.combine(SESSION, time(18)),
            data_version="v1",
            side=TradeSide.BUY,
            account_value=Decimal(1),
            target_position_weight=Decimal("0.1"),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _config(short_turnover_window=4),
            "cannot exceed",
        ),
        (
            lambda: _config(participation_rate=Decimal(0)),
            "participation_rate",
        ),
        (
            lambda: _config(minimum_turnover_rate_percentile=Decimal("1.1")),
            "between zero and one",
        ),
        (
            lambda: InstrumentTradeabilityRule(
                rule_id="r",
                version="v",
                instrument_type=InstrumentType.STOCK,
                effective_from=SESSION,
                price_tick=Decimal(0),
            ),
            "price_tick",
        ),
        (
            lambda: _bar(2, suspended=True),
            "suspended observations",
        ),
    ],
)
def test_contract_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_deterministic_hashes_bind_configuration_and_rule_inputs() -> None:
    first = _analyze()
    repeated = _analyze()
    changed_config = _analyze(config=_config(participation_rate=Decimal("0.05")))
    changed_rule = _analyze(
        rules=(
            InstrumentTradeabilityRule(
                rule_id="rule-stock",
                version="rule-v2",
                instrument_type=InstrumentType.STOCK,
                effective_from=date(2020, 1, 1),
                price_limit_ratio=Decimal("0.10"),
                price_tick=Decimal("0.01"),
            ),
        )
    )

    assert first == repeated
    assert first.input_hash == repeated.input_hash
    assert first.result_hash == repeated.result_hash
    assert first.config_hash != changed_config.config_hash
    assert first.input_hash != changed_config.input_hash
    assert first.rule_hash != changed_rule.rule_hash
    assert first.result_hash != changed_rule.result_hash


@pytest.mark.parametrize(
    ("current", "side", "expected_state", "reason_code"),
    [
        (
            {
                "open_price": "105",
                "high": "110",
                "low": "100",
                "close": "110",
            },
            TradeSide.BUY,
            MarketTradeState.LIMIT_UP,
            TradeabilityReasonCode.LIMIT_UP,
        ),
        (
            {
                "open_price": "90",
                "high": "90",
                "low": "90",
                "close": "90",
            },
            TradeSide.SELL,
            MarketTradeState.ONE_PRICE_LIMIT_DOWN,
            TradeabilityReasonCode.ONE_PRICE_LIMIT_DOWN,
        ),
    ],
)
def test_remaining_limit_states_are_classified(
    current: dict[str, str],
    side: TradeSide,
    expected_state: MarketTradeState,
    reason_code: TradeabilityReasonCode,
) -> None:
    result = _analyze(
        request=_request(side=side),
        bars=(*_bars()[:2], _bar(2, **current)),  # type: ignore[arg-type]
    )

    assert result.market_state is expected_state
    assert not result.eligible
    assert next(reason for reason in result.reasons if reason.code is reason_code).blocking


def test_master_suspension_and_delisting_are_structured() -> None:
    suspended = _analyze(masters=(_master(status=InstrumentStatus.SUSPENDED),))
    delisted = _analyze(masters=(_master(status=InstrumentStatus.DELISTED),))

    assert suspended.market_state is MarketTradeState.SUSPENDED
    assert TradeabilityReasonCode.SUSPENDED in {item.code for item in suspended.reasons}
    assert TradeabilityReasonCode.DELISTED in {item.code for item in delisted.reasons}
    assert not suspended.eligible
    assert not delisted.eligible


def test_zero_capacity_is_finite_and_fails_the_capacity_gate() -> None:
    zero_bars = tuple(_bar(index, amount="0", turnover_rate="0") for index in range(3))
    result = _analyze(
        bars=zero_bars,
        config=_config(
            minimum_average_turnover_short=Decimal(0),
            minimum_average_turnover_long=Decimal(0),
            minimum_turnover_rate_percentile=Decimal(0),
        ),
    )

    assert result.capacity.daily_capacity == 0
    assert result.capacity.capacity_ratio == 0
    assert result.capacity.maximum_account_weight == 0
    assert result.capacity.estimated_trade_days is None
    assert TradeabilityReasonCode.INSUFFICIENT_CAPACITY in {
        reason.code for reason in result.reasons
    }


def test_ambiguous_master_calendar_and_rule_revisions_fail_closed() -> None:
    ambiguous_master = _master(status=InstrumentStatus.SUSPENDED, revision="m2")
    with pytest.raises(TradeabilityInputError, match="ambiguous master"):
        _analyze(masters=(_master(), ambiguous_master))

    calendar_conflict = TradeabilityCalendarSession(
        market="SSE",
        trade_date=SESSION,
        is_open=False,
        available_at=_at(SESSION, 8),
        revision="c2",
        data_version="snapshot-v1",
    )
    with pytest.raises(TradeabilityInputError, match="ambiguous calendar"):
        _analyze(calendar=(*_calendar(), calendar_conflict))

    duplicate_rule = InstrumentTradeabilityRule(
        rule_id="other",
        version="other-v1",
        instrument_type=InstrumentType.STOCK,
        effective_from=date(2020, 1, 1),
    )
    with pytest.raises(TradeabilityInputError, match="ambiguous tradeability rules"):
        _analyze(rules=(_rule(), duplicate_rule))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _request(account_value="0"),
            "account_value",
        ),
        (
            lambda: _request(target_weight="0"),
            "target_position_weight",
        ),
        (
            lambda: TradeabilityRequest(
                instrument_id="A",
                instrument_type=InstrumentType.INDEX,
                market="SSE",
                session_date=SESSION,
                as_of=_at(SESSION, 18),
                data_version="v1",
                side=TradeSide.BUY,
                account_value=Decimal(1),
                target_position_weight=Decimal("0.1"),
            ),
            "stock and ETF",
        ),
        (
            lambda: TradeabilityRequest(
                instrument_id="A",
                instrument_type=InstrumentType.STOCK,
                market="SSE",
                session_date=SESSION,
                as_of=_at(DATES[1], 18),
                data_version="v1",
                side=TradeSide.BUY,
                account_value=Decimal(1),
                target_position_weight=Decimal("0.1"),
            ),
            "cannot be after",
        ),
        (
            lambda: InstrumentTradeabilityState(
                instrument_id="A",
                instrument_type=InstrumentType.STOCK,
                listed_on=SESSION,
                delisted_on=SESSION - timedelta(days=1),
                status=InstrumentStatus.DELISTED,
                effective_from=SESSION,
                effective_to=None,
                available_at=_at(SESSION, 9),
                revision="r",
                data_version="v",
            ),
            "delisted_on",
        ),
        (
            lambda: InstrumentTradeabilityState(
                instrument_id="A",
                instrument_type=InstrumentType.STOCK,
                listed_on=SESSION,
                delisted_on=None,
                status=InstrumentStatus.LISTED,
                effective_from=SESSION,
                effective_to=SESSION - timedelta(days=1),
                available_at=_at(SESSION, 9),
                revision="r",
                data_version="v",
            ),
            "effective_to",
        ),
    ],
)
def test_request_and_master_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _bar(0, close="0", low="0"),
            "prices must be positive",
        ),
        (
            lambda: _bar(0, volume="-1"),
            "non-negative",
        ),
        (
            lambda: _bar(0, high="99"),
            "high must",
        ),
        (
            lambda: _bar(0, low="101"),
            "low must",
        ),
        (
            lambda: _bar(0, turnover_rate="-1"),
            "turnover_rate",
        ),
        (
            lambda: TradeabilityBar(
                instrument_id="A",
                trade_date=SESSION,
                observed_at=_at(DATES[1], 15),
                available_at=_at(SESSION, 16),
                previous_close=Decimal(1),
                open=Decimal(1),
                high=Decimal(1),
                low=Decimal(1),
                close=Decimal(1),
                volume=Decimal(0),
                turnover_amount=Decimal(0),
                turnover_rate=None,
                is_suspended=True,
                revision="r",
                data_version="v",
            ),
            "observed_at date",
        ),
        (
            lambda: TradeabilityBar(
                instrument_id="A",
                trade_date=SESSION,
                observed_at=_at(SESSION, 15),
                available_at=_at(SESSION, 14),
                previous_close=Decimal(1),
                open=Decimal(1),
                high=Decimal(1),
                low=Decimal(1),
                close=Decimal(1),
                volume=Decimal(0),
                turnover_amount=Decimal(0),
                turnover_rate=None,
                is_suspended=True,
                revision="r",
                data_version="v",
            ),
            "cannot precede",
        ),
    ],
)
def test_bar_contract_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: InstrumentTradeabilityRule(
                rule_id="r",
                version="v",
                instrument_type=InstrumentType.INDEX,
                effective_from=SESSION,
            ),
            "stock and ETF",
        ),
        (
            lambda: InstrumentTradeabilityRule(
                rule_id="r",
                version="v",
                instrument_type=InstrumentType.STOCK,
                effective_from=SESSION,
                effective_to=SESSION - timedelta(days=1),
            ),
            "effective_to",
        ),
        (
            lambda: InstrumentTradeabilityRule(
                rule_id="r",
                version="v",
                instrument_type=InstrumentType.STOCK,
                effective_from=SESSION,
                boundary_tolerance=Decimal("0.01"),
                price_tick=Decimal("0.01"),
            ),
            "boundary_tolerance",
        ),
        (
            lambda: InstrumentTradeabilityRule(
                rule_id="r",
                version="v",
                instrument_type=InstrumentType.STOCK,
                effective_from=SESSION,
                price_limit_ratio=Decimal(0),
            ),
            "price_limit_ratio",
        ),
        (
            lambda: InstrumentTradeabilityRule(
                rule_id="r",
                version="v",
                instrument_type=InstrumentType.STOCK,
                effective_from=SESSION,
                blocked_buy_states=frozenset({MarketTradeState.NORMAL}),
            ),
            "normal and suspended",
        ),
    ],
)
def test_rule_contract_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_result_contracts_reject_non_finite_or_inconsistent_outputs() -> None:
    result = _analyze()
    with pytest.raises(ValueError, match="capacity values"):
        replace(result.capacity, capacity_ratio=Decimal("NaN"))
    with pytest.raises(ValueError, match="blocking contribution"):
        replace(result.contributions[0], blocking=True)
    with pytest.raises(ValueError, match="eligible snapshot"):
        replace(
            result,
            reasons=(
                TradeabilityReason(
                    code=TradeabilityReasonCode.SUSPENDED,
                    blocking=True,
                    message="blocked",
                ),
            ),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(result, input_hash="bad")
    with pytest.raises(ValueError, match="estimated_trade_days"):
        CapacityEstimate(
            account_value=Decimal(1),
            target_position_weight=Decimal(1),
            target_notional=Decimal(1),
            average_turnover_short=Decimal(1),
            average_turnover_long=Decimal(1),
            participation_rate=Decimal(1),
            short_window_capacity=Decimal(1),
            long_window_capacity=Decimal(1),
            daily_capacity=Decimal(1),
            capacity_ratio=Decimal(1),
            maximum_account_weight=Decimal(1),
            estimated_trade_days=Decimal("NaN"),
        )
