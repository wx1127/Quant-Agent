"""Tests for effective-dated China stock/ETF mechanics and transaction costs."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from quant_agent.backtest.cn_market import (
    CNFeeSchedule,
    CNMarketRule,
    CNMarketRuleError,
    CNSlippageModel,
    CNSlippageModelBook,
    MarketSessionState,
    MarketStateUnavailableError,
    MatchResult,
    MatchStatus,
    NoFillReason,
    OrderNotTradableError,
    OrderPriceError,
    OrderQuantityError,
    SlippageModelNotFoundError,
    build_cn_fee_rule_book,
)
from quant_agent.backtest.contracts import (
    FeeEvent,
    FillEvent,
    FillStatus,
    OrderEvent,
    OrderStatus,
    OrderType,
    Side,
    TradableInstrumentType,
)
from quant_agent.backtest.engine import EventBacktestConfig, EventDrivenBacktestEngine
from quant_agent.features.tradeability import MarketTradeState

TZ = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 8, 28)
NEXT_SESSION = date(2026, 8, 31)
ORDER_TIME = datetime.combine(DAY, time(10), tzinfo=TZ)


def _state(
    *,
    instrument_type: TradableInstrumentType = TradableInstrumentType.STOCK,
    state: MarketTradeState = MarketTradeState.NORMAL,
    buy_allowed: bool = True,
    sell_allowed: bool = True,
    reference_price: str = "10",
    lower_limit: str | None = "9",
    upper_limit: str | None = "11",
    quantity: str = "1000",
    observed_at: datetime | None = None,
    available_at: datetime | None = None,
    revision: str = "state-v1",
) -> MarketSessionState:
    return MarketSessionState(
        instrument_id="CN.SH.TEST",
        instrument_type=instrument_type,
        trading_day=DAY,
        observed_at=observed_at or datetime.combine(DAY, time(9, 30), tzinfo=TZ),
        available_at=available_at or datetime.combine(DAY, time(9, 31), tzinfo=TZ),
        revision=revision,
        data_version="snapshot-v1",
        state=state,
        buy_allowed=buy_allowed,
        sell_allowed=sell_allowed,
        reference_price=Decimal(reference_price),
        lower_limit_price=Decimal(lower_limit) if lower_limit is not None else None,
        upper_limit_price=Decimal(upper_limit) if upper_limit is not None else None,
        available_quantity=Decimal(quantity),
    )


def _rule(
    *,
    instrument_type: TradableInstrumentType = TradableInstrumentType.STOCK,
    states: tuple[MarketSessionState, ...] | None = None,
    effective_from: date = date(2020, 1, 1),
    effective_to: date | None = None,
    buy_lot: str = "100",
    sell_lot: str = "100",
    allow_odd: bool = True,
    liquidation_order_ids: tuple[str, ...] = (),
    settlement_days: int = 1,
    trading_calendar: tuple[date, ...] = (DAY, NEXT_SESSION, date(2026, 9, 1)),
    participation: str = "0.10",
    version: str = "cn-v1",
) -> CNMarketRule:
    return CNMarketRule(
        rule_id=f"cn-{instrument_type.value.lower()}",
        version=version,
        effective_from=effective_from,
        effective_to=effective_to,
        instrument_type=instrument_type,
        session_states=states if states is not None else (_state(instrument_type=instrument_type),),
        trading_calendar=trading_calendar,
        buy_lot_size=Decimal(buy_lot),
        sell_lot_size=Decimal(sell_lot),
        allow_odd_lot_liquidation=allow_odd,
        odd_lot_liquidation_order_ids=frozenset(liquidation_order_ids),
        settlement_days=settlement_days,
        price_tick=Decimal("0.01"),
        max_participation_rate=Decimal(participation),
    )


def _order(
    *,
    instrument_type: TradableInstrumentType = TradableInstrumentType.STOCK,
    side: Side = Side.BUY,
    quantity: str = "100",
    order_type: OrderType = OrderType.LIMIT,
    limit_price: str | None = "10.50",
    reference_price: str | None = None,
    event_time: datetime = ORDER_TIME,
    sequence: int = 0,
    status: OrderStatus = OrderStatus.CREATED,
    previous_status: OrderStatus | None = None,
    order_id: str = "order-1",
) -> OrderEvent:
    return OrderEvent(
        event_id=f"order-event-{sequence}",
        run_id="run-cn",
        sequence=sequence,
        event_time=event_time,
        trading_day=DAY,
        instrument_id="CN.SH.TEST",
        instrument_type=instrument_type,
        order_id=order_id,
        signal_event_id="signal-1",
        signal_time=ORDER_TIME - timedelta(minutes=1),
        side=side,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        status=status,
        previous_status=previous_status,
        filled_quantity="0",
        reference_price=reference_price,
        price_observed_at=(
            ORDER_TIME - timedelta(minutes=1) if reference_price is not None else None
        ),
        price_available_at=(
            ORDER_TIME - timedelta(minutes=1) if reference_price is not None else None
        ),
    )


def _slippage(
    *,
    base: str = "0",
    impact: str = "0",
    maximum: str = "0.10",
    effective_from: date = date(2020, 1, 1),
    effective_to: date | None = None,
    version: str = "slippage-v1",
) -> CNSlippageModel:
    return CNSlippageModel(
        model_id="cn-linear",
        version=version,
        effective_from=effective_from,
        effective_to=effective_to,
        base_rate=Decimal(base),
        participation_impact_rate=Decimal(impact),
        maximum_rate=Decimal(maximum),
    )


def test_stock_buy_requires_board_lots_and_odd_sell_requires_explicit_switch() -> None:
    rule = _rule()
    rule.validate_order(_order(quantity="100"))
    with pytest.raises(OrderQuantityError, match="integer multiple of 100"):
        rule.validate_order(_order(quantity="150"))

    with pytest.raises(OrderQuantityError, match="full odd-lot liquidation"):
        rule.validate_order(_order(side=Side.SELL, quantity="50"))
    liquidation = _rule(liquidation_order_ids=("order-1",))
    liquidation.validate_order(_order(side=Side.SELL, quantity="50"))
    with pytest.raises(OrderQuantityError, match="full odd-lot liquidation"):
        liquidation.validate_order(_order(side=Side.SELL, quantity="50", order_id="not-authorized"))
    strict = _rule(allow_odd=False)
    with pytest.raises(OrderQuantityError, match="SELL quantity"):
        strict.validate_order(_order(side=Side.SELL, quantity="50"))
    strict.validate_order(_order(side=Side.SELL, quantity="100"))
    with pytest.raises(OrderQuantityError, match="whole number"):
        rule.validate_quantity(side=Side.SELL, quantity=Decimal("0.5"))


def test_stock_t_plus_one_and_etf_units_and_settlement_are_configurable() -> None:
    stock = _rule()
    etf_state = _state(instrument_type=TradableInstrumentType.ETF)
    etf = _rule(
        instrument_type=TradableInstrumentType.ETF,
        states=(etf_state,),
        buy_lot="10",
        sell_lot="10",
        settlement_days=0,
    )

    assert stock.sellable_on(acquired_on=DAY) == NEXT_SESSION
    assert etf.sellable_on(acquired_on=DAY) == DAY
    etf.validate_order(_order(instrument_type=TradableInstrumentType.ETF, quantity="10"))
    with pytest.raises(OrderQuantityError):
        etf.validate_order(_order(instrument_type=TradableInstrumentType.ETF, quantity="15"))


def test_settlement_uses_trading_sessions_and_fails_closed_without_horizon() -> None:
    rule = _rule(
        settlement_days=2,
        trading_calendar=(DAY, NEXT_SESSION, date(2026, 9, 2)),
    )
    assert rule.sellable_on(acquired_on=DAY) == date(2026, 9, 2)
    with pytest.raises(CNMarketRuleError, match="settlement horizon"):
        _rule(trading_calendar=(DAY,)).sellable_on(acquired_on=DAY)
    with pytest.raises(CNMarketRuleError, match="settlement horizon"):
        _rule().sellable_on(acquired_on=date(2026, 8, 27))
    with pytest.raises(ValueError, match="unique ascending"):
        _rule(trading_calendar=(NEXT_SESSION, DAY))
    with pytest.raises(ValueError, match="require the liquidation policy"):
        _rule(allow_odd=False, liquidation_order_ids=("order-1",))


@pytest.mark.parametrize(
    ("state", "side", "message"),
    [
        (
            _state(
                state=MarketTradeState.SUSPENDED,
                buy_allowed=False,
                sell_allowed=False,
            ),
            Side.BUY,
            "suspended",
        ),
        (
            _state(
                state=MarketTradeState.ONE_PRICE_LIMIT_UP,
                reference_price="11",
                buy_allowed=False,
                sell_allowed=True,
            ),
            Side.BUY,
            "blocks BUY",
        ),
        (
            _state(
                state=MarketTradeState.LIMIT_DOWN,
                reference_price="9",
                buy_allowed=True,
                sell_allowed=False,
            ),
            Side.SELL,
            "blocks SELL",
        ),
    ],
)
def test_injected_session_state_blocks_suspension_and_limit_directions(
    state: MarketSessionState,
    side: Side,
    message: str,
) -> None:
    with pytest.raises(OrderNotTradableError, match=message):
        _rule(states=(state,)).validate_order(_order(side=side))


def test_injected_limit_state_can_permit_the_opposite_direction() -> None:
    limit_up = _state(
        state=MarketTradeState.LIMIT_UP,
        reference_price="11",
        buy_allowed=False,
        sell_allowed=True,
    )
    limit_down = _state(
        state=MarketTradeState.ONE_PRICE_LIMIT_DOWN,
        reference_price="9",
        buy_allowed=True,
        sell_allowed=False,
    )

    _rule(states=(limit_up,)).validate_order(_order(side=Side.SELL))
    _rule(states=(limit_down,)).validate_order(_order(side=Side.BUY))


def test_session_state_must_be_known_by_order_time_and_future_revision_is_ignored() -> None:
    known = _state(revision="known")
    future = _state(
        state=MarketTradeState.SUSPENDED,
        buy_allowed=False,
        sell_allowed=False,
        observed_at=ORDER_TIME + timedelta(minutes=1),
        available_at=ORDER_TIME + timedelta(minutes=1),
        revision="future",
    )
    baseline = _rule(states=(known,)).match(_order(), slippage_model=_slippage())
    repeated = _rule(states=(future, known)).match(_order(), slippage_model=_slippage())

    assert repeated == baseline
    assert repeated.state_revision == "known"
    with pytest.raises(MarketStateUnavailableError, match=r"order\.event_time"):
        _rule(states=(future,)).validate_order(_order())


def test_latest_known_revision_wins_and_same_time_conflict_fails_closed() -> None:
    earlier = _state(
        available_at=datetime.combine(DAY, time(9, 31), tzinfo=TZ),
        revision="earlier",
    )
    latest = _state(
        available_at=datetime.combine(DAY, time(9, 45), tzinfo=TZ),
        reference_price="10.10",
        revision="latest",
    )
    result = _rule(states=(earlier, latest)).match(
        _order(limit_price="10.50"),
        slippage_model=_slippage(),
    )
    assert result.reference_price == Decimal("10.10")
    assert result.state_revision == "latest"

    conflict = _state(
        available_at=latest.available_at,
        reference_price="10.20",
        revision="conflict",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        _rule(states=(latest, conflict))


def test_public_state_for_preserves_pit_selection_and_price_validation() -> None:
    earlier = _state(
        available_at=datetime.combine(DAY, time(9, 31), tzinfo=TZ),
        revision="earlier",
    )
    latest = _state(
        observed_at=datetime.combine(DAY, time(9, 40), tzinfo=TZ),
        available_at=datetime.combine(DAY, time(9, 45), tzinfo=TZ),
        reference_price="10.10",
        revision="latest",
    )
    future = _state(
        observed_at=ORDER_TIME + timedelta(minutes=1),
        available_at=ORDER_TIME + timedelta(minutes=1),
        reference_price="10.20",
        revision="future",
    )
    rule = _rule(states=(future, earlier, latest))

    selected = rule.state_for(
        instrument_id="CN.SH.TEST",
        trading_day=DAY,
        as_of=ORDER_TIME,
    )

    assert selected is latest
    with pytest.raises(MarketStateUnavailableError, match="as_of"):
        rule.state_for(
            instrument_id="CN.SH.TEST",
            trading_day=DAY,
            as_of=datetime.combine(DAY, time(9, 30), tzinfo=TZ),
        )
    with pytest.raises(MarketStateUnavailableError, match="as_of"):
        rule.state_for(
            instrument_id="CN.SH.OTHER",
            trading_day=DAY,
            as_of=ORDER_TIME,
        )
    with pytest.raises(OrderPriceError, match="session prices"):
        _rule(states=(_state(reference_price="10.005"),)).state_for(
            instrument_id="CN.SH.TEST",
            trading_day=DAY,
            as_of=ORDER_TIME,
        )


def test_limit_price_tick_and_daily_boundaries_are_inclusive_and_fail_closed() -> None:
    rule = _rule()
    rule.validate_order(_order(limit_price="11.00"))
    rule.validate_order(_order(side=Side.SELL, limit_price="9.00"))
    with pytest.raises(OrderPriceError, match="price_tick"):
        rule.validate_order(_order(limit_price="10.005"))
    with pytest.raises(OrderPriceError, match="upper limit"):
        rule.validate_order(_order(limit_price="11.01"))
    with pytest.raises(OrderPriceError, match="lower limit"):
        rule.validate_order(_order(side=Side.SELL, limit_price="8.99"))


def test_reference_price_and_state_prices_also_obey_tick_and_boundaries() -> None:
    with pytest.raises(OrderPriceError, match="reference_price"):
        _rule().validate_order(_order(reference_price="10.005"))
    bad_state = _state(reference_price="10.005")
    with pytest.raises(OrderPriceError, match="session prices"):
        _rule(states=(bad_state,)).validate_order(_order())


def test_participation_capacity_produces_deterministic_partial_fill() -> None:
    rule = _rule(
        states=(_state(quantity="1000"),),
        participation="0.15",
    )
    result = rule.match(
        _order(quantity="300"),
        slippage_model=_slippage(base="0.001", impact="0.01"),
    )

    assert result.status is MatchStatus.PARTIALLY_FILLED
    assert result.requested_quantity == 300
    assert result.filled_quantity == 100
    assert result.unfilled_quantity == 200
    assert result.participation_rate == Decimal("0.1")
    assert result.execution_price == Decimal("10.02")
    assert result.gross_amount == Decimal("1002")
    assert len(result.state_hash) == len(result.slippage_model_hash) == 64


def test_partial_fill_can_be_disabled_and_sub_lot_capacity_is_no_fill() -> None:
    partial = _rule(states=(_state(quantity="1000"),), participation="0.15")
    disabled = partial.match(
        _order(quantity="300"),
        slippage_model=_slippage(),
        allow_partial=False,
    )
    zero = _rule(states=(_state(quantity="50"),), participation="0.10").match(
        _order(quantity="100"),
        slippage_model=_slippage(),
    )

    assert disabled.status is MatchStatus.NO_FILL
    assert disabled.reason is NoFillReason.PARTIAL_FILL_DISABLED
    assert disabled.execution_price is None
    assert zero.status is MatchStatus.NO_FILL
    assert zero.reason is NoFillReason.ZERO_CAPACITY


def test_odd_lot_liquidation_fills_exactly_or_preserves_lot_sized_partial() -> None:
    full = _rule(
        states=(_state(quantity="1000"),),
        participation="1",
        liquidation_order_ids=("order-1",),
    ).match(
        _order(side=Side.SELL, quantity="50", limit_price="9.50"),
        slippage_model=_slippage(),
    )
    partial = _rule(
        states=(_state(quantity="120"),),
        participation="1",
        liquidation_order_ids=("order-1",),
    ).match(
        _order(side=Side.SELL, quantity="150", limit_price="9.50"),
        slippage_model=_slippage(),
    )

    assert full.status is MatchStatus.FILLED
    assert full.filled_quantity == 50
    assert partial.status is MatchStatus.PARTIALLY_FILLED
    assert partial.filled_quantity == 100
    assert partial.unfilled_quantity == 50


def test_match_accepts_dynamic_full_liquidation_and_false_overrides_registration() -> None:
    dynamic = _rule(
        states=(_state(quantity="1000"),),
        participation="1",
    )
    rule_hash = dynamic.rule_hash

    full = dynamic.match(
        _order(side=Side.SELL, quantity="50", limit_price="9.50"),
        slippage_model=_slippage(),
        is_full_liquidation=True,
    )

    assert full.status is MatchStatus.FILLED
    assert full.filled_quantity == 50
    assert dynamic.rule_hash == rule_hash

    registered = _rule(liquidation_order_ids=("order-1",))
    with pytest.raises(OrderQuantityError, match="full odd-lot liquidation"):
        registered.match(
            _order(side=Side.SELL, quantity="50", limit_price="9.50"),
            slippage_model=_slippage(),
            is_full_liquidation=False,
        )


def test_limit_order_that_cannot_absorb_slippage_is_no_fill() -> None:
    result = _rule().match(
        _order(limit_price="10.00"),
        slippage_model=_slippage(base="0.001"),
    )

    assert result.status is MatchStatus.NO_FILL
    assert result.reason is NoFillReason.LIMIT_NOT_MARKETABLE
    assert result.filled_quantity == 0


def test_slippage_is_adverse_tick_rounded_capped_and_price_limit_bounded() -> None:
    model = _slippage(base="0.0015", impact="0")
    buy = model.execution_price(
        side=Side.BUY,
        reference_price=Decimal("10"),
        participation_rate=Decimal("0.1"),
        price_tick=Decimal("0.01"),
    )
    sell = model.execution_price(
        side=Side.SELL,
        reference_price=Decimal("10"),
        participation_rate=Decimal("0.1"),
        price_tick=Decimal("0.01"),
    )
    assert buy == Decimal("10.02")
    assert sell == Decimal("9.98")

    at_limit = _state(
        state=MarketTradeState.LIMIT_UP,
        reference_price="11",
        buy_allowed=True,
        sell_allowed=True,
    )
    bounded = _rule(states=(at_limit,)).match(
        _order(limit_price="11"),
        slippage_model=_slippage(base="0.05"),
    )
    assert bounded.execution_price == Decimal("11")
    assert _slippage(base="0.01", impact="1", maximum="0.02").rate(Decimal("0.5")) == Decimal(
        "0.02"
    )


def test_slippage_model_book_selects_effective_versions_and_rejects_gaps() -> None:
    old = _slippage(
        effective_from=date(2020, 1, 1),
        effective_to=date(2025, 12, 31),
        version="old",
    )
    current = _slippage(effective_from=date(2026, 1, 1), version="current")
    book = CNSlippageModelBook([current, old])

    assert book.select(date(2025, 1, 1)) is old
    assert book.select(DAY) is current
    with pytest.raises(SlippageModelNotFoundError):
        book.select(date(2019, 1, 1))
    with pytest.raises(SlippageModelNotFoundError):
        _rule().match(
            _order(),
            slippage_model=_slippage(effective_from=DAY + timedelta(days=1)),
        )


def test_fee_schedule_builds_side_specific_minimum_tax_transfer_and_other_fees() -> None:
    schedule = CNFeeSchedule(
        schedule_id="stock-fees",
        version="2026-v1",
        effective_from=date(2026, 1, 1),
        instrument_type=TradableInstrumentType.STOCK,
        commission_rate=Decimal("0.0003"),
        minimum_commission=Decimal("5"),
        buy_stamp_duty_rate=Decimal(0),
        sell_stamp_duty_rate=Decimal("0.001"),
        buy_transfer_fee_rate=Decimal("0.00001"),
        sell_transfer_fee_rate=Decimal("0.00001"),
        buy_other_fee_rate=Decimal("0.00002"),
        sell_other_fee_rate=Decimal("0.00002"),
    )
    buy, sell = schedule.rules()
    buy_fee = buy.assess("1000")
    sell_fee = sell.assess("1000")

    assert buy_fee.commission == Decimal("5.00")
    assert buy_fee.stamp_duty == 0
    assert buy_fee.transfer_fee == Decimal("0.01")
    assert buy_fee.other_fee == Decimal("0.02")
    assert buy_fee.total_amount == Decimal("5.03")
    assert sell_fee.commission == Decimal("5.00")
    assert sell_fee.stamp_duty == Decimal("1.00")
    assert sell_fee.total_amount == Decimal("6.03")
    assert buy.version == schedule.version_for(Side.BUY)
    assert sell.version == schedule.version_for(Side.SELL)


def test_fee_rule_book_switches_by_effective_date_for_stock_and_etf() -> None:
    old = CNFeeSchedule(
        schedule_id="stock",
        version="old",
        effective_from=date(2020, 1, 1),
        instrument_type=TradableInstrumentType.STOCK,
        commission_rate=Decimal("0.001"),
        minimum_commission=Decimal("5"),
        sell_stamp_duty_rate=Decimal("0.001"),
    )
    new = CNFeeSchedule(
        schedule_id="stock",
        version="new",
        effective_from=date(2026, 9, 1),
        instrument_type=TradableInstrumentType.STOCK,
        commission_rate=Decimal("0.0002"),
        minimum_commission=Decimal("3"),
        sell_stamp_duty_rate=Decimal("0.0005"),
    )
    etf = CNFeeSchedule(
        schedule_id="etf",
        version="etf-v1",
        effective_from=date(2020, 1, 1),
        instrument_type=TradableInstrumentType.ETF,
        commission_rate=Decimal("0.0003"),
        minimum_commission=Decimal("5"),
        sell_stamp_duty_rate=Decimal(0),
    )
    book = build_cn_fee_rule_book([new, etf, old])

    assert book.select(
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        trading_day=DAY,
    ).version == old.version_for(Side.SELL)
    assert book.select(
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        trading_day=date(2026, 9, 2),
    ).version == new.version_for(Side.SELL)
    assert (
        book.select(
            instrument_type=TradableInstrumentType.ETF,
            side=Side.SELL,
            trading_day=DAY,
        ).stamp_duty_rate
        == 0
    )


def _order_transition(
    source: OrderEvent,
    *,
    sequence: int,
    status: OrderStatus,
    previous_status: OrderStatus,
) -> OrderEvent:
    return OrderEvent.model_validate(
        source.model_dump(mode="python")
        | {
            "event_id": f"order-event-{sequence}",
            "sequence": sequence,
            "event_time": ORDER_TIME + timedelta(seconds=sequence),
            "status": status,
            "previous_status": previous_status,
        }
    )


def test_generated_fee_rule_book_replays_directly_in_event_engine() -> None:
    schedule = CNFeeSchedule(
        schedule_id="stock",
        version="v1",
        effective_from=date(2020, 1, 1),
        instrument_type=TradableInstrumentType.STOCK,
        commission_rate=Decimal("0.0003"),
        minimum_commission=Decimal("5"),
        sell_stamp_duty_rate=Decimal("0.001"),
    )
    fee_book = build_cn_fee_rule_book([schedule])
    rule = _rule()
    created = _order(limit_price="10")
    submitted = _order_transition(
        created,
        sequence=1,
        status=OrderStatus.SUBMITTED,
        previous_status=OrderStatus.CREATED,
    )
    accepted = _order_transition(
        submitted,
        sequence=2,
        status=OrderStatus.ACCEPTED,
        previous_status=OrderStatus.SUBMITTED,
    )
    fill_time = ORDER_TIME + timedelta(seconds=3)
    fill = FillEvent(
        event_id="fill-event-3",
        run_id="run-cn",
        sequence=3,
        event_time=fill_time,
        trading_day=DAY,
        instrument_id="CN.SH.TEST",
        instrument_type=TradableInstrumentType.STOCK,
        fill_id="fill-1",
        order_id="order-1",
        signal_event_id="signal-1",
        signal_time=created.signal_time,
        order_time=created.event_time,
        side=Side.BUY,
        quantity="100",
        price="10",
        gross_amount="1000",
        price_observed_at=fill_time,
        price_available_at=fill_time,
        status=FillStatus.CREATED,
    )
    confirmed = FillEvent.model_validate(
        fill.model_dump(mode="python")
        | {
            "event_id": "fill-event-4",
            "sequence": 4,
            "event_time": ORDER_TIME + timedelta(seconds=4),
            "status": FillStatus.CONFIRMED,
            "previous_status": FillStatus.CREATED,
        }
    )
    selected_fee = fee_book.select(
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.BUY,
        trading_day=DAY,
    )
    fee = selected_fee.assess("1000")
    fee_event = FeeEvent(
        event_id="fee-event-5",
        run_id="run-cn",
        sequence=5,
        event_time=ORDER_TIME + timedelta(seconds=5),
        trading_day=DAY,
        instrument_id="CN.SH.TEST",
        instrument_type=TradableInstrumentType.STOCK,
        fill_id="fill-1",
        fill_time=fill.event_time,
        side=Side.BUY,
        fee_rule_version=selected_fee.version,
        commission=fee.commission,
        stamp_duty=fee.stamp_duty,
        transfer_fee=fee.transfer_fee,
        other_fee=fee.other_fee,
        total_amount=fee.total_amount,
    )
    engine = EventDrivenBacktestEngine(
        EventBacktestConfig(
            run_id="run-cn",
            account_id="paper-cn",
            initial_cash="2000",
            require_target_for_order=False,
        ),
        [rule],
        fee_book,
    )

    result = engine.process_all([created, submitted, accepted, fill, confirmed, fee_event])

    assert result.cash_balance == Decimal("995")
    assert result.position("CN.SH.TEST") is not None
    assert result.position("CN.SH.TEST").quantity == Decimal("100")  # type: ignore[union-attr]
    assert result.position("CN.SH.TEST").sellable_quantity == 0  # type: ignore[union-attr]
    assert result.position("CN.SH.TEST").average_cost == Decimal("10.05")  # type: ignore[union-attr]


def test_event_engine_selects_latest_effective_cn_market_rule() -> None:
    old = _rule(
        states=(_state(),),
        effective_from=date(2020, 1, 1),
        buy_lot="100",
        version="old",
    )
    new = _rule(
        states=(_state(),),
        effective_from=date(2026, 1, 1),
        buy_lot="10",
        version="new",
    )
    engine = EventDrivenBacktestEngine(
        EventBacktestConfig(
            run_id="run-cn",
            account_id="paper-cn",
            initial_cash="1000",
            require_target_for_order=False,
        ),
        [old, new],
    )

    result = engine.process(_order(quantity="10"))
    assert result.orders[0].quantity == 10


def test_rule_and_model_configuration_are_frozen_hashed_and_finite() -> None:
    rule = _rule()
    model = _slippage()
    with pytest.raises(FrozenInstanceError):
        rule.max_participation_rate = Decimal("0.2")  # type: ignore[misc]
    assert len(rule.rule_hash) == len(model.model_hash) == 64
    with pytest.raises(ValueError, match="finite"):
        _rule(participation="NaN")
    with pytest.raises(ValueError, match="finite"):
        _slippage(base="Infinity")
    with pytest.raises(ValueError, match="finite"):
        CNFeeSchedule(
            schedule_id="fees",
            version="v1",
            effective_from=DAY,
            instrument_type=TradableInstrumentType.STOCK,
            commission_rate=Decimal("NaN"),
            minimum_commission=Decimal(0),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _rule(buy_lot="0"),
            "buy_lot_size",
        ),
        (
            lambda: _rule(settlement_days=-1),
            "settlement_days",
        ),
        (
            lambda: _rule(effective_to=date(2019, 1, 1)),
            "effective_to",
        ),
        (
            lambda: _slippage(base="0.2", maximum="0.1"),
            "base_rate",
        ),
        (
            lambda: _slippage(maximum="1"),
            "maximum_rate",
        ),
        (
            lambda: CNSlippageModelBook([]),
            "at least one",
        ),
        (
            lambda: build_cn_fee_rule_book([]),
            "at least one",
        ),
    ],
)
def test_configuration_boundaries(factory: object, message: str) -> None:
    with pytest.raises((ValueError, SlippageModelNotFoundError), match=message):
        factory()  # type: ignore[operator]


def test_session_state_contract_rejects_impossible_values() -> None:
    with pytest.raises(ValueError, match="whole number"):
        _state(quantity="1.5")
    with pytest.raises(ValueError, match="supplied together"):
        _state(lower_limit=None, upper_limit="11")
    with pytest.raises(ValueError, match="reference_price at the upper"):
        _state(state=MarketTradeState.LIMIT_UP, reference_price="10")
    with pytest.raises(ValueError, match="cannot allow"):
        _state(state=MarketTradeState.SUSPENDED)
    with pytest.raises(ValueError, match="cannot precede"):
        _state(
            observed_at=datetime.combine(DAY, time(9, 31), tzinfo=TZ),
            available_at=datetime.combine(DAY, time(9, 30), tzinfo=TZ),
        )


def test_order_type_effective_date_and_instrument_mismatches_fail_closed() -> None:
    with pytest.raises(CNMarketRuleError, match="instrument_type"):
        _rule().validate_order(_order(instrument_type=TradableInstrumentType.ETF, quantity="100"))
    with pytest.raises(CNMarketRuleError, match="not effective"):
        _rule(effective_from=DAY + timedelta(days=1)).validate_order(_order())
    with pytest.raises(ValidationError):
        _order(order_type=OrderType.MARKET, limit_price="10")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _state(
                observed_at=datetime.combine(DAY - timedelta(days=1), time(10), tzinfo=TZ),
                available_at=datetime.combine(DAY, time(9, 31), tzinfo=TZ),
            ),
            "observed_at date",
        ),
        (
            lambda: _state(reference_price="0"),
            "reference_price must be positive",
        ),
        (
            lambda: _state(quantity="-1"),
            "available_quantity",
        ),
        (
            lambda: _state(lower_limit="11", upper_limit="9"),
            "ascending",
        ),
        (
            lambda: _state(reference_price="12"),
            "inside daily price limits",
        ),
        (
            lambda: _state(state=MarketTradeState.LIMIT_DOWN, reference_price="10"),
            "lower limit",
        ),
        (
            lambda: _state(
                state=MarketTradeState.LIMIT_UP,
                reference_price="10",
                lower_limit=None,
                upper_limit=None,
            ),
            "require daily limit prices",
        ),
    ],
)
def test_additional_session_state_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _slippage(
                effective_from=DAY,
                effective_to=DAY - timedelta(days=1),
            ),
            "effective_to",
        ),
        (
            lambda: _slippage(base="-0.1"),
            "cannot be negative",
        ),
        (
            lambda: _slippage().rate(Decimal("1.1")),
            "between zero and one",
        ),
        (
            lambda: _slippage().execution_price(
                side=Side.BUY,
                reference_price=Decimal(0),
                participation_rate=Decimal(0),
                price_tick=Decimal("0.01"),
            ),
            "must be positive",
        ),
        (
            lambda: _slippage(base="0.9", maximum="0.9").execution_price(
                side=Side.SELL,
                reference_price=Decimal("0.01"),
                participation_rate=Decimal(0),
                price_tick=Decimal("0.01"),
            ),
            "non-positive",
        ),
    ],
)
def test_additional_slippage_boundaries(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_slippage_book_rejects_ambiguous_dates_and_duplicate_versions() -> None:
    first = _slippage(effective_from=date(2020, 1, 1), version="same")
    with pytest.raises(ValueError, match="effective boundary"):
        CNSlippageModelBook([first, _slippage(effective_from=date(2020, 1, 1), version="other")])
    with pytest.raises(ValueError, match="identity and version"):
        CNSlippageModelBook([first, _slippage(effective_from=date(2021, 1, 1), version="same")])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"commission_rate": Decimal("1.1")}, "between zero and one"),
        ({"minimum_commission": Decimal("-1")}, "minimum_commission"),
        ({"rounding_increment": Decimal(0)}, "rounding_increment"),
    ],
)
def test_fee_schedule_rejects_invalid_rates(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "schedule_id": "fees",
        "version": "v1",
        "effective_from": DAY,
        "instrument_type": TradableInstrumentType.STOCK,
        "commission_rate": Decimal("0.001"),
        "minimum_commission": Decimal(0),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        CNFeeSchedule(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sell_lot": "0"}, "sell_lot_size"),
        ({"participation": "1.1"}, "max_participation_rate"),
    ],
)
def test_additional_market_rule_boundaries(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _rule(**kwargs)  # type: ignore[arg-type]

    mismatched_state = _state(instrument_type=TradableInstrumentType.ETF)
    with pytest.raises(ValueError, match="instrument_type"):
        _rule(states=(mismatched_state,))


def test_market_order_matches_without_limit_and_unbounded_state_is_supported() -> None:
    state = _state(lower_limit=None, upper_limit=None)
    result = _rule(states=(state,)).match(
        _order(order_type=OrderType.MARKET, limit_price=None),
        slippage_model=_slippage(),
    )
    assert result.status is MatchStatus.FILLED
    assert result.execution_price == Decimal("10")


def test_match_result_contract_rejects_inconsistent_economics() -> None:
    valid = _rule().match(_order(), slippage_model=_slippage())
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(valid, participation_rate=Decimal("NaN"))
    with pytest.raises(ValueError, match="equal requested_quantity"):
        replace(valid, unfilled_quantity=Decimal(1))
    with pytest.raises(ValueError, match="cannot exceed one"):
        replace(valid, participation_rate=Decimal("1.1"))
    with pytest.raises(ValueError, match="gross_amount"):
        replace(valid, gross_amount=Decimal(1))
    with pytest.raises(ValueError, match="cannot retain"):
        replace(valid, unfilled_quantity=Decimal(1), requested_quantity=Decimal(101))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(valid, state_hash="bad")

    no_fill = _rule(states=(_state(quantity="0"),)).match(
        _order(),
        slippage_model=_slippage(),
    )
    with pytest.raises(ValueError, match="NO_FILL requires"):
        MatchResult(
            order_id=no_fill.order_id,
            trading_day=no_fill.trading_day,
            status=MatchStatus.NO_FILL,
            reason=None,
            requested_quantity=no_fill.requested_quantity,
            filled_quantity=Decimal(0),
            unfilled_quantity=no_fill.requested_quantity,
            reference_price=no_fill.reference_price,
            execution_price=None,
            gross_amount=Decimal(0),
            participation_rate=Decimal(0),
            state_revision=no_fill.state_revision,
            state_available_at=no_fill.state_available_at,
            state_hash=no_fill.state_hash,
            market_rule_version=no_fill.market_rule_version,
            slippage_model_version=no_fill.slippage_model_version,
            slippage_model_hash=no_fill.slippage_model_hash,
        )
