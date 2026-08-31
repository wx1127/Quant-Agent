"""Contract tests for deterministic, replayable stock and ETF backtests."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from quant_agent.backtest import (
    CashDirection,
    CashEvent,
    FeeEvent,
    FeeRule,
    FeeRuleBook,
    FeeRuleNotFoundError,
    FillEvent,
    FillStatus,
    InvalidStateTransition,
    MarketRule,
    OrderEvent,
    OrderStatus,
    OrderType,
    PositionEvent,
    Side,
    SignalDirection,
    SignalEvent,
    TargetEvent,
    TradableInstrumentType,
    deserialize_event,
    deserialize_event_stream,
    ensure_fill_transition,
    ensure_order_transition,
    event_hash,
    serialize_event,
    serialize_event_stream,
    validate_replay_stream,
)

TZ = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
DAY = date(2026, 8, 28)
SIGNAL_TIME = datetime(2026, 8, 28, 9, 31, tzinfo=TZ)
ORDER_TIME = datetime(2026, 8, 28, 9, 32, tzinfo=TZ)
FILL_TIME = datetime(2026, 8, 28, 9, 33, tzinfo=TZ)


def event_base(*, sequence: int, event_time: datetime) -> dict[str, object]:
    return {
        "event_id": f"evt-{sequence}",
        "run_id": "run-1",
        "sequence": sequence,
        "event_time": event_time,
        "trading_day": DAY,
    }


def instrument_base(*, sequence: int, event_time: datetime) -> dict[str, object]:
    return event_base(sequence=sequence, event_time=event_time) | {
        "instrument_id": "CN.SH.510300",
        "instrument_type": TradableInstrumentType.ETF,
    }


def make_signal(
    *, sequence: int = 1, event_time: datetime = SIGNAL_TIME, instrument_type: object = None
) -> SignalEvent:
    payload = instrument_base(sequence=sequence, event_time=event_time)
    if instrument_type is not None:
        payload["instrument_type"] = instrument_type
    return SignalEvent(
        **payload,
        direction=SignalDirection.LONG,
        strength="0.75",
        strategy_version="momentum-v1",
        data_version="snapshot-1",
        reference_price="4.1230",
        price_observed_at=event_time - timedelta(seconds=1),
        price_available_at=event_time,
    )


def make_order(
    *,
    sequence: int = 2,
    event_time: datetime = ORDER_TIME,
    status: OrderStatus = OrderStatus.CREATED,
    previous_status: OrderStatus | None = None,
    filled_quantity: str = "0",
) -> OrderEvent:
    return OrderEvent(
        **instrument_base(sequence=sequence, event_time=event_time),
        order_id="ord-1",
        signal_event_id="evt-1",
        signal_time=SIGNAL_TIME,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity="100",
        limit_price="4.13",
        status=status,
        previous_status=previous_status,
        filled_quantity=filled_quantity,
    )


def make_fill(
    *,
    sequence: int = 3,
    event_time: datetime = FILL_TIME,
    status: FillStatus = FillStatus.CREATED,
    previous_status: FillStatus | None = None,
) -> FillEvent:
    return FillEvent(
        **instrument_base(sequence=sequence, event_time=event_time),
        fill_id="fill-1",
        order_id="ord-1",
        signal_event_id="evt-1",
        signal_time=SIGNAL_TIME,
        order_time=ORDER_TIME,
        side=Side.BUY,
        quantity="100",
        price="4.12",
        gross_amount="412",
        price_observed_at=FILL_TIME - timedelta(seconds=1),
        price_available_at=FILL_TIME,
        status=status,
        previous_status=previous_status,
    )


def test_stock_and_etf_are_supported_but_index_is_rejected() -> None:
    stock = make_signal(instrument_type=TradableInstrumentType.STOCK)
    etf = make_signal(sequence=2, instrument_type=TradableInstrumentType.ETF)

    assert stock.instrument_type is TradableInstrumentType.STOCK
    assert etf.instrument_type is TradableInstrumentType.ETF
    with pytest.raises(ValidationError):
        make_signal(sequence=3, instrument_type="INDEX")


def test_event_requires_aware_time_and_matching_shanghai_trading_day() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        make_signal(event_time=datetime(2026, 8, 28, 9, 31))
    with pytest.raises(ValidationError, match="trading_day"):
        SignalEvent(
            **(
                instrument_base(sequence=1, event_time=SIGNAL_TIME)
                | {"trading_day": date(2026, 8, 27)}
            ),
            direction=SignalDirection.LONG,
            strength="1",
            strategy_version="v1",
            data_version="d1",
        )


@pytest.mark.parametrize(
    ("direction", "strength"),
    [
        (SignalDirection.LONG, "0"),
        (SignalDirection.FLAT, "0.1"),
        (SignalDirection.SHORT, "0.1"),
        (SignalDirection.LONG, "1.1"),
    ],
)
def test_signal_direction_and_strength_are_consistent(
    direction: SignalDirection, strength: str
) -> None:
    with pytest.raises(ValidationError):
        SignalEvent(
            **instrument_base(sequence=1, event_time=SIGNAL_TIME),
            direction=direction,
            strength=strength,
            strategy_version="v1",
            data_version="d1",
        )


def test_exact_decimals_reject_float_and_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="binary floats"):
        SignalEvent(
            **instrument_base(sequence=1, event_time=SIGNAL_TIME),
            direction=SignalDirection.LONG,
            strength=0.5,
            strategy_version="v1",
            data_version="d1",
        )
    with pytest.raises(ValidationError, match="finite"):
        SignalEvent(
            **instrument_base(sequence=1, event_time=SIGNAL_TIME),
            direction=SignalDirection.LONG,
            strength="NaN",
            strategy_version="v1",
            data_version="d1",
        )


def test_price_fields_are_all_or_none_and_cannot_use_future_information() -> None:
    payload = instrument_base(sequence=1, event_time=SIGNAL_TIME)
    common = {
        "direction": SignalDirection.LONG,
        "strength": "1",
        "strategy_version": "v1",
        "data_version": "d1",
    }
    with pytest.raises(ValidationError, match="must be set together"):
        SignalEvent(**payload, **common, reference_price="10")
    with pytest.raises(ValidationError, match="future price"):
        SignalEvent(
            **payload,
            **common,
            reference_price="10",
            price_observed_at=SIGNAL_TIME,
            price_available_at=SIGNAL_TIME + timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="cannot precede"):
        SignalEvent(
            **payload,
            **common,
            reference_price="10",
            price_observed_at=SIGNAL_TIME,
            price_available_at=SIGNAL_TIME - timedelta(seconds=1),
        )


def test_target_links_to_prior_signal_and_has_one_unsigned_magnitude() -> None:
    target = TargetEvent(
        **instrument_base(sequence=2, event_time=ORDER_TIME),
        signal_event_id="evt-1",
        signal_time=SIGNAL_TIME,
        direction=SignalDirection.LONG,
        target_weight="0.6",
    )
    assert target.target_weight == Decimal("0.6")

    with pytest.raises(ValidationError, match="exactly one"):
        TargetEvent(
            **instrument_base(sequence=2, event_time=ORDER_TIME),
            signal_event_id="evt-1",
            signal_time=SIGNAL_TIME,
            direction=SignalDirection.LONG,
        )
    with pytest.raises(ValidationError, match="cannot precede"):
        TargetEvent(
            **instrument_base(sequence=2, event_time=SIGNAL_TIME),
            signal_event_id="evt-1",
            signal_time=ORDER_TIME,
            direction=SignalDirection.LONG,
            target_quantity="100",
        )
    with pytest.raises(ValidationError, match="must be positive"):
        TargetEvent(
            **instrument_base(sequence=2, event_time=ORDER_TIME),
            signal_event_id="evt-1",
            signal_time=SIGNAL_TIME,
            direction=SignalDirection.LONG,
            target_quantity="0",
        )


def test_order_status_transitions_and_fill_quantities_are_validated() -> None:
    created = make_order()
    submitted = make_order(
        sequence=3,
        event_time=ORDER_TIME + timedelta(seconds=1),
        status=OrderStatus.SUBMITTED,
        previous_status=OrderStatus.CREATED,
    )
    partial = make_order(
        sequence=4,
        event_time=ORDER_TIME + timedelta(seconds=2),
        status=OrderStatus.PARTIALLY_FILLED,
        previous_status=OrderStatus.ACCEPTED,
        filled_quantity="40",
    )
    assert (created.status, submitted.status, partial.filled_quantity) == (
        OrderStatus.CREATED,
        OrderStatus.SUBMITTED,
        Decimal("40"),
    )

    with pytest.raises(ValidationError, match="invalid order transition"):
        make_order(status=OrderStatus.FILLED, previous_status=OrderStatus.CREATED)
    with pytest.raises(ValidationError, match="partial filled_quantity"):
        make_order(
            status=OrderStatus.PARTIALLY_FILLED,
            previous_status=OrderStatus.ACCEPTED,
            filled_quantity="100",
        )
    with pytest.raises(ValidationError, match="initial order"):
        make_order(status=OrderStatus.SUBMITTED)


def test_order_instruction_and_amount_constraints() -> None:
    payload = make_order().model_dump()
    with pytest.raises(ValidationError, match="positive limit_price"):
        OrderEvent.model_validate(payload | {"limit_price": None})
    with pytest.raises(ValidationError, match="market orders"):
        OrderEvent.model_validate(payload | {"order_type": "MARKET"})
    with pytest.raises(ValidationError, match="quantity must be positive"):
        OrderEvent.model_validate(payload | {"quantity": "0"})
    with pytest.raises(ValidationError, match="cannot precede its signal"):
        OrderEvent.model_validate(payload | {"event_time": SIGNAL_TIME - timedelta(seconds=1)})


def test_fill_rejects_time_travel_future_prices_and_bad_notional() -> None:
    fill = make_fill()
    assert fill.gross_amount == Decimal("412")

    payload = fill.model_dump()
    with pytest.raises(ValidationError, match="cannot precede its order"):
        FillEvent.model_validate(payload | {"event_time": ORDER_TIME - timedelta(seconds=1)})
    with pytest.raises(ValidationError, match="before the order"):
        FillEvent.model_validate(payload | {"price_observed_at": SIGNAL_TIME})
    with pytest.raises(ValidationError, match="multiplied by price"):
        FillEvent.model_validate(payload | {"gross_amount": "411"})
    with pytest.raises(ValidationError, match="future price"):
        FillEvent.model_validate(payload | {"price_available_at": FILL_TIME + timedelta(seconds=1)})


def test_fill_status_machine_rejects_invalid_transition() -> None:
    confirmed = make_fill(
        sequence=4,
        event_time=FILL_TIME + timedelta(seconds=1),
        status=FillStatus.CONFIRMED,
        previous_status=FillStatus.CREATED,
    )
    assert confirmed.status is FillStatus.CONFIRMED
    with pytest.raises(ValidationError, match="invalid fill transition"):
        make_fill(status=FillStatus.SETTLED, previous_status=FillStatus.CREATED)
    with pytest.raises(ValidationError, match="initial fill"):
        make_fill(status=FillStatus.CONFIRMED)


def test_public_state_helpers_fail_closed() -> None:
    ensure_order_transition(OrderStatus.ACCEPTED, OrderStatus.FILLED)
    ensure_fill_transition(FillStatus.CONFIRMED, FillStatus.SETTLED)
    with pytest.raises(InvalidStateTransition):
        ensure_order_transition(OrderStatus.FILLED, OrderStatus.CANCELED)
    with pytest.raises(InvalidStateTransition):
        ensure_fill_transition(FillStatus.CANCELED, FillStatus.CONFIRMED)


def test_fee_rule_selection_honors_effective_boundary_and_exact_version() -> None:
    old = FeeRule(
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        effective_from=date(2020, 1, 1),
        version="stock-sell-v1",
        commission_rate="0.0003",
        minimum_commission="5",
        stamp_duty_rate="0.001",
    )
    new = old.model_copy(
        update={
            "effective_from": date(2026, 8, 28),
            "version": "stock-sell-v2",
            "stamp_duty_rate": Decimal("0.0005"),
        }
    )
    etf = FeeRule(
        instrument_type=TradableInstrumentType.ETF,
        side=Side.SELL,
        effective_from=date(2020, 1, 1),
        version="etf-sell-v1",
        commission_rate="0.0003",
        minimum_commission="5",
        stamp_duty_rate="0",
    )
    stock_buy = FeeRule(
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.BUY,
        effective_from=date(2020, 1, 1),
        version="stock-buy-v1",
        commission_rate="0.0003",
        minimum_commission="5",
        stamp_duty_rate="0",
    )
    book = FeeRuleBook([old, new, etf, stock_buy])

    assert (
        book.select(
            instrument_type=TradableInstrumentType.STOCK,
            side=Side.SELL,
            trading_day=date(2026, 8, 27),
        ).version
        == "stock-sell-v1"
    )
    assert (
        book.select(
            instrument_type=TradableInstrumentType.STOCK,
            side=Side.SELL,
            trading_day=date(2026, 8, 28),
        ).version
        == "stock-sell-v2"
    )
    assert (
        book.select(
            instrument_type=TradableInstrumentType.STOCK,
            side=Side.SELL,
            trading_day=date(2026, 8, 28),
            version="stock-sell-v1",
        )
        is old
    )
    assert (
        book.select(
            instrument_type=TradableInstrumentType.ETF,
            side=Side.SELL,
            trading_day=DAY,
        ).stamp_duty_rate
        == 0
    )
    assert (
        book.select(
            instrument_type=TradableInstrumentType.STOCK,
            side=Side.BUY,
            trading_day=DAY,
        ).stamp_duty_rate
        == 0
    )


def test_fee_rule_book_fails_closed_for_missing_or_ambiguous_rules() -> None:
    rule = FeeRule(
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.BUY,
        effective_from=DAY,
        version="v1",
    )
    book = FeeRuleBook([rule])
    with pytest.raises(FeeRuleNotFoundError):
        book.select(
            instrument_type=TradableInstrumentType.STOCK,
            side=Side.BUY,
            trading_day=DAY - timedelta(days=1),
        )
    with pytest.raises(FeeRuleNotFoundError):
        book.select(
            instrument_type=TradableInstrumentType.STOCK,
            side=Side.BUY,
            trading_day=DAY,
            version="missing",
        )
    with pytest.raises(ValueError, match="ambiguous"):
        FeeRuleBook([rule, rule.model_copy(update={"version": "v2"})])
    with pytest.raises(ValueError, match="version must be unique"):
        FeeRuleBook([rule, rule.model_copy(update={"effective_from": DAY + timedelta(days=1)})])


def test_fee_assessment_and_event_are_exact_and_auditable() -> None:
    rule = FeeRule(
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        effective_from=DAY,
        version="v2",
        commission_rate="0.0003",
        minimum_commission="5",
        stamp_duty_rate="0.0005",
    )
    fee = rule.assess("10000")
    assert fee.commission == Decimal("5")
    assert fee.stamp_duty == Decimal("5")
    assert fee.total_amount == Decimal("10")
    with pytest.raises(ValueError, match="exact decimal"):
        rule.assess(10000.0)
    with pytest.raises(ValueError, match="valid decimal"):
        rule.assess("not-a-number")

    event = FeeEvent(
        **instrument_base(sequence=4, event_time=FILL_TIME),
        fill_id="fill-1",
        fill_time=FILL_TIME,
        side=Side.SELL,
        fee_rule_version=rule.version,
        commission=fee.commission,
        stamp_duty=fee.stamp_duty,
        transfer_fee=fee.transfer_fee,
        other_fee=fee.other_fee,
        total_amount=fee.total_amount,
        currency="cny",
    )
    assert event.currency == "CNY"
    with pytest.raises(ValidationError, match="component sum"):
        FeeEvent.model_validate(event.model_dump() | {"total_amount": "9"})


def test_position_and_cash_events_use_unsigned_quantities_and_explicit_direction() -> None:
    position = PositionEvent(
        **instrument_base(sequence=5, event_time=FILL_TIME),
        source_event_id="fill-1",
        quantity="100",
        sellable_quantity="0",
        average_cost="4.13",
        market_price="4.12",
        price_observed_at=FILL_TIME,
        price_available_at=FILL_TIME,
    )
    cash = CashEvent(
        **event_base(sequence=6, event_time=FILL_TIME),
        account_id="paper-1",
        source_event_id="fill-1",
        direction=CashDirection.DEBIT,
        amount="422",
        balance_after="9578",
    )
    assert position.sellable_quantity == 0
    assert cash.direction is CashDirection.DEBIT
    with pytest.raises(ValidationError, match="cannot exceed"):
        PositionEvent.model_validate(position.model_dump() | {"sellable_quantity": "101"})
    with pytest.raises(ValidationError, match="must be positive"):
        CashEvent.model_validate(cash.model_dump() | {"amount": "-1"})


def test_market_rule_protocol_carries_stock_etf_mechanics_not_strategy_logic() -> None:
    class EtfRule:
        rule_id = "cn-etf"
        version = "v1"
        effective_from = DAY
        instrument_type = TradableInstrumentType.ETF

        def validate_order(self, order: OrderEvent) -> None:
            if order.quantity % 100:
                raise ValueError("ETF buy quantity must be a board lot")

        def sellable_on(self, *, acquired_on: date) -> date:
            return acquired_on

    class StockRule(EtfRule):
        rule_id = "cn-stock"
        instrument_type = TradableInstrumentType.STOCK

        def sellable_on(self, *, acquired_on: date) -> date:
            return acquired_on + timedelta(days=1)

    etf_rule = EtfRule()
    stock_rule = StockRule()
    assert isinstance(etf_rule, MarketRule)
    assert isinstance(stock_rule, MarketRule)
    etf_rule.validate_order(make_order())
    assert etf_rule.sellable_on(acquired_on=DAY) == DAY
    assert stock_rule.sellable_on(acquired_on=DAY) == DAY + timedelta(days=1)


def test_canonical_serialization_hash_and_stream_round_trip() -> None:
    signal = make_signal()
    order = make_order()
    encoded = serialize_event(signal)
    restored = deserialize_event(encoded)

    assert restored == signal
    assert serialize_event(restored) == encoded
    assert event_hash(restored) == event_hash(signal)
    assert '"event_type":"SIGNAL"' in encoded
    # Equivalent instants and numerically equivalent Decimals have one canonical hash.
    utc_copy = signal.model_copy(
        update={
            "event_time": signal.event_time.astimezone(UTC),
            "price_observed_at": signal.price_observed_at.astimezone(UTC)
            if signal.price_observed_at
            else None,
            "price_available_at": signal.price_available_at.astimezone(UTC)
            if signal.price_available_at
            else None,
            "reference_price": Decimal("4.123"),
        }
    )
    assert event_hash(utc_copy) == event_hash(signal)

    stream = serialize_event_stream([signal, order])
    assert deserialize_event_stream(stream) == (signal, order)


def test_every_event_variant_can_be_serialized_and_restored() -> None:
    signal = make_signal()
    target = TargetEvent(
        **instrument_base(sequence=2, event_time=ORDER_TIME),
        signal_event_id=signal.event_id,
        signal_time=signal.event_time,
        direction=SignalDirection.LONG,
        target_quantity="100",
    )
    order = make_order(sequence=3)
    fill = make_fill(sequence=4)
    fee = FeeEvent(
        **instrument_base(sequence=5, event_time=FILL_TIME),
        fill_id=fill.fill_id,
        fill_time=fill.event_time,
        side=Side.BUY,
        fee_rule_version="etf-buy-v1",
        commission="5",
        total_amount="5",
    )
    position = PositionEvent(
        **instrument_base(sequence=6, event_time=FILL_TIME),
        source_event_id=fill.event_id,
        quantity="100",
        sellable_quantity="100",
        average_cost="4.17",
    )
    cash = CashEvent(
        **event_base(sequence=7, event_time=FILL_TIME),
        account_id="paper-1",
        source_event_id=fill.event_id,
        direction=CashDirection.DEBIT,
        amount="417",
        balance_after="9583",
    )

    events = (signal, target, order, fill, fee, position, cash)
    assert tuple(deserialize_event(serialize_event(event)) for event in events) == events


def test_replay_stream_rejects_mixed_runs_sequence_and_time_regression() -> None:
    signal = make_signal()
    order = make_order()
    assert validate_replay_stream([]) == ()
    with pytest.raises(ValueError, match="run_id"):
        validate_replay_stream([signal, order.model_copy(update={"run_id": "run-2"})])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_replay_stream([signal, order.model_copy(update={"sequence": 1})])
    with pytest.raises(ValueError, match="monotonic"):
        validate_replay_stream(
            [signal, order.model_copy(update={"event_time": SIGNAL_TIME - timedelta(seconds=1)})]
        )
    with pytest.raises(ValueError, match="event_id"):
        validate_replay_stream([signal, order.model_copy(update={"event_id": signal.event_id})])


def test_replay_stream_requires_contiguous_order_and_fill_state() -> None:
    created_order = make_order()
    submitted_order = make_order(
        sequence=3,
        event_time=ORDER_TIME + timedelta(seconds=1),
        status=OrderStatus.SUBMITTED,
        previous_status=OrderStatus.CREATED,
    )
    created_fill = make_fill(sequence=4, event_time=FILL_TIME)
    confirmed_fill = make_fill(
        sequence=5,
        event_time=FILL_TIME + timedelta(seconds=1),
        status=FillStatus.CONFIRMED,
        previous_status=FillStatus.CREATED,
    )
    assert validate_replay_stream(
        [created_order, submitted_order, created_fill, confirmed_fill]
    ) == (created_order, submitted_order, created_fill, confirmed_fill)

    with pytest.raises(ValueError, match="order previous_status"):
        validate_replay_stream([submitted_order])
    with pytest.raises(ValueError, match="fill previous_status"):
        validate_replay_stream([confirmed_fill])
