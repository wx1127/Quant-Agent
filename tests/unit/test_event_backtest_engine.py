from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_agent.backtest.contracts import (
    CashDirection,
    CashEvent,
    FeeEvent,
    FillEvent,
    FillStatus,
    OrderEvent,
    OrderStatus,
    OrderType,
    PositionEvent,
    Side,
    SignalDirection,
    SignalEvent,
    TargetEvent,
    TradableInstrumentType,
)
from quant_agent.backtest.engine import (
    BacktestEngineError,
    EventBacktestConfig,
    EventDrivenBacktestEngine,
    EventLinkError,
    EventOrderError,
    InitialPosition,
    LedgerInvariantError,
    MarketRuleNotFoundError,
)
from quant_agent.backtest.rules import FeeRule, FeeRuleBook

DAY = date(2026, 8, 28)
T0 = datetime(2026, 8, 28, 1, 30, tzinfo=UTC)


@dataclass
class TestMarketRule:
    __test__ = False

    rule_id: str = "test-market"
    version: str = "v1"
    effective_from: date = date(2020, 1, 1)
    instrument_type: TradableInstrumentType = TradableInstrumentType.ETF
    settlement_days: int = 0
    lot_size: Decimal = Decimal(1)
    blocked: bool = False
    validated_orders: list[str] = field(default_factory=list)

    def validate_order(self, order: OrderEvent) -> None:
        self.validated_orders.append(order.order_id)
        if self.blocked:
            raise ValueError("instrument is not tradable")
        if order.quantity % self.lot_size:
            raise ValueError("quantity violates the active market rule")

    def sellable_on(self, *, acquired_on: date) -> date:
        return acquired_on + timedelta(days=self.settlement_days)


def event_base(*, sequence: int, event_time: datetime, event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "run_id": "run-1",
        "sequence": sequence,
        "event_time": event_time,
        "trading_day": DAY,
    }


def instrument_base(*, sequence: int, event_time: datetime, event_id: str) -> dict[str, object]:
    return event_base(sequence=sequence, event_time=event_time, event_id=event_id) | {
        "instrument_id": "510300.SH",
        "instrument_type": TradableInstrumentType.ETF,
    }


def make_target(
    *,
    sequence: int = 1,
    event_time: datetime = T0,
    direction: SignalDirection = SignalDirection.LONG,
    quantity: str = "100",
) -> TargetEvent:
    return TargetEvent(
        **instrument_base(sequence=sequence, event_time=event_time, event_id=f"evt-{sequence}"),
        signal_event_id="signal-1",
        signal_time=T0 - timedelta(seconds=1),
        direction=direction,
        target_quantity=quantity,
    )


def make_created_order(
    *,
    sequence: int = 2,
    event_time: datetime = T0 + timedelta(seconds=1),
    side: Side = Side.BUY,
    quantity: str = "100",
    order_type: OrderType = OrderType.LIMIT,
    limit_price: str | None = "10",
) -> OrderEvent:
    return OrderEvent(
        **instrument_base(sequence=sequence, event_time=event_time, event_id=f"evt-{sequence}"),
        order_id="order-1",
        signal_event_id="signal-1",
        signal_time=T0 - timedelta(seconds=1),
        side=side,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        status=OrderStatus.CREATED,
    )


def transition_order(
    source: OrderEvent,
    *,
    sequence: int,
    status: OrderStatus,
    previous_status: OrderStatus,
    filled_quantity: str = "0",
) -> OrderEvent:
    payload = source.model_dump(mode="python") | {
        "event_id": f"evt-{sequence}",
        "sequence": sequence,
        "event_time": T0 + timedelta(seconds=sequence - 1),
        "status": status,
        "previous_status": previous_status,
        "filled_quantity": filled_quantity,
    }
    return OrderEvent.model_validate(payload)


def accepted_order(created: OrderEvent) -> tuple[OrderEvent, OrderEvent]:
    submitted = transition_order(
        created,
        sequence=3,
        status=OrderStatus.SUBMITTED,
        previous_status=OrderStatus.CREATED,
    )
    accepted = transition_order(
        created,
        sequence=4,
        status=OrderStatus.ACCEPTED,
        previous_status=OrderStatus.SUBMITTED,
    )
    return submitted, accepted


def make_fill(
    order: OrderEvent,
    *,
    sequence: int = 5,
    price: str = "10",
    quantity: str = "100",
) -> FillEvent:
    event_time = T0 + timedelta(seconds=sequence - 1)
    return FillEvent(
        **instrument_base(sequence=sequence, event_time=event_time, event_id=f"evt-{sequence}"),
        fill_id="fill-1",
        order_id=order.order_id,
        signal_event_id=order.signal_event_id,
        signal_time=order.signal_time,
        order_time=order.event_time,
        side=order.side,
        quantity=quantity,
        price=price,
        gross_amount=Decimal(quantity) * Decimal(price),
        price_observed_at=event_time,
        price_available_at=event_time,
        status=FillStatus.CREATED,
    )


def confirm_fill(created: FillEvent, *, sequence: int = 6) -> FillEvent:
    return FillEvent.model_validate(
        created.model_dump(mode="python")
        | {
            "event_id": f"evt-{sequence}",
            "sequence": sequence,
            "event_time": T0 + timedelta(seconds=sequence - 1),
            "status": FillStatus.CONFIRMED,
            "previous_status": FillStatus.CREATED,
        }
    )


def make_cash(
    *,
    sequence: int,
    source_event_id: str,
    direction: CashDirection,
    amount: str,
    balance_after: str,
) -> CashEvent:
    return CashEvent(
        **event_base(
            sequence=sequence,
            event_time=T0 + timedelta(seconds=sequence - 1),
            event_id=f"evt-{sequence}",
        ),
        account_id="paper-1",
        source_event_id=source_event_id,
        direction=direction,
        amount=amount,
        balance_after=balance_after,
    )


def make_position(
    *,
    sequence: int,
    source_event_id: str,
    quantity: str,
    sellable_quantity: str,
    average_cost: str,
) -> PositionEvent:
    return PositionEvent(
        **instrument_base(
            sequence=sequence,
            event_time=T0 + timedelta(seconds=sequence - 1),
            event_id=f"evt-{sequence}",
        ),
        source_event_id=source_event_id,
        quantity=quantity,
        sellable_quantity=sellable_quantity,
        average_cost=average_cost,
    )


def buy_stream() -> tuple[object, ...]:
    target = make_target()
    created = make_created_order()
    submitted, accepted = accepted_order(created)
    fill = make_fill(created)
    confirmed = confirm_fill(fill)
    fill_cash = make_cash(
        sequence=7,
        source_event_id=confirmed.event_id,
        direction=CashDirection.DEBIT,
        amount="1000",
        balance_after="1000",
    )
    fill_position = make_position(
        sequence=8,
        source_event_id=confirmed.event_id,
        quantity="100",
        sellable_quantity="100",
        average_cost="10",
    )
    fee = FeeEvent(
        **instrument_base(
            sequence=9,
            event_time=T0 + timedelta(seconds=8),
            event_id="evt-9",
        ),
        fill_id=fill.fill_id,
        fill_time=fill.event_time,
        side=Side.BUY,
        fee_rule_version="buy-v1",
        commission="5",
        total_amount="5",
    )
    fee_cash = make_cash(
        sequence=10,
        source_event_id=fee.event_id,
        direction=CashDirection.DEBIT,
        amount="5",
        balance_after="995",
    )
    fee_position = make_position(
        sequence=11,
        source_event_id=fee.event_id,
        quantity="100",
        sellable_quantity="100",
        average_cost="10.05",
    )
    filled = transition_order(
        created,
        sequence=12,
        status=OrderStatus.FILLED,
        previous_status=OrderStatus.ACCEPTED,
        filled_quantity="100",
    )
    return (
        target,
        created,
        submitted,
        accepted,
        fill,
        confirmed,
        fill_cash,
        fill_position,
        fee,
        fee_cash,
        fee_position,
        filled,
    )


def config(**updates: object) -> EventBacktestConfig:
    return EventBacktestConfig.model_validate(
        {
            "run_id": "run-1",
            "account_id": "paper-1",
            "initial_cash": "2000",
            **updates,
        }
    )


def fee_book() -> FeeRuleBook:
    return FeeRuleBook(
        [
            FeeRule(
                instrument_type=TradableInstrumentType.ETF,
                side=Side.BUY,
                effective_from=date(2020, 1, 1),
                version="buy-v1",
                commission_rate="0",
                minimum_commission="5",
            )
        ]
    )


def test_complete_buy_replay_is_exact_audited_and_deterministic() -> None:
    events = buy_stream()
    first_rule = TestMarketRule()
    engine = EventDrivenBacktestEngine(config(), [first_rule], fee_book())
    first = engine.run(events, require_complete=True)  # type: ignore[arg-type]

    assert first.cash_balance == Decimal("995")
    assert first.position("510300.SH") is not None
    assert first.position("510300.SH").quantity == Decimal("100")  # type: ignore[union-attr]
    assert first.position("510300.SH").average_cost == Decimal("10.05")  # type: ignore[union-attr]
    assert first.orders[0].status is OrderStatus.FILLED
    assert first.orders[0].confirmed_quantity == Decimal("100")
    assert first_rule.validated_orders == ["order-1"]

    replayed = EventDrivenBacktestEngine(config(), [TestMarketRule()], fee_book()).run(
        events,  # type: ignore[arg-type]
        require_complete=True,
    )
    assert replayed == first
    assert replayed.event_log_hash == first.event_log_hash
    assert '"event_type":"FILL"' in first.event_log


def test_limit_matching_and_future_price_defense_fail_atomically() -> None:
    target = make_target()
    created = make_created_order()
    submitted, accepted = accepted_order(created)
    above_limit = make_fill(created, price="10.01")
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])

    with pytest.raises(LedgerInvariantError, match="limit fill"):
        engine.process_all([target, created, submitted, accepted, above_limit])
    assert engine.result().event_count == 0
    assert engine.cash_balance == Decimal("2000")

    valid_fill = make_fill(created)
    future_copy = valid_fill.model_copy(
        update={"price_available_at": valid_fill.event_time + timedelta(seconds=1)}
    )
    with pytest.raises(ValidationError, match="future price"):
        engine.process_all([target, created, submitted, accepted, future_copy])
    assert engine.result().event_count == 0


def test_strict_clock_and_event_failure_leave_prior_state_untouched() -> None:
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    target = make_target()
    committed = engine.process(target)
    regressed = make_created_order(sequence=1, event_time=T0).model_copy(
        update={"event_id": "regressed-order"}
    )

    with pytest.raises(EventOrderError, match="strictly increasing"):
        engine.process(regressed)
    assert engine.result() == committed

    wrong_run = make_created_order().model_copy(update={"run_id": "other-run"})
    with pytest.raises(EventOrderError, match="run_id"):
        engine.process(wrong_run)
    assert engine.result() == committed


def test_negative_cash_is_denied_unless_explicitly_enabled() -> None:
    target = make_target()
    created = make_created_order()
    submitted, accepted = accepted_order(created)
    fill = make_fill(created)
    confirmed = confirm_fill(fill)
    prefix = [target, created, submitted, accepted, fill]

    denied = EventDrivenBacktestEngine(config(initial_cash="999"), [TestMarketRule()])
    denied.process_all(prefix)
    before = denied.result()
    with pytest.raises(LedgerInvariantError, match="negative balance"):
        denied.process(confirmed)
    assert denied.result() == before

    allowed = EventDrivenBacktestEngine(
        config(initial_cash="999", allow_negative_cash=True),
        [TestMarketRule()],
    )
    allowed.process_all([*prefix, confirmed])
    assert allowed.cash_balance == Decimal("-1")


def sell_prefix(*, target_quantity: str = "0", order_quantity: str = "100") -> list[object]:
    direction = SignalDirection.FLAT if target_quantity == "0" else SignalDirection.SHORT
    target = make_target(direction=direction, quantity=target_quantity)
    created = make_created_order(side=Side.SELL, quantity=order_quantity)
    submitted, accepted = accepted_order(created)
    fill = make_fill(created, quantity=order_quantity)
    confirmed = confirm_fill(fill)
    return [target, created, submitted, accepted, fill, confirmed]


def test_unsellable_and_short_inventory_each_require_an_explicit_policy() -> None:
    opening = InitialPosition(
        instrument_id="510300.SH",
        instrument_type=TradableInstrumentType.ETF,
        quantity="100",
        sellable_quantity="0",
        average_cost="8",
    )
    denied = EventDrivenBacktestEngine(
        config(initial_positions=(opening,)),
        [TestMarketRule()],
    )
    with pytest.raises(LedgerInvariantError, match="sellable_quantity"):
        denied.process_all(sell_prefix())  # type: ignore[arg-type]
    assert denied.result().event_count == 0

    unsettled_allowed = EventDrivenBacktestEngine(
        config(initial_positions=(opening,), allow_unsettled_sales=True),
        [TestMarketRule()],
    )
    result = unsettled_allowed.process_all(sell_prefix())  # type: ignore[arg-type]
    assert result.cash_balance == Decimal("3000")
    assert result.position("510300.SH") is None

    short_events = sell_prefix(target_quantity="50", order_quantity="150")
    short_allowed = EventDrivenBacktestEngine(
        config(
            initial_positions=(opening.model_copy(update={"sellable_quantity": Decimal("100")}),),
            allow_short_sales=True,
        ),
        [TestMarketRule()],
    )
    short_result = short_allowed.process_all(short_events)  # type: ignore[arg-type]
    assert short_result.position("510300.SH").quantity == Decimal("-50")  # type: ignore[union-attr]


def test_market_rule_owns_tradeability_and_quantity_constraints() -> None:
    rule = TestMarketRule(lot_size=Decimal("100"))
    engine = EventDrivenBacktestEngine(config(), [rule])
    invalid_target = make_target(quantity="50")
    invalid_order = make_created_order(quantity="50")
    with pytest.raises(ValueError, match="quantity violates"):
        engine.process_all([invalid_target, invalid_order])
    assert engine.result().event_count == 0

    blocked = EventDrivenBacktestEngine(config(), [TestMarketRule(blocked=True)])
    with pytest.raises(ValueError, match="not tradable"):
        blocked.process_all([make_target(), make_created_order()])
    assert blocked.result().event_count == 0


def test_links_and_order_fill_totals_fail_closed() -> None:
    target = make_target()
    created = make_created_order()
    submitted, accepted = accepted_order(created)
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    engine.process_all([target, created, submitted, accepted])
    before = engine.result()

    wrong_order = make_fill(created).model_copy(update={"order_id": "missing"})
    with pytest.raises(EventLinkError, match="unknown order"):
        engine.process(wrong_order)
    assert engine.result() == before

    too_large = make_fill(created, quantity="101")
    with pytest.raises(LedgerInvariantError, match="exceed the order"):
        engine.process(too_large)
    assert engine.result() == before


def test_tampered_cash_or_position_attestations_do_not_mutate_state() -> None:
    target = make_target()
    created = make_created_order()
    submitted, accepted = accepted_order(created)
    fill = make_fill(created)
    confirmed = confirm_fill(fill)
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    engine.process_all([target, created, submitted, accepted, fill, confirmed])
    before = engine.result()

    wrong_cash = make_cash(
        sequence=7,
        source_event_id=confirmed.event_id,
        direction=CashDirection.DEBIT,
        amount="999",
        balance_after="1",
    )
    with pytest.raises(LedgerInvariantError, match="computed ledger"):
        engine.process(wrong_cash)
    assert engine.result() == before

    wrong_position = make_position(
        sequence=7,
        source_event_id=confirmed.event_id,
        quantity="99",
        sellable_quantity="99",
        average_cost="10",
    )
    with pytest.raises(LedgerInvariantError, match="computed ledger"):
        engine.process(wrong_position)
    assert engine.result() == before


def test_fee_rule_is_pinned_and_missing_attestations_can_be_required() -> None:
    events = list(buy_stream())
    fee = events[8]
    assert isinstance(fee, FeeEvent)
    bad_fee = FeeEvent.model_validate(
        fee.model_dump(mode="python")
        | {
            "commission": Decimal("4"),
            "total_amount": Decimal("4"),
        }
    )
    events[8] = bad_fee
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()], fee_book())
    with pytest.raises(LedgerInvariantError, match="pinned FeeRule"):
        engine.process_all(events)  # type: ignore[arg-type]
    assert engine.result().event_count == 0

    incomplete = buy_stream()[:6]
    strict = EventDrivenBacktestEngine(
        config(require_ledger_events=True),
        [TestMarketRule()],
    )
    with pytest.raises(LedgerInvariantError, match="missing required"):
        strict.run(incomplete)  # type: ignore[arg-type]
    assert strict.result(require_complete=False).event_count == 0

    missing_fee = EventDrivenBacktestEngine(config(), [TestMarketRule()], fee_book())
    with pytest.raises(LedgerInvariantError, match="missing a required fee"):
        missing_fee.run(buy_stream()[:8])  # type: ignore[arg-type]
    assert missing_fee.result(require_complete=False).event_count == 0


def test_fee_cannot_be_recorded_after_its_fill_trading_day() -> None:
    events = buy_stream()
    fee = events[8]
    assert isinstance(fee, FeeEvent)
    next_day = DAY + timedelta(days=1)
    late_fee = FeeEvent.model_validate(
        fee.model_dump(mode="python")
        | {
            "event_time": fee.event_time + timedelta(days=1),
            "trading_day": next_day,
        }
    )
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()], fee_book())
    engine.process_all(events[:8])  # type: ignore[arg-type]
    before = engine.result(require_complete=False)

    with pytest.raises(EventLinkError, match="fill trading day"):
        engine.process(late_fee)
    assert engine.result(require_complete=False) == before


def test_config_rejects_implicit_negative_opening_balances() -> None:
    with pytest.raises(ValidationError, match="negative initial_cash"):
        config(initial_cash="-1")
    short = InitialPosition(
        instrument_id="510300.SH",
        instrument_type=TradableInstrumentType.ETF,
        quantity="-1",
        sellable_quantity="0",
        average_cost="10",
    )
    with pytest.raises(ValidationError, match="requires allow_short_sales"):
        config(initial_positions=(short,))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"instrument_id": " "}, "non-empty"),
        ({"sellable_quantity": "-1"}, "cannot be negative"),
        ({"quantity": "-1", "sellable_quantity": "1"}, "short opening"),
        ({"quantity": "1", "sellable_quantity": "2"}, "cannot exceed"),
        ({"quantity": "0", "sellable_quantity": "0"}, "zero opening"),
    ],
)
def test_initial_position_validation_is_fail_closed(updates: dict[str, str], message: str) -> None:
    payload = {
        "instrument_id": "510300.SH",
        "instrument_type": TradableInstrumentType.ETF,
        "quantity": "1",
        "sellable_quantity": "1",
        "average_cost": "10",
    }
    with pytest.raises(ValidationError, match=message):
        InitialPosition.model_validate(payload | updates)


def test_engine_config_and_market_rule_resolution_reject_ambiguity() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        config(run_id=" ")
    with pytest.raises(ValidationError, match="three-letter"):
        config(currency="yuan")
    opening = InitialPosition(
        instrument_id="510300.SH",
        instrument_type=TradableInstrumentType.ETF,
        quantity="1",
        sellable_quantity="1",
        average_cost="10",
    )
    with pytest.raises(ValidationError, match="unique"):
        config(initial_positions=(opening, opening))
    with pytest.raises(MarketRuleNotFoundError, match="at least one"):
        EventDrivenBacktestEngine(config(), [])
    with pytest.raises(ValueError, match="non-empty"):
        EventDrivenBacktestEngine(config(), [TestMarketRule(rule_id=" ")])
    with pytest.raises(ValueError, match="ambiguous"):
        EventDrivenBacktestEngine(config(), [TestMarketRule(), TestMarketRule(version="v2")])
    with pytest.raises(ValueError, match="identity and version"):
        EventDrivenBacktestEngine(
            config(),
            [
                TestMarketRule(effective_from=date(2020, 1, 1)),
                TestMarketRule(effective_from=date(2021, 1, 1)),
            ],
        )
    future_rule = TestMarketRule(effective_from=DAY + timedelta(days=1))
    engine = EventDrivenBacktestEngine(config(), [future_rule])
    with pytest.raises(MarketRuleNotFoundError, match="no effective"):
        engine.process_all([make_target(), make_created_order()])


def test_target_order_linking_and_sizing_are_strict() -> None:
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    with pytest.raises(LedgerInvariantError, match="SHORT target"):
        engine.process(make_target(direction=SignalDirection.SHORT, quantity="1"))

    with pytest.raises(EventLinkError, match="replayed target"):
        engine.process(make_created_order())

    target = make_target()
    engine.process(target)
    duplicate = make_target(sequence=2, event_time=T0 + timedelta(seconds=1)).model_copy(
        update={"event_id": "duplicate-target"}
    )
    with pytest.raises(EventLinkError, match="only one target"):
        engine.process(duplicate)

    satisfied_opening = InitialPosition(
        instrument_id="510300.SH",
        instrument_type=TradableInstrumentType.ETF,
        quantity="100",
        sellable_quantity="100",
        average_cost="8",
    )
    satisfied = EventDrivenBacktestEngine(
        config(initial_positions=(satisfied_opening,)),
        [TestMarketRule()],
    )
    satisfied.process(make_target())
    with pytest.raises(LedgerInvariantError, match="already satisfied"):
        satisfied.process(make_created_order())

    wrong_size = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    wrong_size.process(make_target())
    with pytest.raises(LedgerInvariantError, match="exactly implement"):
        wrong_size.process(make_created_order(quantity="99"))


def test_order_lifecycle_state_and_instruction_are_immutable() -> None:
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    target = make_target()
    created = make_created_order()
    engine.process_all([target, created])
    before = engine.result()

    skipped = transition_order(
        created,
        sequence=3,
        status=OrderStatus.ACCEPTED,
        previous_status=OrderStatus.SUBMITTED,
    )
    with pytest.raises(EventLinkError, match="previous_status"):
        engine.process(skipped)
    assert engine.result() == before

    changed = transition_order(
        created,
        sequence=3,
        status=OrderStatus.SUBMITTED,
        previous_status=OrderStatus.CREATED,
    ).model_copy(update={"quantity": Decimal("101")})
    with pytest.raises(EventLinkError, match="cannot change"):
        engine.process(changed)

    submitted, accepted = accepted_order(created)
    engine.process_all([submitted, accepted])
    unearned_fill = transition_order(
        created,
        sequence=5,
        status=OrderStatus.FILLED,
        previous_status=OrderStatus.ACCEPTED,
        filled_quantity="100",
    )
    with pytest.raises(LedgerInvariantError, match="confirmed fills"):
        engine.process(unearned_fill)


def test_fill_lifecycle_linking_cancellation_and_reversal_are_safe() -> None:
    target = make_target()
    created = make_created_order()
    submitted, accepted = accepted_order(created)

    not_accepted = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    not_accepted.process_all([target, created])
    with pytest.raises(LedgerInvariantError, match="accepted order"):
        not_accepted.process(make_fill(created))

    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    engine.process_all([target, created, submitted, accepted])
    mismatched = make_fill(created).model_copy(update={"side": Side.SELL})
    with pytest.raises(EventLinkError, match="do not match"):
        engine.process(mismatched)

    fill = make_fill(created)
    engine.process(fill)
    pending_cancel = transition_order(
        created,
        sequence=6,
        status=OrderStatus.CANCELED,
        previous_status=OrderStatus.ACCEPTED,
    )
    with pytest.raises(LedgerInvariantError, match="pending fill"):
        engine.process(pending_cancel)

    changed_confirmation = confirm_fill(fill).model_copy(
        update={"price": Decimal("9"), "gross_amount": Decimal("900")}
    )
    with pytest.raises(EventLinkError, match="cannot change"):
        engine.process(changed_confirmation)

    confirmed = confirm_fill(fill)
    engine.process(confirmed)
    reversed_fill = FillEvent.model_validate(
        fill.model_dump(mode="python")
        | {
            "event_id": "evt-7",
            "sequence": 7,
            "event_time": T0 + timedelta(seconds=6),
            "status": FillStatus.REVERSED,
            "previous_status": FillStatus.CONFIRMED,
        }
    )
    with pytest.raises(LedgerInvariantError, match="compensating"):
        engine.process(reversed_fill)


def test_fee_and_attestation_links_are_not_trusted() -> None:
    unknown_fee = FeeEvent(
        **instrument_base(sequence=1, event_time=T0, event_id="fee-unknown"),
        fill_id="missing",
        fill_time=T0,
        side=Side.BUY,
        fee_rule_version="v1",
        total_amount="0",
    )
    engine = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    with pytest.raises(EventLinkError, match="unknown fill"):
        engine.process(unknown_fee)

    target = make_target()
    created = make_created_order()
    submitted, accepted = accepted_order(created)
    fill = make_fill(created)
    engine.process_all([target, created, submitted, accepted, fill])
    early_fee = FeeEvent(
        **instrument_base(
            sequence=6,
            event_time=T0 + timedelta(seconds=5),
            event_id="early-fee",
        ),
        fill_id=fill.fill_id,
        fill_time=fill.event_time,
        side=Side.BUY,
        fee_rule_version="v1",
        total_amount="0",
    )
    with pytest.raises(LedgerInvariantError, match="confirmed"):
        engine.process(early_fee)

    confirmed = confirm_fill(fill)
    engine.process(confirmed)
    wrong_side = early_fee.model_copy(
        update={
            "event_id": "wrong-side-fee",
            "sequence": 7,
            "event_time": T0 + timedelta(seconds=6),
            "side": Side.SELL,
        }
    )
    with pytest.raises(EventLinkError, match="do not match"):
        engine.process(wrong_side)
    wrong_currency = early_fee.model_copy(
        update={
            "event_id": "wrong-currency-fee",
            "sequence": 7,
            "event_time": T0 + timedelta(seconds=6),
            "currency": "USD",
        }
    )
    with pytest.raises(LedgerInvariantError, match="currency"):
        engine.process(wrong_currency)

    zero_fee = early_fee.model_copy(
        update={
            "event_id": "zero-fee",
            "sequence": 7,
            "event_time": T0 + timedelta(seconds=6),
        }
    )
    engine.process(zero_fee)
    duplicate_fee = zero_fee.model_copy(
        update={
            "event_id": "duplicate-fee",
            "sequence": 8,
            "event_time": T0 + timedelta(seconds=7),
        }
    )
    with pytest.raises(LedgerInvariantError, match="only one fee"):
        engine.process(duplicate_fee)

    no_cash_source = make_cash(
        sequence=8,
        source_event_id="missing",
        direction=CashDirection.DEBIT,
        amount="1",
        balance_after="999",
    )
    with pytest.raises(EventLinkError, match="no unmatched"):
        engine.process(no_cash_source)
    no_position_source = make_position(
        sequence=8,
        source_event_id="missing",
        quantity="100",
        sellable_quantity="100",
        average_cost="10",
    )
    with pytest.raises(EventLinkError, match="no unmatched"):
        engine.process(no_position_source)


def test_partial_sell_and_short_cover_preserve_cost_basis() -> None:
    long_opening = InitialPosition(
        instrument_id="510300.SH",
        instrument_type=TradableInstrumentType.ETF,
        quantity="100",
        sellable_quantity="100",
        average_cost="8",
    )
    target = make_target(quantity="50")
    sell = make_created_order(side=Side.SELL, quantity="50")
    submitted, accepted = accepted_order(sell)
    fill = make_fill(sell, quantity="50")
    confirmed = confirm_fill(fill)
    partial = EventDrivenBacktestEngine(
        config(initial_positions=(long_opening,)),
        [TestMarketRule()],
    ).process_all([target, sell, submitted, accepted, fill, confirmed])
    assert partial.position("510300.SH").quantity == Decimal("50")  # type: ignore[union-attr]
    assert partial.position("510300.SH").average_cost == Decimal("8")  # type: ignore[union-attr]

    short_opening = long_opening.model_copy(
        update={
            "quantity": Decimal("-100"),
            "sellable_quantity": Decimal("0"),
        }
    )
    cover_target = make_target(
        direction=SignalDirection.SHORT,
        quantity="50",
    )
    buy = make_created_order(side=Side.BUY, quantity="50")
    submitted, accepted = accepted_order(buy)
    fill = make_fill(buy, quantity="50")
    confirmed = confirm_fill(fill)
    covered = EventDrivenBacktestEngine(
        config(initial_positions=(short_opening,), allow_short_sales=True),
        [TestMarketRule()],
    ).process_all([cover_target, buy, submitted, accepted, fill, confirmed])
    assert covered.position("510300.SH").quantity == Decimal("-50")  # type: ignore[union-attr]
    assert covered.position("510300.SH").average_cost == Decimal("8")  # type: ignore[union-attr]


def test_public_replay_helpers_and_unsupported_signal_are_deterministic() -> None:
    events = buy_stream()
    engine, result = EventDrivenBacktestEngine.from_replay(
        config=config(),
        market_rules=[TestMarketRule()],
        events=events,  # type: ignore[arg-type]
        fee_rule_book=fee_book(),
        require_complete=True,
    )
    assert engine.config == config()
    assert engine.events == result.events
    with pytest.raises(BacktestEngineError, match="no committed"):
        engine.replay(())

    signal = SignalEvent(
        **instrument_base(sequence=1, event_time=T0, event_id="signal-event"),
        direction=SignalDirection.LONG,
        strength="1",
        strategy_version="v1",
        data_version="d1",
    )
    empty = EventDrivenBacktestEngine(config(), [TestMarketRule()])
    with pytest.raises(BacktestEngineError, match="unsupported"):
        empty.process(signal)
