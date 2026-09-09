"""Contract tests for immutable, credential-free paper execution records."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.config import AppEnvironment, RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.execution.paper.contracts import (
    PAPER_EXECUTION_ENGINE_VERSION,
    PaperAccountState,
    PaperExecutionConfig,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperFill,
    PaperMatchAttempt,
    PaperNoFillReason,
    PaperOrder,
    PaperOrderPolicy,
    PaperOrderStatus,
    PaperPosition,
    PaperPositionLot,
)
from quant_agent.execution.paper.transition import replay_execution_transition
from quant_agent.regime.contracts import stable_hash

AS_OF = datetime(2026, 8, 28, 10, 0, tzinfo=SHANGHAI_TZ)
SOURCE_AS_OF = AS_OF - timedelta(hours=1)
PROCESSED_AT = AS_OF + timedelta(minutes=5)
TRADING_DAY = AS_OF.date()
NEXT_TRADING_DAY = TRADING_DAY + timedelta(days=1)


def _digest(name: str) -> str:
    return stable_hash({"fixture": name})


def _lot(
    *,
    lot_id: str = "lot-a",
    instrument_id: str = "CN.SSE.600000",
    instrument_type: TradableInstrumentType = TradableInstrumentType.STOCK,
    quantity: Decimal = Decimal(100),
    cost_basis: Decimal = Decimal(1000),
    acquired_on: date = TRADING_DAY - timedelta(days=1),
    sellable_on: date = TRADING_DAY,
    externally_frozen: Decimal = Decimal(0),
    source_id: str | None = None,
) -> PaperPositionLot:
    return PaperPositionLot.build(
        lot_id=lot_id,
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        quantity=quantity,
        cost_basis=cost_basis,
        acquired_on=acquired_on,
        sellable_on=sellable_on,
        externally_frozen=externally_frozen,
        source_id=source_id or f"source-{lot_id}",
    )


def _position(
    *,
    as_of: datetime = AS_OF,
    instrument_id: str = "CN.SSE.600000",
    reserved_sell_quantity: Decimal = Decimal(20),
) -> PaperPosition:
    settled = _lot(externally_frozen=Decimal(10))
    unsettled = _lot(
        lot_id="lot-b",
        quantity=Decimal(100),
        cost_basis=Decimal(2000),
        acquired_on=TRADING_DAY,
        sellable_on=NEXT_TRADING_DAY,
    )
    return PaperPosition.build(
        as_of=as_of,
        instrument_id=instrument_id,
        instrument_type=TradableInstrumentType.STOCK,
        lots=(unsettled, settled),
        reserved_sell_quantity=reserved_sell_quantity,
    )


def _order(
    *,
    order_id: str = "order-buy",
    batch_hash: str | None = None,
    instrument_id: str = "CN.SSE.510300",
    instrument_type: TradableInstrumentType = TradableInstrumentType.ETF,
    side: Side = Side.BUY,
    status: PaperOrderStatus = PaperOrderStatus.ACCEPTED,
    quantity: Decimal = Decimal(100),
    filled_quantity: Decimal | None = None,
    cash_reserved: Decimal | None = None,
    sell_reserved_quantity: Decimal | None = None,
    is_full_liquidation: bool = False,
    created_at: datetime = AS_OF,
    expires_at: datetime = AS_OF + timedelta(hours=1),
    last_attempt_id: str | None = None,
) -> PaperOrder:
    active = status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.PARTIALLY_FILLED}
    if filled_quantity is None:
        if status is PaperOrderStatus.PARTIALLY_FILLED:
            filled_quantity = Decimal(40)
        elif status is PaperOrderStatus.FILLED:
            filled_quantity = quantity
        else:
            filled_quantity = Decimal(0)
    remaining = quantity - filled_quantity
    if cash_reserved is None:
        cash_reserved = Decimal(1200) if side is Side.BUY and active else Decimal(0)
    if sell_reserved_quantity is None:
        sell_reserved_quantity = remaining if side is Side.SELL and active else Decimal(0)
    if last_attempt_id is None and filled_quantity > 0:
        last_attempt_id = "attempt-1"
    return PaperOrder.build(
        order_id=order_id,
        batch_hash=batch_hash or _digest(f"batch-{order_id}"),
        line_hash=_digest(f"line-{order_id}"),
        decision_id="decision-20260828",
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        side=side,
        quantity=quantity,
        filled_quantity=filled_quantity,
        is_full_liquidation=is_full_liquidation,
        status=status,
        created_at=created_at,
        expires_at=expires_at,
        cash_reserved=cash_reserved,
        sell_reserved_quantity=sell_reserved_quantity,
        market_rule_version="cn-market-rule-v1",
        market_rule_hash=_digest("market-rule"),
        fee_rule_version="cn-fee-rule-v1",
        fee_rule_hash=_digest("fee-rule"),
        slippage_model_version="cn-slippage-v1",
        slippage_model_hash=_digest("slippage-model"),
        last_attempt_id=last_attempt_id,
    )


def _attempt(
    *,
    attempt_id: str = "attempt-1",
    order_id: str = "order-buy",
    requested_quantity: Decimal = Decimal(100),
    filled_quantity: Decimal = Decimal(100),
    status: PaperOrderStatus = PaperOrderStatus.FILLED,
    no_fill_reason: PaperNoFillReason | None = None,
    execution_price: Decimal | None = Decimal("10.01"),
    participation_rate: Decimal = Decimal("0.01"),
    attempted_at: datetime = AS_OF + timedelta(minutes=2),
    market_rule_version: str = "cn-market-rule-v1",
    market_rule_hash: str | None = None,
    slippage_model_version: str = "cn-slippage-v1",
    slippage_model_hash: str | None = None,
) -> PaperMatchAttempt:
    return PaperMatchAttempt.build(
        attempt_id=attempt_id,
        order_id=order_id,
        attempted_at=attempted_at,
        status=status,
        no_fill_reason=no_fill_reason,
        requested_quantity=requested_quantity,
        filled_quantity=filled_quantity,
        reference_price=Decimal(10),
        execution_price=execution_price,
        participation_rate=participation_rate,
        state_revision="market-state-r1",
        data_version="cn-market-20260828-v1",
        state_hash=_digest("market-state"),
        market_rule_version=market_rule_version,
        market_rule_hash=market_rule_hash or _digest("market-rule"),
        slippage_model_version=slippage_model_version,
        slippage_model_hash=slippage_model_hash or _digest("slippage-model"),
    )


def _no_fill_attempt(
    reason: PaperNoFillReason = PaperNoFillReason.SUSPENDED,
    *,
    attempt_id: str = "attempt-no-fill",
    order_id: str = "order-buy",
) -> PaperMatchAttempt:
    status = (
        PaperOrderStatus.REJECTED
        if reason is PaperNoFillReason.INSUFFICIENT_CASH
        else PaperOrderStatus.ACCEPTED
    )
    return _attempt(
        attempt_id=attempt_id,
        order_id=order_id,
        filled_quantity=Decimal(0),
        status=status,
        no_fill_reason=reason,
        execution_price=None,
        participation_rate=Decimal(0),
    )


def _fill(
    *,
    fill_id: str = "fill-1",
    order_id: str = "order-buy",
    attempt_id: str = "attempt-1",
    instrument_id: str = "CN.SSE.510300",
    instrument_type: TradableInstrumentType = TradableInstrumentType.ETF,
    side: Side = Side.BUY,
    quantity: Decimal = Decimal(100),
    price: Decimal = Decimal("10.01"),
    sellable_on: date | None = NEXT_TRADING_DAY,
    filled_at: datetime = AS_OF + timedelta(minutes=2),
    fee_rule_version: str = "cn-fee-rule-v1",
    fee_rule_hash: str | None = None,
) -> PaperFill:
    return PaperFill.build(
        fill_id=fill_id,
        order_id=order_id,
        attempt_id=attempt_id,
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        filled_at=filled_at,
        side=side,
        quantity=quantity,
        price=price,
        commission=Decimal("1.00"),
        stamp_duty=Decimal("0.50") if side is Side.SELL else Decimal(0),
        transfer_fee=Decimal("0.02"),
        other_fee=Decimal("0.03"),
        sellable_on=sellable_on,
        fee_rule_version=fee_rule_version,
        fee_rule_hash=fee_rule_hash or _digest("fee-rule"),
    )


def _account(
    *,
    as_of: datetime = AS_OF,
    positions: tuple[PaperPosition, ...] = (),
    orders: tuple[PaperOrder, ...] = (),
    processed_batch_hashes: tuple[str, ...] | None = None,
    total_cash: Decimal = Decimal(10_000),
    external_frozen_cash: Decimal = Decimal(50),
    event_log_hash: str | None = None,
    previous_state_hash: str | None = None,
    runtime_mode: RuntimeMode = RuntimeMode.PAPER,
    account_id: str = "paper-account-1",
    source_snapshot_id: str = "account-snapshot-1",
    source_snapshot_hash: str | None = None,
    source_snapshot_as_of: datetime = SOURCE_AS_OF,
) -> PaperAccountState:
    batches = (
        tuple(order.batch_hash for order in orders)
        if processed_batch_hashes is None
        else processed_batch_hashes
    )
    return PaperAccountState.build(
        account_id=account_id,
        as_of=as_of,
        data_version="cn-market-20260828-v1",
        currency="CNY",
        source_snapshot_id=source_snapshot_id,
        source_snapshot_hash=source_snapshot_hash or _digest("source-snapshot"),
        source_snapshot_as_of=source_snapshot_as_of,
        total_cash=total_cash,
        external_frozen_cash=external_frozen_cash,
        positions=positions,
        orders=orders,
        processed_batch_hashes=batches,
        previous_state_hash=previous_state_hash,
        event_log_hash=event_log_hash or _digest("event-log-before"),
        runtime_mode=runtime_mode,
    )


def _request(
    account: PaperAccountState,
    *,
    batch_hash: str | None = None,
    config: PaperExecutionConfig | None = None,
    submitted_at: datetime = AS_OF + timedelta(minutes=1),
) -> PaperExecutionRequest:
    return PaperExecutionRequest.build(
        request_id="paper-request-1",
        idempotency_key="paper-idempotency-1",
        account_id=account.account_id,
        expected_account_state_hash=account.state_hash,
        expected_account_snapshot_hash=account.source_snapshot_hash,
        batch_hash=batch_hash or _digest("receipt-batch"),
        submitted_at=submitted_at,
        config=config or PaperExecutionConfig(version="paper-config-test-v1"),
    )


def _receipt_chain() -> tuple[
    PaperExecutionReceipt,
    PaperExecutionRequest,
    PaperAccountState,
    PaperOrder,
    PaperMatchAttempt,
    PaperFill,
]:
    account_before = _account(external_frozen_cash=Decimal(0))
    batch_hash = _digest("receipt-batch")
    request = _request(account_before, batch_hash=batch_hash)
    order = _order(
        status=PaperOrderStatus.FILLED,
        batch_hash=batch_hash,
        cash_reserved=Decimal(0),
        created_at=request.submitted_at,
    )
    attempt = _attempt()
    fill = _fill()
    event_log_hash = PaperExecutionReceipt.event_log_hash_for(
        previous_event_log_hash=account_before.event_log_hash,
        request_hash=request.request_hash,
        batch_hash=batch_hash,
        orders=(order,),
        attempts=(attempt,),
        fills=(fill,),
    )
    account_after = replay_execution_transition(
        account_before=account_before,
        batch_hash=batch_hash,
        processed_at=PROCESSED_AT,
        event_log_hash=event_log_hash,
        orders=(order,),
        fills=(fill,),
    )
    receipt = PaperExecutionReceipt.build(
        receipt_id="paper-receipt-1",
        request=request,
        account_before=account_before,
        account_after=account_after,
        processed_at=PROCESSED_AT,
        orders=(order,),
        attempts=(attempt,),
        fills=(fill,),
    )
    return receipt, request, account_before, order, attempt, fill


def test_paper_config_is_paper_only_strict_and_content_addressed() -> None:
    config = PaperExecutionConfig(version=" paper-config-v1 ")

    assert config.version == "paper-config-v1"
    assert config.allow_partial
    assert config.order_policy is PaperOrderPolicy.MARKET
    assert config.environment is AppEnvironment.PAPER
    assert config.config_hash == stable_hash(
        {
            "version": "paper-config-v1",
            "allow_partial": True,
            "order_policy": PaperOrderPolicy.MARKET,
            "max_market_state_age_seconds": 14_400,
            "environment": AppEnvironment.PAPER,
        }
    )
    with pytest.raises(ValueError, match="non-empty"):
        PaperExecutionConfig(version=" ")
    with pytest.raises(ValueError, match="bool"):
        PaperExecutionConfig(allow_partial=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="PaperOrderPolicy"):
        PaperExecutionConfig(order_policy="MARKET")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        PaperExecutionConfig(max_market_state_age_seconds=0)
    with pytest.raises(ValueError, match="positive integer"):
        PaperExecutionConfig(max_market_state_age_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="PAPER"):
        PaperExecutionConfig(environment=AppEnvironment.TEST)


def test_position_lot_enforces_whole_quantity_dates_freeze_and_hash() -> None:
    lot = _lot(externally_frozen=Decimal(10))

    assert lot.quantity == Decimal(100)
    assert lot.lot_hash == stable_hash(
        {
            "lot_id": lot.lot_id,
            "instrument_id": lot.instrument_id,
            "instrument_type": lot.instrument_type,
            "quantity": lot.quantity,
            "cost_basis": lot.cost_basis,
            "acquired_on": lot.acquired_on,
            "sellable_on": lot.sellable_on,
            "externally_frozen": lot.externally_frozen,
            "source_id": lot.source_id,
        }
    )
    with pytest.raises(ValueError, match="whole number"):
        replace(lot, quantity=Decimal("1.5"))
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(lot, quantity=100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        replace(lot, cost_basis=Decimal("NaN"))
    with pytest.raises(ValueError, match="cannot precede"):
        _lot(acquired_on=TRADING_DAY, sellable_on=TRADING_DAY - timedelta(days=1))
    with pytest.raises(ValueError, match="must be a date"):
        replace(lot, sellable_on=AS_OF)
    with pytest.raises(ValueError, match="cannot exceed"):
        _lot(externally_frozen=Decimal(101))
    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(lot, instrument_type="STOCK")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(lot, lot_hash="A" * 64)
    with pytest.raises(ValueError, match="does not match"):
        replace(lot, source_id="tampered-source")
    with pytest.raises(ValueError, match="cost_basis cannot be negative"):
        replace(lot, cost_basis=Decimal(-1))


def test_position_derives_t_plus_one_buckets_holds_and_cost_identities() -> None:
    position = _position()

    assert tuple(lot.lot_id for lot in position.lots) == ("lot-a", "lot-b")
    assert position.total_quantity == Decimal(200)
    assert position.unsettled_quantity == Decimal(100)
    assert position.frozen_quantity == Decimal(30)
    assert position.available_quantity == Decimal(70)
    assert position.total_quantity == (
        position.available_quantity + position.frozen_quantity + position.unsettled_quantity
    )
    assert position.cost_basis == Decimal(3000)
    assert position.average_cost == Decimal(15)
    assert position.reserved_sell_quantity == Decimal(20)


def test_position_rejects_duplicate_mixed_or_inconsistent_lots_and_buckets() -> None:
    position = _position()
    settled = _lot()

    with pytest.raises(ValueError, match="unique lot_id"):
        PaperPosition.build(
            as_of=AS_OF,
            instrument_id=settled.instrument_id,
            instrument_type=settled.instrument_type,
            lots=(settled, settled),
        )
    with pytest.raises(ValueError, match="at least one"):
        PaperPosition.build(
            as_of=AS_OF,
            instrument_id="CN.SSE.600000",
            instrument_type=TradableInstrumentType.STOCK,
            lots=(),
        )
    with pytest.raises(ValueError, match="match the paper position instrument"):
        PaperPosition.build(
            as_of=AS_OF,
            instrument_id="CN.SSE.600000",
            instrument_type=TradableInstrumentType.STOCK,
            lots=(_lot(instrument_id="CN.SZSE.000001"),),
        )
    unsettled_and_frozen = _lot(
        lot_id="lot-unsettled",
        acquired_on=TRADING_DAY,
        sellable_on=NEXT_TRADING_DAY,
        externally_frozen=Decimal(1),
    )
    with pytest.raises(ValueError, match="cannot also be externally frozen"):
        PaperPosition.build(
            as_of=AS_OF,
            instrument_id=unsettled_and_frozen.instrument_id,
            instrument_type=unsettled_and_frozen.instrument_type,
            lots=(unsettled_and_frozen,),
        )
    future_lot = _lot(
        lot_id="lot-future",
        acquired_on=NEXT_TRADING_DAY,
        sellable_on=NEXT_TRADING_DAY,
    )
    with pytest.raises(ValueError, match="future-acquired"):
        PaperPosition.build(
            as_of=AS_OF,
            instrument_id=future_lot.instrument_id,
            instrument_type=future_lot.instrument_type,
            lots=(future_lot,),
        )
    with pytest.raises(ValueError, match=r"available_quantity.*non-negative|reserved sell"):
        _position(reserved_sell_quantity=Decimal(91))
    with pytest.raises(ValueError, match="available_quantity"):
        replace(position, available_quantity=Decimal(71))
    with pytest.raises(ValueError, match="cost fields"):
        replace(position, cost_basis=Decimal(3001))
    with pytest.raises(ValueError, match="position_hash"):
        replace(position, position_hash="f" * 64)
    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(position, instrument_type="STOCK")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(position, lots=tuple(reversed(position.lots)))
    with pytest.raises(ValueError, match="costs cannot be negative"):
        replace(position, average_cost=Decimal(-1))
    with pytest.raises(ValueError, match="total_quantity must be positive"):
        replace(position, total_quantity=Decimal(0))
    with pytest.raises(ValueError, match="reserved sell quantity exceeds"):
        replace(position, reserved_sell_quantity=Decimal(91))
    with pytest.raises(ValueError, match="total_quantity must equal"):
        replace(position, total_quantity=Decimal(201))
    with pytest.raises(ValueError, match="unsettled_quantity"):
        replace(position, unsettled_quantity=Decimal(99))
    with pytest.raises(ValueError, match="frozen_quantity"):
        replace(position, frozen_quantity=Decimal(29))


@pytest.mark.parametrize("status", list(PaperOrderStatus))
def test_order_accepts_each_coherent_lifecycle_status(status: PaperOrderStatus) -> None:
    order = _order(status=status)

    assert order.remaining_quantity == order.quantity - order.filled_quantity
    assert order.status is status
    if status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.PARTIALLY_FILLED}:
        assert order.cash_reserved > 0
    else:
        assert order.cash_reserved == 0


def test_order_cash_and_sell_reservations_cover_buy_sell_and_negative_cash_leg() -> None:
    active_buy = _order()
    active_sell = _order(
        order_id="order-sell",
        instrument_id="CN.SSE.600000",
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        cash_reserved=Decimal("2.50"),
        is_full_liquidation=True,
    )

    assert active_buy.cash_reserved == Decimal(1200)
    assert active_buy.sell_reserved_quantity == 0
    assert active_sell.cash_reserved == Decimal("2.50")
    assert active_sell.sell_reserved_quantity == active_sell.remaining_quantity
    with pytest.raises(ValueError, match=r"terminal SELL.*release"):
        _order(
            side=Side.SELL,
            status=PaperOrderStatus.CANCELED,
            cash_reserved=Decimal("0.01"),
        )
    with pytest.raises(ValueError, match="SELL reservation"):
        _order(side=Side.SELL, sell_reserved_quantity=Decimal(99))
    with pytest.raises(ValueError, match="BUY order cannot reserve sell"):
        _order(sell_reserved_quantity=Decimal(1))
    with pytest.raises(ValueError, match=r"active BUY.*positive cash"):
        _order(cash_reserved=Decimal(0))


def test_order_rejects_invalid_status_quantity_time_enum_and_hash_boundaries() -> None:
    order = _order()

    with pytest.raises(ValueError, match="remaining_quantity"):
        replace(order, remaining_quantity=Decimal(99))
    with pytest.raises(ValueError, match="proper partial fill"):
        _order(status=PaperOrderStatus.PARTIALLY_FILLED, filled_quantity=Decimal(0))
    with pytest.raises(ValueError, match="FILLED order"):
        _order(status=PaperOrderStatus.FILLED, filled_quantity=Decimal(99))
    with pytest.raises(ValueError, match=r"last_attempt_id.*non-empty"):
        _order(
            status=PaperOrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal(40),
            last_attempt_id=" ",
        )
    with pytest.raises(ValueError, match="only a SELL"):
        _order(is_full_liquidation=True)
    with pytest.raises(ValueError, match="must follow"):
        _order(expires_at=AS_OF)
    with pytest.raises(ValueError, match="timezone information"):
        replace(order, created_at=AS_OF.replace(tzinfo=None))
    with pytest.raises(ValueError, match="Side"):
        replace(order, side="BUY")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="PaperOrderStatus"):
        replace(order, status="ACCEPTED")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="whole number"):
        replace(order, quantity=Decimal("100.5"))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(order, batch_hash="a" * 63)
    with pytest.raises(ValueError, match="order_hash"):
        replace(order, order_hash="f" * 64)
    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(order, instrument_type="ETF")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a bool"):
        replace(order, is_full_liquidation=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(order, cash_reserved=Decimal(-1))
    with pytest.raises(ValueError, match="cannot exceed quantity"):
        replace(
            order,
            filled_quantity=Decimal(101),
            remaining_quantity=Decimal(0),
        )
    with pytest.raises(ValueError, match="ACCEPTED or REJECTED"):
        replace(
            order,
            filled_quantity=Decimal(1),
            remaining_quantity=Decimal(99),
            last_attempt_id="attempt-1",
        )
    with pytest.raises(ValueError, match="fully filled order must use FILLED"):
        _order(
            status=PaperOrderStatus.CANCELED,
            filled_quantity=Decimal(100),
            last_attempt_id="attempt-1",
        )
    terminal_buy = _order(status=PaperOrderStatus.CANCELED)
    with pytest.raises(ValueError, match=r"terminal BUY.*release"):
        replace(terminal_buy, cash_reserved=Decimal(1))
    partial = _order(status=PaperOrderStatus.PARTIALLY_FILLED)
    with pytest.raises(ValueError, match="last match attempt"):
        replace(partial, last_attempt_id=None)


def test_match_attempts_capture_full_partial_and_every_no_fill_reason() -> None:
    full = _attempt()
    partial = _attempt(
        attempt_id="attempt-partial",
        filled_quantity=Decimal(40),
        status=PaperOrderStatus.PARTIALLY_FILLED,
        participation_rate=Decimal("0.004"),
    )

    assert full.unfilled_quantity == 0
    assert full.gross_amount == Decimal("1001.00")
    assert partial.unfilled_quantity == Decimal(60)
    assert partial.gross_amount == Decimal("400.40")
    for reason in PaperNoFillReason:
        no_fill = _no_fill_attempt(reason)
        expected_status = (
            PaperOrderStatus.REJECTED
            if reason is PaperNoFillReason.INSUFFICIENT_CASH
            else PaperOrderStatus.ACCEPTED
        )
        assert no_fill.status is expected_status
        assert no_fill.gross_amount == 0
        assert no_fill.execution_price is None


def test_match_attempt_rejects_incoherent_fill_reason_amount_and_strict_values() -> None:
    attempt = _attempt()

    with pytest.raises(ValueError, match="requires a no_fill_reason"):
        _attempt(
            filled_quantity=Decimal(0),
            status=PaperOrderStatus.ACCEPTED,
            execution_price=None,
            participation_rate=Decimal(0),
        )
    with pytest.raises(ValueError, match="cannot have a no_fill_reason"):
        _attempt(no_fill_reason=PaperNoFillReason.SUSPENDED)
    with pytest.raises(ValueError, match="reason does not match"):
        _attempt(
            filled_quantity=Decimal(0),
            status=PaperOrderStatus.REJECTED,
            no_fill_reason=PaperNoFillReason.SUSPENDED,
            execution_price=None,
            participation_rate=Decimal(0),
        )
    with pytest.raises(ValueError, match="status does not match"):
        _attempt(filled_quantity=Decimal(40), status=PaperOrderStatus.FILLED)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        replace(attempt, participation_rate=Decimal("1.01"))
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(attempt, reference_price=10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="whole number"):
        replace(attempt, filled_quantity=Decimal("1.5"))
    with pytest.raises(ValueError, match="PaperNoFillReason"):
        replace(_no_fill_attempt(), no_fill_reason="SUSPENDED")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="attempt_hash"):
        replace(attempt, attempt_hash="f" * 64)
    with pytest.raises(ValueError, match="PaperOrderStatus"):
        replace(attempt, status="FILLED")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reference_price must be positive"):
        replace(attempt, reference_price=Decimal(0))
    with pytest.raises(ValueError, match="cannot exceed requested"):
        replace(attempt, filled_quantity=Decimal(101))
    with pytest.raises(ValueError, match="unfilled_quantity"):
        replace(attempt, unfilled_quantity=Decimal(1))
    no_fill = _no_fill_attempt()
    with pytest.raises(ValueError, match="cannot have an execution price or gross"):
        replace(no_fill, execution_price=Decimal(10))
    with pytest.raises(ValueError, match="zero participation"):
        replace(no_fill, participation_rate=Decimal("0.01"))
    with pytest.raises(ValueError, match="positive execution price"):
        replace(attempt, execution_price=None)
    with pytest.raises(ValueError, match="positive participation"):
        replace(attempt, participation_rate=Decimal(0))
    with pytest.raises(ValueError, match="gross_amount"):
        replace(attempt, gross_amount=Decimal(1))


def test_fill_recomputes_fee_cash_and_buy_t_plus_one_or_sell_semantics() -> None:
    buy = _fill()
    sell = _fill(
        fill_id="fill-sell",
        order_id="order-sell",
        attempt_id="attempt-sell",
        instrument_id="CN.SSE.600000",
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        sellable_on=None,
    )

    assert buy.sellable_on == NEXT_TRADING_DAY
    assert buy.gross_amount == Decimal("1001.00")
    assert buy.total_fee == Decimal("1.05")
    assert buy.cash_change == Decimal("-1002.05")
    assert sell.total_fee == Decimal("1.55")
    assert sell.cash_change == Decimal("999.45")
    assert sell.sellable_on is None


def test_fill_rejects_bad_economics_dates_enums_decimals_and_hashes() -> None:
    buy = _fill()

    with pytest.raises(ValueError, match="component sum"):
        replace(buy, total_fee=Decimal("1.06"))
    with pytest.raises(ValueError, match="cash_change"):
        replace(buy, cash_change=Decimal("-1"))
    with pytest.raises(ValueError, match="gross_amount"):
        replace(buy, gross_amount=Decimal(1))
    with pytest.raises(ValueError, match="cannot precede"):
        _fill(sellable_on=TRADING_DAY - timedelta(days=1))
    with pytest.raises(ValueError, match="BUY fill must declare"):
        _fill(sellable_on=None)
    with pytest.raises(ValueError, match="SELL fill cannot create"):
        _fill(side=Side.SELL, sellable_on=NEXT_TRADING_DAY)
    with pytest.raises(ValueError, match="Side"):
        replace(buy, side="BUY")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="whole number"):
        replace(buy, quantity=Decimal("1.5"))
    with pytest.raises(ValueError, match="exact Decimal"):
        replace(buy, price=10.01)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(buy, fee_rule_hash="A" * 64)
    with pytest.raises(ValueError, match="fill_hash"):
        replace(buy, fill_hash="f" * 64)
    with pytest.raises(ValueError, match="TradableInstrumentType"):
        replace(buy, instrument_type="ETF")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be positive"):
        replace(buy, price=Decimal(0))
    with pytest.raises(ValueError, match="fee components cannot be negative"):
        replace(buy, commission=Decimal(-1))


def test_account_rebuilds_sorted_ledgers_and_all_active_cash_and_sell_holds() -> None:
    buy = _order(order_id="order-b", cash_reserved=Decimal(1200))
    sell = _order(
        order_id="order-a",
        instrument_id="CN.SSE.600000",
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        quantity=Decimal(60),
        cash_reserved=Decimal("2.50"),
    )
    position = _position(reserved_sell_quantity=Decimal(0))
    account = _account(positions=(position,), orders=(buy, sell))

    assert tuple(order.order_id for order in account.orders) == ("order-a", "order-b")
    assert account.processed_batch_hashes == tuple(sorted((buy.batch_hash, sell.batch_hash)))
    assert account.frozen_cash == Decimal("1252.50")
    assert account.available_cash == Decimal("8747.50")
    assert account.positions[0].reserved_sell_quantity == Decimal(60)
    assert account.positions[0].frozen_quantity == Decimal(70)
    assert account.positions[0].available_quantity == Decimal(30)


def test_account_rejects_duplicate_unbound_expired_cash_and_runtime_state() -> None:
    order = _order()
    position = _position(reserved_sell_quantity=Decimal(0))
    account = _account(orders=(order,))

    with pytest.raises(ValueError, match="unique order_id"):
        _account(orders=(order, order))
    with pytest.raises(ValueError, match="unique instrument_id"):
        _account(positions=(position, position))
    with pytest.raises(ValueError, match="unique"):
        _account(processed_batch_hashes=(_digest("same"), _digest("same")))
    with pytest.raises(ValueError, match="processed batch"):
        _account(orders=(order,), processed_batch_hashes=(_digest("other"),))
    with pytest.raises(ValueError, match="active SELL hold requires"):
        _account(
            orders=(
                _order(
                    side=Side.SELL,
                    instrument_id="CN.SSE.600000",
                    instrument_type=TradableInstrumentType.STOCK,
                ),
            )
        )
    with pytest.raises(ValueError, match="expired"):
        _account(as_of=AS_OF + timedelta(hours=2), orders=(order,))
    with pytest.raises(ValueError, match="cannot be negative"):
        _account(orders=(order,), total_cash=Decimal(1000))
    with pytest.raises(ValueError, match="PAPER runtime"):
        _account(runtime_mode=RuntimeMode.LIVE_ASSISTED)
    with pytest.raises(ValueError, match="timezone information"):
        replace(account, as_of=AS_OF.replace(tzinfo=None))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(account, previous_state_hash="A" * 64)
    with pytest.raises(ValueError, match="state_hash"):
        replace(account, state_hash="f" * 64)


def test_account_direct_contract_rejects_noncanonical_or_inconsistent_ledger_state() -> None:
    buy = _order(order_id="order-b")
    terminal = _order(order_id="order-a", status=PaperOrderStatus.CANCELED)
    account = _account(orders=(terminal, buy))

    with pytest.raises(ValueError, match="schema_version"):
        replace(account, schema_version="2")
    with pytest.raises(ValueError, match="source snapshot cannot be after"):
        replace(account, source_snapshot_as_of=account.as_of + timedelta(seconds=1))
    with pytest.raises(ValueError, match="positions must be unique and sorted"):
        position = _position(reserved_sell_quantity=Decimal(0))
        replace(account, positions=(position, position))
    with pytest.raises(ValueError, match="align exactly"):
        replace(
            account,
            positions=(_position(as_of=account.as_of - timedelta(seconds=1)),),
        )
    with pytest.raises(ValueError, match="orders must be unique and sorted"):
        replace(account, orders=(buy, terminal))
    with pytest.raises(ValueError, match="future paper orders"):
        replace(account, orders=(_order(created_at=account.as_of + timedelta(seconds=1)),))
    with pytest.raises(ValueError, match=r"processed batch hashes.*sorted"):
        replace(account, processed_batch_hashes=tuple(reversed(account.processed_batch_hashes)))
    with pytest.raises(ValueError, match="frozen_cash"):
        replace(account, frozen_cash=account.frozen_cash + Decimal(1))
    with pytest.raises(ValueError, match="available_cash"):
        replace(account, available_cash=account.available_cash + Decimal(1))
    with pytest.raises(ValueError, match=r"available_cash|cash holds cannot exceed"):
        held = _account(orders=(buy,), total_cash=Decimal(1200), external_frozen_cash=Decimal(0))
        replace(
            held,
            total_cash=Decimal(1199),
            available_cash=Decimal(0),
            frozen_cash=Decimal(1200),
        )
    active_sell = _order(
        order_id="order-sell",
        instrument_id="CN.SSE.600000",
        instrument_type=TradableInstrumentType.STOCK,
        side=Side.SELL,
        cash_reserved=Decimal(0),
    )
    empty = _account()
    with pytest.raises(ValueError, match="active SELL hold requires"):
        replace(
            empty,
            orders=(active_sell,),
            processed_batch_hashes=(active_sell.batch_hash,),
        )
    position = _position(reserved_sell_quantity=Decimal(0))
    with pytest.raises(ValueError, match="position sell holds"):
        replace(
            empty,
            positions=(position,),
            orders=(active_sell,),
            processed_batch_hashes=(active_sell.batch_hash,),
        )


def test_request_binds_cas_snapshot_batch_engine_and_exact_config() -> None:
    account = _account()
    config = PaperExecutionConfig(version="paper-config-v7", allow_partial=False)
    request = _request(account, config=config)

    assert request.expected_account_state_hash == account.state_hash
    assert request.expected_account_snapshot_hash == account.source_snapshot_hash
    assert request.engine_version == PAPER_EXECUTION_ENGINE_VERSION
    assert request.config_hash == config.config_hash
    assert request.allow_partial is False
    assert request.max_market_state_age_seconds == 14_400
    assert request.request_hash == stable_hash(
        {
            "schema_version": request.schema_version,
            "request_id": request.request_id,
            "idempotency_key": request.idempotency_key,
            "account_id": request.account_id,
            "expected_account_state_hash": request.expected_account_state_hash,
            "expected_account_snapshot_hash": request.expected_account_snapshot_hash,
            "batch_hash": request.batch_hash,
            "submitted_at": request.submitted_at,
            "allow_partial": request.allow_partial,
            "order_policy": request.order_policy,
            "max_market_state_age_seconds": request.max_market_state_age_seconds,
            "environment": request.environment,
            "engine_version": request.engine_version,
            "config_version": request.config_version,
            "config_hash": request.config_hash,
        }
    )


def test_request_rejects_lossy_config_enum_time_digest_and_hash_tampering() -> None:
    request = _request(_account())

    with pytest.raises(ValueError, match="schema_version"):
        replace(request, schema_version="2")
    with pytest.raises(ValueError, match="bool"):
        replace(request, allow_partial=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="PaperOrderPolicy"):
        replace(request, order_policy="MARKET")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        replace(request, max_market_state_age_seconds=0)
    with pytest.raises(ValueError, match="PAPER"):
        replace(request, environment=AppEnvironment.TEST)
    with pytest.raises(ValueError, match=r"unknown.*engine"):
        replace(request, engine_version="paper-execution-engine-v2")
    with pytest.raises(ValueError, match="timezone information"):
        replace(request, submitted_at=request.submitted_at.replace(tzinfo=None))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(request, batch_hash="A" * 64)
    with pytest.raises(ValueError, match="config_hash"):
        replace(request, config_hash="f" * 64)
    with pytest.raises(ValueError, match="request_hash"):
        replace(request, request_hash="f" * 64)


def test_receipt_links_request_events_account_cas_and_append_only_log() -> None:
    receipt, request, account_before, order, attempt, fill = _receipt_chain()

    assert receipt.request_hash == request.request_hash
    assert receipt.account_before_hash == account_before.state_hash
    assert receipt.account_after_hash == receipt.account_after.state_hash
    assert receipt.account_after.previous_state_hash == account_before.state_hash
    assert receipt.account_after.event_log_hash == receipt.event_log_hash
    assert request.batch_hash in receipt.account_after.processed_batch_hashes
    assert receipt.orders == (order,)
    assert receipt.attempts == (attempt,)
    assert receipt.fills == (fill,)
    assert receipt.event_log_hash == PaperExecutionReceipt.event_log_hash_for(
        previous_event_log_hash=account_before.event_log_hash,
        request_hash=request.request_hash,
        batch_hash=request.batch_hash,
        orders=(order,),
        attempts=(attempt,),
        fills=(fill,),
    )


def test_receipt_builder_rejects_stale_cas_and_changed_account_identity() -> None:
    receipt, request, account_before, order, attempt, fill = _receipt_chain()

    backdated_request = _request(
        account_before,
        batch_hash=receipt.batch_hash,
        submitted_at=account_before.as_of - timedelta(seconds=1),
    )
    with pytest.raises(PaperExecutionInputError, match="cannot precede account state"):
        PaperExecutionReceipt.build(
            receipt_id="backdated-receipt",
            request=backdated_request,
            account_before=account_before,
            account_after=receipt.account_after,
            processed_at=receipt.processed_at,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )

    stale_request = PaperExecutionRequest.build(
        request_id="stale-request",
        idempotency_key="stale-key",
        account_id=account_before.account_id,
        expected_account_state_hash=_digest("stale-state"),
        expected_account_snapshot_hash=account_before.source_snapshot_hash,
        batch_hash=request.batch_hash,
        submitted_at=request.submitted_at,
        config=PaperExecutionConfig(),
    )
    with pytest.raises(PaperExecutionInputError, match="state hash is stale"):
        PaperExecutionReceipt.build(
            receipt_id="stale-receipt",
            request=stale_request,
            account_before=account_before,
            account_after=receipt.account_after,
            processed_at=PROCESSED_AT,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )
    wrong_account_request = PaperExecutionRequest.build(
        request_id="wrong-account-request",
        idempotency_key="wrong-account-key",
        account_id="different-paper-account",
        expected_account_state_hash=account_before.state_hash,
        expected_account_snapshot_hash=account_before.source_snapshot_hash,
        batch_hash=request.batch_hash,
        submitted_at=request.submitted_at,
        config=PaperExecutionConfig(),
    )
    with pytest.raises(PaperExecutionInputError, match="account_id does not match"):
        PaperExecutionReceipt.build(
            receipt_id="wrong-account-receipt",
            request=wrong_account_request,
            account_before=account_before,
            account_after=receipt.account_after,
            processed_at=PROCESSED_AT,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )
    stale_snapshot_request = PaperExecutionRequest.build(
        request_id="stale-snapshot-request",
        idempotency_key="stale-snapshot-key",
        account_id=account_before.account_id,
        expected_account_state_hash=account_before.state_hash,
        expected_account_snapshot_hash=_digest("stale-snapshot"),
        batch_hash=request.batch_hash,
        submitted_at=request.submitted_at,
        config=PaperExecutionConfig(),
    )
    with pytest.raises(PaperExecutionInputError, match="snapshot hash is stale"):
        PaperExecutionReceipt.build(
            receipt_id="stale-snapshot-receipt",
            request=stale_snapshot_request,
            account_before=account_before,
            account_after=receipt.account_after,
            processed_at=PROCESSED_AT,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )
    changed_account = _account(
        as_of=PROCESSED_AT,
        account_id="different-paper-account",
        orders=(order,),
        processed_batch_hashes=(request.batch_hash,),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=receipt.event_log_hash,
        previous_state_hash=account_before.state_hash,
    )
    with pytest.raises(PaperExecutionInputError, match="identity cannot change"):
        PaperExecutionReceipt.build(
            receipt_id="changed-account-receipt",
            request=request,
            account_before=account_before,
            account_after=changed_account,
            processed_at=PROCESSED_AT,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )
    changed_source = _account(
        as_of=PROCESSED_AT,
        orders=(order,),
        processed_batch_hashes=(request.batch_hash,),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=receipt.event_log_hash,
        previous_state_hash=account_before.state_hash,
        source_snapshot_id="changed-source-snapshot",
    )
    with pytest.raises(PaperExecutionInputError, match="snapshot identity cannot change"):
        PaperExecutionReceipt.build(
            receipt_id="changed-source-receipt",
            request=request,
            account_before=account_before,
            account_after=changed_source,
            processed_at=PROCESSED_AT,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )
    wrong_event_account = _account(
        as_of=PROCESSED_AT,
        orders=(order,),
        processed_batch_hashes=(request.batch_hash,),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=_digest("wrong-event-log"),
        previous_state_hash=account_before.state_hash,
    )
    with pytest.raises(PaperExecutionInputError, match="event log does not match"):
        PaperExecutionReceipt.build(
            receipt_id="wrong-event-receipt",
            request=request,
            account_before=account_before,
            account_after=wrong_event_account,
            processed_at=PROCESSED_AT,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )


def test_receipt_builder_rejects_hash_valid_but_economically_forged_account_state() -> None:
    receipt, request, account_before, order, attempt, fill = _receipt_chain()
    forged_cash = _account(
        as_of=receipt.processed_at,
        positions=receipt.account_after.positions,
        orders=(order,),
        processed_batch_hashes=(receipt.batch_hash,),
        total_cash=receipt.account_after.total_cash + Decimal(1_000_000),
        external_frozen_cash=Decimal(0),
        event_log_hash=receipt.event_log_hash,
        previous_state_hash=account_before.state_hash,
    )
    with pytest.raises(PaperExecutionInputError, match="canonical paper execution replay"):
        PaperExecutionReceipt.build(
            receipt_id="forged-cash-receipt",
            request=request,
            account_before=account_before,
            account_after=forged_cash,
            processed_at=receipt.processed_at,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )

    missing_bought_position = _account(
        as_of=receipt.processed_at,
        orders=(order,),
        processed_batch_hashes=(receipt.batch_hash,),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=receipt.event_log_hash,
        previous_state_hash=account_before.state_hash,
    )
    with pytest.raises(PaperExecutionInputError, match="canonical paper execution replay"):
        PaperExecutionReceipt.build(
            receipt_id="forged-position-receipt",
            request=request,
            account_before=account_before,
            account_after=missing_bought_position,
            processed_at=receipt.processed_at,
            orders=(order,),
            attempts=(attempt,),
            fills=(fill,),
        )


def test_receipt_rejects_sorting_duplicates_or_broken_order_attempt_fill_links() -> None:
    receipt, _, _, order, attempt, _ = _receipt_chain()

    with pytest.raises(ValueError, match="orders must be unique and sorted"):
        replace(receipt, orders=(order, order))
    later_attempt = _no_fill_attempt(attempt_id="attempt-2")
    with pytest.raises(ValueError, match="attempts must be unique and sorted"):
        replace(receipt, attempts=(later_attempt, attempt))
    with pytest.raises(ValueError, match="attempt must bind"):
        replace(receipt, attempts=(_attempt(order_id="missing-order"),))
    missing_attempt_fill = _fill(attempt_id="missing-attempt")
    with pytest.raises(ValueError, match="fill must bind"):
        replace(receipt, fills=(missing_attempt_fill,))
    wrong_identity_fill = _fill(
        instrument_id="CN.SSE.600000",
        instrument_type=TradableInstrumentType.STOCK,
    )
    with pytest.raises(ValueError, match="fill identity"):
        replace(receipt, fills=(wrong_identity_fill,))
    half_fill = _fill(quantity=Decimal(50))
    with pytest.raises(ValueError, match="reconcile"):
        replace(receipt, fills=(half_fill,))
    with pytest.raises(ValueError, match="receipt_hash"):
        replace(receipt, receipt_hash="f" * 64)


def test_receipt_reconciles_each_order_and_pins_event_model_identities() -> None:
    receipt, _, _, _, _, _ = _receipt_chain()

    with pytest.raises(ValueError, match="exactly one match attempt"):
        replace(receipt, attempts=(), fills=())
    partial_attempt = _attempt(
        status=PaperOrderStatus.PARTIALLY_FILLED,
        filled_quantity=Decimal(40),
    )
    with pytest.raises(ValueError, match="quantities and status"):
        replace(
            receipt,
            attempts=(partial_attempt,),
            fills=(_fill(quantity=Decimal(40)),),
        )
    with pytest.raises(ValueError, match="attempt model identities"):
        replace(
            receipt,
            attempts=(_attempt(market_rule_hash=_digest("wrong-market-rule")),),
        )
    with pytest.raises(ValueError, match="fill fee identity"):
        replace(receipt, fills=(_fill(fee_rule_hash=_digest("wrong-fee-rule")),))


def test_receipt_rejects_attempts_and_fills_outside_the_processing_window() -> None:
    receipt, _, _, _, _, _ = _receipt_chain()

    with pytest.raises(ValueError, match="attempt time"):
        replace(
            receipt,
            attempts=(_attempt(attempted_at=PROCESSED_AT + timedelta(seconds=1)),),
        )
    backdated_order = _order(
        status=PaperOrderStatus.FILLED,
        batch_hash=receipt.batch_hash,
        cash_reserved=Decimal(0),
        created_at=receipt.request_submitted_at - timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="created at request submission"):
        replace(receipt, orders=(backdated_order,))
    expired_order = _order(
        status=PaperOrderStatus.FILLED,
        batch_hash=receipt.batch_hash,
        cash_reserved=Decimal(0),
        created_at=receipt.request_submitted_at,
        expires_at=AS_OF + timedelta(seconds=90),
    )
    with pytest.raises(ValueError, match="after order expiry"):
        replace(receipt, orders=(expired_order,))
    fill_expired_order = _order(
        status=PaperOrderStatus.FILLED,
        batch_hash=receipt.batch_hash,
        cash_reserved=Decimal(0),
        created_at=receipt.request_submitted_at,
        expires_at=AS_OF + timedelta(minutes=3),
    )
    with pytest.raises(ValueError, match="fill cannot occur after order expiry"):
        replace(
            receipt,
            orders=(fill_expired_order,),
            fills=(_fill(filled_at=AS_OF + timedelta(minutes=4)),),
        )
    with pytest.raises(ValueError, match="fill time"):
        replace(
            receipt,
            fills=(_fill(filled_at=PROCESSED_AT + timedelta(seconds=1)),),
        )


def test_receipt_rejects_noncanonical_and_unbound_ledger_content() -> None:
    receipt, _, _, order, _, _ = _receipt_chain()

    with pytest.raises(ValueError, match="schema_version"):
        replace(receipt, schema_version="2")
    with pytest.raises(ValueError, match="fills must be unique and sorted"):
        replace(receipt, fills=(receipt.fills[0], receipt.fills[0]))
    with pytest.raises(ValueError, match="belong to its draft batch"):
        replace(receipt, orders=(_order(status=PaperOrderStatus.FILLED),))
    other_attempt = _no_fill_attempt(attempt_id="attempt-2")
    with pytest.raises(ValueError, match="last_attempt_id must bind"):
        replace(receipt, attempts=(other_attempt,), fills=())
    with pytest.raises(ValueError, match="account_after as_of"):
        replace(receipt, processed_at=receipt.processed_at + timedelta(seconds=1))
    wrong_previous = _account(
        as_of=receipt.processed_at,
        orders=(order,),
        processed_batch_hashes=(receipt.batch_hash,),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=receipt.event_log_hash,
        previous_state_hash=_digest("wrong-previous"),
    )
    with pytest.raises(ValueError, match="extend account_before_hash"):
        replace(
            receipt,
            account_after=wrong_previous,
            account_after_hash=wrong_previous.state_hash,
        )
    no_processed_batch = _account(
        as_of=receipt.processed_at,
        processed_batch_hashes=(),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=receipt.event_log_hash,
        previous_state_hash=receipt.account_before_hash,
    )
    with pytest.raises(ValueError, match="record the processed draft batch"):
        replace(
            receipt,
            account_after=no_processed_batch,
            account_after_hash=no_processed_batch.state_hash,
        )
    rejected_order = _order(
        status=PaperOrderStatus.REJECTED,
        batch_hash=receipt.batch_hash,
    )
    wrong_order_ledger = _account(
        as_of=receipt.processed_at,
        orders=(rejected_order,),
        processed_batch_hashes=(receipt.batch_hash,),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=receipt.event_log_hash,
        previous_state_hash=receipt.account_before_hash,
    )
    with pytest.raises(ValueError, match="match the account_after order ledger"):
        replace(
            receipt,
            account_after=wrong_order_ledger,
            account_after_hash=wrong_order_ledger.state_hash,
        )
    forged_event_log = _digest("forged-event-log")
    forged_event_account = _account(
        as_of=receipt.processed_at,
        orders=(order,),
        processed_batch_hashes=(receipt.batch_hash,),
        total_cash=receipt.account_after.total_cash,
        external_frozen_cash=Decimal(0),
        event_log_hash=forged_event_log,
        previous_state_hash=receipt.account_before_hash,
    )
    with pytest.raises(ValueError, match="execution events"):
        replace(
            receipt,
            event_log_hash=forged_event_log,
            account_after=forged_event_account,
            account_after_hash=forged_event_account.state_hash,
        )


def test_receipt_rejects_account_and_event_log_tampering() -> None:
    receipt, _, _, _, _, _ = _receipt_chain()

    with pytest.raises(ValueError, match="account_after_hash"):
        replace(receipt, account_after_hash="f" * 64)
    with pytest.raises(ValueError, match="event logs must match"):
        replace(
            receipt,
            event_log_hash=_digest("tampered-event-log"),
        )
    with pytest.raises(ValueError, match="cannot precede"):
        replace(receipt, processed_at=receipt.request_submitted_at - timedelta(seconds=1))
    with pytest.raises(ValueError, match="timezone information"):
        replace(receipt, processed_at=receipt.processed_at.replace(tzinfo=None))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PaperExecutionReceipt.event_log_hash_for(
            previous_event_log_hash="A" * 64,
            request_hash=receipt.request_hash,
            batch_hash=receipt.batch_hash,
            orders=receipt.orders,
            attempts=receipt.attempts,
            fills=receipt.fills,
        )
