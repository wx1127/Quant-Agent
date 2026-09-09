"""Canonical economic replay for paper-account execution transitions."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, localcontext

from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.portfolio import AccountSnapshot
from quant_agent.regime.contracts import stable_hash

from .contracts import (
    PaperAccountState,
    PaperExecutionInputError,
    PaperFill,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperPositionLot,
)

_ZERO = Decimal(0)


def replay_execution_transition(
    *,
    account_before: PaperAccountState,
    batch_hash: str,
    processed_at: datetime,
    event_log_hash: str,
    orders: tuple[PaperOrder, ...],
    fills: tuple[PaperFill, ...],
) -> PaperAccountState:
    """Derive the only valid account state produced by one execution receipt."""

    processed_at = ensure_aware(processed_at)
    if processed_at < account_before.as_of:
        raise PaperExecutionInputError("paper execution cannot move account time backwards")
    if batch_hash in account_before.processed_batch_hashes:
        raise PaperExecutionInputError("paper draft batch was already processed")
    if not orders:
        raise PaperExecutionInputError("paper execution transition requires at least one order")
    if any(order.batch_hash != batch_hash for order in orders):
        raise PaperExecutionInputError("paper transition orders must belong to its batch")
    prior_order_ids = {order.order_id for order in account_before.orders}
    if any(order.order_id in prior_order_ids for order in orders):
        raise PaperExecutionInputError("paper transition cannot replace an existing order")
    if any(
        order.status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.PARTIALLY_FILLED}
        and processed_at >= order.expires_at
        for order in account_before.orders
    ):
        raise PaperExecutionInputError(
            "expired active paper orders require an account refresh before execution"
        )

    new_orders_by_id = {order.order_id: order for order in orders}
    lots_by_instrument = {
        position.instrument_id: list(position.lots) for position in account_before.positions
    }
    total_cash = account_before.total_cash
    for fill in sorted(fills, key=lambda item: (item.filled_at, item.fill_id)):
        order = new_orders_by_id.get(fill.order_id)
        if order is None:
            raise PaperExecutionInputError("paper fill does not belong to a transition order")
        if (
            fill.instrument_id != order.instrument_id
            or fill.instrument_type is not order.instrument_type
            or fill.side is not order.side
        ):
            raise PaperExecutionInputError("paper fill identity differs from its transition order")
        total_cash += fill.cash_change
        if total_cash < 0:
            raise PaperExecutionInputError("paper execution transition would create negative cash")
        if fill.side is Side.BUY:
            if fill.sellable_on is None:
                raise PaperExecutionInputError("paper BUY fill lacks a settlement date")
            lots_by_instrument.setdefault(fill.instrument_id, []).append(
                PaperPositionLot.build(
                    lot_id=f"paper-lot:{fill.fill_hash}",
                    instrument_id=fill.instrument_id,
                    instrument_type=fill.instrument_type,
                    quantity=fill.quantity,
                    cost_basis=fill.gross_amount + fill.total_fee,
                    acquired_on=fill.filled_at.astimezone(SHANGHAI_TZ).date(),
                    sellable_on=fill.sellable_on,
                    source_id=fill.fill_id,
                )
            )
        else:
            lots_by_instrument[fill.instrument_id] = _consume_sell_lots(
                lots=lots_by_instrument.get(fill.instrument_id, []),
                quantity=fill.quantity,
                trading_day=fill.filled_at.astimezone(SHANGHAI_TZ).date(),
            )

    positions = _positions_from_lots(
        lots_by_instrument=lots_by_instrument,
        as_of=processed_at,
    )
    return PaperAccountState.build(
        account_id=account_before.account_id,
        as_of=processed_at,
        data_version=account_before.data_version,
        currency=account_before.currency,
        source_snapshot_id=account_before.source_snapshot_id,
        source_snapshot_hash=account_before.source_snapshot_hash,
        source_snapshot_as_of=account_before.source_snapshot_as_of,
        total_cash=total_cash,
        external_frozen_cash=account_before.external_frozen_cash,
        positions=positions,
        orders=(*account_before.orders, *orders),
        processed_batch_hashes=(*account_before.processed_batch_hashes, batch_hash),
        previous_state_hash=account_before.state_hash,
        event_log_hash=event_log_hash,
    )


def replay_account_refresh(
    *,
    account_before: PaperAccountState,
    snapshot: AccountSnapshot,
) -> PaperAccountState:
    """Advance settlement and expiry against a new hash-linked account snapshot."""

    try:
        snapshot = AccountSnapshot.from_json(snapshot.to_json())
    except (TypeError, ValueError) as error:
        raise PaperExecutionInputError(str(error)) from error
    if snapshot.runtime_mode.value != "PAPER":
        raise PaperExecutionInputError("paper account refresh requires a PAPER snapshot")
    if snapshot.account_id != account_before.account_id:
        raise PaperExecutionInputError("paper account refresh cannot change account identity")
    if snapshot.as_of <= account_before.as_of:
        raise PaperExecutionInputError("paper account refresh snapshot must be newer")
    if snapshot.currency != account_before.currency:
        raise PaperExecutionInputError("paper account refresh cannot change currency")
    if snapshot.previous_snapshot_hash != account_before.source_snapshot_hash:
        raise PaperExecutionInputError("paper account refresh snapshot lineage is stale")
    if snapshot.source_event_log_hash != account_before.event_log_hash:
        raise PaperExecutionInputError("paper account refresh event-log boundary is stale")

    refreshed_orders = tuple(
        _expire_order(order, as_of=snapshot.as_of) for order in account_before.orders
    )
    positions = tuple(
        PaperPosition.build(
            as_of=snapshot.as_of,
            instrument_id=position.instrument_id,
            instrument_type=position.instrument_type,
            lots=position.lots,
        )
        for position in account_before.positions
    )
    refresh_event_log_hash = stable_hash(
        {
            "event": "PAPER_ACCOUNT_REFRESH",
            "account_before_hash": account_before.state_hash,
            "event_log_before_hash": account_before.event_log_hash,
            "source_snapshot_hash": snapshot.content_hash,
            "order_hashes": [order.order_hash for order in refreshed_orders],
        }
    )
    refreshed = PaperAccountState.build(
        account_id=account_before.account_id,
        as_of=snapshot.as_of,
        data_version=snapshot.data_version,
        currency=account_before.currency,
        source_snapshot_id=snapshot.snapshot_id,
        source_snapshot_hash=snapshot.content_hash,
        source_snapshot_as_of=snapshot.as_of,
        total_cash=account_before.total_cash,
        external_frozen_cash=account_before.external_frozen_cash,
        positions=positions,
        orders=refreshed_orders,
        processed_batch_hashes=account_before.processed_batch_hashes,
        previous_state_hash=account_before.state_hash,
        event_log_hash=refresh_event_log_hash,
    )
    _validate_refresh_snapshot(snapshot=snapshot, refreshed=refreshed)
    return refreshed


def _expire_order(order: PaperOrder, *, as_of: datetime) -> PaperOrder:
    active = order.status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.PARTIALLY_FILLED}
    if not active or as_of < order.expires_at:
        return order
    return PaperOrder.build(
        order_id=order.order_id,
        batch_hash=order.batch_hash,
        line_hash=order.line_hash,
        decision_id=order.decision_id,
        instrument_id=order.instrument_id,
        instrument_type=order.instrument_type,
        side=order.side,
        quantity=order.quantity,
        filled_quantity=order.filled_quantity,
        is_full_liquidation=order.is_full_liquidation,
        status=PaperOrderStatus.EXPIRED,
        created_at=order.created_at,
        expires_at=order.expires_at,
        cash_reserved=_ZERO,
        sell_reserved_quantity=_ZERO,
        market_rule_version=order.market_rule_version,
        market_rule_hash=order.market_rule_hash,
        fee_rule_version=order.fee_rule_version,
        fee_rule_hash=order.fee_rule_hash,
        slippage_model_version=order.slippage_model_version,
        slippage_model_hash=order.slippage_model_hash,
        last_attempt_id=order.last_attempt_id,
    )


def _validate_refresh_snapshot(
    *,
    snapshot: AccountSnapshot,
    refreshed: PaperAccountState,
) -> None:
    if (
        snapshot.cash.total_cash != refreshed.total_cash
        or snapshot.cash.available_cash != refreshed.available_cash
        or snapshot.cash.frozen_cash != refreshed.frozen_cash
    ):
        raise PaperExecutionInputError(
            "paper account refresh cash does not reconcile to the snapshot"
        )
    snapshot_positions = {position.instrument_id: position for position in snapshot.positions}
    if set(snapshot_positions) != {position.instrument_id for position in refreshed.positions}:
        raise PaperExecutionInputError(
            "paper account refresh positions do not reconcile to the snapshot"
        )
    for position in refreshed.positions:
        source = snapshot_positions[position.instrument_id]
        if (
            source.instrument_type is not position.instrument_type
            or source.total_quantity != position.total_quantity
            or source.available_quantity != position.available_quantity
            or source.frozen_quantity != position.frozen_quantity
            or source.unsettled_quantity != position.unsettled_quantity
            or source.cost_basis != position.cost_basis
            or source.average_cost != position.average_cost
        ):
            raise PaperExecutionInputError(
                f"{position.instrument_id} paper refresh position does not reconcile"
            )


def _consume_sell_lots(
    *,
    lots: list[PaperPositionLot],
    quantity: Decimal,
    trading_day: date,
) -> list[PaperPositionLot]:
    remaining = quantity
    result: list[PaperPositionLot] = []
    ordered = sorted(lots, key=lambda lot: (lot.sellable_on, lot.acquired_on, lot.lot_id))
    for lot in ordered:
        free = lot.quantity - lot.externally_frozen if lot.sellable_on <= trading_day else _ZERO
        consumed = min(free, remaining)
        if consumed == 0:
            result.append(lot)
            continue
        remaining_quantity = lot.quantity - consumed
        if remaining_quantity:
            with localcontext() as context:
                context.prec = 50
                remaining_cost = lot.cost_basis * remaining_quantity / lot.quantity
            result.append(
                PaperPositionLot.build(
                    lot_id=lot.lot_id,
                    instrument_id=lot.instrument_id,
                    instrument_type=lot.instrument_type,
                    quantity=remaining_quantity,
                    cost_basis=remaining_cost,
                    acquired_on=lot.acquired_on,
                    sellable_on=lot.sellable_on,
                    externally_frozen=lot.externally_frozen,
                    source_id=lot.source_id,
                )
            )
        remaining -= consumed
    if remaining:
        raise PaperExecutionInputError("paper lot ledger cannot satisfy transition sell fill")
    return result


def _positions_from_lots(
    *,
    lots_by_instrument: dict[str, list[PaperPositionLot]],
    as_of: datetime,
) -> tuple[PaperPosition, ...]:
    result: list[PaperPosition] = []
    for instrument_id, lots in sorted(lots_by_instrument.items()):
        if not lots:
            continue
        instrument_types = {lot.instrument_type for lot in lots}
        if len(instrument_types) != 1:
            raise PaperExecutionInputError("paper lots contain conflicting instrument types")
        instrument_type: TradableInstrumentType = next(iter(instrument_types))
        result.append(
            PaperPosition.build(
                as_of=as_of,
                instrument_id=instrument_id,
                instrument_type=instrument_type,
                lots=tuple(lots),
            )
        )
    return tuple(result)


__all__ = ["replay_account_refresh", "replay_execution_transition"]
