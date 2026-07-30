"""Deterministic event-driven cash, position and order simulation."""

from dataclasses import dataclass
from datetime import date, datetime

from quant_agent.backtest.cn_market_rules import ChinaMarketRules
from quant_agent.backtest.contracts import (
    AssetType,
    Fill,
    MarketBar,
    Order,
    OrderStatus,
    Side,
)
from quant_agent.core.time import ensure_aware


@dataclass(frozen=True, slots=True)
class Position:
    instrument_id: str
    asset_type: AssetType
    quantity: int
    available_quantity: int
    average_cost: float
    last_buy_date: date | None


@dataclass(frozen=True, slots=True)
class BacktestSnapshot:
    as_of: datetime
    cash: float
    positions: tuple[Position, ...]
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    total_fees: float

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)


class EventDrivenBacktest:
    """Process orders against later bars while preserving cash and holdings."""

    def __init__(self, *, initial_cash: float, rules: ChinaMarketRules) -> None:
        if initial_cash < 0:
            raise ValueError("initial cash cannot be negative")
        self._cash = initial_cash
        self._rules = rules
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []

    def submit(self, order: Order) -> Order:
        if order.order_id in self._orders:
            raise ValueError("duplicate order id")
        accepted = order.transition(OrderStatus.ACCEPTED)
        self._orders[order.order_id] = accepted
        return accepted

    def _roll_available(self, trade_date: date) -> None:
        for instrument_id, position in list(self._positions.items()):
            if (
                position.asset_type is AssetType.STOCK
                and position.last_buy_date is not None
                and position.last_buy_date < trade_date
            ):
                self._positions[instrument_id] = Position(
                    instrument_id=position.instrument_id,
                    asset_type=position.asset_type,
                    quantity=position.quantity,
                    available_quantity=position.quantity,
                    average_cost=position.average_cost,
                    last_buy_date=position.last_buy_date,
                )

    def process_bar(self, order_id: str, bar: MarketBar) -> Order:
        order = self._orders[order_id]
        if order.status not in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}:
            raise ValueError("order is not matchable")
        self._roll_available(bar.trade_date)
        position = self._positions.get(order.instrument_id)
        available = position.available_quantity if position is not None else 0
        reason = self._rules.validate(order, bar, available_quantity=available)
        if reason is not None:
            terminal_status = (
                OrderStatus.CANCELLED
                if order.status is OrderStatus.PARTIALLY_FILLED
                else OrderStatus.REJECTED
            )
            terminal = order.transition(
                terminal_status,
                rejection_reason=reason,
            )
            self._orders[order_id] = terminal
            return terminal
        fill = self._rules.match(order, bar, available_quantity=available)
        if fill is None:
            return order
        cash_change = fill.gross_amount + fill.commission + fill.tax
        if fill.side is Side.BUY and cash_change > self._cash:
            rejected = order.transition(OrderStatus.REJECTED, rejection_reason="insufficient cash")
            self._orders[order_id] = rejected
            return rejected
        if fill.side is Side.BUY:
            self._cash -= cash_change
            old_quantity = position.quantity if position else 0
            old_cost = position.average_cost * old_quantity if position else 0.0
            quantity = old_quantity + fill.quantity
            available_quantity = (position.available_quantity if position else 0) + (
                fill.quantity if order.asset_type is AssetType.ETF else 0
            )
            self._positions[order.instrument_id] = Position(
                order.instrument_id,
                order.asset_type,
                quantity,
                available_quantity,
                (old_cost + fill.gross_amount + fill.commission) / quantity,
                bar.trade_date,
            )
        else:
            assert position is not None
            self._cash += fill.gross_amount - fill.commission - fill.tax
            quantity = position.quantity - fill.quantity
            if quantity:
                self._positions[order.instrument_id] = Position(
                    position.instrument_id,
                    position.asset_type,
                    quantity,
                    position.available_quantity - fill.quantity,
                    position.average_cost,
                    position.last_buy_date,
                )
            else:
                del self._positions[order.instrument_id]
        self._fills.append(fill)
        total_filled = order.filled_quantity + fill.quantity
        status = (
            OrderStatus.FILLED if total_filled == order.quantity else OrderStatus.PARTIALLY_FILLED
        )
        updated = order.transition(status, filled_quantity=total_filled)
        self._orders[order_id] = updated
        return updated

    def snapshot(self, *, as_of: datetime) -> BacktestSnapshot:
        ensure_aware(as_of)
        return BacktestSnapshot(
            as_of=as_of,
            cash=self._cash,
            positions=tuple(self._positions[key] for key in sorted(self._positions)),
            orders=tuple(self._orders[key] for key in sorted(self._orders)),
            fills=tuple(self._fills),
            total_fees=sum(fill.total_fees for fill in self._fills),
        )
