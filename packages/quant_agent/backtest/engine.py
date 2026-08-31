"""Fail-closed event-driven order matching and portfolio accounting.

The engine consumes the immutable contracts from :mod:`quant_agent.backtest.contracts`.
Economic events (confirmed fills and fees) update the private ledger exactly once;
``CashEvent`` and ``PositionEvent`` records then attest to the resulting state instead
of applying the same movement a second time.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, localcontext
from typing import Self

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator, model_validator

from quant_agent.backtest.contracts import (
    BacktestEvent,
    CashDirection,
    CashEvent,
    ExactDecimal,
    FeeEvent,
    FillEvent,
    FillStatus,
    OrderEvent,
    OrderStatus,
    OrderType,
    PositionEvent,
    Side,
    SignalDirection,
    TargetEvent,
    TradableInstrumentType,
)
from quant_agent.backtest.rules import FeeRuleBook, MarketRule
from quant_agent.backtest.serialization import serialize_event_stream

_EVENT_ADAPTER: TypeAdapter[BacktestEvent] = TypeAdapter(BacktestEvent)


class BacktestEngineError(ValueError):
    """Base error for a rejected event or invalid engine configuration."""


class EventOrderError(BacktestEngineError):
    """Raised when an event regresses the engine clock or replay sequence."""


class EventLinkError(BacktestEngineError):
    """Raised when an event does not match the entity it claims to reference."""


class LedgerInvariantError(BacktestEngineError):
    """Raised before a cash or position invariant could be violated."""


class MarketRuleNotFoundError(BacktestEngineError):
    """Raised when no unambiguous market rule is effective for an event."""


class InitialPosition(BaseModel):
    """A deterministic opening position supplied outside the replay event stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str
    instrument_type: TradableInstrumentType
    quantity: ExactDecimal
    sellable_quantity: ExactDecimal
    average_cost: ExactDecimal

    @field_validator("instrument_id")
    @classmethod
    def validate_instrument_id(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("instrument_id must be non-empty")
        return result

    @model_validator(mode="after")
    def validate_position(self) -> InitialPosition:
        if self.sellable_quantity < 0 or self.average_cost < 0:
            raise ValueError("sellable_quantity and average_cost cannot be negative")
        if self.quantity < 0 and self.sellable_quantity != 0:
            raise ValueError("a short opening position cannot have sellable long inventory")
        if self.quantity >= 0 and self.sellable_quantity > self.quantity:
            raise ValueError("sellable_quantity cannot exceed a long opening quantity")
        if self.quantity == 0 and self.average_cost != 0:
            raise ValueError("a zero opening position must have zero average_cost")
        return self


class EventBacktestConfig(BaseModel):
    """Run-level safety policy; permissive behavior must be explicitly enabled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    account_id: str
    initial_cash: ExactDecimal
    currency: str = "CNY"
    allow_negative_cash: bool = False
    allow_short_sales: bool = False
    allow_unsettled_sales: bool = False
    require_target_for_order: bool = True
    require_ledger_events: bool = False
    initial_positions: tuple[InitialPosition, ...] = ()

    @field_validator("run_id", "account_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("run_id and account_id must be non-empty")
        return result

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        result = value.strip().upper()
        if len(result) != 3 or not result.isalpha():
            raise ValueError("currency must be a three-letter alphabetic code")
        return result

    @model_validator(mode="after")
    def validate_opening_ledger(self) -> EventBacktestConfig:
        if self.initial_cash < 0 and not self.allow_negative_cash:
            raise ValueError("negative initial_cash requires allow_negative_cash")
        identities: set[tuple[str, TradableInstrumentType]] = set()
        for position in self.initial_positions:
            identity = (position.instrument_id, position.instrument_type)
            if identity in identities:
                raise ValueError("initial positions must be unique by instrument identity")
            if position.quantity < 0 and not self.allow_short_sales:
                raise ValueError("a short opening position requires allow_short_sales")
            identities.add(identity)
        return self


# Short aliases keep the public API pleasant without creating a second implementation.
EngineConfig = EventBacktestConfig


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """One signed internal position in a deterministic result snapshot."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    quantity: Decimal
    sellable_quantity: Decimal
    average_cost: Decimal


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    """Latest order lifecycle state plus economically confirmed quantity."""

    order_id: str
    instrument_id: str
    instrument_type: TradableInstrumentType
    side: Side
    order_type: OrderType
    quantity: Decimal
    status: OrderStatus
    filled_quantity: Decimal
    confirmed_quantity: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable replay result and its canonical event-log identity."""

    run_id: str
    account_id: str
    currency: str
    initial_cash: Decimal
    cash_balance: Decimal
    positions: tuple[PositionSnapshot, ...]
    orders: tuple[OrderSnapshot, ...]
    events: tuple[BacktestEvent, ...]
    event_log: str
    event_log_hash: str

    @property
    def event_count(self) -> int:
        """Return the number of committed replay events."""

        return len(self.events)

    def position(self, instrument_id: str) -> PositionSnapshot | None:
        """Look up an unambiguous position by instrument identifier."""

        matches = tuple(item for item in self.positions if item.instrument_id == instrument_id)
        if len(matches) > 1:
            raise ValueError("instrument_id is ambiguous across instrument types")
        return matches[0] if matches else None


@dataclass(slots=True)
class _Lot:
    quantity: Decimal
    sellable_on: date


@dataclass(slots=True)
class _PositionLedger:
    instrument_type: TradableInstrumentType
    quantity: Decimal = Decimal(0)
    cost_basis: Decimal = Decimal(0)
    lots: list[_Lot] = field(default_factory=list)


@dataclass(slots=True)
class _OrderRecord:
    created: OrderEvent
    latest: OrderEvent
    confirmed_quantity: Decimal = Decimal(0)


@dataclass(slots=True)
class _FillRecord:
    created: FillEvent
    latest: FillEvent
    economic_applied: bool = False
    opened_long_quantity: Decimal = Decimal(0)
    position_quantity_after: Decimal | None = None
    position_cost_basis_after: Decimal | None = None
    fee: FeeEvent | None = None


@dataclass(frozen=True, slots=True)
class _ExpectedCash:
    source_time: datetime
    trading_day: date
    direction: CashDirection
    amount: Decimal
    balance_after: Decimal


@dataclass(frozen=True, slots=True)
class _ExpectedPosition:
    source_time: datetime
    trading_day: date
    snapshot: PositionSnapshot


@dataclass(slots=True)
class _EngineState:
    cash: Decimal
    positions: dict[tuple[str, TradableInstrumentType], _PositionLedger]
    targets: dict[tuple[str, str, TradableInstrumentType], TargetEvent] = field(
        default_factory=dict
    )
    orders: dict[str, _OrderRecord] = field(default_factory=dict)
    fills: dict[str, _FillRecord] = field(default_factory=dict)
    events: list[BacktestEvent] = field(default_factory=list)
    event_ids: set[str] = field(default_factory=set)
    expected_cash: dict[str, _ExpectedCash] = field(default_factory=dict)
    expected_positions: dict[str, _ExpectedPosition] = field(default_factory=dict)
    last_sequence: int = -1
    last_event_time: datetime | None = None
    last_trading_day: date | None = None


class _MarketRuleResolver:
    def __init__(self, rules: Iterable[MarketRule]) -> None:
        self._rules = tuple(rules)
        if not self._rules:
            raise MarketRuleNotFoundError("at least one MarketRule is required")
        identities: set[tuple[TradableInstrumentType, date]] = set()
        versions: set[tuple[TradableInstrumentType, str, str]] = set()
        for rule in self._rules:
            if not rule.rule_id.strip() or not rule.version.strip():
                raise ValueError("market rule_id and version must be non-empty")
            identity = (rule.instrument_type, rule.effective_from)
            version_identity = (rule.instrument_type, rule.rule_id, rule.version)
            if identity in identities:
                raise ValueError("market rules are ambiguous at an effective_from boundary")
            if version_identity in versions:
                raise ValueError("market rule identity and version must be unique")
            identities.add(identity)
            versions.add(version_identity)

    def select(
        self,
        *,
        instrument_type: TradableInstrumentType,
        trading_day: date,
    ) -> MarketRule:
        candidates = tuple(
            rule
            for rule in self._rules
            if rule.instrument_type is instrument_type and rule.effective_from <= trading_day
        )
        if not candidates:
            raise MarketRuleNotFoundError(
                f"no effective {instrument_type.value} market rule for {trading_day.isoformat()}"
            )
        return max(candidates, key=lambda rule: rule.effective_from)


class EventDrivenBacktestEngine:
    """Consume an event stream using a strict clock and atomic private ledger."""

    def __init__(
        self,
        config: EventBacktestConfig,
        market_rules: Iterable[MarketRule],
        fee_rule_book: FeeRuleBook | None = None,
    ) -> None:
        self._config = config
        self._market_rules = _MarketRuleResolver(market_rules)
        self._fee_rule_book = fee_rule_book
        positions: dict[tuple[str, TradableInstrumentType], _PositionLedger] = {}
        for opening in config.initial_positions:
            ledger = _PositionLedger(
                instrument_type=opening.instrument_type,
                quantity=opening.quantity,
                cost_basis=abs(opening.quantity) * opening.average_cost,
            )
            if opening.quantity > 0:
                if opening.sellable_quantity:
                    ledger.lots.append(_Lot(opening.sellable_quantity, date.min))
                unsettled = opening.quantity - opening.sellable_quantity
                if unsettled:
                    ledger.lots.append(_Lot(unsettled, date.max))
            positions[(opening.instrument_id, opening.instrument_type)] = ledger
        self._state = _EngineState(cash=config.initial_cash, positions=positions)

    @property
    def config(self) -> EventBacktestConfig:
        """Return the immutable run configuration."""

        return self._config

    @property
    def cash_balance(self) -> Decimal:
        """Return the currently committed cash balance."""

        return self._state.cash

    @property
    def events(self) -> tuple[BacktestEvent, ...]:
        """Return committed events in canonical replay order."""

        return tuple(self._state.events)

    def process(self, event: BacktestEvent) -> BacktestResult:
        """Atomically validate and commit one event."""

        staged = copy.deepcopy(self._state)
        self._apply_event(staged, event)
        result = self._result(staged)
        self._state = staged
        return result

    process_event = process

    def process_all(self, events: Iterable[BacktestEvent]) -> BacktestResult:
        """Atomically commit a complete iterable, preserving its supplied order."""

        staged = copy.deepcopy(self._state)
        for event in tuple(events):
            self._apply_event(staged, event)
        result = self._result(staged)
        self._state = staged
        return result

    def run(
        self,
        events: Iterable[BacktestEvent],
        *,
        require_complete: bool | None = None,
    ) -> BacktestResult:
        """Atomically consume events and optionally require all ledger attestations."""

        staged = copy.deepcopy(self._state)
        for event in tuple(events):
            self._apply_event(staged, event)
        self._ensure_complete(staged, require_complete=require_complete)
        result = self._result(staged)
        self._state = staged
        return result

    def result(self, *, require_complete: bool | None = None) -> BacktestResult:
        """Snapshot committed state, optionally failing on missing ledger events."""

        self._ensure_complete(self._state, require_complete=require_complete)
        return self._result(self._state)

    def replay(self, events: Iterable[BacktestEvent]) -> BacktestResult:
        """Replay into an empty engine and return the deterministic result."""

        if self._state.events:
            raise BacktestEngineError("replay requires an engine with no committed events")
        return self.run(events)

    def _apply_event(self, state: _EngineState, event: BacktestEvent) -> None:
        # Revalidate at the trust boundary. Pydantic's model_copy/model_construct can
        # intentionally bypass validators, but a replay engine must not inherit that risk.
        event = _EVENT_ADAPTER.validate_python(event.model_dump(mode="python"))
        self._validate_clock(state, event)
        if isinstance(event, TargetEvent):
            self._apply_target(state, event)
        elif isinstance(event, OrderEvent):
            self._apply_order(state, event)
        elif isinstance(event, FillEvent):
            self._apply_fill(state, event)
        elif isinstance(event, FeeEvent):
            self._apply_fee(state, event)
        elif isinstance(event, CashEvent):
            self._apply_cash_attestation(state, event)
        elif isinstance(event, PositionEvent):
            self._apply_position_attestation(state, event)
        else:  # SignalEvent is a valid contract but not an engine instruction.
            raise BacktestEngineError(f"unsupported engine event type: {event.event_type}")
        state.event_ids.add(event.event_id)
        state.events.append(event)
        state.last_sequence = event.sequence
        state.last_event_time = event.event_time
        state.last_trading_day = event.trading_day

    def _validate_clock(self, state: _EngineState, event: BacktestEvent) -> None:
        if event.run_id != self._config.run_id:
            raise EventOrderError("event run_id does not match engine run_id")
        if event.event_id in state.event_ids:
            raise EventOrderError("event_id values must be unique")
        if event.sequence <= state.last_sequence:
            raise EventOrderError("event sequence must be strictly increasing")
        if state.last_event_time is not None and event.event_time < state.last_event_time:
            raise EventOrderError("event_time cannot regress")

    def _apply_target(self, state: _EngineState, event: TargetEvent) -> None:
        if event.direction is SignalDirection.SHORT and not self._config.allow_short_sales:
            raise LedgerInvariantError("SHORT target requires allow_short_sales")
        key = (event.signal_event_id, event.instrument_id, event.instrument_type)
        if key in state.targets:
            raise EventLinkError("a signal may create only one target per instrument")
        state.targets[key] = event

    def _apply_order(self, state: _EngineState, event: OrderEvent) -> None:
        record = state.orders.get(event.order_id)
        if record is None:
            self._create_order(state, event)
            return
        if event.previous_status is not record.latest.status:
            raise EventLinkError("order previous_status does not match engine state")
        if self._order_instruction(event) != self._order_instruction(record.created):
            raise EventLinkError("order instruction fields cannot change across transitions")
        if (
            event.status
            in {
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELED,
            }
            and event.filled_quantity != record.confirmed_quantity
        ):
            raise LedgerInvariantError(
                "order filled_quantity must equal economically confirmed fills"
            )
        if event.status is OrderStatus.CANCELED and self._pending_fill_quantity(
            state, event.order_id
        ):
            raise LedgerInvariantError(
                "an order with pending fill confirmations cannot be canceled"
            )
        record.latest = event

    def _create_order(self, state: _EngineState, event: OrderEvent) -> None:
        if event.previous_status is not None or event.status is not OrderStatus.CREATED:
            raise EventLinkError("an unknown order must begin in CREATED status")
        target = state.targets.get(
            (event.signal_event_id, event.instrument_id, event.instrument_type)
        )
        if self._config.require_target_for_order and target is None:
            raise EventLinkError("order does not reference a replayed target")
        if target is not None:
            self._validate_target_order(state, target, event)
            for existing in state.orders.values():
                if (
                    existing.created.signal_event_id == event.signal_event_id
                    and existing.created.instrument_id == event.instrument_id
                    and existing.created.instrument_type is event.instrument_type
                    and existing.latest.status
                    not in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED}
                ):
                    raise LedgerInvariantError("a target cannot have overlapping active orders")
        rule = self._market_rules.select(
            instrument_type=event.instrument_type,
            trading_day=event.trading_day,
        )
        rule.validate_order(event)
        state.orders[event.order_id] = _OrderRecord(created=event, latest=event)

    def _validate_target_order(
        self,
        state: _EngineState,
        target: TargetEvent,
        order: OrderEvent,
    ) -> None:
        if target.event_time > order.event_time:
            raise EventLinkError("order cannot precede its target")
        if target.target_quantity is None:
            return
        desired = target.target_quantity
        if target.direction is SignalDirection.FLAT:
            desired = Decimal(0)
        elif target.direction is SignalDirection.SHORT:
            desired = -desired
        ledger = state.positions.get((order.instrument_id, order.instrument_type))
        current = ledger.quantity if ledger is not None else Decimal(0)
        delta = desired - current
        if delta == 0:
            raise LedgerInvariantError("target is already satisfied")
        expected_side = Side.BUY if delta > 0 else Side.SELL
        if order.side is not expected_side or order.quantity != abs(delta):
            raise LedgerInvariantError("order does not exactly implement target_quantity")

    @staticmethod
    def _order_instruction(event: OrderEvent) -> tuple[object, ...]:
        return (
            event.instrument_id,
            event.instrument_type,
            event.signal_event_id,
            event.signal_time,
            event.side,
            event.order_type,
            event.quantity,
            event.limit_price,
            event.reference_price,
            event.price_observed_at,
            event.price_available_at,
        )

    def _apply_fill(self, state: _EngineState, event: FillEvent) -> None:
        record = state.fills.get(event.fill_id)
        if record is None:
            self._create_fill(state, event)
            return
        if event.previous_status is not record.latest.status:
            raise EventLinkError("fill previous_status does not match engine state")
        if self._fill_instruction(event) != self._fill_instruction(record.created):
            raise EventLinkError("fill execution fields cannot change across transitions")
        if event.status is FillStatus.CONFIRMED:
            self._confirm_fill(state, record, event)
        elif event.status is FillStatus.REVERSED:
            raise LedgerInvariantError("fill reversals require explicit compensating events")
        record.latest = event

    def _create_fill(self, state: _EngineState, event: FillEvent) -> None:
        if event.previous_status is not None or event.status is not FillStatus.CREATED:
            raise EventLinkError("an unknown fill must begin in CREATED status")
        order = state.orders.get(event.order_id)
        if order is None:
            raise EventLinkError("fill references an unknown order")
        if order.latest.status not in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}:
            raise LedgerInvariantError("fills require an accepted order")
        self._validate_fill_order(event, order)
        active = self._active_fill_quantity(state, event.order_id)
        if active + event.quantity > order.created.quantity:
            raise LedgerInvariantError("fills exceed the order quantity")
        state.fills[event.fill_id] = _FillRecord(created=event, latest=event)

    def _confirm_fill(
        self,
        state: _EngineState,
        record: _FillRecord,
        event: FillEvent,
    ) -> None:
        if record.economic_applied:
            raise LedgerInvariantError("fill economics can be applied only once")
        order = state.orders.get(event.order_id)
        if order is None:  # pragma: no cover - protected by fill creation
            raise EventLinkError("fill references an unknown order")
        if order.latest.status not in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}:
            raise LedgerInvariantError("fill confirmation requires an accepted order")
        if order.confirmed_quantity + event.quantity > order.created.quantity:
            raise LedgerInvariantError("confirmed fills exceed the order quantity")
        if self._fee_rule_book is not None and any(
            item.economic_applied and item.fee is None for item in state.fills.values()
        ):
            raise LedgerInvariantError(
                "a prior confirmed fill requires its fee before another fill can confirm"
            )
        opened_long = self._apply_trade(state, record.created, source_event=event)
        record.economic_applied = True
        record.opened_long_quantity = opened_long
        ledger = state.positions[(event.instrument_id, event.instrument_type)]
        record.position_quantity_after = ledger.quantity
        record.position_cost_basis_after = ledger.cost_basis
        order.confirmed_quantity += event.quantity

    def _validate_fill_order(self, fill: FillEvent, order: _OrderRecord) -> None:
        created = order.created
        if (
            fill.instrument_id != created.instrument_id
            or fill.instrument_type is not created.instrument_type
            or fill.signal_event_id != created.signal_event_id
            or fill.signal_time != created.signal_time
            or fill.side is not created.side
            or fill.order_time != created.event_time
        ):
            raise EventLinkError("fill fields do not match the referenced order")
        if created.order_type is OrderType.LIMIT:
            assert created.limit_price is not None
            if created.side is Side.BUY and fill.price > created.limit_price:
                raise LedgerInvariantError("BUY limit fill price exceeds limit_price")
            if created.side is Side.SELL and fill.price < created.limit_price:
                raise LedgerInvariantError("SELL limit fill price is below limit_price")

    @staticmethod
    def _fill_instruction(event: FillEvent) -> tuple[object, ...]:
        return (
            event.instrument_id,
            event.instrument_type,
            event.order_id,
            event.signal_event_id,
            event.signal_time,
            event.order_time,
            event.side,
            event.quantity,
            event.price,
            event.gross_amount,
            event.price_observed_at,
            event.price_available_at,
        )

    @staticmethod
    def _active_fill_quantity(state: _EngineState, order_id: str) -> Decimal:
        return sum(
            (
                record.created.quantity
                for record in state.fills.values()
                if record.created.order_id == order_id
                and record.latest.status not in {FillStatus.CANCELED, FillStatus.REVERSED}
            ),
            Decimal(0),
        )

    @staticmethod
    def _pending_fill_quantity(state: _EngineState, order_id: str) -> Decimal:
        return sum(
            (
                record.created.quantity
                for record in state.fills.values()
                if record.created.order_id == order_id
                and record.latest.status is FillStatus.CREATED
            ),
            Decimal(0),
        )

    def _apply_trade(
        self,
        state: _EngineState,
        fill: FillEvent,
        *,
        source_event: FillEvent,
    ) -> Decimal:
        cash_delta = fill.gross_amount if fill.side is Side.SELL else -fill.gross_amount
        self._move_cash(state, cash_delta, source_event)
        key = (fill.instrument_id, fill.instrument_type)
        ledger = state.positions.setdefault(
            key,
            _PositionLedger(instrument_type=fill.instrument_type),
        )
        opened_long = Decimal(0)
        if fill.side is Side.BUY:
            opened_long = self._apply_buy(ledger, fill)
        else:
            self._apply_sell(ledger, fill)
        self._record_position_expectation(state, source_event, fill.instrument_id, ledger)
        return opened_long

    def _apply_buy(self, ledger: _PositionLedger, fill: FillEvent) -> Decimal:
        quantity = fill.quantity
        old_quantity = ledger.quantity
        if old_quantity < 0:
            covered = min(quantity, -old_quantity)
            remaining_short = -old_quantity - covered
            if remaining_short:
                ledger.cost_basis = self._pro_rata(
                    ledger.cost_basis,
                    remaining_short,
                    -old_quantity,
                )
            else:
                ledger.cost_basis = Decimal(0)
            ledger.quantity += covered
            quantity -= covered
        if quantity:
            if ledger.quantity < 0:  # pragma: no cover - quantity was exhausted covering
                return Decimal(0)
            ledger.quantity += quantity
            ledger.cost_basis += quantity * fill.price
            rule = self._market_rules.select(
                instrument_type=fill.instrument_type,
                trading_day=fill.trading_day,
            )
            sellable_on = rule.sellable_on(acquired_on=fill.trading_day)
            if sellable_on < fill.trading_day:
                raise LedgerInvariantError("MarketRule.sellable_on cannot precede acquisition")
            ledger.lots.append(_Lot(quantity=quantity, sellable_on=sellable_on))
        return quantity

    def _apply_sell(self, ledger: _PositionLedger, fill: FillEvent) -> None:
        long_quantity = max(ledger.quantity, Decimal(0))
        closing = min(fill.quantity, long_quantity)
        sellable = self._sellable_quantity(ledger, fill.trading_day)
        if closing > sellable and not self._config.allow_unsettled_sales:
            raise LedgerInvariantError("sell fill exceeds sellable_quantity")
        excess = fill.quantity - long_quantity
        if excess > 0 and not self._config.allow_short_sales:
            raise LedgerInvariantError("sell fill exceeds owned quantity")
        if closing:
            self._consume_lots(
                ledger,
                closing,
                trading_day=fill.trading_day,
                allow_unsettled=self._config.allow_unsettled_sales,
            )
            remaining = long_quantity - closing
            ledger.cost_basis = (
                self._pro_rata(ledger.cost_basis, remaining, long_quantity)
                if remaining
                else Decimal(0)
            )
            ledger.quantity -= closing
        if excess > 0:
            ledger.quantity -= excess
            ledger.cost_basis += excess * fill.price
            ledger.lots.clear()

    @staticmethod
    def _consume_lots(
        ledger: _PositionLedger,
        quantity: Decimal,
        *,
        trading_day: date,
        allow_unsettled: bool,
    ) -> None:
        eligible = [lot for lot in ledger.lots if lot.sellable_on <= trading_day]
        ineligible = [lot for lot in ledger.lots if lot.sellable_on > trading_day]
        ordered = eligible + ineligible if allow_unsettled else eligible
        remaining = quantity
        for lot in ordered:
            consumed = min(lot.quantity, remaining)
            lot.quantity -= consumed
            remaining -= consumed
            if not remaining:
                break
        if remaining:
            raise LedgerInvariantError("position lot ledger cannot satisfy sell fill")
        ledger.lots = [lot for lot in ledger.lots if lot.quantity]

    @staticmethod
    def _pro_rata(value: Decimal, numerator: Decimal, denominator: Decimal) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            return value * numerator / denominator

    def _apply_fee(self, state: _EngineState, event: FeeEvent) -> None:
        fill = state.fills.get(event.fill_id)
        if fill is None:
            raise EventLinkError("fee references an unknown fill")
        if not fill.economic_applied or fill.latest.status not in {
            FillStatus.CONFIRMED,
            FillStatus.SETTLED,
        }:
            raise LedgerInvariantError("fee requires a confirmed, active fill")
        if fill.fee is not None:
            raise LedgerInvariantError("a fill may have only one fee event")
        created = fill.created
        if (
            event.instrument_id != created.instrument_id
            or event.instrument_type is not created.instrument_type
            or event.side is not created.side
            or event.fill_time != created.event_time
        ):
            raise EventLinkError("fee fields do not match the referenced fill")
        if event.trading_day != created.trading_day:
            raise EventLinkError("fee must be recorded on its fill trading day")
        if event.currency != self._config.currency:
            raise LedgerInvariantError("fee currency does not match engine currency")
        if self._fee_rule_book is not None:
            rule = self._fee_rule_book.select(
                instrument_type=event.instrument_type,
                side=event.side,
                trading_day=created.trading_day,
                version=event.fee_rule_version,
            )
            expected = rule.assess(created.gross_amount)
            actual = (
                event.commission,
                event.stamp_duty,
                event.transfer_fee,
                event.other_fee,
                event.total_amount,
            )
            required = (
                expected.commission,
                expected.stamp_duty,
                expected.transfer_fee,
                expected.other_fee,
                expected.total_amount,
            )
            if actual != required:
                raise LedgerInvariantError("fee does not match its pinned FeeRule")
        if event.total_amount:
            self._move_cash(state, -event.total_amount, event)
        fill.fee = event
        if event.side is Side.BUY and fill.opened_long_quantity and event.total_amount:
            ledger = state.positions[(event.instrument_id, event.instrument_type)]
            if (
                ledger.quantity != fill.position_quantity_after
                or ledger.cost_basis != fill.position_cost_basis_after
            ):
                raise LedgerInvariantError(
                    "buy fee cannot be delayed past another position-changing fill"
                )
            allocated = self._pro_rata(
                event.total_amount,
                fill.opened_long_quantity,
                created.quantity,
            )
            ledger.cost_basis += allocated
            self._record_position_expectation(state, event, event.instrument_id, ledger)

    def _move_cash(
        self,
        state: _EngineState,
        delta: Decimal,
        source_event: BacktestEvent,
    ) -> None:
        balance = state.cash + delta
        if balance < 0 and not self._config.allow_negative_cash:
            raise LedgerInvariantError("cash movement would create a negative balance")
        state.cash = balance
        if not delta:
            return
        if source_event.event_id in state.expected_cash:
            raise LedgerInvariantError("source event produced duplicate cash movements")
        state.expected_cash[source_event.event_id] = _ExpectedCash(
            source_time=source_event.event_time,
            trading_day=source_event.trading_day,
            direction=CashDirection.CREDIT if delta > 0 else CashDirection.DEBIT,
            amount=abs(delta),
            balance_after=balance,
        )

    def _record_position_expectation(
        self,
        state: _EngineState,
        source_event: BacktestEvent,
        instrument_id: str,
        ledger: _PositionLedger,
    ) -> None:
        if source_event.event_id in state.expected_positions:
            raise LedgerInvariantError("source event produced duplicate position movements")
        state.expected_positions[source_event.event_id] = _ExpectedPosition(
            source_time=source_event.event_time,
            trading_day=source_event.trading_day,
            snapshot=self._position_snapshot(
                instrument_id,
                ledger,
                trading_day=source_event.trading_day,
            ),
        )

    def _apply_cash_attestation(self, state: _EngineState, event: CashEvent) -> None:
        expected = state.expected_cash.get(event.source_event_id)
        if expected is None:
            raise EventLinkError("cash event has no unmatched economic source")
        if event.account_id != self._config.account_id or event.currency != self._config.currency:
            raise LedgerInvariantError("cash account or currency does not match engine")
        if event.event_time < expected.source_time or event.trading_day != expected.trading_day:
            raise EventLinkError("cash event must attest on its source trading day")
        if (
            event.direction is not expected.direction
            or event.amount != expected.amount
            or event.balance_after != expected.balance_after
            or event.balance_after != state.cash
        ):
            raise LedgerInvariantError("cash event does not match the computed ledger movement")
        del state.expected_cash[event.source_event_id]

    def _apply_position_attestation(self, state: _EngineState, event: PositionEvent) -> None:
        expected = state.expected_positions.get(event.source_event_id)
        if expected is None:
            raise EventLinkError("position event has no unmatched economic source")
        if event.event_time < expected.source_time or event.trading_day != expected.trading_day:
            raise EventLinkError("position event must attest on its source trading day")
        key = (event.instrument_id, event.instrument_type)
        ledger = state.positions.get(key)
        current = (
            self._position_snapshot(event.instrument_id, ledger, event.trading_day)
            if ledger is not None
            else PositionSnapshot(
                instrument_id=event.instrument_id,
                instrument_type=event.instrument_type,
                quantity=Decimal(0),
                sellable_quantity=Decimal(0),
                average_cost=Decimal(0),
            )
        )
        actual = (event.quantity, event.sellable_quantity, event.average_cost)
        expected_values = (
            expected.snapshot.quantity,
            expected.snapshot.sellable_quantity,
            expected.snapshot.average_cost,
        )
        current_values = (current.quantity, current.sellable_quantity, current.average_cost)
        if (
            event.instrument_id != expected.snapshot.instrument_id
            or event.instrument_type is not expected.snapshot.instrument_type
            or actual != expected_values
            or actual != current_values
        ):
            raise LedgerInvariantError("position event does not match the computed ledger state")
        del state.expected_positions[event.source_event_id]

    def _ensure_complete(
        self,
        state: _EngineState,
        *,
        require_complete: bool | None,
    ) -> None:
        required = (
            self._config.require_ledger_events if require_complete is None else require_complete
        )
        if required and (state.expected_cash or state.expected_positions):
            raise LedgerInvariantError("replay is missing required cash or position events")
        require_fees = self._fee_rule_book is not None and require_complete is not False
        if require_fees and any(
            fill.economic_applied and fill.fee is None for fill in state.fills.values()
        ):
            raise LedgerInvariantError("replay is missing a required fee event")

    def _result(self, state: _EngineState) -> BacktestResult:
        as_of_day = state.last_trading_day or date.min
        positions = tuple(
            self._position_snapshot(instrument_id, ledger, as_of_day)
            for (instrument_id, _), ledger in sorted(
                state.positions.items(),
                key=lambda item: (item[0][0], item[0][1].value),
            )
            if ledger.quantity
        )
        orders = tuple(
            OrderSnapshot(
                order_id=order_id,
                instrument_id=record.created.instrument_id,
                instrument_type=record.created.instrument_type,
                side=record.created.side,
                order_type=record.created.order_type,
                quantity=record.created.quantity,
                status=record.latest.status,
                filled_quantity=record.latest.filled_quantity,
                confirmed_quantity=record.confirmed_quantity,
            )
            for order_id, record in sorted(state.orders.items())
        )
        serialized = serialize_event_stream(state.events) if state.events else "[]"
        return BacktestResult(
            run_id=self._config.run_id,
            account_id=self._config.account_id,
            currency=self._config.currency,
            initial_cash=self._config.initial_cash,
            cash_balance=state.cash,
            positions=positions,
            orders=orders,
            events=tuple(state.events),
            event_log=serialized,
            event_log_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_replay(
        cls,
        *,
        config: EventBacktestConfig,
        market_rules: Iterable[MarketRule],
        events: Iterable[BacktestEvent],
        fee_rule_book: FeeRuleBook | None = None,
        require_complete: bool | None = None,
    ) -> tuple[Self, BacktestResult]:
        """Construct a new engine and atomically replay a supplied stream."""

        engine = cls(config, market_rules, fee_rule_book)
        return engine, engine.run(events, require_complete=require_complete)

    @staticmethod
    def _sellable_quantity(ledger: _PositionLedger, trading_day: date) -> Decimal:
        if ledger.quantity <= 0:
            return Decimal(0)
        return sum(
            (lot.quantity for lot in ledger.lots if lot.sellable_on <= trading_day),
            Decimal(0),
        )

    def _position_snapshot(
        self,
        instrument_id: str,
        ledger: _PositionLedger,
        trading_day: date,
    ) -> PositionSnapshot:
        absolute_quantity = abs(ledger.quantity)
        average_cost = (
            self._pro_rata(ledger.cost_basis, Decimal(1), absolute_quantity)
            if absolute_quantity
            else Decimal(0)
        )
        return PositionSnapshot(
            instrument_id=instrument_id,
            instrument_type=ledger.instrument_type,
            quantity=ledger.quantity,
            sellable_quantity=self._sellable_quantity(ledger, trading_day),
            average_cost=average_cost,
        )


BacktestEngine = EventDrivenBacktestEngine


__all__ = [
    "BacktestEngine",
    "BacktestEngineError",
    "BacktestResult",
    "EngineConfig",
    "EventBacktestConfig",
    "EventDrivenBacktestEngine",
    "EventLinkError",
    "EventOrderError",
    "InitialPosition",
    "LedgerInvariantError",
    "MarketRuleNotFoundError",
    "OrderSnapshot",
    "PositionSnapshot",
]
