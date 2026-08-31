"""Immutable, strategy-neutral event contracts for deterministic backtests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware


def _parse_decimal(value: object) -> Decimal:
    """Parse exact numeric inputs while rejecting binary floats and non-finite values."""

    if isinstance(value, (bool, float)):
        raise ValueError("decimal values must not be booleans or binary floats")
    if not isinstance(value, (Decimal, int, str)):
        raise ValueError("decimal values must be Decimal, integer, or string")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal value") from error
    if not result.is_finite():
        raise ValueError("decimal values must be finite")
    return result


ExactDecimal = Annotated[Decimal, BeforeValidator(_parse_decimal)]


def _non_empty(value: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError("identifier and version fields must be non-empty")
    return result


def _validate_known_price(
    *,
    price: Decimal | None,
    observed_at: datetime | None,
    available_at: datetime | None,
    decision_time: datetime,
) -> None:
    fields = (price, observed_at, available_at)
    if all(value is None for value in fields):
        return
    if any(value is None for value in fields):
        raise ValueError("price, price_observed_at, and price_available_at must be set together")
    assert price is not None
    assert observed_at is not None
    assert available_at is not None
    ensure_aware(observed_at)
    ensure_aware(available_at)
    if price <= 0:
        raise ValueError("price must be positive")
    if observed_at > available_at:
        raise ValueError("price_available_at cannot precede price_observed_at")
    if available_at > decision_time:
        raise ValueError("future price information is not allowed")


class TradableInstrumentType(StrEnum):
    """Instrument types supported by the first backtest contract."""

    STOCK = "STOCK"
    ETF = "ETF"


class Side(StrEnum):
    """Unambiguous order and fill direction."""

    BUY = "BUY"
    SELL = "SELL"


class SignalDirection(StrEnum):
    """Signed signal direction; market eligibility belongs to a market rule."""

    LONG = "LONG"
    FLAT = "FLAT"
    SHORT = "SHORT"


class OrderType(StrEnum):
    """Supported order pricing instructions."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    """Lifecycle states for an order."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class FillStatus(StrEnum):
    """Lifecycle states for a fill record."""

    CREATED = "CREATED"
    CONFIRMED = "CONFIRMED"
    SETTLED = "SETTLED"
    CANCELED = "CANCELED"
    REVERSED = "REVERSED"


class CashDirection(StrEnum):
    """Direction of a positive cash movement amount."""

    CREDIT = "CREDIT"
    DEBIT = "DEBIT"


ORDER_STATE_TRANSITIONS: Mapping[OrderStatus, frozenset[OrderStatus]] = MappingProxyType(
    {
        OrderStatus.CREATED: frozenset({OrderStatus.SUBMITTED, OrderStatus.CANCELED}),
        OrderStatus.SUBMITTED: frozenset(
            {OrderStatus.ACCEPTED, OrderStatus.CANCELED, OrderStatus.REJECTED}
        ),
        OrderStatus.ACCEPTED: frozenset(
            {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED}
        ),
        OrderStatus.PARTIALLY_FILLED: frozenset(
            {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED}
        ),
        OrderStatus.FILLED: frozenset(),
        OrderStatus.CANCELED: frozenset(),
        OrderStatus.REJECTED: frozenset(),
    }
)

FILL_STATE_TRANSITIONS: Mapping[FillStatus, frozenset[FillStatus]] = MappingProxyType(
    {
        FillStatus.CREATED: frozenset({FillStatus.CONFIRMED, FillStatus.CANCELED}),
        FillStatus.CONFIRMED: frozenset({FillStatus.SETTLED, FillStatus.REVERSED}),
        FillStatus.SETTLED: frozenset({FillStatus.REVERSED}),
        FillStatus.CANCELED: frozenset(),
        FillStatus.REVERSED: frozenset(),
    }
)


class BacktestEventBase(BaseModel):
    """Common replay identity and market-clock boundary for every event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str
    schema_version: Literal["1"] = "1"
    event_id: str
    run_id: str
    sequence: int = Field(ge=0)
    event_time: datetime
    trading_day: date

    @field_validator("event_id", "run_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("event_time")
    @classmethod
    def validate_event_time(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @model_validator(mode="after")
    def validate_trading_day(self) -> BacktestEventBase:
        if self.event_time.astimezone(SHANGHAI_TZ).date() != self.trading_day:
            raise ValueError("trading_day must match event_time in Asia/Shanghai")
        return self


class InstrumentEventBase(BacktestEventBase):
    """Common security identity shared by instrument-level events."""

    instrument_id: str
    instrument_type: TradableInstrumentType

    @field_validator("instrument_id")
    @classmethod
    def validate_instrument_id(cls, value: str) -> str:
        return _non_empty(value)


class SignalEvent(InstrumentEventBase):
    """Strategy output using only price information available by event_time."""

    event_type: Literal["SIGNAL"] = "SIGNAL"
    direction: SignalDirection
    strength: ExactDecimal
    strategy_version: str
    data_version: str
    reference_price: ExactDecimal | None = None
    price_observed_at: datetime | None = None
    price_available_at: datetime | None = None

    @field_validator("strategy_version", "data_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _non_empty(value)

    @model_validator(mode="after")
    def validate_signal(self) -> SignalEvent:
        if self.strength < -1 or self.strength > 1:
            raise ValueError("signal strength must be between -1 and 1")
        if self.direction is SignalDirection.LONG and self.strength <= 0:
            raise ValueError("LONG signal strength must be positive")
        if self.direction is SignalDirection.FLAT and self.strength != 0:
            raise ValueError("FLAT signal strength must be zero")
        if self.direction is SignalDirection.SHORT and self.strength >= 0:
            raise ValueError("SHORT signal strength must be negative")
        _validate_known_price(
            price=self.reference_price,
            observed_at=self.price_observed_at,
            available_at=self.price_available_at,
            decision_time=self.event_time,
        )
        return self


class TargetEvent(InstrumentEventBase):
    """Desired exposure derived from a prior signal, independent of execution rules."""

    event_type: Literal["TARGET"] = "TARGET"
    signal_event_id: str
    signal_time: datetime
    direction: SignalDirection
    target_quantity: ExactDecimal | None = None
    target_weight: ExactDecimal | None = None

    @field_validator("signal_event_id")
    @classmethod
    def validate_signal_event_id(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("signal_time")
    @classmethod
    def validate_signal_time(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @model_validator(mode="after")
    def validate_target(self) -> TargetEvent:
        if self.signal_time > self.event_time:
            raise ValueError("target event cannot precede its signal")
        if (self.target_quantity is None) == (self.target_weight is None):
            raise ValueError("exactly one target_quantity or target_weight is required")
        magnitude = self.target_quantity if self.target_quantity is not None else self.target_weight
        assert magnitude is not None
        if magnitude < 0:
            raise ValueError("target magnitude cannot be negative; use direction for sign")
        if self.direction is SignalDirection.FLAT and magnitude != 0:
            raise ValueError("FLAT target magnitude must be zero")
        if self.direction is not SignalDirection.FLAT and magnitude == 0:
            raise ValueError("non-FLAT target magnitude must be positive")
        return self


class OrderEvent(InstrumentEventBase):
    """One immutable order lifecycle transition."""

    event_type: Literal["ORDER"] = "ORDER"
    order_id: str
    signal_event_id: str
    signal_time: datetime
    side: Side
    order_type: OrderType
    quantity: ExactDecimal
    limit_price: ExactDecimal | None = None
    status: OrderStatus
    previous_status: OrderStatus | None = None
    filled_quantity: ExactDecimal = Decimal(0)
    reference_price: ExactDecimal | None = None
    price_observed_at: datetime | None = None
    price_available_at: datetime | None = None

    @field_validator("order_id", "signal_event_id")
    @classmethod
    def validate_order_identifier(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("signal_time")
    @classmethod
    def validate_signal_time(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @model_validator(mode="after")
    def validate_order(self) -> OrderEvent:
        if self.signal_time > self.event_time:
            raise ValueError("order event cannot precede its signal")
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if self.filled_quantity < 0 or self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity must be between zero and order quantity")
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("limit orders require a positive limit_price")
        elif self.limit_price is not None:
            raise ValueError("market orders cannot include limit_price")
        self._validate_status()
        _validate_known_price(
            price=self.reference_price,
            observed_at=self.price_observed_at,
            available_at=self.price_available_at,
            decision_time=self.event_time,
        )
        return self

    def _validate_status(self) -> None:
        if self.previous_status is None:
            if self.status is not OrderStatus.CREATED:
                raise ValueError("an initial order event must have CREATED status")
        elif self.status not in ORDER_STATE_TRANSITIONS[self.previous_status]:
            raise ValueError(
                f"invalid order transition: {self.previous_status.value} -> {self.status.value}"
            )
        if self.status is OrderStatus.PARTIALLY_FILLED and not (
            0 < self.filled_quantity < self.quantity
        ):
            raise ValueError("PARTIALLY_FILLED requires a partial filled_quantity")
        if self.status is OrderStatus.FILLED and self.filled_quantity != self.quantity:
            raise ValueError("FILLED requires filled_quantity equal to quantity")
        if (
            self.status
            in {
                OrderStatus.CREATED,
                OrderStatus.SUBMITTED,
                OrderStatus.ACCEPTED,
                OrderStatus.REJECTED,
            }
            and self.filled_quantity != 0
        ):
            raise ValueError(f"{self.status.value} order cannot have a filled quantity")


class FillEvent(InstrumentEventBase):
    """One immutable fill lifecycle transition tied to known execution data."""

    event_type: Literal["FILL"] = "FILL"
    fill_id: str
    order_id: str
    signal_event_id: str
    signal_time: datetime
    order_time: datetime
    side: Side
    quantity: ExactDecimal
    price: ExactDecimal
    gross_amount: ExactDecimal
    price_observed_at: datetime
    price_available_at: datetime
    status: FillStatus
    previous_status: FillStatus | None = None

    @field_validator("fill_id", "order_id", "signal_event_id")
    @classmethod
    def validate_fill_identifier(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("signal_time", "order_time", "price_observed_at", "price_available_at")
    @classmethod
    def validate_linked_time(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @model_validator(mode="after")
    def validate_fill(self) -> FillEvent:
        if self.signal_time > self.order_time:
            raise ValueError("fill order_time cannot precede signal_time")
        if self.order_time > self.event_time:
            raise ValueError("fill event cannot precede its order")
        if self.quantity <= 0 or self.price <= 0 or self.gross_amount <= 0:
            raise ValueError("fill quantity, price, and gross_amount must be positive")
        if self.gross_amount != self.quantity * self.price:
            raise ValueError("gross_amount must equal quantity multiplied by price")
        if self.price_observed_at < self.order_time:
            raise ValueError("fill price cannot be observed before the order")
        _validate_known_price(
            price=self.price,
            observed_at=self.price_observed_at,
            available_at=self.price_available_at,
            decision_time=self.event_time,
        )
        if self.previous_status is None:
            if self.status is not FillStatus.CREATED:
                raise ValueError("an initial fill event must have CREATED status")
        elif self.status not in FILL_STATE_TRANSITIONS[self.previous_status]:
            raise ValueError(
                f"invalid fill transition: {self.previous_status.value} -> {self.status.value}"
            )
        return self


class FeeEvent(InstrumentEventBase):
    """Auditable fee assessment linked to a fill and a versioned fee rule."""

    event_type: Literal["FEE"] = "FEE"
    fill_id: str
    fill_time: datetime
    side: Side
    fee_rule_version: str
    commission: ExactDecimal = Decimal(0)
    stamp_duty: ExactDecimal = Decimal(0)
    transfer_fee: ExactDecimal = Decimal(0)
    other_fee: ExactDecimal = Decimal(0)
    total_amount: ExactDecimal
    currency: str = "CNY"

    @field_validator("fill_id", "fee_rule_version")
    @classmethod
    def validate_fee_identifier(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("fill_time")
    @classmethod
    def validate_fill_time(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        result = value.strip().upper()
        if len(result) != 3 or not result.isalpha():
            raise ValueError("currency must be a three-letter alphabetic code")
        return result

    @model_validator(mode="after")
    def validate_fee(self) -> FeeEvent:
        if self.fill_time > self.event_time:
            raise ValueError("fee event cannot precede its fill")
        components = (self.commission, self.stamp_duty, self.transfer_fee, self.other_fee)
        if any(amount < 0 for amount in components) or self.total_amount < 0:
            raise ValueError("fee amounts cannot be negative")
        if sum(components, Decimal(0)) != self.total_amount:
            raise ValueError("total_amount must equal the fee component sum")
        return self


class PositionEvent(InstrumentEventBase):
    """Position snapshot after applying a referenced event."""

    event_type: Literal["POSITION"] = "POSITION"
    source_event_id: str
    quantity: ExactDecimal
    sellable_quantity: ExactDecimal
    average_cost: ExactDecimal
    market_price: ExactDecimal | None = None
    price_observed_at: datetime | None = None
    price_available_at: datetime | None = None

    @field_validator("source_event_id")
    @classmethod
    def validate_source_event_id(cls, value: str) -> str:
        return _non_empty(value)

    @model_validator(mode="after")
    def validate_position(self) -> PositionEvent:
        if self.quantity < 0 or self.sellable_quantity < 0 or self.average_cost < 0:
            raise ValueError("position quantities and average_cost cannot be negative")
        if self.sellable_quantity > self.quantity:
            raise ValueError("sellable_quantity cannot exceed quantity")
        _validate_known_price(
            price=self.market_price,
            observed_at=self.price_observed_at,
            available_at=self.price_available_at,
            decision_time=self.event_time,
        )
        return self


class CashEvent(BacktestEventBase):
    """Cash ledger movement represented by an explicit direction and positive amount."""

    event_type: Literal["CASH"] = "CASH"
    account_id: str
    source_event_id: str
    direction: CashDirection
    amount: ExactDecimal
    balance_after: ExactDecimal
    currency: str = "CNY"

    @field_validator("account_id", "source_event_id")
    @classmethod
    def validate_cash_identifier(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("currency")
    @classmethod
    def validate_cash_currency(cls, value: str) -> str:
        result = value.strip().upper()
        if len(result) != 3 or not result.isalpha():
            raise ValueError("currency must be a three-letter alphabetic code")
        return result

    @model_validator(mode="after")
    def validate_cash(self) -> CashEvent:
        if self.amount <= 0:
            raise ValueError("cash movement amount must be positive")
        return self


BacktestEvent = Annotated[
    SignalEvent | TargetEvent | OrderEvent | FillEvent | FeeEvent | PositionEvent | CashEvent,
    Field(discriminator="event_type"),
]
