"""Serializable, strategy-independent backtest event contracts."""

import json
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol

from quant_agent.core.time import ensure_aware


class AssetType(StrEnum):
    STOCK = "STOCK"
    ETF = "ETF"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    CREATED = "CREATED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


_LEGAL_TRANSITIONS = {
    OrderStatus.CREATED: {OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.CANCELLED},
    OrderStatus.ACCEPTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.CANCELLED: set(),
}


@dataclass(frozen=True, slots=True)
class SignalEvent:
    signal_id: str
    strategy_id: str
    instrument_id: str
    asset_type: AssetType
    generated_at: datetime
    target_weight: float
    data_version: str
    strategy_version: str
    parameter_version: str

    def __post_init__(self) -> None:
        ensure_aware(self.generated_at)
        if not 0 <= self.target_weight <= 1:
            raise ValueError("target weight must be in [0, 1]")

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "SignalEvent":
        values = json.loads(payload)
        values["asset_type"] = AssetType(values["asset_type"])
        values["generated_at"] = datetime.fromisoformat(values["generated_at"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    signal_id: str
    instrument_id: str
    asset_type: AssetType
    side: Side
    quantity: int
    order_type: OrderType
    created_at: datetime
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = 0
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        ensure_aware(self.created_at)
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit price must be positive")
        if not 0 <= self.filled_quantity <= self.quantity:
            raise ValueError("filled quantity is invalid")

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    def transition(
        self,
        status: OrderStatus,
        *,
        filled_quantity: int | None = None,
        rejection_reason: str | None = None,
    ) -> "Order":
        if status not in _LEGAL_TRANSITIONS[self.status]:
            raise ValueError(f"illegal order transition {self.status} -> {status}")
        total_filled = self.filled_quantity if filled_quantity is None else filled_quantity
        if status is OrderStatus.FILLED and total_filled != self.quantity:
            raise ValueError("filled order must have full quantity")
        if status is OrderStatus.PARTIALLY_FILLED and not (0 < total_filled < self.quantity):
            raise ValueError("partial fill quantity must be inside order quantity")
        return replace(
            self,
            status=status,
            filled_quantity=total_filled,
            rejection_reason=rejection_reason,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "Order":
        values = json.loads(payload)
        values["asset_type"] = AssetType(values["asset_type"])
        values["side"] = Side(values["side"])
        values["order_type"] = OrderType(values["order_type"])
        values["status"] = OrderStatus(values["status"])
        values["created_at"] = datetime.fromisoformat(values["created_at"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MarketBar:
    instrument_id: str
    asset_type: AssetType
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: float
    available_at: datetime
    suspended: bool = False
    limit_up_price: float | None = None
    limit_down_price: float | None = None

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("market prices must be positive")
        if self.volume < 0 or self.turnover < 0:
            raise ValueError("volume and turnover cannot be negative")


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    instrument_id: str
    side: Side
    quantity: int
    price: float
    gross_amount: float
    commission: float
    tax: float
    slippage: float
    filled_at: datetime
    fee_version: str

    def __post_init__(self) -> None:
        ensure_aware(self.filled_at)
        if self.quantity <= 0 or self.price <= 0:
            raise ValueError("fill quantity and price must be positive")

    @property
    def total_fees(self) -> float:
        return self.commission + self.tax + self.slippage

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "Fill":
        values = json.loads(payload)
        values["side"] = Side(values["side"])
        values["filled_at"] = datetime.fromisoformat(values["filled_at"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    version: str
    effective_from: date
    commission_rate: float
    minimum_commission: float
    stock_sell_tax_rate: float
    slippage_bps: float

    def __post_init__(self) -> None:
        values = (
            self.commission_rate,
            self.minimum_commission,
            self.stock_sell_tax_rate,
            self.slippage_bps,
        )
        if any(value < 0 for value in values):
            raise ValueError("fee values cannot be negative")


class MarketRule(Protocol):
    """Market-specific order validation and matching interface."""

    def validate(self, order: Order, bar: MarketBar, *, available_quantity: int) -> str | None: ...

    def match(
        self,
        order: Order,
        bar: MarketBar,
        *,
        available_quantity: int,
    ) -> Fill | None: ...
