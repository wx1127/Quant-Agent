"""Effective-dated China stock/ETF mechanics, costs, and deterministic matching.

The module supplies one concrete :class:`~quant_agent.backtest.rules.MarketRule`
implementation.  It intentionally does not assert that any tax, lot, settlement, or
price-limit convention is permanent: callers inject effective-dated rules, session
states, fee schedules, and slippage models for the period being replayed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    Decimal,
    InvalidOperation,
)
from enum import StrEnum
from typing import Any

from quant_agent.backtest.contracts import (
    OrderEvent,
    OrderType,
    Side,
    TradableInstrumentType,
)
from quant_agent.backtest.rules import FeeRule, FeeRuleBook
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.features.tradeability import MarketTradeState

_ZERO = Decimal(0)
_ONE = Decimal(1)


class CNMarketRuleError(ValueError):
    """Base class for fail-closed China market-rule errors."""


class MarketStateUnavailableError(CNMarketRuleError):
    """Raised when no unambiguous session state was known at order time."""


class OrderNotTradableError(CNMarketRuleError):
    """Raised when an injected session state blocks the requested order side."""


class OrderQuantityError(CNMarketRuleError):
    """Raised when order quantity violates the active unit rule."""


class OrderPriceError(CNMarketRuleError):
    """Raised when an order price violates tick or daily-boundary rules."""


class SlippageModelNotFoundError(LookupError):
    """Raised when no effective slippage model covers a trading day."""


class MatchStatus(StrEnum):
    """Outcome of the deterministic capacity and price helper."""

    NO_FILL = "NO_FILL"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"


class NoFillReason(StrEnum):
    """Stable reasons returned by a non-executing match helper."""

    ZERO_CAPACITY = "ZERO_CAPACITY"
    PARTIAL_FILL_DISABLED = "PARTIAL_FILL_DISABLED"
    LIMIT_NOT_MARKETABLE = "LIMIT_NOT_MARKETABLE"


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _decimal(value: Decimal | int | str, field_name: str) -> Decimal:
    if isinstance(value, bool | float):
        raise ValueError(f"{field_name} must be an exact decimal input")
    try:
        normalized = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f"{field_name} must be a valid decimal") from error
    if not normalized.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _utc(value: datetime) -> str:
    return ensure_aware(value).astimezone(UTC).isoformat(timespec="microseconds")


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_integral(value: Decimal) -> bool:
    return value == value.to_integral_value()


def _is_tick_aligned(value: Decimal, tick: Decimal) -> bool:
    return value % tick == 0


def _floor_to_unit(value: Decimal, unit: Decimal) -> Decimal:
    units = (value / unit).to_integral_value(rounding=ROUND_FLOOR)
    return units * unit


@dataclass(frozen=True, slots=True)
class MarketSessionState:
    """Immutable order-time market state and matching capacity for one instrument day."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    trading_day: date
    observed_at: datetime
    available_at: datetime
    revision: str
    data_version: str
    state: MarketTradeState
    buy_allowed: bool
    sell_allowed: bool
    reference_price: Decimal
    lower_limit_price: Decimal | None
    upper_limit_price: Decimal | None
    available_quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            _non_empty(self.instrument_id, "instrument_id"),
        )
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        ensure_aware(self.observed_at)
        ensure_aware(self.available_at)
        if self.observed_at.astimezone(SHANGHAI_TZ).date() != self.trading_day:
            raise ValueError("observed_at date must equal trading_day in Asia/Shanghai")
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot precede observed_at")
        reference_price = _decimal(self.reference_price, "reference_price")
        available_quantity = _decimal(self.available_quantity, "available_quantity")
        object.__setattr__(self, "reference_price", reference_price)
        object.__setattr__(self, "available_quantity", available_quantity)
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        if available_quantity < 0 or not _is_integral(available_quantity):
            raise ValueError("available_quantity must be a non-negative whole number")
        if (self.lower_limit_price is None) != (self.upper_limit_price is None):
            raise ValueError("lower and upper limit prices must be supplied together")
        if self.lower_limit_price is not None and self.upper_limit_price is not None:
            lower = _decimal(self.lower_limit_price, "lower_limit_price")
            upper = _decimal(self.upper_limit_price, "upper_limit_price")
            object.__setattr__(self, "lower_limit_price", lower)
            object.__setattr__(self, "upper_limit_price", upper)
            if lower <= 0 or upper <= lower:
                raise ValueError("daily price limits must be positive and ascending")
            if reference_price < lower or reference_price > upper:
                raise ValueError("reference_price must lie inside daily price limits")
            if (
                self.state
                in {
                    MarketTradeState.ONE_PRICE_LIMIT_UP,
                    MarketTradeState.LIMIT_UP,
                }
                and reference_price != upper
            ):
                raise ValueError("limit-up state requires reference_price at the upper limit")
            if (
                self.state
                in {
                    MarketTradeState.ONE_PRICE_LIMIT_DOWN,
                    MarketTradeState.LIMIT_DOWN,
                }
                and reference_price != lower
            ):
                raise ValueError("limit-down state requires reference_price at the lower limit")
        elif self.state in {
            MarketTradeState.ONE_PRICE_LIMIT_UP,
            MarketTradeState.LIMIT_UP,
            MarketTradeState.ONE_PRICE_LIMIT_DOWN,
            MarketTradeState.LIMIT_DOWN,
        }:
            raise ValueError("price-limit states require daily limit prices")
        if self.state is MarketTradeState.SUSPENDED and (self.buy_allowed or self.sell_allowed):
            raise ValueError("suspended state cannot allow either order side")

    def fingerprint_payload(self) -> dict[str, str | bool | None]:
        """Return canonical state content used by matching identities."""

        return {
            "available_at": _utc(self.available_at),
            "available_quantity": canonical_decimal(self.available_quantity),
            "buy_allowed": self.buy_allowed,
            "data_version": self.data_version,
            "instrument_id": self.instrument_id,
            "instrument_type": self.instrument_type.value,
            "lower_limit_price": (
                canonical_decimal(self.lower_limit_price)
                if self.lower_limit_price is not None
                else None
            ),
            "observed_at": _utc(self.observed_at),
            "reference_price": canonical_decimal(self.reference_price),
            "revision": self.revision,
            "sell_allowed": self.sell_allowed,
            "state": self.state.value,
            "trading_day": self.trading_day.isoformat(),
            "upper_limit_price": (
                canonical_decimal(self.upper_limit_price)
                if self.upper_limit_price is not None
                else None
            ),
        }

    @property
    def state_hash(self) -> str:
        """Return a stable content identity for the injected market state."""

        return _hash(self.fingerprint_payload())


@dataclass(frozen=True, slots=True)
class CNSlippageModel:
    """Versioned adverse price-impact model with an explicit effective interval."""

    model_id: str
    version: str
    effective_from: date
    effective_to: date | None = None
    base_rate: Decimal = Decimal(0)
    participation_impact_rate: Decimal = Decimal(0)
    maximum_rate: Decimal = Decimal("0.10")

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _non_empty(self.model_id, "model_id"))
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        for field_name in ("base_rate", "participation_impact_rate", "maximum_rate"):
            normalized = _decimal(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
        if self.base_rate < 0 or self.participation_impact_rate < 0:
            raise ValueError("slippage rates cannot be negative")
        if self.maximum_rate < 0 or self.maximum_rate >= 1:
            raise ValueError("maximum_rate must be between zero and one")
        if self.base_rate > self.maximum_rate:
            raise ValueError("base_rate cannot exceed maximum_rate")

    def applies_on(self, trading_day: date) -> bool:
        """Return whether the model covers a trading day."""

        return self.effective_from <= trading_day and (
            self.effective_to is None or trading_day <= self.effective_to
        )

    def rate(self, participation_rate: Decimal | int | str) -> Decimal:
        """Return capped adverse slippage for an exact participation ratio."""

        participation = _decimal(participation_rate, "participation_rate")
        if participation < 0 or participation > 1:
            raise ValueError("participation_rate must be between zero and one")
        return min(
            self.base_rate + self.participation_impact_rate * participation,
            self.maximum_rate,
        )

    def execution_price(
        self,
        *,
        side: Side,
        reference_price: Decimal,
        participation_rate: Decimal,
        price_tick: Decimal,
    ) -> Decimal:
        """Apply adverse slippage and round against the trader to a valid tick."""

        reference = _decimal(reference_price, "reference_price")
        tick = _decimal(price_tick, "price_tick")
        if reference <= 0 or tick <= 0:
            raise ValueError("reference_price and price_tick must be positive")
        slippage = self.rate(participation_rate)
        raw = reference * (_ONE + slippage) if side is Side.BUY else reference * (_ONE - slippage)
        rounding = ROUND_CEILING if side is Side.BUY else ROUND_FLOOR
        units = (raw / tick).to_integral_value(rounding=rounding)
        result = units * tick
        if result <= 0:
            raise ValueError("slippage produced a non-positive execution price")
        return result

    @property
    def model_hash(self) -> str:
        """Return a stable identity for the complete model configuration."""

        return _hash(
            {
                "base_rate": canonical_decimal(self.base_rate),
                "effective_from": self.effective_from.isoformat(),
                "effective_to": self.effective_to.isoformat() if self.effective_to else None,
                "maximum_rate": canonical_decimal(self.maximum_rate),
                "model_id": self.model_id,
                "participation_impact_rate": canonical_decimal(self.participation_impact_rate),
                "version": self.version,
            }
        )


class CNSlippageModelBook:
    """Select the latest unambiguous effective slippage configuration."""

    def __init__(self, models: Iterable[CNSlippageModel]) -> None:
        self._models = tuple(models)
        if not self._models:
            raise SlippageModelNotFoundError("at least one slippage model is required")
        identities: set[date] = set()
        versions: set[tuple[str, str]] = set()
        for model in self._models:
            if model.effective_from in identities:
                raise ValueError("slippage models are ambiguous at an effective boundary")
            identity = (model.model_id, model.version)
            if identity in versions:
                raise ValueError("slippage model identity and version must be unique")
            identities.add(model.effective_from)
            versions.add(identity)

    def select(self, trading_day: date) -> CNSlippageModel:
        """Select the latest model that actually covers ``trading_day``."""

        candidates = tuple(model for model in self._models if model.applies_on(trading_day))
        if not candidates:
            raise SlippageModelNotFoundError(
                f"no effective slippage model for {trading_day.isoformat()}"
            )
        return max(candidates, key=lambda model: model.effective_from)


@dataclass(frozen=True, slots=True)
class CNFeeSchedule:
    """Effective-dated fee inputs used to construct shared side-specific FeeRules."""

    schedule_id: str
    version: str
    effective_from: date
    instrument_type: TradableInstrumentType
    commission_rate: Decimal
    minimum_commission: Decimal
    buy_stamp_duty_rate: Decimal = Decimal(0)
    sell_stamp_duty_rate: Decimal = Decimal(0)
    buy_transfer_fee_rate: Decimal = Decimal(0)
    sell_transfer_fee_rate: Decimal = Decimal(0)
    buy_other_fee_rate: Decimal = Decimal(0)
    sell_other_fee_rate: Decimal = Decimal(0)
    rounding_increment: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schedule_id",
            _non_empty(self.schedule_id, "schedule_id"),
        )
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        rate_fields = (
            "commission_rate",
            "buy_stamp_duty_rate",
            "sell_stamp_duty_rate",
            "buy_transfer_fee_rate",
            "sell_transfer_fee_rate",
            "buy_other_fee_rate",
            "sell_other_fee_rate",
        )
        for field_name in (*rate_fields, "minimum_commission", "rounding_increment"):
            normalized = _decimal(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
        if any(
            getattr(self, field_name) < 0 or getattr(self, field_name) > 1
            for field_name in rate_fields
        ):
            raise ValueError("fee rates must be between zero and one")
        if self.minimum_commission < 0:
            raise ValueError("minimum_commission cannot be negative")
        if self.rounding_increment <= 0:
            raise ValueError("rounding_increment must be positive")

    def version_for(self, side: Side) -> str:
        """Return the immutable side-specific version stored on FeeRule/FeeEvent."""

        return f"{self.schedule_id}/{self.version}/{side.value}"

    def rules(self) -> tuple[FeeRule, FeeRule]:
        """Build BUY and SELL FeeRules without assuming rates are permanent."""

        common: dict[str, Any] = {
            "commission_rate": self.commission_rate,
            "effective_from": self.effective_from,
            "instrument_type": self.instrument_type,
            "minimum_commission": self.minimum_commission,
            "rounding_increment": self.rounding_increment,
        }
        buy = FeeRule(
            **common,
            side=Side.BUY,
            version=self.version_for(Side.BUY),
            stamp_duty_rate=self.buy_stamp_duty_rate,
            transfer_fee_rate=self.buy_transfer_fee_rate,
            other_fee_rate=self.buy_other_fee_rate,
        )
        sell = FeeRule(
            **common,
            side=Side.SELL,
            version=self.version_for(Side.SELL),
            stamp_duty_rate=self.sell_stamp_duty_rate,
            transfer_fee_rate=self.sell_transfer_fee_rate,
            other_fee_rate=self.sell_other_fee_rate,
        )
        return buy, sell


def build_cn_fee_rule_book(schedules: Iterable[CNFeeSchedule]) -> FeeRuleBook:
    """Build the shared FeeRuleBook from stock/ETF effective-dated schedules."""

    frozen = tuple(schedules)
    if not frozen:
        raise ValueError("at least one CN fee schedule is required")
    rules = [
        rule
        for schedule in sorted(
            frozen,
            key=lambda item: (item.instrument_type.value, item.effective_from, item.version),
        )
        for rule in schedule.rules()
    ]
    return FeeRuleBook(rules)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Pure deterministic match helper output; it does not mutate an engine ledger."""

    order_id: str
    trading_day: date
    status: MatchStatus
    reason: NoFillReason | None
    requested_quantity: Decimal
    filled_quantity: Decimal
    unfilled_quantity: Decimal
    reference_price: Decimal
    execution_price: Decimal | None
    gross_amount: Decimal
    participation_rate: Decimal
    state_revision: str
    state_available_at: datetime
    state_hash: str
    market_rule_version: str
    slippage_model_version: str
    slippage_model_hash: str

    def __post_init__(self) -> None:
        values = (
            self.requested_quantity,
            self.filled_quantity,
            self.unfilled_quantity,
            self.reference_price,
            self.gross_amount,
            self.participation_rate,
        )
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("match numeric values must be finite and non-negative")
        if self.requested_quantity <= 0 or self.reference_price <= 0:
            raise ValueError("requested_quantity and reference_price must be positive")
        if self.filled_quantity + self.unfilled_quantity != self.requested_quantity:
            raise ValueError("filled and unfilled quantities must equal requested_quantity")
        if self.participation_rate > 1:
            raise ValueError("participation_rate cannot exceed one")
        if self.status is MatchStatus.NO_FILL:
            if self.reason is None or self.filled_quantity != 0 or self.execution_price is not None:
                raise ValueError("NO_FILL requires a reason, zero fill, and no execution price")
            if self.gross_amount != 0:
                raise ValueError("NO_FILL gross_amount must be zero")
        else:
            if self.reason is not None or self.execution_price is None or self.filled_quantity <= 0:
                raise ValueError(
                    "executed match requires price and quantity without a no-fill reason"
                )
            if self.gross_amount != self.filled_quantity * self.execution_price:
                raise ValueError("gross_amount must equal filled quantity times execution price")
            if self.status is MatchStatus.FILLED and self.unfilled_quantity != 0:
                raise ValueError("FILLED match cannot retain unfilled quantity")
            if self.status is MatchStatus.PARTIALLY_FILLED and self.unfilled_quantity <= 0:
                raise ValueError("PARTIALLY_FILLED match requires unfilled quantity")
        for digest in (self.state_hash, self.slippage_model_hash):
            if len(digest) != 64:
                raise ValueError("match hashes must be SHA-256 digests")


@dataclass(frozen=True, slots=True)
class CNMarketRule:
    """Effective-dated stock/ETF market mechanics implementing the shared protocol."""

    rule_id: str
    version: str
    effective_from: date
    instrument_type: TradableInstrumentType
    session_states: tuple[MarketSessionState, ...]
    trading_calendar: tuple[date, ...] = ()
    effective_to: date | None = None
    buy_lot_size: Decimal = Decimal(100)
    sell_lot_size: Decimal = Decimal(100)
    allow_odd_lot_liquidation: bool = True
    odd_lot_liquidation_order_ids: frozenset[str] = frozenset()
    settlement_days: int = 1
    price_tick: Decimal = Decimal("0.01")
    max_participation_rate: Decimal = Decimal("0.10")

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _non_empty(self.rule_id, "rule_id"))
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        if self.settlement_days < 0:
            raise ValueError("settlement_days cannot be negative")
        for field_name in (
            "buy_lot_size",
            "sell_lot_size",
            "price_tick",
            "max_participation_rate",
        ):
            normalized = _decimal(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
        if self.buy_lot_size <= 0 or not _is_integral(self.buy_lot_size):
            raise ValueError("buy_lot_size must be a positive whole number")
        if self.sell_lot_size <= 0 or not _is_integral(self.sell_lot_size):
            raise ValueError("sell_lot_size must be a positive whole number")
        if self.price_tick <= 0:
            raise ValueError("price_tick must be positive")
        if self.max_participation_rate <= 0 or self.max_participation_rate > 1:
            raise ValueError("max_participation_rate must be between zero exclusive and one")
        calendar = tuple(self.trading_calendar)
        if tuple(sorted(set(calendar))) != calendar:
            raise ValueError("trading_calendar must contain unique ascending trading dates")
        object.__setattr__(self, "trading_calendar", calendar)
        liquidation_ids = frozenset(
            _non_empty(value, "odd-lot liquidation order_id")
            for value in self.odd_lot_liquidation_order_ids
        )
        if liquidation_ids and not self.allow_odd_lot_liquidation:
            raise ValueError("odd-lot liquidation order IDs require the liquidation policy")
        object.__setattr__(self, "odd_lot_liquidation_order_ids", liquidation_ids)
        states = tuple(
            sorted(
                self.session_states,
                key=lambda item: (
                    item.trading_day,
                    item.instrument_id,
                    item.available_at,
                    item.revision,
                ),
            )
        )
        object.__setattr__(self, "session_states", states)
        identities: dict[tuple[str, date, datetime], MarketSessionState] = {}
        for state in states:
            if state.instrument_type is not self.instrument_type:
                raise ValueError("session state instrument_type does not match market rule")
            identity = (state.instrument_id, state.trading_day, state.available_at)
            existing = identities.get(identity)
            if existing is not None and existing != state:
                raise ValueError("session states are ambiguous at the same available_at")
            identities[identity] = state

    def applies_on(self, trading_day: date) -> bool:
        """Return whether this market-rule version covers a trading day."""

        return self.effective_from <= trading_day and (
            self.effective_to is None or trading_day <= self.effective_to
        )

    def sellable_on(self, *, acquired_on: date) -> date:
        """Return the configured first sellable trading date for a new acquisition."""

        if self.settlement_days == 0:
            return acquired_on
        try:
            acquired_index = self.trading_calendar.index(acquired_on)
            return self.trading_calendar[acquired_index + self.settlement_days]
        except (ValueError, IndexError) as error:
            raise CNMarketRuleError(
                "trading_calendar cannot resolve the configured settlement horizon"
            ) from error

    def validate_order(self, order: OrderEvent) -> None:
        """Reject an order unless quantity, price, side, and PIT state are valid."""

        self._validate_order(order)

    def validate_quantity(
        self,
        *,
        side: Side,
        quantity: Decimal | int | str,
        is_full_liquidation: bool = False,
    ) -> None:
        """Validate units independently for order-draft and sizing consumers."""

        normalized = _decimal(quantity, "quantity")
        if normalized <= 0 or not _is_integral(normalized):
            raise OrderQuantityError("order quantity must be a positive whole number")
        if side is Side.BUY and normalized % self.buy_lot_size:
            raise OrderQuantityError(
                f"BUY quantity must be an integer multiple of {self.buy_lot_size}"
            )
        if (
            side is Side.SELL
            and normalized % self.sell_lot_size
            and not (self.allow_odd_lot_liquidation and is_full_liquidation)
        ):
            raise OrderQuantityError(
                f"SELL quantity must be an integer multiple of {self.sell_lot_size} "
                "unless explicitly authorized as a full odd-lot liquidation"
            )

    def match(
        self,
        order: OrderEvent,
        *,
        slippage_model: CNSlippageModel,
        allow_partial: bool = True,
    ) -> MatchResult:
        """Return a deterministic full/partial/no-fill result without ledger mutation."""

        state = self._validate_order(order)
        if not slippage_model.applies_on(order.trading_day):
            raise SlippageModelNotFoundError(
                f"slippage model is not effective on {order.trading_day.isoformat()}"
            )
        requested = order.quantity - order.filled_quantity
        capacity = state.available_quantity * self.max_participation_rate
        candidate = min(requested, capacity)
        unit = self.buy_lot_size if order.side is Side.BUY else self.sell_lot_size
        is_full_liquidation = order.order_id in self.odd_lot_liquidation_order_ids
        if order.side is Side.SELL and self.allow_odd_lot_liquidation and is_full_liquidation:
            fill_quantity = requested if candidate >= requested else _floor_to_unit(candidate, unit)
        else:
            fill_quantity = _floor_to_unit(candidate, unit)
        if fill_quantity <= 0:
            return self._no_fill(
                order=order,
                state=state,
                requested=requested,
                reason=NoFillReason.ZERO_CAPACITY,
                slippage_model=slippage_model,
            )
        if fill_quantity < requested and not allow_partial:
            return self._no_fill(
                order=order,
                state=state,
                requested=requested,
                reason=NoFillReason.PARTIAL_FILL_DISABLED,
                slippage_model=slippage_model,
            )
        participation = fill_quantity / state.available_quantity
        execution_price = slippage_model.execution_price(
            side=order.side,
            reference_price=state.reference_price,
            participation_rate=participation,
            price_tick=self.price_tick,
        )
        if state.upper_limit_price is not None:
            execution_price = min(execution_price, state.upper_limit_price)
        if state.lower_limit_price is not None:
            execution_price = max(execution_price, state.lower_limit_price)
        if order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            marketable = (
                execution_price <= order.limit_price
                if order.side is Side.BUY
                else execution_price >= order.limit_price
            )
            if not marketable:
                return self._no_fill(
                    order=order,
                    state=state,
                    requested=requested,
                    reason=NoFillReason.LIMIT_NOT_MARKETABLE,
                    slippage_model=slippage_model,
                )
        unfilled = requested - fill_quantity
        status = MatchStatus.FILLED if unfilled == 0 else MatchStatus.PARTIALLY_FILLED
        return MatchResult(
            order_id=order.order_id,
            trading_day=order.trading_day,
            status=status,
            reason=None,
            requested_quantity=requested,
            filled_quantity=fill_quantity,
            unfilled_quantity=unfilled,
            reference_price=state.reference_price,
            execution_price=execution_price,
            gross_amount=fill_quantity * execution_price,
            participation_rate=participation,
            state_revision=state.revision,
            state_available_at=state.available_at,
            state_hash=state.state_hash,
            market_rule_version=self.version,
            slippage_model_version=slippage_model.version,
            slippage_model_hash=slippage_model.model_hash,
        )

    def _validate_order(self, order: OrderEvent) -> MarketSessionState:
        if order.instrument_type is not self.instrument_type:
            raise CNMarketRuleError("order instrument_type does not match market rule")
        if not self.applies_on(order.trading_day):
            raise CNMarketRuleError("market rule is not effective on order trading_day")
        state = self._state_for(order)
        if state.state is MarketTradeState.SUSPENDED:
            raise OrderNotTradableError("instrument is suspended")
        allowed = state.buy_allowed if order.side is Side.BUY else state.sell_allowed
        if not allowed:
            raise OrderNotTradableError(
                f"session state {state.state.value} blocks {order.side.value}"
            )
        self.validate_quantity(
            side=order.side,
            quantity=order.quantity,
            is_full_liquidation=order.order_id in self.odd_lot_liquidation_order_ids,
        )
        if order.limit_price is not None:
            self._validate_price(order.limit_price, state, field_name="limit_price")
        if order.reference_price is not None:
            self._validate_price(order.reference_price, state, field_name="reference_price")
        return state

    def _state_for(self, order: OrderEvent) -> MarketSessionState:
        candidates = tuple(
            state
            for state in self.session_states
            if state.instrument_id == order.instrument_id
            and state.trading_day == order.trading_day
            and state.observed_at <= order.event_time
            and state.available_at <= order.event_time
        )
        if not candidates:
            raise MarketStateUnavailableError(
                "no market session state was available by order.event_time"
            )
        latest_available = max(state.available_at for state in candidates)
        latest = tuple(state for state in candidates if state.available_at == latest_available)
        if any(state != latest[0] for state in latest[1:]):
            raise MarketStateUnavailableError(
                "ambiguous market session revisions at order.event_time"
            )
        state = latest[0]
        self._validate_state_prices(state)
        return state

    def _validate_state_prices(self, state: MarketSessionState) -> None:
        prices = (
            state.reference_price,
            state.lower_limit_price,
            state.upper_limit_price,
        )
        if any(
            value is not None and not _is_tick_aligned(value, self.price_tick) for value in prices
        ):
            raise OrderPriceError("market session prices must align to price_tick")

    def _validate_price(
        self,
        value: Decimal,
        state: MarketSessionState,
        *,
        field_name: str,
    ) -> None:
        if not _is_tick_aligned(value, self.price_tick):
            raise OrderPriceError(f"{field_name} must align to price_tick {self.price_tick}")
        if state.lower_limit_price is not None and value < state.lower_limit_price:
            raise OrderPriceError(f"{field_name} is below the daily lower limit")
        if state.upper_limit_price is not None and value > state.upper_limit_price:
            raise OrderPriceError(f"{field_name} exceeds the daily upper limit")

    def _no_fill(
        self,
        *,
        order: OrderEvent,
        state: MarketSessionState,
        requested: Decimal,
        reason: NoFillReason,
        slippage_model: CNSlippageModel,
    ) -> MatchResult:
        return MatchResult(
            order_id=order.order_id,
            trading_day=order.trading_day,
            status=MatchStatus.NO_FILL,
            reason=reason,
            requested_quantity=requested,
            filled_quantity=_ZERO,
            unfilled_quantity=requested,
            reference_price=state.reference_price,
            execution_price=None,
            gross_amount=_ZERO,
            participation_rate=_ZERO,
            state_revision=state.revision,
            state_available_at=state.available_at,
            state_hash=state.state_hash,
            market_rule_version=self.version,
            slippage_model_version=slippage_model.version,
            slippage_model_hash=slippage_model.model_hash,
        )

    @property
    def rule_hash(self) -> str:
        """Return a stable identity for the effective market mechanics, excluding state."""

        return _hash(
            {
                "allow_odd_lot_liquidation": self.allow_odd_lot_liquidation,
                "buy_lot_size": canonical_decimal(self.buy_lot_size),
                "effective_from": self.effective_from.isoformat(),
                "effective_to": self.effective_to.isoformat() if self.effective_to else None,
                "instrument_type": self.instrument_type.value,
                "max_participation_rate": canonical_decimal(self.max_participation_rate),
                "odd_lot_liquidation_order_ids": sorted(self.odd_lot_liquidation_order_ids),
                "price_tick": canonical_decimal(self.price_tick),
                "rule_id": self.rule_id,
                "sell_lot_size": canonical_decimal(self.sell_lot_size),
                "settlement_days": self.settlement_days,
                "trading_calendar": [value.isoformat() for value in self.trading_calendar],
                "version": self.version,
            }
        )


__all__ = [
    "CNFeeSchedule",
    "CNMarketRule",
    "CNMarketRuleError",
    "CNSlippageModel",
    "CNSlippageModelBook",
    "MarketSessionState",
    "MarketStateUnavailableError",
    "MatchResult",
    "MatchStatus",
    "NoFillReason",
    "OrderNotTradableError",
    "OrderPriceError",
    "OrderQuantityError",
    "SlippageModelNotFoundError",
    "build_cn_fee_rule_book",
]
