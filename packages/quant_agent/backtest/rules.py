"""State, market, and versioned fee rules for backtest event consumers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from quant_agent.backtest.contracts import (
    FILL_STATE_TRANSITIONS,
    ORDER_STATE_TRANSITIONS,
    ExactDecimal,
    FillStatus,
    OrderEvent,
    OrderStatus,
    Side,
    TradableInstrumentType,
)


class InvalidStateTransition(ValueError):
    """Raised when replay attempts an impossible lifecycle transition."""


class FeeRuleNotFoundError(LookupError):
    """Raised when no unambiguous fee rule was effective at the trade date."""


def ensure_order_transition(current: OrderStatus, target: OrderStatus) -> None:
    """Fail closed unless target is a legal successor of current."""

    if target not in ORDER_STATE_TRANSITIONS[current]:
        raise InvalidStateTransition(f"invalid order transition: {current.value} -> {target.value}")


def ensure_fill_transition(current: FillStatus, target: FillStatus) -> None:
    """Fail closed unless target is a legal successor of current."""

    if target not in FILL_STATE_TRANSITIONS[current]:
        raise InvalidStateTransition(f"invalid fill transition: {current.value} -> {target.value}")


@runtime_checkable
class MarketRule(Protocol):
    """Injectable market mechanics; it deliberately contains no strategy decisions.

    Implementations may express differences such as A-share buy lots, odd-lot sells,
    stock T+1 inventory, or an ETF's applicable settlement convention.
    """

    rule_id: str
    version: str
    effective_from: date
    instrument_type: TradableInstrumentType

    def validate_order(self, order: OrderEvent) -> None:
        """Reject an order that violates this market's mechanical constraints."""

    def sellable_on(self, *, acquired_on: date) -> date:
        """Return the first trading date on which newly acquired units may be sold."""


class FeeBreakdown(BaseModel):
    """Exact, non-negative components produced by one fee-rule version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commission: ExactDecimal
    stamp_duty: ExactDecimal
    transfer_fee: ExactDecimal
    other_fee: ExactDecimal
    total_amount: ExactDecimal

    @model_validator(mode="after")
    def validate_components(self) -> FeeBreakdown:
        components = (self.commission, self.stamp_duty, self.transfer_fee, self.other_fee)
        if any(value < 0 for value in components):
            raise ValueError("fee components cannot be negative")
        if self.total_amount != sum(components, Decimal(0)):
            raise ValueError("total_amount must equal the fee component sum")
        return self


class FeeRule(BaseModel):
    """One side- and instrument-specific fee policy effective from a date."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_type: TradableInstrumentType
    side: Side
    effective_from: date
    version: str
    commission_rate: ExactDecimal = Decimal(0)
    minimum_commission: ExactDecimal = Decimal(0)
    stamp_duty_rate: ExactDecimal = Decimal(0)
    transfer_fee_rate: ExactDecimal = Decimal(0)
    other_fee_rate: ExactDecimal = Decimal(0)
    rounding_increment: ExactDecimal = Decimal("0.01")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("fee rule version must be non-empty")
        return result

    @model_validator(mode="after")
    def validate_rates(self) -> FeeRule:
        rates = (
            self.commission_rate,
            self.stamp_duty_rate,
            self.transfer_fee_rate,
            self.other_fee_rate,
        )
        if any(rate < 0 or rate > 1 for rate in rates):
            raise ValueError("fee rates must be between zero and one")
        if self.minimum_commission < 0:
            raise ValueError("minimum_commission cannot be negative")
        if self.rounding_increment <= 0:
            raise ValueError("rounding_increment must be positive")
        return self

    def assess(self, gross_amount: Decimal | int | str) -> FeeBreakdown:
        """Calculate deterministic fee components without hard-coded market differences."""

        if isinstance(gross_amount, (bool, float)):
            raise ValueError("gross_amount must be an exact decimal input")
        try:
            amount = gross_amount if isinstance(gross_amount, Decimal) else Decimal(gross_amount)
        except InvalidOperation as error:
            raise ValueError("gross_amount must be a valid decimal") from error
        if not amount.is_finite() or amount <= 0:
            raise ValueError("gross_amount must be finite and positive")

        def rounded(value: Decimal) -> Decimal:
            units = (value / self.rounding_increment).quantize(Decimal(1), rounding=ROUND_HALF_UP)
            return units * self.rounding_increment

        commission = rounded(max(amount * self.commission_rate, self.minimum_commission))
        stamp_duty = rounded(amount * self.stamp_duty_rate)
        transfer_fee = rounded(amount * self.transfer_fee_rate)
        other_fee = rounded(amount * self.other_fee_rate)
        total = commission + stamp_duty + transfer_fee + other_fee
        return FeeBreakdown(
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            other_fee=other_fee,
            total_amount=total,
        )


class FeeRuleBook:
    """Select the latest effective fee rule, with optional exact version pinning."""

    def __init__(self, rules: Iterable[FeeRule]) -> None:
        self._rules = tuple(rules)
        identities: set[tuple[TradableInstrumentType, Side, date]] = set()
        versions: set[tuple[TradableInstrumentType, Side, str]] = set()
        for rule in self._rules:
            identity = (rule.instrument_type, rule.side, rule.effective_from)
            version_identity = (rule.instrument_type, rule.side, rule.version)
            if identity in identities:
                raise ValueError("fee rules are ambiguous at an effective_from boundary")
            if version_identity in versions:
                raise ValueError("fee rule version must be unique per instrument type and side")
            identities.add(identity)
            versions.add(version_identity)

    def select(
        self,
        *,
        instrument_type: TradableInstrumentType,
        side: Side,
        trading_day: date,
        version: str | None = None,
    ) -> FeeRule:
        """Select a rule effective on trading_day or fail closed if none exists."""

        candidates = [
            rule
            for rule in self._rules
            if rule.instrument_type is instrument_type
            and rule.side is side
            and rule.effective_from <= trading_day
            and (version is None or rule.version == version)
        ]
        if not candidates:
            suffix = f" version {version!r}" if version is not None else ""
            raise FeeRuleNotFoundError(
                f"no effective {instrument_type.value} {side.value} fee rule{suffix} "
                f"for {trading_day.isoformat()}"
            )
        return max(candidates, key=lambda rule: rule.effective_from)
