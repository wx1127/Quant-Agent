"""Immutable contracts for point-in-time liquidity and tradeability features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.data.domain import InstrumentStatus, InstrumentType
from quant_agent.features.core.identity import canonical_decimal

_ZERO = Decimal(0)
_ONE = Decimal(1)


class TradeabilityInputError(ValueError):
    """Raised when inputs cannot be aligned without making an unsafe assumption."""


class InsufficientTradeabilityData(TradeabilityInputError):
    """Raised when an authoritative calendar, master record, or window is incomplete."""


class TradeSide(StrEnum):
    """Direction whose practical executability is being evaluated."""

    BUY = "BUY"
    SELL = "SELL"


class TradeabilityEligibility(StrEnum):
    """Final gate after structural and numeric constraints are combined."""

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


class MarketTradeState(StrEnum):
    """Current-session market state derived using the supplied market rule."""

    NORMAL = "NORMAL"
    SUSPENDED = "SUSPENDED"
    ONE_PRICE_LIMIT_UP = "ONE_PRICE_LIMIT_UP"
    ONE_PRICE_LIMIT_DOWN = "ONE_PRICE_LIMIT_DOWN"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"


class TradeabilityReasonCode(StrEnum):
    """Stable machine-readable explanation codes for eligibility consumers."""

    NOT_YET_LISTED = "NOT_YET_LISTED"
    DELISTED = "DELISTED"
    SUSPENDED = "SUSPENDED"
    ONE_PRICE_LIMIT_UP = "ONE_PRICE_LIMIT_UP"
    ONE_PRICE_LIMIT_DOWN = "ONE_PRICE_LIMIT_DOWN"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"
    NEWLY_LISTED = "NEWLY_LISTED"
    LOW_AVERAGE_TURNOVER_20 = "LOW_AVERAGE_TURNOVER_20"
    LOW_AVERAGE_TURNOVER_60 = "LOW_AVERAGE_TURNOVER_60"
    LOW_TURNOVER_RATE_PERCENTILE = "LOW_TURNOVER_RATE_PERCENTILE"
    INSUFFICIENT_CAPACITY = "INSUFFICIENT_CAPACITY"


class TradeabilityContributionCode(StrEnum):
    """Numeric gates that contribute to the final eligibility decision."""

    AVERAGE_TURNOVER_SHORT = "AVERAGE_TURNOVER_SHORT"
    AVERAGE_TURNOVER_LONG = "AVERAGE_TURNOVER_LONG"
    TURNOVER_RATE_PERCENTILE = "TURNOVER_RATE_PERCENTILE"
    LISTING_DAYS = "LISTING_DAYS"
    CAPACITY_RATIO = "CAPACITY_RATIO"


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


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


@dataclass(frozen=True, slots=True)
class TradeabilityRequest:
    """One account-sized tradeability question at a point-in-time boundary."""

    instrument_id: str
    instrument_type: InstrumentType
    market: str
    session_date: date
    as_of: datetime
    data_version: str
    side: TradeSide
    account_value: Decimal
    target_position_weight: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            _non_empty(self.instrument_id, "instrument_id"),
        )
        object.__setattr__(self, "market", _non_empty(self.market, "market"))
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        ensure_aware(self.as_of)
        if self.instrument_type not in (InstrumentType.STOCK, InstrumentType.ETF):
            raise ValueError("tradeability supports stock and ETF instruments")
        if self.session_date > self.as_of.astimezone(SHANGHAI_TZ).date():
            raise ValueError("session_date cannot be after as_of in Asia/Shanghai")
        account_value = _decimal(self.account_value, "account_value")
        target_weight = _decimal(self.target_position_weight, "target_position_weight")
        object.__setattr__(self, "account_value", account_value)
        object.__setattr__(self, "target_position_weight", target_weight)
        if account_value <= 0:
            raise ValueError("account_value must be positive")
        if target_weight <= 0 or target_weight > 1:
            raise ValueError("target_position_weight must be between zero exclusive and one")

    @property
    def target_notional(self) -> Decimal:
        """Requested trade notional used by the capacity gate."""

        return self.account_value * self.target_position_weight

    def fingerprint_payload(self) -> dict[str, str]:
        """Return canonical request fields used in result identities."""

        return {
            "account_value": canonical_decimal(self.account_value),
            "as_of": _utc(self.as_of),
            "data_version": self.data_version,
            "instrument_id": self.instrument_id,
            "instrument_type": self.instrument_type.value,
            "market": self.market,
            "session_date": self.session_date.isoformat(),
            "side": self.side.value,
            "target_position_weight": canonical_decimal(self.target_position_weight),
        }


@dataclass(frozen=True, slots=True)
class InstrumentTradeabilityState:
    """One revisable PIT master/status record effective over a date interval."""

    instrument_id: str
    instrument_type: InstrumentType
    listed_on: date
    delisted_on: date | None
    status: InstrumentStatus
    effective_from: date
    effective_to: date | None
    available_at: datetime
    revision: str
    data_version: str

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
        ensure_aware(self.available_at)
        if self.delisted_on is not None and self.delisted_on < self.listed_on:
            raise ValueError("delisted_on cannot precede listed_on")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")

    def applies_on(self, value: date) -> bool:
        """Return whether this historical status record covers ``value``."""

        return self.effective_from <= value and (
            self.effective_to is None or value <= self.effective_to
        )

    def fingerprint_payload(self) -> dict[str, str | None]:
        """Return stable master-data identity fields."""

        return {
            "available_at": _utc(self.available_at),
            "data_version": self.data_version,
            "delisted_on": self.delisted_on.isoformat() if self.delisted_on else None,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "instrument_id": self.instrument_id,
            "instrument_type": self.instrument_type.value,
            "listed_on": self.listed_on.isoformat(),
            "revision": self.revision,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class TradeabilityCalendarSession:
    """One revisable market-calendar observation with explicit availability."""

    market: str
    trade_date: date
    is_open: bool
    available_at: datetime
    revision: str
    data_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", _non_empty(self.market, "market"))
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        ensure_aware(self.available_at)

    def fingerprint_payload(self) -> dict[str, str | bool]:
        """Return stable calendar identity fields."""

        return {
            "available_at": _utc(self.available_at),
            "data_version": self.data_version,
            "is_open": self.is_open,
            "market": self.market,
            "revision": self.revision,
            "trade_date": self.trade_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TradeabilityBar:
    """One PIT daily bar with fields required for liquidity and limit detection."""

    instrument_id: str
    trade_date: date
    observed_at: datetime
    available_at: datetime
    previous_close: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover_amount: Decimal
    turnover_rate: Decimal | None
    is_suspended: bool
    revision: str
    data_version: str

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
        if self.observed_at.astimezone(SHANGHAI_TZ).date() != self.trade_date:
            raise ValueError("observed_at date must equal trade_date in Asia/Shanghai")
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot precede observed_at")
        price_fields = ("previous_close", "open", "high", "low", "close")
        for field_name in price_fields:
            normalized = _decimal(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
            if normalized <= 0:
                raise ValueError("tradeability prices must be positive")
        volume = _decimal(self.volume, "volume")
        turnover_amount = _decimal(self.turnover_amount, "turnover_amount")
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "turnover_amount", turnover_amount)
        if volume < 0 or turnover_amount < 0:
            raise ValueError("volume and turnover_amount must be non-negative")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be at least the other OHLC prices")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be at most the other OHLC prices")
        if self.turnover_rate is not None:
            turnover_rate = _decimal(self.turnover_rate, "turnover_rate")
            object.__setattr__(self, "turnover_rate", turnover_rate)
            if turnover_rate < 0:
                raise ValueError("turnover_rate must be non-negative")
        if self.is_suspended and (volume != 0 or turnover_amount != 0):
            raise ValueError("suspended observations must have zero volume and turnover_amount")

    def fingerprint_payload(self) -> dict[str, str | bool | None]:
        """Return stable bar fields used by the input hash."""

        return {
            "available_at": _utc(self.available_at),
            "close": canonical_decimal(self.close),
            "data_version": self.data_version,
            "high": canonical_decimal(self.high),
            "instrument_id": self.instrument_id,
            "is_suspended": self.is_suspended,
            "low": canonical_decimal(self.low),
            "observed_at": _utc(self.observed_at),
            "open": canonical_decimal(self.open),
            "previous_close": canonical_decimal(self.previous_close),
            "revision": self.revision,
            "trade_date": self.trade_date.isoformat(),
            "turnover_amount": canonical_decimal(self.turnover_amount),
            "turnover_rate": (
                canonical_decimal(self.turnover_rate) if self.turnover_rate is not None else None
            ),
            "volume": canonical_decimal(self.volume),
        }


_DEFAULT_BUY_BLOCKS = frozenset({MarketTradeState.ONE_PRICE_LIMIT_UP, MarketTradeState.LIMIT_UP})
_DEFAULT_SELL_BLOCKS = frozenset(
    {MarketTradeState.ONE_PRICE_LIMIT_DOWN, MarketTradeState.LIMIT_DOWN}
)


@dataclass(frozen=True, slots=True)
class InstrumentTradeabilityRule:
    """Versioned instrument mechanics supplied to, rather than encoded in, the analyzer."""

    rule_id: str
    version: str
    instrument_type: InstrumentType
    effective_from: date
    effective_to: date | None = None
    price_limit_ratio: Decimal | None = Decimal("0.10")
    price_tick: Decimal = Decimal("0.01")
    boundary_tolerance: Decimal = Decimal(0)
    blocked_buy_states: frozenset[MarketTradeState] = _DEFAULT_BUY_BLOCKS
    blocked_sell_states: frozenset[MarketTradeState] = _DEFAULT_SELL_BLOCKS

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _non_empty(self.rule_id, "rule_id"))
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        if self.instrument_type not in (InstrumentType.STOCK, InstrumentType.ETF):
            raise ValueError("tradeability rules support stock and ETF instruments")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        price_tick = _decimal(self.price_tick, "price_tick")
        tolerance = _decimal(self.boundary_tolerance, "boundary_tolerance")
        object.__setattr__(self, "price_tick", price_tick)
        object.__setattr__(self, "boundary_tolerance", tolerance)
        if price_tick <= 0:
            raise ValueError("price_tick must be positive")
        if tolerance < 0 or tolerance >= price_tick:
            raise ValueError("boundary_tolerance must be non-negative and below price_tick")
        if self.price_limit_ratio is not None:
            ratio = _decimal(self.price_limit_ratio, "price_limit_ratio")
            object.__setattr__(self, "price_limit_ratio", ratio)
            if ratio <= 0 or ratio >= 1:
                raise ValueError("price_limit_ratio must be between zero and one")
        buy_states = frozenset(self.blocked_buy_states)
        sell_states = frozenset(self.blocked_sell_states)
        object.__setattr__(self, "blocked_buy_states", buy_states)
        object.__setattr__(self, "blocked_sell_states", sell_states)
        invalid = {MarketTradeState.NORMAL, MarketTradeState.SUSPENDED}
        if buy_states & invalid or sell_states & invalid:
            raise ValueError("normal and suspended states cannot be configurable side blocks")

    def applies_on(self, value: date) -> bool:
        """Return whether this rule is effective on ``value``."""

        return self.effective_from <= value and (
            self.effective_to is None or value <= self.effective_to
        )

    def blocks(self, side: TradeSide, state: MarketTradeState) -> bool:
        """Return whether the supplied rule blocks this side in the derived state."""

        states = self.blocked_buy_states if side is TradeSide.BUY else self.blocked_sell_states
        return state in states

    @property
    def rule_hash(self) -> str:
        """Return a stable identity for every effective market-rule parameter."""

        return _sha256(
            {
                "blocked_buy_states": sorted(value.value for value in self.blocked_buy_states),
                "blocked_sell_states": sorted(value.value for value in self.blocked_sell_states),
                "boundary_tolerance": canonical_decimal(self.boundary_tolerance),
                "effective_from": self.effective_from.isoformat(),
                "effective_to": self.effective_to.isoformat() if self.effective_to else None,
                "instrument_type": self.instrument_type.value,
                "price_limit_ratio": (
                    canonical_decimal(self.price_limit_ratio)
                    if self.price_limit_ratio is not None
                    else None
                ),
                "price_tick": canonical_decimal(self.price_tick),
                "rule_id": self.rule_id,
                "version": self.version,
            }
        )


@dataclass(frozen=True, slots=True)
class TradeabilityConfig:
    """Versioned liquidity, listing-age, and account-capacity thresholds."""

    version: str = "tradeability-v1"
    short_turnover_window: int = 20
    long_turnover_window: int = 60
    minimum_average_turnover_short: Decimal = Decimal("20000000")
    minimum_average_turnover_long: Decimal = Decimal("10000000")
    turnover_rate_percentile_window: int = 60
    minimum_turnover_rate_observations: int = 20
    minimum_turnover_rate_percentile: Decimal | None = None
    minimum_listing_days: int = 60
    participation_rate: Decimal = Decimal("0.05")
    minimum_capacity_ratio: Decimal = Decimal(1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        if self.short_turnover_window <= 0 or self.long_turnover_window <= 0:
            raise ValueError("turnover windows must be positive")
        if self.short_turnover_window > self.long_turnover_window:
            raise ValueError("short_turnover_window cannot exceed long_turnover_window")
        if not 1 <= self.turnover_rate_percentile_window <= self.long_turnover_window:
            raise ValueError("turnover_rate_percentile_window must be within the long window")
        if not 1 <= self.minimum_turnover_rate_observations <= self.turnover_rate_percentile_window:
            raise ValueError("minimum turnover-rate observations must be within its window")
        if self.minimum_listing_days <= 0:
            raise ValueError("minimum_listing_days must be positive")
        for field_name in (
            "minimum_average_turnover_short",
            "minimum_average_turnover_long",
            "participation_rate",
            "minimum_capacity_ratio",
        ):
            normalized = _decimal(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
        if self.minimum_average_turnover_short < 0 or self.minimum_average_turnover_long < 0:
            raise ValueError("average-turnover thresholds must be non-negative")
        if self.participation_rate <= 0 or self.participation_rate > 1:
            raise ValueError("participation_rate must be between zero exclusive and one")
        if self.minimum_capacity_ratio <= 0:
            raise ValueError("minimum_capacity_ratio must be positive")
        if self.minimum_turnover_rate_percentile is not None:
            percentile = _decimal(
                self.minimum_turnover_rate_percentile,
                "minimum_turnover_rate_percentile",
            )
            object.__setattr__(self, "minimum_turnover_rate_percentile", percentile)
            if percentile < 0 or percentile > 1:
                raise ValueError("minimum_turnover_rate_percentile must be between zero and one")

    @property
    def config_hash(self) -> str:
        """Return a stable hash for all thresholds and window semantics."""

        return _sha256(
            {
                "long_turnover_window": self.long_turnover_window,
                "minimum_average_turnover_long": canonical_decimal(
                    self.minimum_average_turnover_long
                ),
                "minimum_average_turnover_short": canonical_decimal(
                    self.minimum_average_turnover_short
                ),
                "minimum_capacity_ratio": canonical_decimal(self.minimum_capacity_ratio),
                "minimum_listing_days": self.minimum_listing_days,
                "minimum_turnover_rate_observations": (self.minimum_turnover_rate_observations),
                "minimum_turnover_rate_percentile": (
                    canonical_decimal(self.minimum_turnover_rate_percentile)
                    if self.minimum_turnover_rate_percentile is not None
                    else None
                ),
                "participation_rate": canonical_decimal(self.participation_rate),
                "short_turnover_window": self.short_turnover_window,
                "turnover_rate_percentile_window": self.turnover_rate_percentile_window,
                "version": self.version,
            }
        )


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    """Account-relative capacity derived from short/long average traded amounts."""

    account_value: Decimal
    target_position_weight: Decimal
    target_notional: Decimal
    average_turnover_short: Decimal
    average_turnover_long: Decimal
    participation_rate: Decimal
    short_window_capacity: Decimal
    long_window_capacity: Decimal
    daily_capacity: Decimal
    capacity_ratio: Decimal
    maximum_account_weight: Decimal
    estimated_trade_days: Decimal | None

    def __post_init__(self) -> None:
        values = (
            self.account_value,
            self.target_position_weight,
            self.target_notional,
            self.average_turnover_short,
            self.average_turnover_long,
            self.participation_rate,
            self.short_window_capacity,
            self.long_window_capacity,
            self.daily_capacity,
            self.capacity_ratio,
            self.maximum_account_weight,
        )
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("capacity values must be finite and non-negative")
        if self.account_value <= 0 or self.target_notional <= 0:
            raise ValueError("capacity account and target notionals must be positive")
        if self.estimated_trade_days is not None and (
            not self.estimated_trade_days.is_finite() or self.estimated_trade_days < 0
        ):
            raise ValueError("estimated_trade_days must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TradeabilityContribution:
    """One auditable numeric gate contribution to final eligibility."""

    code: TradeabilityContributionCode
    value: Decimal
    threshold: Decimal | None
    passed: bool | None
    blocking: bool
    description: str

    def __post_init__(self) -> None:
        if not self.value.is_finite():
            raise ValueError("contribution value must be finite")
        if self.threshold is not None and not self.threshold.is_finite():
            raise ValueError("contribution threshold must be finite")
        if not self.description.strip():
            raise ValueError("contribution description must be non-empty")
        if (self.threshold is None) != (self.passed is None):
            raise ValueError("contribution threshold and passed must be present together")
        if self.blocking and self.passed is not False:
            raise ValueError("a blocking contribution must explicitly fail")


@dataclass(frozen=True, slots=True)
class TradeabilityReason:
    """Structured exceptional market state or failed gate explanation."""

    code: TradeabilityReasonCode
    blocking: bool
    message: str
    value: Decimal | None = None
    threshold: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("reason message must be non-empty")
        for value in (self.value, self.threshold):
            if value is not None and not value.is_finite():
                raise ValueError("reason numeric values must be finite")


@dataclass(frozen=True, slots=True)
class TradeabilitySnapshot:
    """PIT, rule-bound liquidity and practical-execution assessment."""

    request: TradeabilityRequest
    feature_version: str
    config_hash: str
    rule_id: str
    rule_version: str
    rule_hash: str
    market_state: MarketTradeState
    eligibility: TradeabilityEligibility
    reasons: tuple[TradeabilityReason, ...]
    contributions: tuple[TradeabilityContribution, ...]
    listing_days: int
    average_turnover_short: Decimal
    average_turnover_long: Decimal
    turnover_rate_percentile: Decimal | None
    turnover_rate_observation_count: int
    short_window_dates: tuple[date, ...]
    long_window_dates: tuple[date, ...]
    upper_limit_price: Decimal | None
    lower_limit_price: Decimal | None
    capacity: CapacityEstimate
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        if (
            not self.feature_version.strip()
            or not self.rule_id.strip()
            or not self.rule_version.strip()
        ):
            raise ValueError("snapshot versions and rule_id must be non-empty")
        for field_name, digest in (
            ("config_hash", self.config_hash),
            ("rule_hash", self.rule_hash),
            ("input_hash", self.input_hash),
            ("result_hash", self.result_hash),
        ):
            _validate_hash(digest, field_name)
        if self.listing_days < 0:
            raise ValueError("listing_days cannot be negative")
        if self.turnover_rate_observation_count < 0:
            raise ValueError("turnover_rate_observation_count cannot be negative")
        for turnover in (self.average_turnover_short, self.average_turnover_long):
            if not turnover.is_finite() or turnover < 0:
                raise ValueError("average turnover values must be finite and non-negative")
        if self.turnover_rate_percentile is not None and not (
            _ZERO <= self.turnover_rate_percentile <= _ONE
        ):
            raise ValueError("turnover_rate_percentile must be between zero and one")
        if not self.short_window_dates or not self.long_window_dates:
            raise ValueError("snapshot windows must not be empty")
        if self.short_window_dates != self.long_window_dates[-len(self.short_window_dates) :]:
            raise ValueError("short window must be the trailing portion of the long window")
        if self.long_window_dates[-1] != self.request.session_date:
            raise ValueError("long window must end on request session_date")
        if self.eligibility is TradeabilityEligibility.ELIGIBLE and any(
            reason.blocking for reason in self.reasons
        ):
            raise ValueError("eligible snapshot cannot contain blocking reasons")

    @property
    def eligible(self) -> bool:
        """Convenience boolean for filters that do not need the enum value."""

        return self.eligibility is TradeabilityEligibility.ELIGIBLE

    @property
    def cache_key(self) -> str:
        """Compatibility identity for feature-cache consumers."""

        return self.result_hash


__all__ = [
    "CapacityEstimate",
    "InstrumentTradeabilityRule",
    "InstrumentTradeabilityState",
    "InsufficientTradeabilityData",
    "MarketTradeState",
    "TradeSide",
    "TradeabilityBar",
    "TradeabilityCalendarSession",
    "TradeabilityConfig",
    "TradeabilityContribution",
    "TradeabilityContributionCode",
    "TradeabilityEligibility",
    "TradeabilityInputError",
    "TradeabilityReason",
    "TradeabilityReasonCode",
    "TradeabilityRequest",
    "TradeabilitySnapshot",
]
