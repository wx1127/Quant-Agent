"""Immutable contracts for deterministic, credential-free paper execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import Any

from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.config import AppEnvironment, RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.regime.contracts import stable_hash

PAPER_EXECUTION_ENGINE_VERSION = "paper-execution-engine-v1"

_ZERO = Decimal(0)
_ACTIVE_ORDER_STATUSES = frozenset(
    {
        "ACCEPTED",
        "PARTIALLY_FILLED",
    }
)


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _exact_decimal(value: Decimal, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be an exact Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return value


def _whole(value: Decimal, field_name: str, *, positive: bool = False) -> Decimal:
    result = _exact_decimal(value, field_name)
    if result != result.to_integral_value():
        raise ValueError(f"{field_name} must be a whole number")
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field_name} must be {qualifier}")
    return result


def _sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: str | None, field_name: str) -> None:
    if value is not None:
        _sha256(value, field_name)


def _strict_date(value: date, field_name: str) -> date:
    if type(value) is not date:
        raise ValueError(f"{field_name} must be a date")
    return value


def _content_payload(instance: Any, hash_field: str) -> dict[str, Any]:
    return {key: value for key, value in asdict(instance).items() if key != hash_field}


def _is_active(status: PaperOrderStatus) -> bool:
    return status.value in _ACTIVE_ORDER_STATUSES


class PaperExecutionInputError(ValueError):
    """Raised when paper execution cannot safely consume the supplied inputs."""


class PaperIdempotencyConflict(PaperExecutionInputError):
    """Raised when one idempotency key is reused for different request content."""


class PaperOrderPolicy(StrEnum):
    """Pricing instructions supported by the first paper-execution engine."""

    MARKET = "MARKET"


class PaperOrderStatus(StrEnum):
    """Durable paper-order lifecycle states."""

    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class PaperNoFillReason(StrEnum):
    """Closed vocabulary for deterministic no-fill outcomes."""

    SUSPENDED = "SUSPENDED"
    SIDE_BLOCKED = "SIDE_BLOCKED"
    ZERO_CAPACITY = "ZERO_CAPACITY"
    PARTIAL_FILL_DISABLED = "PARTIAL_FILL_DISABLED"
    LIMIT_NOT_MARKETABLE = "LIMIT_NOT_MARKETABLE"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"


@dataclass(frozen=True, slots=True)
class PaperExecutionConfig:
    """Versioned behavior for a local paper-only execution engine."""

    version: str = "paper-execution-config-v1"
    allow_partial: bool = True
    order_policy: PaperOrderPolicy = PaperOrderPolicy.MARKET
    max_market_state_age_seconds: int = 14_400
    environment: AppEnvironment = AppEnvironment.PAPER

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "paper config version"))
        if not isinstance(self.allow_partial, bool):
            raise ValueError("allow_partial must be a bool")
        if not isinstance(self.order_policy, PaperOrderPolicy):
            raise ValueError("order_policy must be a PaperOrderPolicy value")
        if (
            isinstance(self.max_market_state_age_seconds, bool)
            or not isinstance(self.max_market_state_age_seconds, int)
            or self.max_market_state_age_seconds <= 0
        ):
            raise ValueError("max_market_state_age_seconds must be a positive integer")
        if self.environment is not AppEnvironment.PAPER:
            raise ValueError("paper execution requires the PAPER app environment")

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class PaperPositionLot:
    """One immutable acquisition lot used for T+N sellability and cost basis."""

    lot_id: str
    instrument_id: str
    instrument_type: TradableInstrumentType
    quantity: Decimal
    cost_basis: Decimal
    acquired_on: date
    sellable_on: date
    externally_frozen: Decimal
    source_id: str
    lot_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lot_id", _non_empty(self.lot_id, "lot_id"))
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "source_id", _non_empty(self.source_id, "lot source_id"))
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("instrument_type must be a TradableInstrumentType value")
        _whole(self.quantity, "lot quantity", positive=True)
        _whole(self.externally_frozen, "lot externally_frozen")
        _exact_decimal(self.cost_basis, "lot cost_basis")
        if self.cost_basis < 0:
            raise ValueError("lot cost_basis cannot be negative")
        _strict_date(self.acquired_on, "lot acquired_on")
        _strict_date(self.sellable_on, "lot sellable_on")
        if self.sellable_on < self.acquired_on:
            raise ValueError("lot sellable_on cannot precede acquired_on")
        if self.externally_frozen > self.quantity:
            raise ValueError("lot externally_frozen cannot exceed quantity")
        _sha256(self.lot_hash, "lot_hash")
        if self.lot_hash != stable_hash(self._content_payload()):
            raise ValueError("lot_hash does not match paper position lot content")

    @classmethod
    def build(
        cls,
        *,
        lot_id: str,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        quantity: Decimal,
        cost_basis: Decimal,
        acquired_on: date,
        sellable_on: date,
        source_id: str,
        externally_frozen: Decimal = _ZERO,
    ) -> PaperPositionLot:
        values: dict[str, object] = {
            "lot_id": lot_id,
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "quantity": quantity,
            "cost_basis": cost_basis,
            "acquired_on": acquired_on,
            "sellable_on": sellable_on,
            "externally_frozen": externally_frozen,
            "source_id": source_id,
        }
        return cls(**values, lot_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "lot_hash")


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """A paper position whose availability buckets are derived from immutable lots."""

    as_of: datetime
    instrument_id: str
    instrument_type: TradableInstrumentType
    lots: tuple[PaperPositionLot, ...]
    reserved_sell_quantity: Decimal
    total_quantity: Decimal
    available_quantity: Decimal
    frozen_quantity: Decimal
    unsettled_quantity: Decimal
    cost_basis: Decimal
    average_cost: Decimal
    position_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("instrument_type must be a TradableInstrumentType value")
        if not self.lots:
            raise ValueError("a paper position requires at least one positive lot")
        lot_ids = tuple(lot.lot_id for lot in self.lots)
        if lot_ids != tuple(sorted(set(lot_ids))):
            raise ValueError("paper position lots must be unique and sorted by lot_id")
        if any(
            lot.instrument_id != self.instrument_id
            or lot.instrument_type is not self.instrument_type
            for lot in self.lots
        ):
            raise ValueError("all lots must match the paper position instrument")
        for field_name in (
            "reserved_sell_quantity",
            "total_quantity",
            "available_quantity",
            "frozen_quantity",
            "unsettled_quantity",
        ):
            _whole(getattr(self, field_name), f"position {field_name}")
        for field_name in ("cost_basis", "average_cost"):
            value = _exact_decimal(getattr(self, field_name), f"position {field_name}")
            if value < 0:
                raise ValueError("paper position costs cannot be negative")
        if self.total_quantity <= 0:
            raise ValueError("paper position total_quantity must be positive")
        trading_day = self.as_of.astimezone(SHANGHAI_TZ).date()
        if any(lot.acquired_on > trading_day for lot in self.lots):
            raise ValueError("paper position cannot contain a future-acquired lot")
        expected_total = sum((lot.quantity for lot in self.lots), _ZERO)
        expected_unsettled = sum(
            (lot.quantity for lot in self.lots if lot.sellable_on > trading_day),
            _ZERO,
        )
        if any(lot.externally_frozen > 0 and lot.sellable_on > trading_day for lot in self.lots):
            raise ValueError("an unsettled lot cannot also be externally frozen")
        externally_frozen = sum(
            (lot.externally_frozen for lot in self.lots if lot.sellable_on <= trading_day),
            _ZERO,
        )
        settled_free = expected_total - expected_unsettled - externally_frozen
        if self.reserved_sell_quantity > settled_free:
            raise ValueError("reserved sell quantity exceeds settled unfrozen quantity")
        expected_frozen = externally_frozen + self.reserved_sell_quantity
        expected_available = expected_total - expected_unsettled - expected_frozen
        expected_cost_basis = sum((lot.cost_basis for lot in self.lots), _ZERO)
        with localcontext() as context:
            context.prec = 50
            expected_average_cost = expected_cost_basis / expected_total
        if self.total_quantity != expected_total:
            raise ValueError("position total_quantity must equal its lot quantities")
        if self.unsettled_quantity != expected_unsettled:
            raise ValueError("position unsettled_quantity does not match lot sellable dates")
        if self.frozen_quantity != expected_frozen:
            raise ValueError("position frozen_quantity does not match external and order holds")
        if self.available_quantity != expected_available:
            raise ValueError("position available_quantity does not match derived buckets")
        if self.total_quantity != (
            self.available_quantity + self.frozen_quantity + self.unsettled_quantity
        ):
            raise ValueError("position quantity buckets must conserve total quantity")
        if self.cost_basis != expected_cost_basis or self.average_cost != expected_average_cost:
            raise ValueError("position cost fields do not match its lots")
        _sha256(self.position_hash, "position_hash")
        if self.position_hash != stable_hash(self._content_payload()):
            raise ValueError("position_hash does not match paper position content")

    @classmethod
    def build(
        cls,
        *,
        as_of: datetime,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        lots: tuple[PaperPositionLot, ...],
        reserved_sell_quantity: Decimal = _ZERO,
    ) -> PaperPosition:
        ordered_lots = tuple(sorted(lots, key=lambda lot: lot.lot_id))
        if len({lot.lot_id for lot in ordered_lots}) != len(ordered_lots):
            raise ValueError("paper position lots must have unique lot_id values")
        trading_day = ensure_aware(as_of).astimezone(SHANGHAI_TZ).date()
        total = sum((lot.quantity for lot in ordered_lots), _ZERO)
        unsettled = sum(
            (lot.quantity for lot in ordered_lots if lot.sellable_on > trading_day),
            _ZERO,
        )
        externally_frozen = sum(
            (lot.externally_frozen for lot in ordered_lots if lot.sellable_on <= trading_day),
            _ZERO,
        )
        frozen = externally_frozen + reserved_sell_quantity
        available = total - unsettled - frozen
        cost_basis = sum((lot.cost_basis for lot in ordered_lots), _ZERO)
        with localcontext() as context:
            context.prec = 50
            average_cost = cost_basis / total if total > 0 else _ZERO
        values: dict[str, object] = {
            "as_of": as_of,
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "lots": [asdict(lot) for lot in ordered_lots],
            "reserved_sell_quantity": reserved_sell_quantity,
            "total_quantity": total,
            "available_quantity": available,
            "frozen_quantity": frozen,
            "unsettled_quantity": unsettled,
            "cost_basis": cost_basis,
            "average_cost": average_cost,
        }
        return cls(
            as_of=as_of,
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            lots=ordered_lots,
            reserved_sell_quantity=reserved_sell_quantity,
            total_quantity=total,
            available_quantity=available,
            frozen_quantity=frozen,
            unsettled_quantity=unsettled,
            cost_basis=cost_basis,
            average_cost=average_cost,
            position_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "position_hash")


@dataclass(frozen=True, slots=True)
class PaperOrder:
    """A paper-only order bound to one reviewed draft line and sizing identities."""

    order_id: str
    batch_hash: str
    line_hash: str
    decision_id: str
    instrument_id: str
    instrument_type: TradableInstrumentType
    side: Side
    quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    is_full_liquidation: bool
    status: PaperOrderStatus
    created_at: datetime
    expires_at: datetime
    cash_reserved: Decimal
    sell_reserved_quantity: Decimal
    market_rule_version: str
    market_rule_hash: str
    fee_rule_version: str
    fee_rule_hash: str
    slippage_model_version: str
    slippage_model_hash: str
    last_attempt_id: str | None
    order_hash: str

    def __post_init__(self) -> None:
        for field_name in ("order_id", "decision_id", "instrument_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        for field_name in (
            "market_rule_version",
            "fee_rule_version",
            "slippage_model_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        for value, field_name in (
            (self.batch_hash, "batch_hash"),
            (self.line_hash, "line_hash"),
            (self.market_rule_hash, "market_rule_hash"),
            (self.fee_rule_hash, "fee_rule_hash"),
            (self.slippage_model_hash, "slippage_model_hash"),
            (self.order_hash, "order_hash"),
        ):
            _sha256(value, field_name)
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("instrument_type must be a TradableInstrumentType value")
        if not isinstance(self.side, Side):
            raise ValueError("side must be a Side value")
        if not isinstance(self.status, PaperOrderStatus):
            raise ValueError("status must be a PaperOrderStatus value")
        if not isinstance(self.is_full_liquidation, bool):
            raise ValueError("is_full_liquidation must be a bool")
        ensure_aware(self.created_at)
        ensure_aware(self.expires_at)
        if self.expires_at <= self.created_at:
            raise ValueError("paper order expires_at must follow created_at")
        _whole(self.quantity, "order quantity", positive=True)
        _whole(self.filled_quantity, "order filled_quantity")
        _whole(self.remaining_quantity, "order remaining_quantity")
        _whole(self.sell_reserved_quantity, "order sell_reserved_quantity")
        _exact_decimal(self.cash_reserved, "order cash_reserved")
        if self.cash_reserved < 0:
            raise ValueError("order cash_reserved cannot be negative")
        if self.filled_quantity > self.quantity:
            raise ValueError("order filled_quantity cannot exceed quantity")
        if self.remaining_quantity != self.quantity - self.filled_quantity:
            raise ValueError("order remaining_quantity must equal quantity minus fills")
        if self.is_full_liquidation and self.side is not Side.SELL:
            raise ValueError("only a SELL order may be a full liquidation")
        if self.status is PaperOrderStatus.FILLED:
            if self.remaining_quantity != 0:
                raise ValueError("a FILLED order cannot have remaining quantity")
        elif self.status is PaperOrderStatus.PARTIALLY_FILLED:
            if not (_ZERO < self.filled_quantity < self.quantity):
                raise ValueError("a PARTIALLY_FILLED order requires a proper partial fill")
        elif self.status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.REJECTED}:
            if self.filled_quantity != 0:
                raise ValueError("an ACCEPTED or REJECTED order cannot already be filled")
        elif self.remaining_quantity == 0:
            raise ValueError("a fully filled order must use FILLED status")
        active = _is_active(self.status)
        if self.side is Side.BUY:
            if self.sell_reserved_quantity != 0:
                raise ValueError("a BUY order cannot reserve sell quantity")
            if active and self.cash_reserved <= 0:
                raise ValueError("an active BUY order must reserve positive cash")
            if not active and self.cash_reserved != 0:
                raise ValueError("a terminal BUY order must release reserved cash")
        else:
            if not active and self.cash_reserved != 0:
                raise ValueError("a terminal SELL order must release reserved cash")
            expected_sell_hold = self.remaining_quantity if active else _ZERO
            if self.sell_reserved_quantity != expected_sell_hold:
                raise ValueError("SELL reservation must equal active remaining quantity")
        if self.last_attempt_id is not None:
            object.__setattr__(
                self,
                "last_attempt_id",
                _non_empty(self.last_attempt_id, "last_attempt_id"),
            )
        if self.filled_quantity > 0 and self.last_attempt_id is None:
            raise ValueError("an order with fills must identify its last match attempt")
        if self.order_hash != stable_hash(self._content_payload()):
            raise ValueError("order_hash does not match paper order content")

    @classmethod
    def build(
        cls,
        *,
        order_id: str,
        batch_hash: str,
        line_hash: str,
        decision_id: str,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        side: Side,
        quantity: Decimal,
        filled_quantity: Decimal,
        is_full_liquidation: bool,
        status: PaperOrderStatus,
        created_at: datetime,
        expires_at: datetime,
        cash_reserved: Decimal,
        sell_reserved_quantity: Decimal,
        market_rule_version: str,
        market_rule_hash: str,
        fee_rule_version: str,
        fee_rule_hash: str,
        slippage_model_version: str,
        slippage_model_hash: str,
        last_attempt_id: str | None = None,
    ) -> PaperOrder:
        remaining_quantity = quantity - filled_quantity
        values: dict[str, object] = {
            "order_id": order_id,
            "batch_hash": batch_hash,
            "line_hash": line_hash,
            "decision_id": decision_id,
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "side": side,
            "quantity": quantity,
            "filled_quantity": filled_quantity,
            "remaining_quantity": remaining_quantity,
            "is_full_liquidation": is_full_liquidation,
            "status": status,
            "created_at": created_at,
            "expires_at": expires_at,
            "cash_reserved": cash_reserved,
            "sell_reserved_quantity": sell_reserved_quantity,
            "market_rule_version": market_rule_version,
            "market_rule_hash": market_rule_hash,
            "fee_rule_version": fee_rule_version,
            "fee_rule_hash": fee_rule_hash,
            "slippage_model_version": slippage_model_version,
            "slippage_model_hash": slippage_model_hash,
            "last_attempt_id": last_attempt_id,
        }
        return cls(**values, order_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "order_hash")


@dataclass(frozen=True, slots=True)
class PaperMatchAttempt:
    """One auditable match attempt, including durable no-fill evidence."""

    attempt_id: str
    order_id: str
    attempted_at: datetime
    status: PaperOrderStatus
    no_fill_reason: PaperNoFillReason | None
    requested_quantity: Decimal
    filled_quantity: Decimal
    unfilled_quantity: Decimal
    reference_price: Decimal
    execution_price: Decimal | None
    gross_amount: Decimal
    participation_rate: Decimal
    state_revision: str
    data_version: str
    state_hash: str
    market_rule_version: str
    market_rule_hash: str
    slippage_model_version: str
    slippage_model_hash: str
    attempt_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.attempted_at)
        for field_name in (
            "attempt_id",
            "order_id",
            "state_revision",
            "data_version",
            "market_rule_version",
            "slippage_model_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        for value, field_name in (
            (self.state_hash, "state_hash"),
            (self.market_rule_hash, "market_rule_hash"),
            (self.slippage_model_hash, "slippage_model_hash"),
            (self.attempt_hash, "attempt_hash"),
        ):
            _sha256(value, field_name)
        if not isinstance(self.status, PaperOrderStatus):
            raise ValueError("attempt status must be a PaperOrderStatus value")
        if self.no_fill_reason is not None and not isinstance(
            self.no_fill_reason, PaperNoFillReason
        ):
            raise ValueError("no_fill_reason must be a PaperNoFillReason value")
        _whole(self.requested_quantity, "attempt requested_quantity", positive=True)
        _whole(self.filled_quantity, "attempt filled_quantity")
        _whole(self.unfilled_quantity, "attempt unfilled_quantity")
        _exact_decimal(self.reference_price, "attempt reference_price")
        _exact_decimal(self.gross_amount, "attempt gross_amount")
        _exact_decimal(self.participation_rate, "attempt participation_rate")
        if self.execution_price is not None:
            _exact_decimal(self.execution_price, "attempt execution_price")
        if self.reference_price <= 0:
            raise ValueError("attempt reference_price must be positive")
        if self.filled_quantity > self.requested_quantity:
            raise ValueError("attempt filled_quantity cannot exceed requested_quantity")
        if self.unfilled_quantity != self.requested_quantity - self.filled_quantity:
            raise ValueError("attempt unfilled_quantity must equal the unfilled request")
        if self.participation_rate < 0 or self.participation_rate > 1:
            raise ValueError("attempt participation_rate must be within [0, 1]")
        if self.filled_quantity == 0:
            if self.no_fill_reason is None:
                raise ValueError("a zero-fill attempt requires a no_fill_reason")
            if self.execution_price is not None or self.gross_amount != 0:
                raise ValueError("a zero-fill attempt cannot have an execution price or gross")
            if self.participation_rate != 0:
                raise ValueError("a zero-fill attempt must have zero participation")
            expected_status = (
                PaperOrderStatus.REJECTED
                if self.no_fill_reason is PaperNoFillReason.INSUFFICIENT_CASH
                else PaperOrderStatus.ACCEPTED
            )
            if self.status is not expected_status:
                raise ValueError("no-fill reason does not match the resulting order status")
        else:
            if self.no_fill_reason is not None:
                raise ValueError("a filled match attempt cannot have a no_fill_reason")
            if self.execution_price is None or self.execution_price <= 0:
                raise ValueError("a filled attempt requires a positive execution price")
            if self.participation_rate <= 0:
                raise ValueError("a filled attempt requires positive participation")
            if self.gross_amount != self.filled_quantity * self.execution_price:
                raise ValueError("attempt gross_amount must equal fill quantity times price")
            expected_status = (
                PaperOrderStatus.FILLED
                if self.unfilled_quantity == 0
                else PaperOrderStatus.PARTIALLY_FILLED
            )
            if self.status is not expected_status:
                raise ValueError("filled attempt status does not match its quantities")
        if self.attempt_hash != stable_hash(self._content_payload()):
            raise ValueError("attempt_hash does not match paper match attempt content")

    @classmethod
    def build(
        cls,
        *,
        attempt_id: str,
        order_id: str,
        attempted_at: datetime,
        status: PaperOrderStatus,
        requested_quantity: Decimal,
        filled_quantity: Decimal,
        reference_price: Decimal,
        execution_price: Decimal | None,
        participation_rate: Decimal,
        state_revision: str,
        data_version: str,
        state_hash: str,
        market_rule_version: str,
        market_rule_hash: str,
        slippage_model_version: str,
        slippage_model_hash: str,
        no_fill_reason: PaperNoFillReason | None = None,
    ) -> PaperMatchAttempt:
        unfilled_quantity = requested_quantity - filled_quantity
        gross_amount = filled_quantity * execution_price if execution_price is not None else _ZERO
        values: dict[str, object] = {
            "attempt_id": attempt_id,
            "order_id": order_id,
            "attempted_at": attempted_at,
            "status": status,
            "no_fill_reason": no_fill_reason,
            "requested_quantity": requested_quantity,
            "filled_quantity": filled_quantity,
            "unfilled_quantity": unfilled_quantity,
            "reference_price": reference_price,
            "execution_price": execution_price,
            "gross_amount": gross_amount,
            "participation_rate": participation_rate,
            "state_revision": state_revision,
            "data_version": data_version,
            "state_hash": state_hash,
            "market_rule_version": market_rule_version,
            "market_rule_hash": market_rule_hash,
            "slippage_model_version": slippage_model_version,
            "slippage_model_hash": slippage_model_hash,
        }
        return cls(**values, attempt_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "attempt_hash")


@dataclass(frozen=True, slots=True)
class PaperFill:
    """One immutable paper fill with exact fees and cash movement."""

    fill_id: str
    order_id: str
    attempt_id: str
    instrument_id: str
    instrument_type: TradableInstrumentType
    filled_at: datetime
    side: Side
    quantity: Decimal
    price: Decimal
    gross_amount: Decimal
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    other_fee: Decimal
    total_fee: Decimal
    cash_change: Decimal
    sellable_on: date | None
    fee_rule_version: str
    fee_rule_hash: str
    fill_hash: str

    def __post_init__(self) -> None:
        ensure_aware(self.filled_at)
        for field_name in (
            "fill_id",
            "order_id",
            "attempt_id",
            "instrument_id",
            "fee_rule_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("instrument_type must be a TradableInstrumentType value")
        if not isinstance(self.side, Side):
            raise ValueError("fill side must be a Side value")
        _whole(self.quantity, "fill quantity", positive=True)
        for field_name in (
            "price",
            "gross_amount",
            "commission",
            "stamp_duty",
            "transfer_fee",
            "other_fee",
            "total_fee",
            "cash_change",
        ):
            _exact_decimal(getattr(self, field_name), f"fill {field_name}")
        if self.price <= 0 or self.gross_amount <= 0:
            raise ValueError("fill price and gross_amount must be positive")
        if min(self.commission, self.stamp_duty, self.transfer_fee, self.other_fee) < 0:
            raise ValueError("fill fee components cannot be negative")
        if self.gross_amount != self.quantity * self.price:
            raise ValueError("fill gross_amount must equal quantity times price")
        expected_fee = self.commission + self.stamp_duty + self.transfer_fee + self.other_fee
        if self.total_fee != expected_fee:
            raise ValueError("fill total_fee must equal its fee component sum")
        expected_cash_change = (
            -(self.gross_amount + self.total_fee)
            if self.side is Side.BUY
            else self.gross_amount - self.total_fee
        )
        if self.cash_change != expected_cash_change:
            raise ValueError("fill cash_change does not match side, gross, and fees")
        if self.side is Side.BUY:
            if self.sellable_on is None:
                raise ValueError("a BUY fill must declare its sellable_on date")
            _strict_date(self.sellable_on, "fill sellable_on")
            if self.sellable_on < self.filled_at.astimezone(SHANGHAI_TZ).date():
                raise ValueError("fill sellable_on cannot precede its local fill date")
        elif self.sellable_on is not None:
            raise ValueError("a SELL fill cannot create a sellable_on date")
        _sha256(self.fee_rule_hash, "fee_rule_hash")
        _sha256(self.fill_hash, "fill_hash")
        if self.fill_hash != stable_hash(self._content_payload()):
            raise ValueError("fill_hash does not match paper fill content")

    @classmethod
    def build(
        cls,
        *,
        fill_id: str,
        order_id: str,
        attempt_id: str,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        filled_at: datetime,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        stamp_duty: Decimal,
        transfer_fee: Decimal,
        other_fee: Decimal,
        sellable_on: date | None,
        fee_rule_version: str,
        fee_rule_hash: str,
    ) -> PaperFill:
        gross_amount = quantity * price
        total_fee = commission + stamp_duty + transfer_fee + other_fee
        cash_change = -(gross_amount + total_fee) if side is Side.BUY else gross_amount - total_fee
        values: dict[str, object] = {
            "fill_id": fill_id,
            "order_id": order_id,
            "attempt_id": attempt_id,
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "filled_at": filled_at,
            "side": side,
            "quantity": quantity,
            "price": price,
            "gross_amount": gross_amount,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "transfer_fee": transfer_fee,
            "other_fee": other_fee,
            "total_fee": total_fee,
            "cash_change": cash_change,
            "sellable_on": sellable_on,
            "fee_rule_version": fee_rule_version,
            "fee_rule_hash": fee_rule_hash,
        }
        return cls(**values, fill_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "fill_hash")


@dataclass(frozen=True, slots=True)
class PaperAccountState:
    """Content-addressed cash, lots, holds, and orders for one paper account boundary."""

    schema_version: str
    account_id: str
    runtime_mode: RuntimeMode
    as_of: datetime
    data_version: str
    currency: str
    source_snapshot_id: str
    source_snapshot_hash: str
    source_snapshot_as_of: datetime
    total_cash: Decimal
    available_cash: Decimal
    frozen_cash: Decimal
    external_frozen_cash: Decimal
    positions: tuple[PaperPosition, ...]
    orders: tuple[PaperOrder, ...]
    processed_batch_hashes: tuple[str, ...]
    previous_state_hash: str | None
    event_log_hash: str
    state_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("paper account state schema_version must be '1'")
        for field_name in (
            "account_id",
            "data_version",
            "currency",
            "source_snapshot_id",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if self.runtime_mode is not RuntimeMode.PAPER:
            raise ValueError("paper account state requires PAPER runtime mode")
        ensure_aware(self.as_of)
        ensure_aware(self.source_snapshot_as_of)
        if self.source_snapshot_as_of > self.as_of:
            raise ValueError("source snapshot cannot be after paper account state")
        for value, field_name in (
            (self.source_snapshot_hash, "source_snapshot_hash"),
            (self.event_log_hash, "event_log_hash"),
            (self.state_hash, "state_hash"),
        ):
            _sha256(value, field_name)
        _optional_sha256(self.previous_state_hash, "previous_state_hash")
        for field_name in (
            "total_cash",
            "available_cash",
            "frozen_cash",
            "external_frozen_cash",
        ):
            cash_value = _exact_decimal(getattr(self, field_name), f"account {field_name}")
            if cash_value < 0:
                raise ValueError("paper account cash balances cannot be negative")
        position_ids = tuple(position.instrument_id for position in self.positions)
        if position_ids != tuple(sorted(set(position_ids))):
            raise ValueError("paper positions must be unique and sorted by instrument_id")
        if any(position.as_of != self.as_of for position in self.positions):
            raise ValueError("paper positions must align exactly to account state as_of")
        order_ids = tuple(order.order_id for order in self.orders)
        if order_ids != tuple(sorted(set(order_ids))):
            raise ValueError("paper orders must be unique and sorted by order_id")
        if any(order.created_at > self.as_of for order in self.orders):
            raise ValueError("future paper orders cannot enter an account state")
        if any(
            _is_active(order.status) and self.as_of >= order.expires_at for order in self.orders
        ):
            raise ValueError("expired paper orders must be transitioned out of active status")
        if self.processed_batch_hashes != tuple(sorted(set(self.processed_batch_hashes))):
            raise ValueError("processed batch hashes must be unique and sorted")
        for digest in self.processed_batch_hashes:
            _sha256(digest, "processed batch hash")
        if any(order.batch_hash not in self.processed_batch_hashes for order in self.orders):
            raise ValueError("every paper order must belong to a processed batch")
        active_order_cash = sum(
            (order.cash_reserved for order in self.orders if _is_active(order.status)),
            _ZERO,
        )
        expected_frozen_cash = self.external_frozen_cash + active_order_cash
        if self.frozen_cash != expected_frozen_cash:
            raise ValueError("frozen_cash must equal external and active order holds")
        if self.available_cash != self.total_cash - self.frozen_cash:
            raise ValueError("available_cash must equal total_cash minus frozen_cash")
        if self.frozen_cash > self.total_cash:
            raise ValueError("paper cash holds cannot exceed total_cash")
        expected_sell_holds: dict[str, Decimal] = {}
        for order in self.orders:
            if order.side is Side.SELL and _is_active(order.status):
                expected_sell_holds[order.instrument_id] = (
                    expected_sell_holds.get(order.instrument_id, _ZERO)
                    + order.sell_reserved_quantity
                )
        positions_by_id = {position.instrument_id: position for position in self.positions}
        if any(instrument_id not in positions_by_id for instrument_id in expected_sell_holds):
            raise ValueError("an active SELL hold requires a matching paper position")
        if any(
            position.reserved_sell_quantity
            != expected_sell_holds.get(position.instrument_id, _ZERO)
            for position in self.positions
        ):
            raise ValueError("position sell holds must equal active SELL order reservations")
        if self.state_hash != stable_hash(self._content_payload()):
            raise ValueError("state_hash does not match paper account state content")

    @classmethod
    def build(
        cls,
        *,
        account_id: str,
        as_of: datetime,
        data_version: str,
        currency: str,
        source_snapshot_id: str,
        source_snapshot_hash: str,
        source_snapshot_as_of: datetime,
        total_cash: Decimal,
        external_frozen_cash: Decimal,
        positions: tuple[PaperPosition, ...],
        orders: tuple[PaperOrder, ...],
        processed_batch_hashes: tuple[str, ...],
        event_log_hash: str,
        previous_state_hash: str | None = None,
        runtime_mode: RuntimeMode = RuntimeMode.PAPER,
    ) -> PaperAccountState:
        ordered_orders = tuple(sorted(orders, key=lambda order: order.order_id))
        if len({order.order_id for order in ordered_orders}) != len(ordered_orders):
            raise ValueError("paper orders must have unique order_id values")
        sell_holds: dict[str, Decimal] = {}
        for order in ordered_orders:
            if order.side is Side.SELL and _is_active(order.status):
                sell_holds[order.instrument_id] = (
                    sell_holds.get(order.instrument_id, _ZERO) + order.sell_reserved_quantity
                )
        provided_positions = tuple(sorted(positions, key=lambda position: position.instrument_id))
        if len({position.instrument_id for position in provided_positions}) != len(
            provided_positions
        ):
            raise ValueError("paper positions must have unique instrument_id values")
        rebuilt_positions = tuple(
            PaperPosition.build(
                as_of=as_of,
                instrument_id=position.instrument_id,
                instrument_type=position.instrument_type,
                lots=position.lots,
                reserved_sell_quantity=sell_holds.pop(position.instrument_id, _ZERO),
            )
            for position in provided_positions
        )
        if sell_holds:
            raise ValueError("an active SELL hold requires a matching paper position")
        batch_hashes = tuple(sorted(processed_batch_hashes))
        if len(set(batch_hashes)) != len(batch_hashes):
            raise ValueError("processed batch hashes must be unique")
        active_order_cash = sum(
            (order.cash_reserved for order in ordered_orders if _is_active(order.status)),
            _ZERO,
        )
        frozen_cash = external_frozen_cash + active_order_cash
        available_cash = total_cash - frozen_cash
        values: dict[str, object] = {
            "schema_version": "1",
            "account_id": account_id,
            "runtime_mode": runtime_mode,
            "as_of": as_of,
            "data_version": data_version,
            "currency": currency,
            "source_snapshot_id": source_snapshot_id,
            "source_snapshot_hash": source_snapshot_hash,
            "source_snapshot_as_of": source_snapshot_as_of,
            "total_cash": total_cash,
            "available_cash": available_cash,
            "frozen_cash": frozen_cash,
            "external_frozen_cash": external_frozen_cash,
            "positions": [asdict(position) for position in rebuilt_positions],
            "orders": [asdict(order) for order in ordered_orders],
            "processed_batch_hashes": batch_hashes,
            "previous_state_hash": previous_state_hash,
            "event_log_hash": event_log_hash,
        }
        return cls(
            schema_version="1",
            account_id=account_id,
            runtime_mode=runtime_mode,
            as_of=as_of,
            data_version=data_version,
            currency=currency,
            source_snapshot_id=source_snapshot_id,
            source_snapshot_hash=source_snapshot_hash,
            source_snapshot_as_of=source_snapshot_as_of,
            total_cash=total_cash,
            available_cash=available_cash,
            frozen_cash=frozen_cash,
            external_frozen_cash=external_frozen_cash,
            positions=rebuilt_positions,
            orders=ordered_orders,
            processed_batch_hashes=batch_hashes,
            previous_state_hash=previous_state_hash,
            event_log_hash=event_log_hash,
            state_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "state_hash")


@dataclass(frozen=True, slots=True)
class PaperExecutionRequest:
    """A CAS-bound, idempotent request to process one immutable draft batch."""

    schema_version: str
    request_id: str
    idempotency_key: str
    account_id: str
    expected_account_state_hash: str
    expected_account_snapshot_hash: str
    batch_hash: str
    submitted_at: datetime
    allow_partial: bool
    order_policy: PaperOrderPolicy
    max_market_state_age_seconds: int
    environment: AppEnvironment
    engine_version: str
    config_version: str
    config_hash: str
    request_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("paper execution request schema_version must be '1'")
        for field_name in (
            "request_id",
            "idempotency_key",
            "account_id",
            "engine_version",
            "config_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.submitted_at)
        if not isinstance(self.allow_partial, bool):
            raise ValueError("allow_partial must be a bool")
        if not isinstance(self.order_policy, PaperOrderPolicy):
            raise ValueError("order_policy must be a PaperOrderPolicy value")
        if (
            isinstance(self.max_market_state_age_seconds, bool)
            or not isinstance(self.max_market_state_age_seconds, int)
            or self.max_market_state_age_seconds <= 0
        ):
            raise ValueError("max_market_state_age_seconds must be a positive integer")
        if self.environment is not AppEnvironment.PAPER:
            raise ValueError("paper execution request requires the PAPER app environment")
        if self.engine_version != PAPER_EXECUTION_ENGINE_VERSION:
            raise ValueError("unknown paper execution engine version")
        for value, field_name in (
            (self.expected_account_state_hash, "expected_account_state_hash"),
            (self.expected_account_snapshot_hash, "expected_account_snapshot_hash"),
            (self.batch_hash, "batch_hash"),
            (self.config_hash, "config_hash"),
            (self.request_hash, "request_hash"),
        ):
            _sha256(value, field_name)
        expected_config_hash = stable_hash(
            {
                "version": self.config_version,
                "allow_partial": self.allow_partial,
                "order_policy": self.order_policy,
                "max_market_state_age_seconds": self.max_market_state_age_seconds,
                "environment": self.environment,
            }
        )
        if self.config_hash != expected_config_hash:
            raise ValueError("request config_hash does not match execution configuration")
        if self.request_hash != stable_hash(self._content_payload()):
            raise ValueError("request_hash does not match paper execution request content")

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        idempotency_key: str,
        account_id: str,
        expected_account_state_hash: str,
        expected_account_snapshot_hash: str,
        batch_hash: str,
        submitted_at: datetime,
        config: PaperExecutionConfig,
    ) -> PaperExecutionRequest:
        values: dict[str, object] = {
            "schema_version": "1",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "account_id": account_id,
            "expected_account_state_hash": expected_account_state_hash,
            "expected_account_snapshot_hash": expected_account_snapshot_hash,
            "batch_hash": batch_hash,
            "submitted_at": submitted_at,
            "allow_partial": config.allow_partial,
            "order_policy": config.order_policy,
            "max_market_state_age_seconds": config.max_market_state_age_seconds,
            "environment": config.environment,
            "engine_version": PAPER_EXECUTION_ENGINE_VERSION,
            "config_version": config.version,
            "config_hash": config.config_hash,
        }
        return cls(**values, request_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "request_hash")


@dataclass(frozen=True, slots=True)
class PaperExecutionReceipt:
    """Hash-bound result of one idempotent paper-execution request."""

    schema_version: str
    receipt_id: str
    request_id: str
    request_hash: str
    idempotency_key: str
    batch_hash: str
    request_submitted_at: datetime
    processed_at: datetime
    account_before_hash: str
    account_after_hash: str
    event_log_before_hash: str
    event_log_hash: str
    orders: tuple[PaperOrder, ...]
    attempts: tuple[PaperMatchAttempt, ...]
    fills: tuple[PaperFill, ...]
    account_before: PaperAccountState
    account_after: PaperAccountState
    receipt_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("paper execution receipt schema_version must be '1'")
        for field_name in ("receipt_id", "request_id", "idempotency_key"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.request_submitted_at)
        ensure_aware(self.processed_at)
        if self.processed_at < self.request_submitted_at:
            raise ValueError("paper receipt cannot precede request submission")
        for value, field_name in (
            (self.request_hash, "request_hash"),
            (self.batch_hash, "batch_hash"),
            (self.account_before_hash, "account_before_hash"),
            (self.account_after_hash, "account_after_hash"),
            (self.event_log_before_hash, "event_log_before_hash"),
            (self.event_log_hash, "event_log_hash"),
            (self.receipt_hash, "receipt_hash"),
        ):
            _sha256(value, field_name)
        order_ids = tuple(order.order_id for order in self.orders)
        attempt_ids = tuple(attempt.attempt_id for attempt in self.attempts)
        fill_ids = tuple(fill.fill_id for fill in self.fills)
        if order_ids != tuple(sorted(set(order_ids))):
            raise ValueError("receipt orders must be unique and sorted by order_id")
        if attempt_ids != tuple(sorted(set(attempt_ids))):
            raise ValueError("receipt attempts must be unique and sorted by attempt_id")
        if fill_ids != tuple(sorted(set(fill_ids))):
            raise ValueError("receipt fills must be unique and sorted by fill_id")
        if any(order.batch_hash != self.batch_hash for order in self.orders):
            raise ValueError("receipt orders must belong to its draft batch")
        if any(order.created_at != self.request_submitted_at for order in self.orders):
            raise ValueError("receipt orders must be created at request submission")
        orders_by_id = {order.order_id: order for order in self.orders}
        attempts_by_id = {attempt.attempt_id: attempt for attempt in self.attempts}
        if any(attempt.order_id not in orders_by_id for attempt in self.attempts):
            raise ValueError("every receipt attempt must bind to a receipt order")
        if any(
            fill.order_id not in orders_by_id or fill.attempt_id not in attempts_by_id
            for fill in self.fills
        ):
            raise ValueError("every receipt fill must bind to its order and attempt")
        for fill in self.fills:
            order = orders_by_id[fill.order_id]
            attempt = attempts_by_id[fill.attempt_id]
            if (
                fill.instrument_id != order.instrument_id
                or fill.instrument_type is not order.instrument_type
                or fill.side is not order.side
                or attempt.order_id != order.order_id
            ):
                raise ValueError("receipt fill identity must match its paper order and attempt")
        fills_by_attempt: dict[str, Decimal] = {}
        for fill in self.fills:
            fills_by_attempt[fill.attempt_id] = (
                fills_by_attempt.get(fill.attempt_id, _ZERO) + fill.quantity
            )
        if any(
            fills_by_attempt.get(attempt.attempt_id, _ZERO) != attempt.filled_quantity
            for attempt in self.attempts
        ):
            raise ValueError("receipt fills must reconcile to each match attempt")
        if any(
            order.last_attempt_id is not None
            and order.last_attempt_id not in attempts_by_id
            and order.order_id in {attempt.order_id for attempt in self.attempts}
            for order in self.orders
        ):
            raise ValueError("receipt order last_attempt_id must bind to its receipt attempts")
        attempts_by_order: dict[str, list[PaperMatchAttempt]] = {}
        for attempt in self.attempts:
            attempts_by_order.setdefault(attempt.order_id, []).append(attempt)
        fills_by_order: dict[str, Decimal] = {}
        for fill in self.fills:
            fills_by_order[fill.order_id] = fills_by_order.get(fill.order_id, _ZERO) + fill.quantity
        for order in self.orders:
            order_attempts = attempts_by_order.get(order.order_id, [])
            if len(order_attempts) != 1:
                raise ValueError("every receipt order must have exactly one match attempt")
            attempt = order_attempts[0]
            if order.last_attempt_id != attempt.attempt_id:
                raise ValueError("receipt order last_attempt_id must identify its match attempt")
            if (
                attempt.status is not order.status
                or attempt.requested_quantity != order.quantity
                or attempt.filled_quantity != order.filled_quantity
                or fills_by_order.get(order.order_id, _ZERO) != order.filled_quantity
            ):
                raise ValueError("receipt order quantities and status must reconcile to its events")
            if (
                attempt.market_rule_version != order.market_rule_version
                or attempt.market_rule_hash != order.market_rule_hash
                or attempt.slippage_model_version != order.slippage_model_version
                or attempt.slippage_model_hash != order.slippage_model_hash
            ):
                raise ValueError("receipt attempt model identities must match its paper order")
            if not (order.created_at <= attempt.attempted_at <= self.processed_at):
                raise ValueError("receipt attempt time must fall within the processing window")
            if attempt.attempted_at > order.expires_at:
                raise ValueError("receipt attempt cannot occur after order expiry")
        for fill in self.fills:
            order = orders_by_id[fill.order_id]
            attempt = attempts_by_id[fill.attempt_id]
            if (
                fill.fee_rule_version != order.fee_rule_version
                or fill.fee_rule_hash != order.fee_rule_hash
            ):
                raise ValueError("receipt fill fee identity must match its paper order")
            if fill.filled_at > order.expires_at:
                raise ValueError("receipt fill cannot occur after order expiry")
            if not (attempt.attempted_at <= fill.filled_at <= self.processed_at):
                raise ValueError("receipt fill time must follow its attempt within processing")
        if self.account_before_hash != self.account_before.state_hash:
            raise ValueError("account_before_hash must equal account_before.state_hash")
        if self.event_log_before_hash != self.account_before.event_log_hash:
            raise ValueError("receipt event_log_before_hash must match account_before")
        if self.request_submitted_at < self.account_before.as_of:
            raise ValueError("receipt request cannot precede account_before")
        if self.account_after_hash != self.account_after.state_hash:
            raise ValueError("account_after_hash must equal account_after.state_hash")
        if self.account_after.as_of != self.processed_at:
            raise ValueError("account_after as_of must equal receipt processed_at")
        if self.account_after.previous_state_hash != self.account_before_hash:
            raise ValueError("account_after must extend account_before_hash")
        if self.account_after.event_log_hash != self.event_log_hash:
            raise ValueError("receipt and account_after event logs must match")
        if self.batch_hash not in self.account_after.processed_batch_hashes:
            raise ValueError("account_after must record the processed draft batch")
        account_orders = {order.order_id: order.order_hash for order in self.account_after.orders}
        if any(account_orders.get(order.order_id) != order.order_hash for order in self.orders):
            raise ValueError("receipt orders must match the account_after order ledger")
        expected_event_log_hash = self.event_log_hash_for(
            previous_event_log_hash=self.event_log_before_hash,
            request_hash=self.request_hash,
            batch_hash=self.batch_hash,
            orders=self.orders,
            attempts=self.attempts,
            fills=self.fills,
        )
        if self.event_log_hash != expected_event_log_hash:
            raise ValueError("event_log_hash does not match receipt execution events")
        from .transition import replay_execution_transition

        expected_account_after = replay_execution_transition(
            account_before=self.account_before,
            batch_hash=self.batch_hash,
            processed_at=self.processed_at,
            event_log_hash=self.event_log_hash,
            orders=self.orders,
            fills=self.fills,
        )
        if self.account_after != expected_account_after:
            raise ValueError("account_after does not match canonical paper execution replay")
        if self.receipt_hash != stable_hash(self._content_payload()):
            raise ValueError("receipt_hash does not match paper execution receipt content")

    @staticmethod
    def event_log_hash_for(
        *,
        previous_event_log_hash: str,
        request_hash: str,
        batch_hash: str,
        orders: tuple[PaperOrder, ...],
        attempts: tuple[PaperMatchAttempt, ...],
        fills: tuple[PaperFill, ...],
    ) -> str:
        """Return the deterministic append-only event-log identity for one request."""

        _sha256(previous_event_log_hash, "previous_event_log_hash")
        _sha256(request_hash, "request_hash")
        _sha256(batch_hash, "batch_hash")
        ordered_orders = tuple(sorted(orders, key=lambda order: order.order_id))
        ordered_attempts = tuple(sorted(attempts, key=lambda attempt: attempt.attempt_id))
        ordered_fills = tuple(sorted(fills, key=lambda fill: fill.fill_id))
        return stable_hash(
            {
                "previous_event_log_hash": previous_event_log_hash,
                "request_hash": request_hash,
                "batch_hash": batch_hash,
                "order_hashes": [order.order_hash for order in ordered_orders],
                "attempt_hashes": [attempt.attempt_hash for attempt in ordered_attempts],
                "fill_hashes": [fill.fill_hash for fill in ordered_fills],
            }
        )

    @classmethod
    def build(
        cls,
        *,
        receipt_id: str,
        request: PaperExecutionRequest,
        account_before: PaperAccountState,
        account_after: PaperAccountState,
        processed_at: datetime,
        orders: tuple[PaperOrder, ...],
        attempts: tuple[PaperMatchAttempt, ...],
        fills: tuple[PaperFill, ...],
    ) -> PaperExecutionReceipt:
        if request.account_id != account_before.account_id:
            raise PaperExecutionInputError("request account_id does not match account state")
        if request.submitted_at < account_before.as_of:
            raise PaperExecutionInputError("request submission cannot precede account state")
        if request.expected_account_state_hash != account_before.state_hash:
            raise PaperExecutionInputError("expected account state hash is stale")
        if request.expected_account_snapshot_hash != account_before.source_snapshot_hash:
            raise PaperExecutionInputError("expected account snapshot hash is stale")
        if account_after.account_id != account_before.account_id:
            raise PaperExecutionInputError("paper account identity cannot change during execution")
        if (
            account_after.source_snapshot_id != account_before.source_snapshot_id
            or account_after.source_snapshot_hash != account_before.source_snapshot_hash
            or account_after.source_snapshot_as_of != account_before.source_snapshot_as_of
        ):
            raise PaperExecutionInputError("source account snapshot identity cannot change")
        ordered_orders = tuple(sorted(orders, key=lambda order: order.order_id))
        ordered_attempts = tuple(sorted(attempts, key=lambda attempt: attempt.attempt_id))
        ordered_fills = tuple(sorted(fills, key=lambda fill: fill.fill_id))
        event_log_hash = cls.event_log_hash_for(
            previous_event_log_hash=account_before.event_log_hash,
            request_hash=request.request_hash,
            batch_hash=request.batch_hash,
            orders=ordered_orders,
            attempts=ordered_attempts,
            fills=ordered_fills,
        )
        if account_after.event_log_hash != event_log_hash:
            raise PaperExecutionInputError(
                "account_after event log does not match the request execution events"
            )
        from .transition import replay_execution_transition

        expected_account_after = replay_execution_transition(
            account_before=account_before,
            batch_hash=request.batch_hash,
            processed_at=processed_at,
            event_log_hash=event_log_hash,
            orders=ordered_orders,
            fills=ordered_fills,
        )
        if account_after != expected_account_after:
            raise PaperExecutionInputError(
                "account_after does not match canonical paper execution replay"
            )
        values: dict[str, object] = {
            "schema_version": "1",
            "receipt_id": receipt_id,
            "request_id": request.request_id,
            "request_hash": request.request_hash,
            "idempotency_key": request.idempotency_key,
            "batch_hash": request.batch_hash,
            "request_submitted_at": request.submitted_at,
            "processed_at": processed_at,
            "account_before_hash": account_before.state_hash,
            "account_after_hash": account_after.state_hash,
            "event_log_before_hash": account_before.event_log_hash,
            "event_log_hash": event_log_hash,
            "orders": [asdict(order) for order in ordered_orders],
            "attempts": [asdict(attempt) for attempt in ordered_attempts],
            "fills": [asdict(fill) for fill in ordered_fills],
            "account_before": asdict(account_before),
            "account_after": asdict(account_after),
        }
        return cls(
            schema_version="1",
            receipt_id=receipt_id,
            request_id=request.request_id,
            request_hash=request.request_hash,
            idempotency_key=request.idempotency_key,
            batch_hash=request.batch_hash,
            request_submitted_at=request.submitted_at,
            processed_at=processed_at,
            account_before_hash=account_before.state_hash,
            account_after_hash=account_after.state_hash,
            event_log_before_hash=account_before.event_log_hash,
            event_log_hash=event_log_hash,
            orders=ordered_orders,
            attempts=ordered_attempts,
            fills=ordered_fills,
            account_before=account_before,
            account_after=account_after,
            receipt_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "receipt_hash")


__all__ = [
    "PAPER_EXECUTION_ENGINE_VERSION",
    "PaperAccountState",
    "PaperExecutionConfig",
    "PaperExecutionInputError",
    "PaperExecutionReceipt",
    "PaperExecutionRequest",
    "PaperFill",
    "PaperIdempotencyConflict",
    "PaperMatchAttempt",
    "PaperNoFillReason",
    "PaperOrder",
    "PaperOrderPolicy",
    "PaperOrderStatus",
    "PaperPosition",
    "PaperPositionLot",
]
