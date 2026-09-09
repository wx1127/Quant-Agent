"""Immutable account, cash, and position snapshots for portfolio services."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum, StrEnum
from typing import Any

from quant_agent.backtest import TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _decimal(value: Decimal | int | str, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field_name} must be an exact decimal input")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f"{field_name} must be a valid decimal") from error
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _validate_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        ensure_aware(value)
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("cannot serialize non-finite Decimal")
        if value == 0:
            return "0"
        rendered = format(value, "f")
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(
        _canonical(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ValuationStatus(StrEnum):
    """Whether a position mark is current, stale, or retained through suspension."""

    FRESH = "FRESH"
    STALE = "STALE"
    SUSPENDED = "SUSPENDED"


@dataclass(frozen=True, slots=True)
class CashSnapshot:
    """Cash known at one account boundary, split into available and frozen amounts."""

    as_of: datetime
    currency: str
    total_cash: Decimal
    available_cash: Decimal
    frozen_cash: Decimal

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        object.__setattr__(self, "currency", _non_empty(self.currency, "cash currency"))
        for field_name in ("total_cash", "available_cash", "frozen_cash"):
            value = _decimal(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
            if value < 0:
                raise ValueError("cash balances cannot be negative")
        if self.total_cash != self.available_cash + self.frozen_cash:
            raise ValueError("total_cash must equal available_cash plus frozen_cash")


@dataclass(frozen=True, slots=True)
class PortfolioPositionSnapshot:
    """One valued long position with sellable, frozen, and unsettled quantities."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    total_quantity: Decimal
    available_quantity: Decimal
    frozen_quantity: Decimal
    unsettled_quantity: Decimal
    average_cost: Decimal
    valuation_price: Decimal
    price_observed_at: datetime
    price_available_at: datetime
    position_as_of: datetime
    price_data_version: str
    price_source_hash: str
    valuation_status: ValuationStatus
    mark_policy_version: str
    mark_policy_hash: str
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pnl: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        quantity_fields = (
            "total_quantity",
            "available_quantity",
            "frozen_quantity",
            "unsettled_quantity",
        )
        for field_name in quantity_fields:
            value = _decimal(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
            if value < 0:
                raise ValueError("position quantities must be non-negative")
        if self.total_quantity <= 0:
            raise ValueError("position total_quantity must be positive")
        if self.total_quantity != (
            self.available_quantity + self.frozen_quantity + self.unsettled_quantity
        ):
            raise ValueError(
                "total_quantity must equal available, frozen, and unsettled quantities"
            )
        for field_name in (
            "average_cost",
            "valuation_price",
            "market_value",
            "cost_basis",
            "unrealized_pnl",
        ):
            object.__setattr__(
                self,
                field_name,
                _decimal(getattr(self, field_name), field_name),
            )
        if self.average_cost < 0 or self.valuation_price <= 0:
            raise ValueError("average_cost cannot be negative and valuation_price must be positive")
        ensure_aware(self.price_observed_at)
        ensure_aware(self.price_available_at)
        ensure_aware(self.position_as_of)
        if self.price_observed_at > self.price_available_at:
            raise ValueError("price_available_at cannot precede price_observed_at")
        object.__setattr__(
            self,
            "price_data_version",
            _non_empty(self.price_data_version, "price_data_version"),
        )
        _validate_hash(self.price_source_hash, "price_source_hash")
        object.__setattr__(
            self,
            "mark_policy_version",
            _non_empty(self.mark_policy_version, "mark_policy_version"),
        )
        _validate_hash(self.mark_policy_hash, "mark_policy_hash")
        expected_market_value = self.total_quantity * self.valuation_price
        expected_cost_basis = self.total_quantity * self.average_cost
        if self.market_value != expected_market_value:
            raise ValueError("market_value must equal quantity times valuation_price")
        if self.cost_basis != expected_cost_basis:
            raise ValueError("cost_basis must equal quantity times average_cost")
        if self.unrealized_pnl != self.market_value - self.cost_basis:
            raise ValueError("unrealized_pnl must equal market_value minus cost_basis")

    @classmethod
    def build(
        cls,
        *,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        available_quantity: Decimal,
        frozen_quantity: Decimal,
        unsettled_quantity: Decimal,
        average_cost: Decimal,
        valuation_price: Decimal,
        price_observed_at: datetime,
        price_available_at: datetime,
        position_as_of: datetime,
        price_data_version: str,
        price_source_hash: str,
        mark_policy_hash: str,
        valuation_status: ValuationStatus = ValuationStatus.FRESH,
        mark_policy_version: str = "position-mark-v1",
    ) -> PortfolioPositionSnapshot:
        available = _decimal(available_quantity, "available_quantity")
        frozen = _decimal(frozen_quantity, "frozen_quantity")
        unsettled = _decimal(unsettled_quantity, "unsettled_quantity")
        quantity = available + frozen + unsettled
        cost = _decimal(average_cost, "average_cost")
        price = _decimal(valuation_price, "valuation_price")
        market_value = quantity * price
        cost_basis = quantity * cost
        return cls(
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            total_quantity=quantity,
            available_quantity=available,
            frozen_quantity=frozen,
            unsettled_quantity=unsettled,
            average_cost=cost,
            valuation_price=price,
            price_observed_at=price_observed_at,
            price_available_at=price_available_at,
            position_as_of=position_as_of,
            price_data_version=price_data_version,
            price_source_hash=price_source_hash,
            valuation_status=valuation_status,
            mark_policy_version=mark_policy_version,
            mark_policy_hash=mark_policy_hash,
            market_value=market_value,
            cost_basis=cost_basis,
            unrealized_pnl=market_value - cost_basis,
        )


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Content-addressed account and portfolio valuation at one exact boundary."""

    schema_version: str
    snapshot_id: str
    account_id: str
    runtime_mode: RuntimeMode
    as_of: datetime
    valuation_at: datetime
    data_version: str
    currency: str
    cash: CashSnapshot
    positions: tuple[PortfolioPositionSnapshot, ...]
    position_market_value: Decimal
    total_equity: Decimal
    gross_exposure: Decimal
    cash_weight: Decimal
    previous_snapshot_hash: str | None
    source_event_log_hash: str | None
    content_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("account snapshot schema_version must be '1'")
        for field_name in ("snapshot_id", "account_id", "data_version", "currency"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.as_of)
        ensure_aware(self.valuation_at)
        if not isinstance(self.runtime_mode, RuntimeMode):
            raise ValueError("account runtime_mode must be a RuntimeMode value")
        if self.valuation_at > self.as_of:
            raise ValueError("valuation_at cannot be after account snapshot as_of")
        if self.cash.as_of != self.as_of or self.cash.currency != self.currency:
            raise ValueError("cash time and currency must align exactly to account snapshot")
        position_ids = tuple(item.instrument_id for item in self.positions)
        if tuple(sorted(set(position_ids))) != position_ids:
            raise ValueError("account positions must be unique and sorted by instrument_id")
        for position in self.positions:
            if position.position_as_of != self.as_of:
                raise ValueError("position_as_of must align exactly to account snapshot as_of")
            if position.price_observed_at > self.valuation_at:
                raise ValueError("position price_observed_at cannot follow account valuation_at")
            if (
                position.valuation_status is ValuationStatus.FRESH
                and position.price_observed_at != self.valuation_at
            ):
                raise ValueError("FRESH position marks must be observed at valuation_at")
            if position.price_available_at > self.as_of:
                raise ValueError("future valuation prices cannot enter an account snapshot")
        for field_name in (
            "position_market_value",
            "total_equity",
            "gross_exposure",
            "cash_weight",
        ):
            object.__setattr__(
                self,
                field_name,
                _decimal(getattr(self, field_name), field_name),
            )
        expected_market_value = sum(
            (item.market_value for item in self.positions),
            Decimal(0),
        )
        expected_equity = self.cash.total_cash + expected_market_value
        if self.position_market_value != expected_market_value:
            raise ValueError("position_market_value must equal valued positions")
        if self.total_equity != expected_equity:
            raise ValueError("total_equity must equal cash plus position market value")
        if self.total_equity <= 0:
            raise ValueError("account snapshot total_equity must be positive")
        with localcontext() as context:
            context.prec = 50
            expected_exposure = expected_market_value / self.total_equity
            expected_cash_weight = self.cash.total_cash / self.total_equity
        if self.gross_exposure != expected_exposure or self.cash_weight != expected_cash_weight:
            raise ValueError("account exposure and cash weight must match valued totals")
        if self.gross_exposure + self.cash_weight != Decimal(1):
            raise ValueError("account exposure and cash weight must sum to one")
        for optional_hash, field_name in (
            (self.previous_snapshot_hash, "previous_snapshot_hash"),
            (self.source_event_log_hash, "source_event_log_hash"),
        ):
            if optional_hash is not None:
                _validate_hash(optional_hash, field_name)
        _validate_hash(self.content_hash, "content_hash")
        if self.content_hash != _stable_hash(self._content_payload()):
            raise ValueError("account snapshot content_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        snapshot_id: str,
        account_id: str,
        runtime_mode: RuntimeMode,
        as_of: datetime,
        valuation_at: datetime,
        data_version: str,
        currency: str,
        cash: CashSnapshot,
        positions: tuple[PortfolioPositionSnapshot, ...],
        previous_snapshot_hash: str | None = None,
        source_event_log_hash: str | None = None,
    ) -> AccountSnapshot:
        market_value = sum((item.market_value for item in positions), Decimal(0))
        equity = cash.total_cash + market_value
        if equity <= 0:
            raise ValueError("account snapshot total_equity must be positive")
        with localcontext() as context:
            context.prec = 50
            exposure = market_value / equity
            cash_weight = cash.total_cash / equity
        values: dict[str, Any] = {
            "schema_version": "1",
            "snapshot_id": snapshot_id,
            "account_id": account_id,
            "runtime_mode": runtime_mode,
            "as_of": as_of,
            "valuation_at": valuation_at,
            "data_version": data_version,
            "currency": currency,
            "cash": asdict(cash),
            "positions": [asdict(item) for item in positions],
            "position_market_value": market_value,
            "total_equity": equity,
            "gross_exposure": exposure,
            "cash_weight": cash_weight,
            "previous_snapshot_hash": previous_snapshot_hash,
            "source_event_log_hash": source_event_log_hash,
        }
        return cls(
            schema_version="1",
            snapshot_id=snapshot_id,
            account_id=account_id,
            runtime_mode=runtime_mode,
            as_of=as_of,
            valuation_at=valuation_at,
            data_version=data_version,
            currency=currency,
            cash=cash,
            positions=positions,
            position_market_value=market_value,
            total_equity=equity,
            gross_exposure=exposure,
            cash_weight=cash_weight,
            previous_snapshot_hash=previous_snapshot_hash,
            source_event_log_hash=source_event_log_hash,
            content_hash=_stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "account_id": self.account_id,
            "runtime_mode": self.runtime_mode,
            "as_of": self.as_of,
            "valuation_at": self.valuation_at,
            "data_version": self.data_version,
            "currency": self.currency,
            "cash": asdict(self.cash),
            "positions": [asdict(item) for item in self.positions],
            "position_market_value": self.position_market_value,
            "total_equity": self.total_equity,
            "gross_exposure": self.gross_exposure,
            "cash_weight": self.cash_weight,
            "previous_snapshot_hash": self.previous_snapshot_hash,
            "source_event_log_hash": self.source_event_log_hash,
        }

    def to_json(self) -> str:
        """Return canonical JSON with exact decimal strings and the content hash."""

        return _json({**self._content_payload(), "content_hash": self.content_hash})

    @classmethod
    def from_json(cls, value: str) -> AccountSnapshot:
        """Parse canonical snapshot JSON and revalidate all economic identities."""

        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("account snapshot JSON is invalid") from error
        if not isinstance(payload, dict):
            raise ValueError("account snapshot JSON must contain an object")
        expected_root_keys = {
            "schema_version",
            "snapshot_id",
            "account_id",
            "runtime_mode",
            "as_of",
            "valuation_at",
            "data_version",
            "currency",
            "cash",
            "positions",
            "position_market_value",
            "total_equity",
            "gross_exposure",
            "cash_weight",
            "previous_snapshot_hash",
            "source_event_log_hash",
            "content_hash",
        }
        if set(payload) != expected_root_keys:
            raise ValueError("account snapshot JSON fields do not match schema_version 1")
        cash_payload = payload.get("cash")
        positions_payload = payload.get("positions")
        if not isinstance(cash_payload, dict) or not isinstance(positions_payload, list):
            raise ValueError("account snapshot JSON cash and positions are invalid")
        if set(cash_payload) != {
            "as_of",
            "currency",
            "total_cash",
            "available_cash",
            "frozen_cash",
        }:
            raise ValueError("account snapshot JSON cash fields do not match schema")
        expected_position_keys = {
            "instrument_id",
            "instrument_type",
            "total_quantity",
            "available_quantity",
            "frozen_quantity",
            "unsettled_quantity",
            "average_cost",
            "valuation_price",
            "price_observed_at",
            "price_available_at",
            "position_as_of",
            "price_data_version",
            "price_source_hash",
            "valuation_status",
            "mark_policy_version",
            "mark_policy_hash",
            "market_value",
            "cost_basis",
            "unrealized_pnl",
        }
        if any(
            not isinstance(item, dict) or set(item) != expected_position_keys
            for item in positions_payload
        ):
            raise ValueError("account snapshot JSON position fields do not match schema")
        cash = CashSnapshot(
            as_of=datetime.fromisoformat(cash_payload["as_of"]),
            currency=cash_payload["currency"],
            total_cash=cash_payload["total_cash"],
            available_cash=cash_payload["available_cash"],
            frozen_cash=cash_payload["frozen_cash"],
        )
        positions = tuple(
            PortfolioPositionSnapshot(
                instrument_id=item["instrument_id"],
                instrument_type=TradableInstrumentType(item["instrument_type"]),
                total_quantity=item["total_quantity"],
                available_quantity=item["available_quantity"],
                frozen_quantity=item["frozen_quantity"],
                unsettled_quantity=item["unsettled_quantity"],
                average_cost=item["average_cost"],
                valuation_price=item["valuation_price"],
                price_observed_at=datetime.fromisoformat(item["price_observed_at"]),
                price_available_at=datetime.fromisoformat(item["price_available_at"]),
                position_as_of=datetime.fromisoformat(item["position_as_of"]),
                price_data_version=item["price_data_version"],
                price_source_hash=item["price_source_hash"],
                valuation_status=ValuationStatus(item["valuation_status"]),
                mark_policy_version=item["mark_policy_version"],
                mark_policy_hash=item["mark_policy_hash"],
                market_value=item["market_value"],
                cost_basis=item["cost_basis"],
                unrealized_pnl=item["unrealized_pnl"],
            )
            for item in positions_payload
        )
        return cls(
            schema_version=payload["schema_version"],
            snapshot_id=payload["snapshot_id"],
            account_id=payload["account_id"],
            runtime_mode=RuntimeMode(payload["runtime_mode"]),
            as_of=datetime.fromisoformat(payload["as_of"]),
            valuation_at=datetime.fromisoformat(payload["valuation_at"]),
            data_version=payload["data_version"],
            currency=payload["currency"],
            cash=cash,
            positions=positions,
            position_market_value=payload["position_market_value"],
            total_equity=payload["total_equity"],
            gross_exposure=payload["gross_exposure"],
            cash_weight=payload["cash_weight"],
            previous_snapshot_hash=payload["previous_snapshot_hash"],
            source_event_log_hash=payload["source_event_log_hash"],
            content_hash=payload["content_hash"],
        )

    def identity_payload(self) -> dict[str, str]:
        return {
            "account_id": self.account_id,
            "as_of": self.as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "content_hash": self.content_hash,
            "data_version": self.data_version,
            "snapshot_id": self.snapshot_id,
        }


__all__ = [
    "AccountSnapshot",
    "CashSnapshot",
    "PortfolioPositionSnapshot",
    "ValuationStatus",
]
