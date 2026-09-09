"""Immutable contracts for independent order, fill, cash, and position reconciliation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import Any

from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware
from quant_agent.execution.order_drafts import OrderDraftBatch
from quant_agent.execution.paper import PaperExecutionReceipt, PaperOrderStatus
from quant_agent.portfolio import AccountSnapshot
from quant_agent.regime.contracts import stable_hash

RECONCILIATION_ENGINE_VERSION = "reconciliation-engine-v1"
_ZERO = Decimal(0)
_MAX_DECIMAL_DIGITS = 200
_MAX_DECIMAL_ABS_EXPONENT = 200
_EXACT_ARITHMETIC_PRECISION = (_MAX_DECIMAL_DIGITS * 4) + 32
_MAX_RECONCILIATION_PRECISION = 512
_MAX_OBSERVATION_LAG_SECONDS = 86_400


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


def _optional_sha256(value: str | None, field_name: str) -> None:
    if value is not None:
        _sha256(value, field_name)


def _exact_decimal(value: Decimal, field_name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{field_name} must be a finite Decimal")
    _, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"{field_name} has an invalid Decimal exponent")
    if len(digits) > _MAX_DECIMAL_DIGITS or abs(value.adjusted()) > _MAX_DECIMAL_ABS_EXPONENT:
        raise ValueError(f"{field_name} exceeds the reconciliation numeric boundary")
    return value


def _whole(value: Decimal, field_name: str, *, positive: bool = False) -> None:
    _exact_decimal(value, field_name)
    minimum = Decimal(1) if positive else _ZERO
    if value < minimum or value != value.to_integral_value():
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{field_name} must be a {qualifier}whole Decimal")


def _content_payload(value: Any, hash_field: str) -> dict[str, Any]:
    return {key: item for key, item in asdict(value).items() if key != hash_field}


def _difference_matches(*, expected: Decimal, observed: Decimal, delta: Decimal) -> bool:
    with localcontext() as context:
        context.prec = _EXACT_ARITHMETIC_PRECISION
        return delta == observed - expected


def _validate_nested_decimal_boundaries(value: object, path: str) -> None:
    """Reject nested numeric inputs that the reconciliation engine cannot represent."""

    if isinstance(value, Decimal):
        _exact_decimal(value, path)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _validate_nested_decimal_boundaries(
                getattr(value, field.name),
                f"{path}.{field.name}",
            )
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_nested_decimal_boundaries(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_nested_decimal_boundaries(item, f"{path}[{key!r}]")


def _nested_decimal_values(value: object) -> list[Decimal]:
    values: list[Decimal] = []

    def collect(item: object) -> None:
        if isinstance(item, Decimal):
            values.append(item)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                collect(getattr(item, field.name))
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)

    collect(value)
    return values


def _observed_fill_economics(
    *,
    side: Side,
    quantity: Decimal,
    price: Decimal,
    commission: Decimal,
    stamp_duty: Decimal,
    transfer_fee: Decimal,
    other_fee: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    """Derive exact fill economics independently of the process Decimal context."""

    inputs = {
        "quantity": quantity,
        "price": price,
        "commission": commission,
        "stamp_duty": stamp_duty,
        "transfer_fee": transfer_fee,
        "other_fee": other_fee,
    }
    for field_name, value in inputs.items():
        _exact_decimal(value, f"fill report {field_name}")
    with localcontext() as context:
        context.prec = _EXACT_ARITHMETIC_PRECISION
        gross_amount = quantity * price
        total_fee = commission + stamp_duty + transfer_fee + other_fee
        cash_change = -(gross_amount + total_fee) if side is Side.BUY else gross_amount - total_fee
    for field_name, value in (
        ("gross_amount", gross_amount),
        ("total_fee", total_fee),
        ("cash_change", cash_change),
    ):
        _exact_decimal(value, f"fill report {field_name}")
    return gross_amount, total_fee, cash_change


class ReconciliationStatus(StrEnum):
    """Overall conclusion at one exact observation boundary."""

    MATCHED = "MATCHED"
    RECONCILED_WITH_VARIANCE = "RECONCILED_WITH_VARIANCE"
    MISMATCH = "MISMATCH"
    UNRECONCILABLE = "UNRECONCILABLE"


class ReconciliationSeverity(StrEnum):
    """Ordered severity carried by every explicit reconciliation difference."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


_SEVERITY_RANK = {
    ReconciliationSeverity.INFO: 0,
    ReconciliationSeverity.WARNING: 1,
    ReconciliationSeverity.ERROR: 2,
    ReconciliationSeverity.CRITICAL: 3,
}


def severity_rank(value: ReconciliationSeverity) -> int:
    """Return the stable ordering used by policies and result derivation."""

    return _SEVERITY_RANK[value]


class ReconciliationCheckStatus(StrEnum):
    """One detailed comparison outcome."""

    MATCH = "MATCH"
    EXPECTED_VARIANCE = "EXPECTED_VARIANCE"
    MISSING = "MISSING"
    UNEXPECTED = "UNEXPECTED"
    MISMATCH = "MISMATCH"
    CONFLICT = "CONFLICT"


_CHECK_STATUS_RANK = {
    ReconciliationCheckStatus.MATCH: 0,
    ReconciliationCheckStatus.EXPECTED_VARIANCE: 1,
    ReconciliationCheckStatus.MISSING: 2,
    ReconciliationCheckStatus.UNEXPECTED: 3,
    ReconciliationCheckStatus.MISMATCH: 4,
    ReconciliationCheckStatus.CONFLICT: 5,
}


class ReconciliationDomain(StrEnum):
    """Evidence domain to which a finding belongs."""

    INPUT = "INPUT"
    REPORT = "REPORT"
    ORDER = "ORDER"
    FILL = "FILL"
    CASH = "CASH"
    POSITION = "POSITION"


class ReconciliationDifferenceCode(StrEnum):
    """Closed vocabulary for findings consumed by monitoring and the kill switch."""

    DRAFT_RECEIPT_BATCH_MISMATCH = "DRAFT_RECEIPT_BATCH_MISMATCH"
    DECISION_MISMATCH = "DECISION_MISMATCH"
    ACCOUNT_IDENTITY_MISMATCH = "ACCOUNT_IDENTITY_MISMATCH"
    SOURCE_SYSTEM_MISMATCH = "SOURCE_SYSTEM_MISMATCH"
    RUNTIME_MODE_MISMATCH = "RUNTIME_MODE_MISMATCH"
    DATA_VERSION_MISMATCH = "DATA_VERSION_MISMATCH"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    SOURCE_SNAPSHOT_MISMATCH = "SOURCE_SNAPSHOT_MISMATCH"
    SNAPSHOT_BOUNDARY_MISMATCH = "SNAPSHOT_BOUNDARY_MISMATCH"
    EVENT_LOG_MISMATCH = "EVENT_LOG_MISMATCH"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    FUTURE_EVIDENCE = "FUTURE_EVIDENCE"
    UNLINKED_ORDER_REPORT = "UNLINKED_ORDER_REPORT"
    UNLINKED_FILL_REPORT = "UNLINKED_FILL_REPORT"
    ORDER_REPORT_REGRESSION = "ORDER_REPORT_REGRESSION"
    ORDER_MISSING = "ORDER_MISSING"
    ORDER_UNEXPECTED = "ORDER_UNEXPECTED"
    ORDER_IDENTITY_MISMATCH = "ORDER_IDENTITY_MISMATCH"
    ORDER_STATUS_MISMATCH = "ORDER_STATUS_MISMATCH"
    ORDER_QUANTITY_MISMATCH = "ORDER_QUANTITY_MISMATCH"
    ORDER_REASON_MISMATCH = "ORDER_REASON_MISMATCH"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_PARTIAL = "ORDER_PARTIAL"
    ORDER_NO_FILL = "ORDER_NO_FILL"
    FILL_MISSING = "FILL_MISSING"
    FILL_UNEXPECTED = "FILL_UNEXPECTED"
    FILL_IDENTITY_MISMATCH = "FILL_IDENTITY_MISMATCH"
    FILL_QUANTITY_MISMATCH = "FILL_QUANTITY_MISMATCH"
    FILL_GROSS_MISMATCH = "FILL_GROSS_MISMATCH"
    FILL_PRICE_MISMATCH = "FILL_PRICE_MISMATCH"
    FILL_FEE_MISMATCH = "FILL_FEE_MISMATCH"
    FILL_CASH_MISMATCH = "FILL_CASH_MISMATCH"
    FILL_OVERFILL = "FILL_OVERFILL"
    CASH_TOTAL_MISMATCH = "CASH_TOTAL_MISMATCH"
    CASH_AVAILABLE_MISMATCH = "CASH_AVAILABLE_MISMATCH"
    CASH_FROZEN_MISMATCH = "CASH_FROZEN_MISMATCH"
    POSITION_MISSING = "POSITION_MISSING"
    POSITION_UNEXPECTED = "POSITION_UNEXPECTED"
    POSITION_TYPE_MISMATCH = "POSITION_TYPE_MISMATCH"
    POSITION_QUANTITY_MISMATCH = "POSITION_QUANTITY_MISMATCH"
    POSITION_BUCKET_MISMATCH = "POSITION_BUCKET_MISMATCH"
    POSITION_COST_MISMATCH = "POSITION_COST_MISMATCH"
    IDENTICAL_DUPLICATE_REPORT = "IDENTICAL_DUPLICATE_REPORT"
    CONFLICTING_DUPLICATE_REPORT = "CONFLICTING_DUPLICATE_REPORT"


class ReconciliationReportKind(StrEnum):
    ORDER = "ORDER"
    FILL = "FILL"


class DuplicateClassification(StrEnum):
    IDENTICAL = "IDENTICAL"
    CONFLICTING = "CONFLICTING"


class DuplicateKeyKind(StrEnum):
    """Namespace for identities that can independently form duplicate groups."""

    SOURCE_IDENTITY = "SOURCE_IDENTITY"
    REPORT_ID = "REPORT_ID"


class ReconciliationStopAction(StrEnum):
    NONE = "NONE"
    STOP_NEW_ORDERS = "STOP_NEW_ORDERS"


class ReconciliationStopScope(StrEnum):
    ACCOUNT = "ACCOUNT"
    GLOBAL = "GLOBAL"


@dataclass(frozen=True, slots=True)
class ObservedOrderReport:
    """One independent order-state report; duplicate envelopes remain observable."""

    schema_version: str
    source_system: str
    account_id: str
    report_id: str
    source_order_id: str
    client_order_id: str | None
    revision: int
    event_time: datetime
    available_at: datetime
    instrument_id: str
    instrument_type: TradableInstrumentType
    side: Side
    status: PaperOrderStatus
    requested_quantity: Decimal
    cumulative_filled_quantity: Decimal
    remaining_quantity: Decimal
    outcome_code: str | None
    source_payload_hash: str
    report_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("observed order report schema_version must be '1'")
        for field_name in (
            "source_system",
            "account_id",
            "report_id",
            "source_order_id",
            "instrument_id",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if self.client_order_id is not None:
            object.__setattr__(
                self,
                "client_order_id",
                _non_empty(self.client_order_id, "client_order_id"),
            )
        if self.outcome_code is not None:
            object.__setattr__(
                self,
                "outcome_code",
                _non_empty(self.outcome_code, "outcome_code"),
            )
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("order report revision must be a positive integer")
        ensure_aware(self.event_time)
        ensure_aware(self.available_at)
        if self.event_time > self.available_at:
            raise ValueError("order report available_at cannot precede event_time")
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("order report instrument_type is invalid")
        if not isinstance(self.side, Side):
            raise ValueError("order report side is invalid")
        if not isinstance(self.status, PaperOrderStatus):
            raise ValueError("order report status is invalid")
        _whole(self.requested_quantity, "order report requested_quantity", positive=True)
        _whole(self.cumulative_filled_quantity, "order report cumulative_filled_quantity")
        _whole(self.remaining_quantity, "order report remaining_quantity")
        if self.cumulative_filled_quantity > self.requested_quantity:
            raise ValueError("order report cumulative fill cannot exceed requested quantity")
        if not _difference_matches(
            expected=self.cumulative_filled_quantity,
            observed=self.requested_quantity,
            delta=self.remaining_quantity,
        ):
            raise ValueError("order report remaining quantity does not conserve requested quantity")
        if self.status is PaperOrderStatus.FILLED and self.remaining_quantity != 0:
            raise ValueError("FILLED order report must have zero remaining quantity")
        if self.status is PaperOrderStatus.PARTIALLY_FILLED and not (
            _ZERO < self.cumulative_filled_quantity < self.requested_quantity
        ):
            raise ValueError("PARTIALLY_FILLED order report requires a proper partial fill")
        if self.status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.REJECTED} and (
            self.cumulative_filled_quantity != 0
        ):
            raise ValueError("ACCEPTED or REJECTED order report cannot contain fills")
        if self.status in {PaperOrderStatus.ACCEPTED, PaperOrderStatus.REJECTED} and (
            self.outcome_code is None
        ):
            raise ValueError("ACCEPTED or REJECTED order report requires an outcome_code")
        if self.status in {PaperOrderStatus.FILLED, PaperOrderStatus.PARTIALLY_FILLED} and (
            self.outcome_code is not None
        ):
            raise ValueError("a filled order report cannot carry outcome_code")
        _sha256(self.source_payload_hash, "order source_payload_hash")
        _sha256(self.report_hash, "order report_hash")
        if self.source_payload_hash != stable_hash(self._source_payload()):
            raise ValueError("order source_payload_hash does not match report semantics")
        if self.report_hash != stable_hash(self._report_payload()):
            raise ValueError("order report_hash does not match its envelope")

    @classmethod
    def build(
        cls,
        *,
        source_system: str,
        account_id: str,
        report_id: str,
        source_order_id: str,
        client_order_id: str | None,
        revision: int,
        event_time: datetime,
        available_at: datetime,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        side: Side,
        status: PaperOrderStatus,
        requested_quantity: Decimal,
        cumulative_filled_quantity: Decimal,
        remaining_quantity: Decimal,
        outcome_code: str | None = None,
    ) -> ObservedOrderReport:
        source_payload = {
            "schema_version": "1",
            "source_system": source_system,
            "account_id": account_id,
            "source_order_id": source_order_id,
            "client_order_id": client_order_id,
            "revision": revision,
            "event_time": event_time,
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "side": side,
            "status": status,
            "requested_quantity": requested_quantity,
            "cumulative_filled_quantity": cumulative_filled_quantity,
            "remaining_quantity": remaining_quantity,
            "outcome_code": outcome_code,
        }
        source_hash = stable_hash(source_payload)
        report_payload = {
            "schema_version": "1",
            "report_id": report_id,
            "available_at": available_at,
            "source_payload_hash": source_hash,
        }
        return cls(
            schema_version="1",
            source_system=source_system,
            account_id=account_id,
            report_id=report_id,
            source_order_id=source_order_id,
            client_order_id=client_order_id,
            revision=revision,
            event_time=event_time,
            available_at=available_at,
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            side=side,
            status=status,
            requested_quantity=requested_quantity,
            cumulative_filled_quantity=cumulative_filled_quantity,
            remaining_quantity=remaining_quantity,
            outcome_code=outcome_code,
            source_payload_hash=source_hash,
            report_hash=stable_hash(report_payload),
        )

    @property
    def dedup_key(self) -> tuple[str, str, str, str]:
        return (
            self.source_system,
            self.account_id,
            self.source_order_id,
            str(self.revision),
        )

    def _source_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_system": self.source_system,
            "account_id": self.account_id,
            "source_order_id": self.source_order_id,
            "client_order_id": self.client_order_id,
            "revision": self.revision,
            "event_time": self.event_time,
            "instrument_id": self.instrument_id,
            "instrument_type": self.instrument_type,
            "side": self.side,
            "status": self.status,
            "requested_quantity": self.requested_quantity,
            "cumulative_filled_quantity": self.cumulative_filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "outcome_code": self.outcome_code,
        }

    def _report_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "available_at": self.available_at,
            "source_payload_hash": self.source_payload_hash,
        }


@dataclass(frozen=True, slots=True)
class ObservedFillReport:
    """One independent economic fill report with exact fee and cash identities."""

    schema_version: str
    source_system: str
    account_id: str
    report_id: str
    source_fill_id: str
    source_order_id: str
    client_order_id: str | None
    client_fill_id: str | None
    filled_at: datetime
    available_at: datetime
    instrument_id: str
    instrument_type: TradableInstrumentType
    side: Side
    currency: str
    quantity: Decimal
    price: Decimal
    gross_amount: Decimal
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    other_fee: Decimal
    total_fee: Decimal
    cash_change: Decimal
    source_payload_hash: str
    report_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("observed fill report schema_version must be '1'")
        for field_name in (
            "source_system",
            "account_id",
            "report_id",
            "source_fill_id",
            "source_order_id",
            "instrument_id",
            "currency",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        for field_name in ("client_order_id", "client_fill_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_empty(value, field_name))
        ensure_aware(self.filled_at)
        ensure_aware(self.available_at)
        if self.filled_at > self.available_at:
            raise ValueError("fill report available_at cannot precede filled_at")
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("fill report instrument_type is invalid")
        if not isinstance(self.side, Side):
            raise ValueError("fill report side is invalid")
        _whole(self.quantity, "fill report quantity", positive=True)
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
            _exact_decimal(getattr(self, field_name), f"fill report {field_name}")
        if self.price <= 0 or self.gross_amount <= 0:
            raise ValueError("fill report price and gross_amount must be positive")
        if min(self.commission, self.stamp_duty, self.transfer_fee, self.other_fee) < 0:
            raise ValueError("fill report fee components cannot be negative")
        expected_gross, expected_fee, expected_cash = _observed_fill_economics(
            side=self.side,
            quantity=self.quantity,
            price=self.price,
            commission=self.commission,
            stamp_duty=self.stamp_duty,
            transfer_fee=self.transfer_fee,
            other_fee=self.other_fee,
        )
        if self.gross_amount != expected_gross:
            raise ValueError("fill report gross_amount must equal quantity times price")
        if self.total_fee != expected_fee:
            raise ValueError("fill report total_fee must equal its fee components")
        if self.cash_change != expected_cash:
            raise ValueError("fill report cash_change does not match side, gross, and fees")
        _sha256(self.source_payload_hash, "fill source_payload_hash")
        _sha256(self.report_hash, "fill report_hash")
        if self.source_payload_hash != stable_hash(self._source_payload()):
            raise ValueError("fill source_payload_hash does not match report semantics")
        if self.report_hash != stable_hash(self._report_payload()):
            raise ValueError("fill report_hash does not match its envelope")

    @classmethod
    def build(
        cls,
        *,
        source_system: str,
        account_id: str,
        report_id: str,
        source_fill_id: str,
        source_order_id: str,
        client_order_id: str | None,
        client_fill_id: str | None,
        filled_at: datetime,
        available_at: datetime,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        side: Side,
        currency: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        stamp_duty: Decimal,
        transfer_fee: Decimal,
        other_fee: Decimal,
    ) -> ObservedFillReport:
        gross_amount, total_fee, cash_change = _observed_fill_economics(
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            other_fee=other_fee,
        )
        source_payload = {
            "schema_version": "1",
            "source_system": source_system,
            "account_id": account_id,
            "source_fill_id": source_fill_id,
            "source_order_id": source_order_id,
            "client_order_id": client_order_id,
            "client_fill_id": client_fill_id,
            "filled_at": filled_at,
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "side": side,
            "currency": currency,
            "quantity": quantity,
            "price": price,
            "gross_amount": gross_amount,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "transfer_fee": transfer_fee,
            "other_fee": other_fee,
            "total_fee": total_fee,
            "cash_change": cash_change,
        }
        source_hash = stable_hash(source_payload)
        report_payload = {
            "schema_version": "1",
            "report_id": report_id,
            "available_at": available_at,
            "source_payload_hash": source_hash,
        }
        return cls(
            schema_version="1",
            source_system=source_system,
            account_id=account_id,
            report_id=report_id,
            source_fill_id=source_fill_id,
            source_order_id=source_order_id,
            client_order_id=client_order_id,
            client_fill_id=client_fill_id,
            filled_at=filled_at,
            available_at=available_at,
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            side=side,
            currency=currency,
            quantity=quantity,
            price=price,
            gross_amount=gross_amount,
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            other_fee=other_fee,
            total_fee=total_fee,
            cash_change=cash_change,
            source_payload_hash=source_hash,
            report_hash=stable_hash(report_payload),
        )

    @property
    def dedup_key(self) -> tuple[str, str, str]:
        return (self.source_system, self.account_id, self.source_fill_id)

    def _source_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_system": self.source_system,
            "account_id": self.account_id,
            "source_fill_id": self.source_fill_id,
            "source_order_id": self.source_order_id,
            "client_order_id": self.client_order_id,
            "client_fill_id": self.client_fill_id,
            "filled_at": self.filled_at,
            "instrument_id": self.instrument_id,
            "instrument_type": self.instrument_type,
            "side": self.side,
            "currency": self.currency,
            "quantity": self.quantity,
            "price": self.price,
            "gross_amount": self.gross_amount,
            "commission": self.commission,
            "stamp_duty": self.stamp_duty,
            "transfer_fee": self.transfer_fee,
            "other_fee": self.other_fee,
            "total_fee": self.total_fee,
            "cash_change": self.cash_change,
        }

    def _report_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "available_at": self.available_at,
            "source_payload_hash": self.source_payload_hash,
        }


def _canonical_fills_for_aggregate_validation(
    reports: tuple[ObservedFillReport, ...],
) -> tuple[ObservedFillReport, ...]:
    """Mirror duplicate exclusion so numeric validation covers engine aggregates."""

    envelopes: dict[tuple[str, str, str], list[ObservedFillReport]] = {}
    for report in reports:
        key = (report.source_system, report.account_id, report.report_id)
        envelopes.setdefault(key, []).append(report)
    excluded_hashes = {
        report.report_hash
        for values in envelopes.values()
        if len({item.report_hash for item in values}) > 1
        for report in values
    }
    semantic_groups: dict[tuple[str, str, str], list[ObservedFillReport]] = {}
    for report in reports:
        if report.report_hash not in excluded_hashes:
            semantic_groups.setdefault(report.dedup_key, []).append(report)
    canonical: list[ObservedFillReport] = []
    for values in semantic_groups.values():
        if len({item.source_payload_hash for item in values}) == 1:
            canonical.append(min(values, key=lambda item: item.report_hash))
    return tuple(sorted(canonical, key=lambda item: (*item.dedup_key, item.report_hash)))


@dataclass(frozen=True, slots=True)
class ObservedExecutionEvidence:
    """Raw normalized reports and one authoritative account snapshot."""

    schema_version: str
    source_system: str
    account_id: str
    observed_at: datetime
    available_at: datetime
    account_snapshot: AccountSnapshot
    order_reports: tuple[ObservedOrderReport, ...]
    fill_reports: tuple[ObservedFillReport, ...]
    source_cursor: str | None
    evidence_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("observed evidence schema_version must be '1'")
        for field_name in ("source_system", "account_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if self.source_cursor is not None:
            object.__setattr__(
                self,
                "source_cursor",
                _non_empty(self.source_cursor, "source_cursor"),
            )
        ensure_aware(self.observed_at)
        ensure_aware(self.available_at)
        if self.observed_at > self.available_at:
            raise ValueError("observed evidence available_at cannot precede observed_at")
        _validate_nested_decimal_boundaries(
            self.account_snapshot,
            "observed.account_snapshot",
        )
        canonical_orders = tuple(
            sorted(
                self.order_reports,
                key=lambda item: (*item.dedup_key, item.source_payload_hash, item.report_hash),
            )
        )
        canonical_fills = tuple(
            sorted(
                self.fill_reports,
                key=lambda item: (*item.dedup_key, item.source_payload_hash, item.report_hash),
            )
        )
        if self.order_reports != canonical_orders or self.fill_reports != canonical_fills:
            raise ValueError("observed reports must use deterministic canonical ordering")
        _sha256(self.evidence_hash, "evidence_hash")
        if self.evidence_hash != stable_hash(self._content_payload()):
            raise ValueError("evidence_hash does not match observed evidence")

    @classmethod
    def build(
        cls,
        *,
        source_system: str,
        account_id: str,
        observed_at: datetime,
        available_at: datetime,
        account_snapshot: AccountSnapshot,
        order_reports: tuple[ObservedOrderReport, ...],
        fill_reports: tuple[ObservedFillReport, ...],
        source_cursor: str | None = None,
    ) -> ObservedExecutionEvidence:
        ordered_orders = tuple(
            sorted(
                order_reports,
                key=lambda item: (*item.dedup_key, item.source_payload_hash, item.report_hash),
            )
        )
        ordered_fills = tuple(
            sorted(
                fill_reports,
                key=lambda item: (*item.dedup_key, item.source_payload_hash, item.report_hash),
            )
        )
        values: dict[str, object] = {
            "schema_version": "1",
            "source_system": source_system,
            "account_id": account_id,
            "observed_at": observed_at,
            "available_at": available_at,
            "account_snapshot_hash": account_snapshot.content_hash,
            "order_report_hashes": [item.report_hash for item in ordered_orders],
            "fill_report_hashes": [item.report_hash for item in ordered_fills],
            "source_cursor": source_cursor,
        }
        return cls(
            schema_version="1",
            source_system=source_system,
            account_id=account_id,
            observed_at=observed_at,
            available_at=available_at,
            account_snapshot=account_snapshot,
            order_reports=ordered_orders,
            fill_reports=ordered_fills,
            source_cursor=source_cursor,
            evidence_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_system": self.source_system,
            "account_id": self.account_id,
            "observed_at": self.observed_at,
            "available_at": self.available_at,
            "account_snapshot_hash": self.account_snapshot.content_hash,
            "order_report_hashes": [item.report_hash for item in self.order_reports],
            "fill_report_hashes": [item.report_hash for item in self.fill_reports],
            "source_cursor": self.source_cursor,
        }


def _validate_fill_aggregate_boundaries(
    receipt: PaperExecutionReceipt,
    observed: ObservedExecutionEvidence,
) -> None:
    """Ensure every per-order aggregate and delta remains inside result boundaries."""

    field_names = (
        "quantity",
        "gross_amount",
        "commission",
        "stamp_duty",
        "transfer_fee",
        "other_fee",
        "total_fee",
        "cash_change",
    )
    expected_groups: dict[str, list[object]] = {}
    for expected_fill in receipt.fills:
        expected_groups.setdefault(expected_fill.order_id, []).append(expected_fill)
    observed_groups: dict[str, list[object]] = {}
    for observed_fill in _canonical_fills_for_aggregate_validation(observed.fill_reports):
        if observed_fill.client_order_id is not None:
            observed_groups.setdefault(observed_fill.client_order_id, []).append(observed_fill)

    def aggregates(
        groups: dict[str, list[object]],
        label: str,
    ) -> dict[str, dict[str, Decimal]]:
        result: dict[str, dict[str, Decimal]] = {}
        for order_id, fills in groups.items():
            values: dict[str, Decimal] = {}
            for field_name in field_names:
                with localcontext() as context:
                    context.prec = _EXACT_ARITHMETIC_PRECISION
                    total = sum(
                        (getattr(item, field_name) for item in fills),
                        _ZERO,
                    )
                values[field_name] = _exact_decimal(
                    total,
                    f"{label}[{order_id}].{field_name}",
                )
            result[order_id] = values
        return result

    expected = aggregates(expected_groups, "receipt.fill_aggregate")
    actual = aggregates(observed_groups, "observed.fill_aggregate")
    for order_id in set(expected) | set(actual):
        for field_name in field_names:
            expected_value = expected.get(order_id, {}).get(field_name, _ZERO)
            observed_value = actual.get(order_id, {}).get(field_name, _ZERO)
            with localcontext() as context:
                context.prec = _EXACT_ARITHMETIC_PRECISION
                delta = observed_value - expected_value
            _exact_decimal(delta, f"fill_aggregate_delta[{order_id}].{field_name}")


@dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    """Versioned tolerances and stop threshold; quantity always remains exact."""

    version: str = "reconciliation-policy-v1"
    max_observation_lag_seconds: int = 300
    cash_tolerance: Decimal = _ZERO
    fee_tolerance: Decimal = _ZERO
    price_tolerance: Decimal = _ZERO
    cost_tolerance: Decimal = _ZERO
    stop_on_severity: ReconciliationSeverity = ReconciliationSeverity.CRITICAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "policy version"))
        if (
            not isinstance(self.max_observation_lag_seconds, int)
            or isinstance(self.max_observation_lag_seconds, bool)
            or self.max_observation_lag_seconds <= 0
            or self.max_observation_lag_seconds > _MAX_OBSERVATION_LAG_SECONDS
        ):
            raise ValueError("max_observation_lag_seconds must be an integer within [1, 86400]")
        for field_name in (
            "cash_tolerance",
            "fee_tolerance",
            "price_tolerance",
            "cost_tolerance",
        ):
            value = _exact_decimal(getattr(self, field_name), field_name)
            if value < 0:
                raise ValueError("reconciliation tolerances cannot be negative")
        if not isinstance(self.stop_on_severity, ReconciliationSeverity):
            raise ValueError("stop_on_severity must be a ReconciliationSeverity value")
        if self.stop_on_severity is ReconciliationSeverity.INFO:
            raise ValueError("stop_on_severity cannot be INFO because a clean match has no trigger")

    @property
    def policy_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class ReconciliationRequest:
    """Expected and observed evidence at one deterministic reconciliation boundary."""

    schema_version: str
    request_id: str
    idempotency_key: str
    reconciled_at: datetime
    draft: OrderDraftBatch
    receipt: PaperExecutionReceipt
    observed: ObservedExecutionEvidence
    policy: ReconciliationPolicy
    previous_result_hash: str | None
    input_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reconciliation request schema_version must be '1'")
        for field_name in ("request_id", "idempotency_key"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.reconciled_at)
        for value, path in (
            (self.draft, "draft"),
            (self.receipt, "receipt"),
            (self.observed, "observed"),
        ):
            _validate_nested_decimal_boundaries(value, path)
        _validate_fill_aggregate_boundaries(self.receipt, self.observed)
        _required_reconciliation_precision(self)
        _optional_sha256(self.previous_result_hash, "previous_result_hash")
        _sha256(self.input_hash, "input_hash")
        if self.input_hash != stable_hash(self._input_payload()):
            raise ValueError("reconciliation input_hash does not match its evidence")

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        idempotency_key: str,
        reconciled_at: datetime,
        draft: OrderDraftBatch,
        receipt: PaperExecutionReceipt,
        observed: ObservedExecutionEvidence,
        policy: ReconciliationPolicy | None = None,
        previous_result_hash: str | None = None,
    ) -> ReconciliationRequest:
        selected_policy = policy or ReconciliationPolicy()
        values: dict[str, object] = {
            "schema_version": "1",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "reconciled_at": reconciled_at,
            "draft_hash": draft.batch_hash,
            "receipt_hash": receipt.receipt_hash,
            "evidence_hash": observed.evidence_hash,
            "policy_hash": selected_policy.policy_hash,
            "previous_result_hash": previous_result_hash,
        }
        return cls(
            schema_version="1",
            request_id=request_id,
            idempotency_key=idempotency_key,
            reconciled_at=reconciled_at,
            draft=draft,
            receipt=receipt,
            observed=observed,
            policy=selected_policy,
            previous_result_hash=previous_result_hash,
            input_hash=stable_hash(values),
        )

    @property
    def is_executable(self) -> bool:
        return False

    def _input_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "reconciled_at": self.reconciled_at,
            "draft_hash": self.draft.batch_hash,
            "receipt_hash": self.receipt.receipt_hash,
            "evidence_hash": self.observed.evidence_hash,
            "policy_hash": self.policy.policy_hash,
            "previous_result_hash": self.previous_result_hash,
        }


def _required_reconciliation_precision(request: ReconciliationRequest) -> int:
    """Return one deterministic context size, rejecting unsafe cross-field combinations."""

    values = _nested_decimal_values(request)
    maximum_integer_digits = 1
    maximum_fraction_digits = 0
    for value in values:
        _, digits, exponent = value.as_tuple()
        if not isinstance(exponent, int):
            raise ValueError("reconciliation input has an invalid Decimal exponent")
        maximum_integer_digits = max(
            maximum_integer_digits,
            max(len(digits) + exponent, 1),
        )
        maximum_fraction_digits = max(maximum_fraction_digits, max(-exponent, 0))
    accumulation_digits = len(str(max(len(values), 1)))
    required = max(
        80,
        maximum_integer_digits + maximum_fraction_digits + accumulation_digits + 20,
    )
    if required > _MAX_RECONCILIATION_PRECISION:
        raise ValueError("reconciliation numeric precision exceeds the safe boundary")
    return required


@dataclass(frozen=True, slots=True)
class ReconciliationFinding:
    """One explicit, content-addressed difference that is never silently discarded."""

    finding_id: str
    domain: ReconciliationDomain
    code: ReconciliationDifferenceCode
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    entity_id: str
    message: str
    decision_id: str
    batch_hash: str
    line_hash: str | None
    order_id: str | None
    fill_id: str | None
    instrument_id: str | None
    expected_value: str | None
    observed_value: str | None
    delta: str | None
    evidence_hashes: tuple[str, ...]
    finding_hash: str

    def __post_init__(self) -> None:
        for field_name in ("finding_id", "entity_id", "message", "decision_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if not isinstance(self.domain, ReconciliationDomain):
            raise ValueError("finding domain is invalid")
        if not isinstance(self.code, ReconciliationDifferenceCode):
            raise ValueError("finding code is invalid")
        if not isinstance(self.status, ReconciliationCheckStatus) or (
            self.status is ReconciliationCheckStatus.MATCH
        ):
            raise ValueError("a finding cannot have MATCH status")
        if not isinstance(self.severity, ReconciliationSeverity):
            raise ValueError("finding severity is invalid")
        if self.status is ReconciliationCheckStatus.EXPECTED_VARIANCE:
            if self.severity is not ReconciliationSeverity.WARNING:
                raise ValueError("expected variance findings must have WARNING severity")
        elif self.status in {
            ReconciliationCheckStatus.CONFLICT,
            ReconciliationCheckStatus.MISSING,
            ReconciliationCheckStatus.UNEXPECTED,
        }:
            if self.severity is not ReconciliationSeverity.CRITICAL:
                raise ValueError("conflict, missing, and unexpected findings must be CRITICAL")
        elif self.status is ReconciliationCheckStatus.MISMATCH and severity_rank(
            self.severity
        ) < severity_rank(ReconciliationSeverity.ERROR):
            raise ValueError("mismatch findings must have ERROR or CRITICAL severity")
        if (
            self.code in _UNRECONCILABLE_CODES
            and self.severity is not ReconciliationSeverity.CRITICAL
        ):
            raise ValueError("unreconcilable finding codes must have CRITICAL severity")
        _sha256(self.batch_hash, "finding batch_hash")
        for value, field_name in (
            (self.line_hash, "finding line_hash"),
            (self.evidence_hashes, "finding evidence_hash"),
        ):
            values = (value,) if isinstance(value, str) else value
            if values is None:
                continue
            for digest in values:
                _sha256(digest, field_name)
        if self.evidence_hashes != tuple(sorted(set(self.evidence_hashes))):
            raise ValueError("finding evidence hashes must be unique and sorted")
        for field_name in ("order_id", "fill_id", "instrument_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_empty(value, field_name))
        _sha256(self.finding_hash, "finding_hash")
        identity_hash = stable_hash(
            {
                "domain": self.domain,
                "code": self.code,
                "entity_id": self.entity_id,
                "decision_id": self.decision_id,
                "batch_hash": self.batch_hash,
                "line_hash": self.line_hash,
                "order_id": self.order_id,
                "fill_id": self.fill_id,
                "instrument_id": self.instrument_id,
            }
        )
        if self.finding_id != f"finding:{identity_hash}":
            raise ValueError("finding_id does not match finding identity")
        if self.finding_hash != stable_hash(self._content_payload()):
            raise ValueError("finding_hash does not match finding content")

    @classmethod
    def build(
        cls,
        *,
        domain: ReconciliationDomain,
        code: ReconciliationDifferenceCode,
        status: ReconciliationCheckStatus,
        severity: ReconciliationSeverity,
        entity_id: str,
        message: str,
        decision_id: str,
        batch_hash: str,
        line_hash: str | None = None,
        order_id: str | None = None,
        fill_id: str | None = None,
        instrument_id: str | None = None,
        expected_value: str | None = None,
        observed_value: str | None = None,
        delta: str | None = None,
        evidence_hashes: tuple[str, ...] = (),
    ) -> ReconciliationFinding:
        ordered_evidence = tuple(sorted(set(evidence_hashes)))
        identity = {
            "domain": domain,
            "code": code,
            "entity_id": entity_id,
            "decision_id": decision_id,
            "batch_hash": batch_hash,
            "line_hash": line_hash,
            "order_id": order_id,
            "fill_id": fill_id,
            "instrument_id": instrument_id,
        }
        finding_id = f"finding:{stable_hash(identity)}"
        values: dict[str, object] = {
            "finding_id": finding_id,
            "domain": domain,
            "code": code,
            "status": status,
            "severity": severity,
            "entity_id": entity_id,
            "message": message,
            "decision_id": decision_id,
            "batch_hash": batch_hash,
            "line_hash": line_hash,
            "order_id": order_id,
            "fill_id": fill_id,
            "instrument_id": instrument_id,
            "expected_value": expected_value,
            "observed_value": observed_value,
            "delta": delta,
            "evidence_hashes": ordered_evidence,
        }
        return cls(**values, finding_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "finding_hash")


def _derived_check_status(
    findings: tuple[ReconciliationFinding, ...],
) -> ReconciliationCheckStatus:
    if not findings:
        return ReconciliationCheckStatus.MATCH
    return max((item.status for item in findings), key=lambda value: _CHECK_STATUS_RANK[value])


def _derived_severity(
    findings: tuple[ReconciliationFinding, ...],
) -> ReconciliationSeverity:
    if not findings:
        return ReconciliationSeverity.INFO
    return max((item.severity for item in findings), key=severity_rank)


def _ordered_findings(
    findings: tuple[ReconciliationFinding, ...],
) -> tuple[ReconciliationFinding, ...]:
    ordered = tuple(sorted(findings, key=lambda item: item.finding_hash))
    if len({item.finding_hash for item in ordered}) != len(ordered):
        raise ValueError("reconciliation findings must be unique")
    if len({item.finding_id for item in ordered}) != len(ordered):
        raise ValueError("reconciliation finding IDs must be unique")
    return ordered


@dataclass(frozen=True, slots=True)
class OrderReconciliation:
    """Plan, expected order, and observed order comparison for one identity."""

    check_id: str
    input_hash: str
    decision_id: str
    batch_hash: str
    line_hash: str | None
    order_id: str | None
    source_order_id: str | None
    instrument_id: str | None
    planned_side: Side | None
    planned_quantity: Decimal | None
    expected_status: PaperOrderStatus | None
    expected_outcome_code: str | None
    expected_filled_quantity: Decimal | None
    expected_remaining_quantity: Decimal | None
    observed_status: PaperOrderStatus | None
    observed_outcome_code: str | None
    observed_requested_quantity: Decimal | None
    observed_filled_quantity: Decimal | None
    observed_remaining_quantity: Decimal | None
    report_hashes: tuple[str, ...]
    findings: tuple[ReconciliationFinding, ...]
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    check_hash: str

    def __post_init__(self) -> None:
        for field_name in ("check_id", "decision_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        _sha256(self.input_hash, "order check input_hash")
        _sha256(self.batch_hash, "order check batch_hash")
        _optional_sha256(self.line_hash, "order check line_hash")
        for field_name in ("order_id", "source_order_id", "instrument_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_empty(value, field_name))
        for field_name in ("expected_outcome_code", "observed_outcome_code"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_empty(value, field_name))
        for field_name in (
            "planned_quantity",
            "expected_filled_quantity",
            "expected_remaining_quantity",
            "observed_requested_quantity",
            "observed_filled_quantity",
            "observed_remaining_quantity",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _whole(value, f"order check {field_name}")
        if self.planned_side is not None and not isinstance(self.planned_side, Side):
            raise ValueError("order check planned_side is invalid")
        if self.report_hashes != tuple(sorted(set(self.report_hashes))):
            raise ValueError("order check report hashes must be unique and sorted")
        for digest in self.report_hashes:
            _sha256(digest, "order check report_hash")
        ordered = _ordered_findings(self.findings)
        if self.findings != ordered:
            raise ValueError("order findings must use canonical ordering")
        if self.status is not _derived_check_status(self.findings):
            raise ValueError("order check status must be derived from findings")
        if self.severity is not _derived_severity(self.findings):
            raise ValueError("order check severity must be derived from findings")
        _sha256(self.check_hash, "order check_hash")
        if self.check_hash != stable_hash(self._content_payload()):
            raise ValueError("order check_hash does not match content")

    @classmethod
    def build(cls, **kwargs: Any) -> OrderReconciliation:
        findings = _ordered_findings(tuple(kwargs.pop("findings", ())))
        kwargs["report_hashes"] = tuple(sorted(set(kwargs.get("report_hashes", ()))))
        kwargs["findings"] = findings
        kwargs["status"] = _derived_check_status(findings)
        kwargs["severity"] = _derived_severity(findings)
        values = dict(kwargs)
        hash_values = {**values, "findings": tuple(asdict(item) for item in findings)}
        return cls(**values, check_hash=stable_hash(hash_values))

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "check_hash")


@dataclass(frozen=True, slots=True)
class FillReconciliation:
    """Aggregated expected and observed fills for one explicit order identity."""

    check_id: str
    input_hash: str
    decision_id: str
    batch_hash: str
    line_hash: str | None
    order_id: str
    instrument_id: str
    expected_fill_ids: tuple[str, ...]
    observed_source_fill_ids: tuple[str, ...]
    expected_quantity: Decimal
    observed_quantity: Decimal
    quantity_delta: Decimal
    expected_gross: Decimal
    observed_gross: Decimal
    gross_delta: Decimal
    expected_total_fee: Decimal
    observed_total_fee: Decimal
    fee_delta: Decimal
    expected_commission: Decimal
    observed_commission: Decimal
    commission_delta: Decimal
    expected_stamp_duty: Decimal
    observed_stamp_duty: Decimal
    stamp_duty_delta: Decimal
    expected_transfer_fee: Decimal
    observed_transfer_fee: Decimal
    transfer_fee_delta: Decimal
    expected_other_fee: Decimal
    observed_other_fee: Decimal
    other_fee_delta: Decimal
    expected_cash_change: Decimal
    observed_cash_change: Decimal
    cash_delta: Decimal
    expected_vwap: Decimal | None
    observed_vwap: Decimal | None
    report_hashes: tuple[str, ...]
    findings: tuple[ReconciliationFinding, ...]
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    check_hash: str

    def __post_init__(self) -> None:
        for field_name in ("check_id", "decision_id", "order_id", "instrument_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        _sha256(self.input_hash, "fill check input_hash")
        _sha256(self.batch_hash, "fill check batch_hash")
        _optional_sha256(self.line_hash, "fill check line_hash")
        for values, field_name in (
            (self.expected_fill_ids, "expected_fill_ids"),
            (self.observed_source_fill_ids, "observed_source_fill_ids"),
            (self.report_hashes, "report_hashes"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be unique and sorted")
        for digest in self.report_hashes:
            _sha256(digest, "fill check report_hash")
        for field_name in (
            "expected_quantity",
            "observed_quantity",
            "quantity_delta",
            "expected_gross",
            "observed_gross",
            "gross_delta",
            "expected_total_fee",
            "observed_total_fee",
            "fee_delta",
            "expected_commission",
            "observed_commission",
            "commission_delta",
            "expected_stamp_duty",
            "observed_stamp_duty",
            "stamp_duty_delta",
            "expected_transfer_fee",
            "observed_transfer_fee",
            "transfer_fee_delta",
            "expected_other_fee",
            "observed_other_fee",
            "other_fee_delta",
            "expected_cash_change",
            "observed_cash_change",
            "cash_delta",
        ):
            _exact_decimal(getattr(self, field_name), f"fill check {field_name}")
        if self.expected_quantity < 0 or self.observed_quantity < 0:
            raise ValueError("fill quantities cannot be negative")
        if not _difference_matches(
            expected=self.expected_quantity,
            observed=self.observed_quantity,
            delta=self.quantity_delta,
        ):
            raise ValueError("fill quantity_delta is inconsistent")
        if not _difference_matches(
            expected=self.expected_gross,
            observed=self.observed_gross,
            delta=self.gross_delta,
        ):
            raise ValueError("fill gross_delta is inconsistent")
        if not _difference_matches(
            expected=self.expected_total_fee,
            observed=self.observed_total_fee,
            delta=self.fee_delta,
        ):
            raise ValueError("fill fee_delta is inconsistent")
        component_triples = (
            (self.expected_commission, self.observed_commission, self.commission_delta),
            (self.expected_stamp_duty, self.observed_stamp_duty, self.stamp_duty_delta),
            (self.expected_transfer_fee, self.observed_transfer_fee, self.transfer_fee_delta),
            (self.expected_other_fee, self.observed_other_fee, self.other_fee_delta),
        )
        if any(
            not _difference_matches(expected=expected, observed=observed, delta=delta)
            for expected, observed, delta in component_triples
        ):
            raise ValueError("fill fee-component deltas are inconsistent")
        if not _difference_matches(
            expected=self.expected_cash_change,
            observed=self.observed_cash_change,
            delta=self.cash_delta,
        ):
            raise ValueError("fill cash_delta is inconsistent")
        for field_name in ("expected_vwap", "observed_vwap"):
            value = getattr(self, field_name)
            if value is not None:
                _exact_decimal(value, f"fill check {field_name}")
                if value <= 0:
                    raise ValueError("fill VWAP values must be positive")
        ordered = _ordered_findings(self.findings)
        if self.findings != ordered:
            raise ValueError("fill findings must use canonical ordering")
        if self.status is not _derived_check_status(self.findings):
            raise ValueError("fill check status must be derived from findings")
        if self.severity is not _derived_severity(self.findings):
            raise ValueError("fill check severity must be derived from findings")
        _sha256(self.check_hash, "fill check_hash")
        if self.check_hash != stable_hash(self._content_payload()):
            raise ValueError("fill check_hash does not match content")

    @classmethod
    def build(cls, **kwargs: Any) -> FillReconciliation:
        findings = _ordered_findings(tuple(kwargs.pop("findings", ())))
        for field_name in ("expected_fill_ids", "observed_source_fill_ids", "report_hashes"):
            kwargs[field_name] = tuple(sorted(set(kwargs.get(field_name, ()))))
        kwargs["findings"] = findings
        kwargs["status"] = _derived_check_status(findings)
        kwargs["severity"] = _derived_severity(findings)
        values = dict(kwargs)
        hash_values = {**values, "findings": tuple(asdict(item) for item in findings)}
        return cls(**values, check_hash=stable_hash(hash_values))

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "check_hash")


@dataclass(frozen=True, slots=True)
class CashReconciliation:
    """Exact expected-versus-observed cash bucket comparison."""

    check_id: str
    input_hash: str
    decision_id: str
    batch_hash: str
    account_id: str
    currency: str
    expected_total_cash: Decimal
    observed_total_cash: Decimal
    total_delta: Decimal
    expected_available_cash: Decimal
    observed_available_cash: Decimal
    available_delta: Decimal
    expected_frozen_cash: Decimal
    observed_frozen_cash: Decimal
    frozen_delta: Decimal
    mismatched_fields: tuple[str, ...]
    findings: tuple[ReconciliationFinding, ...]
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    check_hash: str

    def __post_init__(self) -> None:
        for field_name in ("check_id", "decision_id", "account_id", "currency"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        _sha256(self.input_hash, "cash check input_hash")
        _sha256(self.batch_hash, "cash check batch_hash")
        for field_name in (
            "expected_total_cash",
            "observed_total_cash",
            "total_delta",
            "expected_available_cash",
            "observed_available_cash",
            "available_delta",
            "expected_frozen_cash",
            "observed_frozen_cash",
            "frozen_delta",
        ):
            _exact_decimal(getattr(self, field_name), f"cash check {field_name}")
        if not _difference_matches(
            expected=self.expected_total_cash,
            observed=self.observed_total_cash,
            delta=self.total_delta,
        ):
            raise ValueError("cash total_delta is inconsistent")
        if not _difference_matches(
            expected=self.expected_available_cash,
            observed=self.observed_available_cash,
            delta=self.available_delta,
        ):
            raise ValueError("cash available_delta is inconsistent")
        if not _difference_matches(
            expected=self.expected_frozen_cash,
            observed=self.observed_frozen_cash,
            delta=self.frozen_delta,
        ):
            raise ValueError("cash frozen_delta is inconsistent")
        if self.mismatched_fields != tuple(sorted(set(self.mismatched_fields))):
            raise ValueError("cash mismatched_fields must be unique and sorted")
        expected_mismatched_fields = tuple(
            sorted(
                field_name
                for field_name, delta in (
                    ("available_cash", self.available_delta),
                    ("frozen_cash", self.frozen_delta),
                    ("total_cash", self.total_delta),
                )
                if delta != 0
            )
        )
        if self.mismatched_fields != expected_mismatched_fields:
            raise ValueError("cash mismatched_fields must exactly identify non-zero deltas")
        ordered = _ordered_findings(self.findings)
        if self.findings != ordered:
            raise ValueError("cash findings must use canonical ordering")
        if self.status is not _derived_check_status(self.findings):
            raise ValueError("cash check status must be derived from findings")
        if self.severity is not _derived_severity(self.findings):
            raise ValueError("cash check severity must be derived from findings")
        code_by_field = {
            "available_cash": ReconciliationDifferenceCode.CASH_AVAILABLE_MISMATCH,
            "frozen_cash": ReconciliationDifferenceCode.CASH_FROZEN_MISMATCH,
            "total_cash": ReconciliationDifferenceCode.CASH_TOTAL_MISMATCH,
        }
        expected_finding_codes = tuple(
            sorted(
                (code_by_field[item] for item in self.mismatched_fields),
                key=lambda item: item.value,
            )
        )
        actual_finding_codes = tuple(
            sorted((item.code for item in self.findings), key=lambda item: item.value)
        )
        if actual_finding_codes != expected_finding_codes:
            raise ValueError("cash findings must exactly cover every mismatched field")
        _sha256(self.check_hash, "cash check_hash")
        if self.check_hash != stable_hash(self._content_payload()):
            raise ValueError("cash check_hash does not match content")

    @classmethod
    def build(cls, **kwargs: Any) -> CashReconciliation:
        findings = _ordered_findings(tuple(kwargs.pop("findings", ())))
        kwargs["mismatched_fields"] = tuple(sorted(set(kwargs.get("mismatched_fields", ()))))
        kwargs["findings"] = findings
        kwargs["status"] = _derived_check_status(findings)
        kwargs["severity"] = _derived_severity(findings)
        values = dict(kwargs)
        hash_values = {**values, "findings": tuple(asdict(item) for item in findings)}
        return cls(**values, check_hash=stable_hash(hash_values))

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "check_hash")


@dataclass(frozen=True, slots=True)
class PositionReconciliation:
    """Quantity-bucket and cost comparison for one instrument union member."""

    check_id: str
    input_hash: str
    decision_id: str
    batch_hash: str
    instrument_id: str
    expected_instrument_type: TradableInstrumentType | None
    observed_instrument_type: TradableInstrumentType | None
    expected_total: Decimal | None
    observed_total: Decimal | None
    total_delta: Decimal | None
    expected_available: Decimal | None
    observed_available: Decimal | None
    available_delta: Decimal | None
    expected_frozen: Decimal | None
    observed_frozen: Decimal | None
    frozen_delta: Decimal | None
    expected_unsettled: Decimal | None
    observed_unsettled: Decimal | None
    unsettled_delta: Decimal | None
    expected_cost_basis: Decimal | None
    observed_cost_basis: Decimal | None
    cost_delta: Decimal | None
    expected_average_cost: Decimal | None
    observed_average_cost: Decimal | None
    average_cost_delta: Decimal | None
    related_order_ids: tuple[str, ...]
    related_fill_ids: tuple[str, ...]
    findings: tuple[ReconciliationFinding, ...]
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    check_hash: str

    def __post_init__(self) -> None:
        for field_name in ("check_id", "decision_id", "instrument_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        _sha256(self.input_hash, "position check input_hash")
        _sha256(self.batch_hash, "position check batch_hash")
        for field_name in (
            "expected_total",
            "observed_total",
            "total_delta",
            "expected_available",
            "observed_available",
            "available_delta",
            "expected_frozen",
            "observed_frozen",
            "frozen_delta",
            "expected_unsettled",
            "observed_unsettled",
            "unsettled_delta",
            "expected_cost_basis",
            "observed_cost_basis",
            "cost_delta",
            "expected_average_cost",
            "observed_average_cost",
            "average_cost_delta",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _exact_decimal(value, f"position check {field_name}")
        triples = (
            (self.expected_total, self.observed_total, self.total_delta),
            (self.expected_available, self.observed_available, self.available_delta),
            (self.expected_frozen, self.observed_frozen, self.frozen_delta),
            (self.expected_unsettled, self.observed_unsettled, self.unsettled_delta),
            (self.expected_cost_basis, self.observed_cost_basis, self.cost_delta),
            (self.expected_average_cost, self.observed_average_cost, self.average_cost_delta),
        )
        for expected, observed, delta in triples:
            if expected is not None and observed is not None:
                if delta is None or not _difference_matches(
                    expected=expected,
                    observed=observed,
                    delta=delta,
                ):
                    raise ValueError("position delta does not match expected and observed values")
            elif delta is not None:
                raise ValueError("position delta requires expected and observed values")
        for field_name in ("related_order_ids", "related_fill_ids"):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be unique and sorted")
        ordered = _ordered_findings(self.findings)
        if self.findings != ordered:
            raise ValueError("position findings must use canonical ordering")
        if self.status is not _derived_check_status(self.findings):
            raise ValueError("position check status must be derived from findings")
        if self.severity is not _derived_severity(self.findings):
            raise ValueError("position check severity must be derived from findings")
        _sha256(self.check_hash, "position check_hash")
        if self.check_hash != stable_hash(self._content_payload()):
            raise ValueError("position check_hash does not match content")

    @classmethod
    def build(cls, **kwargs: Any) -> PositionReconciliation:
        findings = _ordered_findings(tuple(kwargs.pop("findings", ())))
        for field_name in ("related_order_ids", "related_fill_ids"):
            kwargs[field_name] = tuple(sorted(set(kwargs.get(field_name, ()))))
        kwargs["findings"] = findings
        kwargs["status"] = _derived_check_status(findings)
        kwargs["severity"] = _derived_severity(findings)
        values = dict(kwargs)
        hash_values = {**values, "findings": tuple(asdict(item) for item in findings)}
        return cls(**values, check_hash=stable_hash(hash_values))

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "check_hash")


@dataclass(frozen=True, slots=True)
class DuplicateReportGroup:
    """Auditable duplicate classification without selecting conflicting content."""

    group_id: str
    report_kind: ReconciliationReportKind
    key_kind: DuplicateKeyKind
    source_system: str
    dedup_key: tuple[str, ...]
    occurrence_count: int
    report_ids: tuple[str, ...]
    report_hashes: tuple[str, ...]
    content_hashes: tuple[str, ...]
    canonical_report_hash: str | None
    classification: DuplicateClassification
    severity: ReconciliationSeverity
    group_hash: str

    def __post_init__(self) -> None:
        for field_name in ("group_id", "source_system"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if not isinstance(self.report_kind, ReconciliationReportKind):
            raise ValueError("duplicate report_kind is invalid")
        if not isinstance(self.key_kind, DuplicateKeyKind):
            raise ValueError("duplicate key_kind is invalid")
        if not self.dedup_key or any(not item.strip() for item in self.dedup_key):
            raise ValueError("duplicate dedup_key must contain non-empty values")
        if self.occurrence_count < 2:
            raise ValueError("a duplicate group requires at least two occurrences")
        if not (
            len(self.report_ids)
            == len(self.report_hashes)
            == len(self.content_hashes)
            == self.occurrence_count
        ):
            raise ValueError("duplicate occurrence evidence counts must align")
        if self.report_ids != tuple(sorted(self.report_ids)):
            raise ValueError("duplicate report_ids must be sorted while preserving occurrences")
        if self.report_hashes != tuple(sorted(self.report_hashes)):
            raise ValueError("duplicate report_hashes must be sorted while preserving occurrences")
        if self.content_hashes != tuple(sorted(self.content_hashes)):
            raise ValueError("duplicate content_hashes must be sorted while preserving occurrences")
        for digest in (*self.report_hashes, *self.content_hashes):
            _sha256(digest, "duplicate evidence hash")
        _optional_sha256(self.canonical_report_hash, "canonical_report_hash")
        unique_content = set(self.content_hashes)
        if self.classification is DuplicateClassification.IDENTICAL:
            if len(unique_content) != 1 or self.canonical_report_hash is None:
                raise ValueError("identical duplicates require one content and a canonical report")
            if self.severity is not ReconciliationSeverity.WARNING:
                raise ValueError("identical duplicates must retain WARNING severity")
        elif self.classification is DuplicateClassification.CONFLICTING:
            if len(unique_content) < 2 or self.canonical_report_hash is not None:
                raise ValueError("conflicting duplicates cannot select canonical content")
            if self.severity is not ReconciliationSeverity.CRITICAL:
                raise ValueError("conflicting duplicates must be CRITICAL")
        else:
            raise ValueError("duplicate classification is invalid")
        _sha256(self.group_hash, "duplicate group_hash")
        identity_hash = stable_hash(
            {
                "report_kind": self.report_kind,
                "key_kind": self.key_kind,
                "source_system": self.source_system,
                "key": self.dedup_key,
            }
        )
        if self.group_id != f"duplicate:{identity_hash}":
            raise ValueError("duplicate group_id does not match its identity")
        if self.group_hash != stable_hash(self._content_payload()):
            raise ValueError("duplicate group_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        report_kind: ReconciliationReportKind,
        key_kind: DuplicateKeyKind,
        source_system: str,
        dedup_key: tuple[str, ...],
        report_ids: tuple[str, ...],
        report_hashes: tuple[str, ...],
        content_hashes: tuple[str, ...],
    ) -> DuplicateReportGroup:
        ordered_ids = tuple(sorted(report_ids))
        ordered_reports = tuple(sorted(report_hashes))
        ordered_content = tuple(sorted(content_hashes))
        classification = (
            DuplicateClassification.IDENTICAL
            if len(set(ordered_content)) == 1
            else DuplicateClassification.CONFLICTING
        )
        canonical = (
            min(ordered_reports) if classification is DuplicateClassification.IDENTICAL else None
        )
        severity = (
            ReconciliationSeverity.WARNING
            if classification is DuplicateClassification.IDENTICAL
            else ReconciliationSeverity.CRITICAL
        )
        group_identity = {
            "report_kind": report_kind,
            "key_kind": key_kind,
            "source_system": source_system,
            "key": dedup_key,
        }
        group_id = f"duplicate:{stable_hash(group_identity)}"
        values: dict[str, object] = {
            "group_id": group_id,
            "report_kind": report_kind,
            "key_kind": key_kind,
            "source_system": source_system,
            "dedup_key": dedup_key,
            "occurrence_count": len(ordered_ids),
            "report_ids": ordered_ids,
            "report_hashes": ordered_reports,
            "content_hashes": ordered_content,
            "canonical_report_hash": canonical,
            "classification": classification,
            "severity": severity,
        }
        return cls(**values, group_hash=stable_hash(values))  # type: ignore[arg-type]

    @property
    def difference_code(self) -> ReconciliationDifferenceCode:
        return (
            ReconciliationDifferenceCode.IDENTICAL_DUPLICATE_REPORT
            if self.classification is DuplicateClassification.IDENTICAL
            else ReconciliationDifferenceCode.CONFLICTING_DUPLICATE_REPORT
        )

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "group_hash")


@dataclass(frozen=True, slots=True)
class ReconciliationStopSignal:
    """Immutable request for P5-T08; it has no activation or recovery authority."""

    account_id: str
    engine_version: str
    policy_hash: str
    reconciliation_input_hash: str
    decision_id: str
    batch_hash: str
    required: bool
    action: ReconciliationStopAction
    scope: ReconciliationStopScope
    severity: ReconciliationSeverity
    reason_codes: tuple[ReconciliationDifferenceCode, ...]
    trigger_hashes: tuple[str, ...]
    generated_at: datetime
    signal_hash: str

    def __post_init__(self) -> None:
        for field_name in ("account_id", "engine_version", "decision_id"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        if self.engine_version != RECONCILIATION_ENGINE_VERSION:
            raise ValueError("stop signal engine_version is unknown")
        _sha256(self.policy_hash, "stop policy_hash")
        _sha256(self.reconciliation_input_hash, "stop reconciliation_input_hash")
        _sha256(self.batch_hash, "stop batch_hash")
        ensure_aware(self.generated_at)
        if self.scope is not ReconciliationStopScope.ACCOUNT:
            raise ValueError("single-account reconciliation cannot emit a GLOBAL stop")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("stop reason_codes must be unique and sorted")
        if self.trigger_hashes != tuple(sorted(set(self.trigger_hashes))):
            raise ValueError("stop trigger_hashes must be unique and sorted")
        for digest in self.trigger_hashes:
            _sha256(digest, "stop trigger_hash")
        if self.required:
            if self.action is not ReconciliationStopAction.STOP_NEW_ORDERS:
                raise ValueError("required reconciliation stop must stop new orders")
            if not self.reason_codes or not self.trigger_hashes:
                raise ValueError("required reconciliation stop needs reasons and triggers")
        elif (
            self.action is not ReconciliationStopAction.NONE
            or self.reason_codes
            or self.trigger_hashes
            or self.severity is not ReconciliationSeverity.INFO
        ):
            raise ValueError("non-required stop signal must be an empty INFO/NONE signal")
        _sha256(self.signal_hash, "stop signal_hash")
        if self.signal_hash != stable_hash(self._content_payload()):
            raise ValueError("stop signal_hash does not match content")

    @classmethod
    def build(
        cls,
        *,
        account_id: str,
        policy_hash: str,
        reconciliation_input_hash: str,
        decision_id: str,
        batch_hash: str,
        required: bool,
        severity: ReconciliationSeverity,
        reason_codes: tuple[ReconciliationDifferenceCode, ...],
        trigger_hashes: tuple[str, ...],
        generated_at: datetime,
    ) -> ReconciliationStopSignal:
        ordered_codes = tuple(sorted(set(reason_codes), key=lambda item: item.value))
        ordered_triggers = tuple(sorted(set(trigger_hashes)))
        values: dict[str, object] = {
            "account_id": account_id,
            "engine_version": RECONCILIATION_ENGINE_VERSION,
            "policy_hash": policy_hash,
            "reconciliation_input_hash": reconciliation_input_hash,
            "decision_id": decision_id,
            "batch_hash": batch_hash,
            "required": required,
            "action": (
                ReconciliationStopAction.STOP_NEW_ORDERS
                if required
                else ReconciliationStopAction.NONE
            ),
            "scope": ReconciliationStopScope.ACCOUNT,
            "severity": severity if required else ReconciliationSeverity.INFO,
            "reason_codes": ordered_codes if required else (),
            "trigger_hashes": ordered_triggers if required else (),
            "generated_at": generated_at,
        }
        return cls(**values, signal_hash=stable_hash(values))  # type: ignore[arg-type]

    @property
    def is_executable(self) -> bool:
        return False

    def _content_payload(self) -> dict[str, Any]:
        return _content_payload(self, "signal_hash")


_UNRECONCILABLE_CODES = {
    ReconciliationDifferenceCode.DRAFT_RECEIPT_BATCH_MISMATCH,
    ReconciliationDifferenceCode.DECISION_MISMATCH,
    ReconciliationDifferenceCode.ACCOUNT_IDENTITY_MISMATCH,
    ReconciliationDifferenceCode.SOURCE_SYSTEM_MISMATCH,
    ReconciliationDifferenceCode.RUNTIME_MODE_MISMATCH,
    ReconciliationDifferenceCode.DATA_VERSION_MISMATCH,
    ReconciliationDifferenceCode.CURRENCY_MISMATCH,
    ReconciliationDifferenceCode.SOURCE_SNAPSHOT_MISMATCH,
    ReconciliationDifferenceCode.SNAPSHOT_BOUNDARY_MISMATCH,
    ReconciliationDifferenceCode.EVENT_LOG_MISMATCH,
    ReconciliationDifferenceCode.STALE_EVIDENCE,
    ReconciliationDifferenceCode.FUTURE_EVIDENCE,
    ReconciliationDifferenceCode.UNLINKED_ORDER_REPORT,
    ReconciliationDifferenceCode.UNLINKED_FILL_REPORT,
    ReconciliationDifferenceCode.ORDER_REPORT_REGRESSION,
    ReconciliationDifferenceCode.CONFLICTING_DUPLICATE_REPORT,
}


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Complete, deterministic read-only result and downstream stop request."""

    schema_version: str
    reconciliation_id: str
    engine_version: str
    request_id: str
    input_hash: str
    account_id: str
    currency: str
    runtime_mode: RuntimeMode
    decision_id: str
    batch_hash: str
    draft_hash: str
    receipt_id: str
    receipt_hash: str
    expected_account_before_hash: str
    expected_account_after_hash: str
    observed_snapshot_id: str
    observed_snapshot_hash: str
    observed_as_of: datetime
    reconciled_at: datetime
    policy_version: str
    policy_hash: str
    stop_on_severity: ReconciliationSeverity
    status: ReconciliationStatus
    max_severity: ReconciliationSeverity
    input_findings: tuple[ReconciliationFinding, ...]
    order_checks: tuple[OrderReconciliation, ...]
    fill_checks: tuple[FillReconciliation, ...]
    cash_check: CashReconciliation
    position_checks: tuple[PositionReconciliation, ...]
    duplicate_groups: tuple[DuplicateReportGroup, ...]
    difference_hashes: tuple[str, ...]
    stop_signal: ReconciliationStopSignal
    previous_result_hash: str | None
    result_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reconciliation result schema_version must be '1'")
        if self.engine_version != RECONCILIATION_ENGINE_VERSION:
            raise ValueError("unknown reconciliation engine version")
        for field_name in (
            "reconciliation_id",
            "request_id",
            "account_id",
            "currency",
            "decision_id",
            "receipt_id",
            "observed_snapshot_id",
            "policy_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.observed_as_of)
        ensure_aware(self.reconciled_at)
        for value, field_name in (
            (self.input_hash, "result input_hash"),
            (self.batch_hash, "result batch_hash"),
            (self.draft_hash, "result draft_hash"),
            (self.receipt_hash, "result receipt_hash"),
            (self.expected_account_before_hash, "expected_account_before_hash"),
            (self.expected_account_after_hash, "expected_account_after_hash"),
            (self.observed_snapshot_hash, "observed_snapshot_hash"),
            (self.policy_hash, "result policy_hash"),
            (self.result_hash, "result_hash"),
        ):
            _sha256(value, field_name)
        _optional_sha256(self.previous_result_hash, "result previous_result_hash")
        if not isinstance(self.runtime_mode, RuntimeMode):
            raise ValueError("result runtime_mode is invalid")
        ordered_inputs = _ordered_findings(self.input_findings)
        if self.input_findings != ordered_inputs:
            raise ValueError("input findings must use canonical ordering")
        if self.order_checks != tuple(sorted(self.order_checks, key=lambda item: item.check_id)):
            raise ValueError("order checks must be sorted by check_id")
        if self.fill_checks != tuple(sorted(self.fill_checks, key=lambda item: item.check_id)):
            raise ValueError("fill checks must be sorted by check_id")
        if self.position_checks != tuple(
            sorted(self.position_checks, key=lambda item: item.instrument_id)
        ):
            raise ValueError("position checks must be sorted by instrument_id")
        if self.duplicate_groups != tuple(
            sorted(self.duplicate_groups, key=lambda item: item.group_id)
        ):
            raise ValueError("duplicate groups must be sorted by group_id")
        identity_groups = (
            (tuple(item.check_id for item in self.order_checks), "order check IDs"),
            (tuple(item.check_id for item in self.fill_checks), "fill check IDs"),
            (
                tuple(item.instrument_id for item in self.position_checks),
                "position check instruments",
            ),
            (tuple(item.group_id for item in self.duplicate_groups), "duplicate group IDs"),
        )
        for identities, field_name in identity_groups:
            if len(set(identities)) != len(identities):
                raise ValueError(f"result {field_name} must be unique")
        checks_are_unbound = (
            any(
                item.input_hash != self.input_hash
                or item.decision_id != self.decision_id
                or item.batch_hash != self.batch_hash
                for item in self.order_checks
            )
            or any(
                item.input_hash != self.input_hash
                or item.decision_id != self.decision_id
                or item.batch_hash != self.batch_hash
                for item in self.fill_checks
            )
            or any(
                item.input_hash != self.input_hash
                or item.decision_id != self.decision_id
                or item.batch_hash != self.batch_hash
                for item in self.position_checks
            )
        )
        if checks_are_unbound:
            raise ValueError("result checks must bind its input, decision, and batch")
        if (
            self.cash_check.input_hash != self.input_hash
            or self.cash_check.decision_id != self.decision_id
            or self.cash_check.batch_hash != self.batch_hash
            or self.cash_check.account_id != self.account_id
            or self.cash_check.currency != self.currency
            or self.cash_check.check_id != f"cash:{self.account_id}"
        ):
            raise ValueError("cash check must bind the result input, account, and currency")
        if any(item.check_id != f"position:{item.instrument_id}" for item in self.position_checks):
            raise ValueError("position check IDs must bind their instruments")
        if any(
            item.decision_id != self.decision_id or item.batch_hash != self.batch_hash
            for item in self.all_findings
        ):
            raise ValueError("result findings must bind its decision and batch")
        if any(
            item.domain not in {ReconciliationDomain.INPUT, ReconciliationDomain.REPORT}
            for item in self.input_findings
        ):
            raise ValueError("input findings must use INPUT or REPORT domain")
        for order_check in self.order_checks:
            if any(
                finding.domain is not ReconciliationDomain.ORDER
                or (finding.order_id is not None and finding.order_id != order_check.order_id)
                or (finding.line_hash is not None and finding.line_hash != order_check.line_hash)
                or (
                    finding.instrument_id is not None
                    and finding.instrument_id != order_check.instrument_id
                )
                for finding in order_check.findings
            ):
                raise ValueError("order findings must bind their containing check")
        for fill_check in self.fill_checks:
            if any(
                finding.domain is not ReconciliationDomain.FILL
                or finding.order_id != fill_check.order_id
                or (finding.line_hash is not None and finding.line_hash != fill_check.line_hash)
                or (
                    finding.instrument_id is not None
                    and finding.instrument_id != fill_check.instrument_id
                )
                or (
                    finding.fill_id is not None
                    and finding.fill_id not in fill_check.expected_fill_ids
                )
                for finding in fill_check.findings
            ):
                raise ValueError("fill findings must bind their containing check")
        if any(
            finding.domain is not ReconciliationDomain.CASH for finding in self.cash_check.findings
        ):
            raise ValueError("cash findings must use CASH domain")
        for position_check in self.position_checks:
            if any(
                finding.domain is not ReconciliationDomain.POSITION
                or finding.instrument_id != position_check.instrument_id
                for finding in position_check.findings
            ):
                raise ValueError("position findings must bind their containing check")
        findings = self.all_findings
        if len({item.finding_id for item in findings}) != len(findings) or len(
            {item.finding_hash for item in findings}
        ) != len(findings):
            raise ValueError("result findings must have unique IDs and hashes")
        severities = [item.severity for item in findings]
        severities.extend(item.severity for item in self.duplicate_groups)
        expected_max = (
            max(severities, key=severity_rank) if severities else ReconciliationSeverity.INFO
        )
        if self.max_severity is not expected_max:
            raise ValueError("result max_severity must be derived from all differences")
        expected_difference_hashes = tuple(
            sorted(
                {
                    *(item.finding_hash for item in findings),
                    *(item.group_hash for item in self.duplicate_groups),
                }
            )
        )
        if self.difference_hashes != expected_difference_hashes:
            raise ValueError("result difference_hashes must cover every difference")
        codes = {item.code for item in findings}
        codes.update(item.difference_code for item in self.duplicate_groups)
        if any(item.status is ReconciliationCheckStatus.CONFLICT for item in findings) or (
            codes & _UNRECONCILABLE_CODES
        ):
            expected_status = ReconciliationStatus.UNRECONCILABLE
        elif severity_rank(expected_max) >= severity_rank(ReconciliationSeverity.ERROR):
            expected_status = ReconciliationStatus.MISMATCH
        elif self.difference_hashes:
            expected_status = ReconciliationStatus.RECONCILED_WITH_VARIANCE
        else:
            expected_status = ReconciliationStatus.MATCHED
        if self.status is not expected_status:
            raise ValueError("result status must be derived from its differences")
        required = severity_rank(expected_max) >= severity_rank(self.stop_on_severity)
        if self.stop_signal.required != required:
            raise ValueError("result stop signal must follow policy severity threshold")
        if (
            self.stop_signal.account_id != self.account_id
            or self.stop_signal.engine_version != self.engine_version
            or self.stop_signal.policy_hash != self.policy_hash
            or self.stop_signal.reconciliation_input_hash != self.input_hash
            or self.stop_signal.decision_id != self.decision_id
            or self.stop_signal.batch_hash != self.batch_hash
            or self.stop_signal.generated_at != self.reconciled_at
        ):
            raise ValueError("stop signal does not bind the reconciliation result inputs")
        expected_signal_severity = self.max_severity if required else ReconciliationSeverity.INFO
        if self.stop_signal.severity is not expected_signal_severity:
            raise ValueError("stop signal severity does not match the reconciliation result")
        if required:
            expected_triggers = tuple(
                sorted(
                    item.finding_hash
                    for item in findings
                    if severity_rank(item.severity) >= severity_rank(self.stop_on_severity)
                )
            )
            duplicate_triggers = tuple(
                sorted(
                    item.group_hash
                    for item in self.duplicate_groups
                    if severity_rank(item.severity) >= severity_rank(self.stop_on_severity)
                )
            )
            if self.stop_signal.trigger_hashes != tuple(
                sorted({*expected_triggers, *duplicate_triggers})
            ):
                raise ValueError("stop signal triggers do not cover severe differences")
            expected_reason_codes = tuple(
                sorted(
                    {
                        *(
                            item.code
                            for item in findings
                            if severity_rank(item.severity) >= severity_rank(self.stop_on_severity)
                        ),
                        *(
                            item.difference_code
                            for item in self.duplicate_groups
                            if severity_rank(item.severity) >= severity_rank(self.stop_on_severity)
                        ),
                    },
                    key=lambda item: item.value,
                )
            )
            if self.stop_signal.reason_codes != expected_reason_codes:
                raise ValueError("stop signal reasons do not cover severe differences")
        expected_identity = {
            "request_id": self.request_id,
            "input_hash": self.input_hash,
        }
        expected_id = f"reconciliation:{stable_hash(expected_identity)}"
        if self.reconciliation_id != expected_id:
            raise ValueError("reconciliation_id does not match request identity")
        if self.result_hash != stable_hash(self._content_payload()):
            raise ValueError("result_hash does not match reconciliation content")

    @property
    def all_findings(self) -> tuple[ReconciliationFinding, ...]:
        return (
            *self.input_findings,
            *(finding for item in self.order_checks for finding in item.findings),
            *(finding for item in self.fill_checks for finding in item.findings),
            *self.cash_check.findings,
            *(finding for item in self.position_checks for finding in item.findings),
        )

    @property
    def is_executable(self) -> bool:
        return False

    @staticmethod
    def _validate_required_children(
        *,
        request: ReconciliationRequest,
        order_checks: tuple[OrderReconciliation, ...],
        fill_checks: tuple[FillReconciliation, ...],
        cash_check: CashReconciliation,
        position_checks: tuple[PositionReconciliation, ...],
    ) -> None:
        """Prove that callers cannot omit a required comparison and report MATCHED."""

        order_checks_by_id = {item.check_id: item for item in order_checks}
        if len(order_checks_by_id) != len(order_checks):
            raise ValueError("result order check IDs must be unique")
        receipt_order_check_ids = {f"order:{item.order_id}" for item in request.receipt.orders}
        if not receipt_order_check_ids.issubset(order_checks_by_id):
            raise ValueError("every receipt order requires an order reconciliation check")
        represented_lines = {item.line_hash for item in request.receipt.orders}
        missing_line_check_ids = {
            f"draft-line:{item.line_hash}"
            for item in request.draft.lines
            if item.line_hash not in represented_lines
        }
        if not missing_line_check_ids.issubset(order_checks_by_id):
            raise ValueError("every unrepresented draft line requires an order check")
        for order in request.receipt.orders:
            order_check = order_checks_by_id[f"order:{order.order_id}"]
            if (
                order_check.order_id != order.order_id
                or order_check.line_hash != order.line_hash
                or order_check.instrument_id != order.instrument_id
                or order_check.expected_status is not order.status
                or order_check.expected_filled_quantity != order.filled_quantity
                or order_check.expected_remaining_quantity != order.remaining_quantity
            ):
                raise ValueError("receipt order check does not bind its expected order")

        fill_checks_by_order = {item.order_id: item for item in fill_checks}
        if len(fill_checks_by_order) != len(fill_checks):
            raise ValueError("result fill checks must be unique by order")
        receipt_order_ids = {item.order_id for item in request.receipt.orders}
        if not receipt_order_ids.issubset(fill_checks_by_order):
            raise ValueError("every receipt order requires a fill reconciliation check")
        for order in request.receipt.orders:
            fill_check = fill_checks_by_order[order.order_id]
            expected_fills = tuple(
                item for item in request.receipt.fills if item.order_id == order.order_id
            )
            if (
                fill_check.check_id != f"fill-order:{order.order_id}"
                or fill_check.line_hash != order.line_hash
                or fill_check.instrument_id != order.instrument_id
                or fill_check.expected_fill_ids
                != tuple(sorted(item.fill_id for item in expected_fills))
            ):
                raise ValueError("receipt order fill check does not bind its expected fills")

        expected = request.receipt.account_after
        observed = request.observed.account_snapshot.cash
        if (
            cash_check.input_hash != request.input_hash
            or cash_check.decision_id != request.draft.decision_id
            or cash_check.batch_hash != request.draft.batch_hash
        ):
            raise ValueError("cash check must bind the reconciliation request input")
        if (
            cash_check.check_id != f"cash:{expected.account_id}"
            or cash_check.account_id != expected.account_id
        ):
            raise ValueError("cash check identity does not match the expected account")
        if cash_check.currency != expected.currency:
            raise ValueError("cash check currency does not match the expected account")
        if (
            cash_check.expected_total_cash != expected.total_cash
            or cash_check.expected_available_cash != expected.available_cash
            or cash_check.expected_frozen_cash != expected.frozen_cash
            or cash_check.observed_total_cash != observed.total_cash
            or cash_check.observed_available_cash != observed.available_cash
            or cash_check.observed_frozen_cash != observed.frozen_cash
        ):
            raise ValueError("cash check values do not cover the reconciliation request")

        expected_position_ids = {
            *(item.instrument_id for item in request.receipt.account_after.positions),
            *(item.instrument_id for item in request.observed.account_snapshot.positions),
        }
        actual_position_ids = {item.instrument_id for item in position_checks}
        if actual_position_ids != expected_position_ids:
            raise ValueError("position checks must cover the expected-observed instrument union")

    @classmethod
    def build(
        cls,
        *,
        request: ReconciliationRequest,
        input_findings: tuple[ReconciliationFinding, ...],
        order_checks: tuple[OrderReconciliation, ...],
        fill_checks: tuple[FillReconciliation, ...],
        cash_check: CashReconciliation,
        position_checks: tuple[PositionReconciliation, ...],
        duplicate_groups: tuple[DuplicateReportGroup, ...],
    ) -> ReconciliationResult:
        """Verify supplied components against a fresh authoritative engine evaluation."""

        cls._validate_required_children(
            request=request,
            order_checks=order_checks,
            fill_checks=fill_checks,
            cash_check=cash_check,
            position_checks=position_checks,
        )
        from .engine import ReconciliationEngine

        authoritative = ReconciliationEngine().reconcile(request)
        supplied = (
            input_findings,
            order_checks,
            fill_checks,
            cash_check,
            position_checks,
            duplicate_groups,
        )
        expected = (
            authoritative.input_findings,
            authoritative.order_checks,
            authoritative.fill_checks,
            authoritative.cash_check,
            authoritative.position_checks,
            authoritative.duplicate_groups,
        )
        if supplied != expected:
            raise ValueError(
                "supplied reconciliation components do not exactly match "
                "the authoritative engine output"
            )
        return authoritative

    @classmethod
    def _build_from_engine(
        cls,
        *,
        request: ReconciliationRequest,
        input_findings: tuple[ReconciliationFinding, ...],
        order_checks: tuple[OrderReconciliation, ...],
        fill_checks: tuple[FillReconciliation, ...],
        cash_check: CashReconciliation,
        position_checks: tuple[PositionReconciliation, ...],
        duplicate_groups: tuple[DuplicateReportGroup, ...],
    ) -> ReconciliationResult:
        """Assemble components already computed by the package's pure engine."""

        cls._validate_required_children(
            request=request,
            order_checks=order_checks,
            fill_checks=fill_checks,
            cash_check=cash_check,
            position_checks=position_checks,
        )
        ordered_inputs = _ordered_findings(input_findings)
        ordered_orders = tuple(sorted(order_checks, key=lambda item: item.check_id))
        ordered_fills = tuple(sorted(fill_checks, key=lambda item: item.check_id))
        ordered_positions = tuple(sorted(position_checks, key=lambda item: item.instrument_id))
        ordered_duplicates = tuple(sorted(duplicate_groups, key=lambda item: item.group_id))
        findings = (
            *ordered_inputs,
            *(finding for item in ordered_orders for finding in item.findings),
            *(finding for item in ordered_fills for finding in item.findings),
            *cash_check.findings,
            *(finding for item in ordered_positions for finding in item.findings),
        )
        severities = [item.severity for item in findings]
        severities.extend(item.severity for item in ordered_duplicates)
        max_severity = (
            max(severities, key=severity_rank) if severities else ReconciliationSeverity.INFO
        )
        codes = {item.code for item in findings}
        codes.update(item.difference_code for item in ordered_duplicates)
        difference_hashes = tuple(
            sorted(
                {
                    *(item.finding_hash for item in findings),
                    *(item.group_hash for item in ordered_duplicates),
                }
            )
        )
        if any(item.status is ReconciliationCheckStatus.CONFLICT for item in findings) or (
            codes & _UNRECONCILABLE_CODES
        ):
            status = ReconciliationStatus.UNRECONCILABLE
        elif severity_rank(max_severity) >= severity_rank(ReconciliationSeverity.ERROR):
            status = ReconciliationStatus.MISMATCH
        elif difference_hashes:
            status = ReconciliationStatus.RECONCILED_WITH_VARIANCE
        else:
            status = ReconciliationStatus.MATCHED
        required = severity_rank(max_severity) >= severity_rank(request.policy.stop_on_severity)
        trigger_findings = tuple(
            item
            for item in findings
            if severity_rank(item.severity) >= severity_rank(request.policy.stop_on_severity)
        )
        trigger_groups = tuple(
            item
            for item in ordered_duplicates
            if severity_rank(item.severity) >= severity_rank(request.policy.stop_on_severity)
        )
        reason_codes = tuple(
            sorted(
                {
                    *(item.code for item in trigger_findings),
                    *(item.difference_code for item in trigger_groups),
                },
                key=lambda item: item.value,
            )
        )
        trigger_hashes = tuple(
            sorted(
                {
                    *(item.finding_hash for item in trigger_findings),
                    *(item.group_hash for item in trigger_groups),
                }
            )
        )
        account_id = request.receipt.account_after.account_id
        decision_id = request.draft.decision_id
        batch_hash = request.draft.batch_hash
        stop_signal = ReconciliationStopSignal.build(
            account_id=account_id,
            policy_hash=request.policy.policy_hash,
            reconciliation_input_hash=request.input_hash,
            decision_id=decision_id,
            batch_hash=batch_hash,
            required=required,
            severity=max_severity,
            reason_codes=reason_codes,
            trigger_hashes=trigger_hashes,
            generated_at=request.reconciled_at,
        )
        reconciliation_identity = {
            "request_id": request.request_id,
            "input_hash": request.input_hash,
        }
        reconciliation_id = f"reconciliation:{stable_hash(reconciliation_identity)}"
        snapshot = request.observed.account_snapshot
        values: dict[str, object] = {
            "schema_version": "1",
            "reconciliation_id": reconciliation_id,
            "engine_version": RECONCILIATION_ENGINE_VERSION,
            "request_id": request.request_id,
            "input_hash": request.input_hash,
            "account_id": account_id,
            "currency": request.receipt.account_after.currency,
            "runtime_mode": request.receipt.account_after.runtime_mode,
            "decision_id": decision_id,
            "batch_hash": batch_hash,
            "draft_hash": request.draft.batch_hash,
            "receipt_id": request.receipt.receipt_id,
            "receipt_hash": request.receipt.receipt_hash,
            "expected_account_before_hash": request.receipt.account_before_hash,
            "expected_account_after_hash": request.receipt.account_after_hash,
            "observed_snapshot_id": snapshot.snapshot_id,
            "observed_snapshot_hash": snapshot.content_hash,
            "observed_as_of": snapshot.as_of,
            "reconciled_at": request.reconciled_at,
            "policy_version": request.policy.version,
            "policy_hash": request.policy.policy_hash,
            "stop_on_severity": request.policy.stop_on_severity,
            "status": status,
            "max_severity": max_severity,
            "input_findings": [asdict(item) for item in ordered_inputs],
            "order_checks": [asdict(item) for item in ordered_orders],
            "fill_checks": [asdict(item) for item in ordered_fills],
            "cash_check": asdict(cash_check),
            "position_checks": [asdict(item) for item in ordered_positions],
            "duplicate_groups": [asdict(item) for item in ordered_duplicates],
            "difference_hashes": difference_hashes,
            "stop_signal": asdict(stop_signal),
            "previous_result_hash": request.previous_result_hash,
        }
        return cls(
            schema_version="1",
            reconciliation_id=reconciliation_id,
            engine_version=RECONCILIATION_ENGINE_VERSION,
            request_id=request.request_id,
            input_hash=request.input_hash,
            account_id=account_id,
            currency=request.receipt.account_after.currency,
            runtime_mode=request.receipt.account_after.runtime_mode,
            decision_id=decision_id,
            batch_hash=batch_hash,
            draft_hash=request.draft.batch_hash,
            receipt_id=request.receipt.receipt_id,
            receipt_hash=request.receipt.receipt_hash,
            expected_account_before_hash=request.receipt.account_before_hash,
            expected_account_after_hash=request.receipt.account_after_hash,
            observed_snapshot_id=snapshot.snapshot_id,
            observed_snapshot_hash=snapshot.content_hash,
            observed_as_of=snapshot.as_of,
            reconciled_at=request.reconciled_at,
            policy_version=request.policy.version,
            policy_hash=request.policy.policy_hash,
            stop_on_severity=request.policy.stop_on_severity,
            status=status,
            max_severity=max_severity,
            input_findings=ordered_inputs,
            order_checks=ordered_orders,
            fill_checks=ordered_fills,
            cash_check=cash_check,
            position_checks=ordered_positions,
            duplicate_groups=ordered_duplicates,
            difference_hashes=difference_hashes,
            stop_signal=stop_signal,
            previous_result_hash=request.previous_result_hash,
            result_hash=stable_hash(values),
        )

    def _content_payload(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in _content_payload(self, "result_hash").items()
                if key
                not in {
                    "input_findings",
                    "order_checks",
                    "fill_checks",
                    "cash_check",
                    "position_checks",
                    "duplicate_groups",
                    "stop_signal",
                }
            },
            "input_findings": [asdict(item) for item in self.input_findings],
            "order_checks": [asdict(item) for item in self.order_checks],
            "fill_checks": [asdict(item) for item in self.fill_checks],
            "cash_check": asdict(self.cash_check),
            "position_checks": [asdict(item) for item in self.position_checks],
            "duplicate_groups": [asdict(item) for item in self.duplicate_groups],
            "stop_signal": asdict(self.stop_signal),
        }


__all__ = [
    "RECONCILIATION_ENGINE_VERSION",
    "CashReconciliation",
    "DuplicateClassification",
    "DuplicateReportGroup",
    "FillReconciliation",
    "ObservedExecutionEvidence",
    "ObservedFillReport",
    "ObservedOrderReport",
    "OrderReconciliation",
    "PositionReconciliation",
    "ReconciliationCheckStatus",
    "ReconciliationDifferenceCode",
    "ReconciliationDomain",
    "ReconciliationFinding",
    "ReconciliationPolicy",
    "ReconciliationReportKind",
    "ReconciliationRequest",
    "ReconciliationResult",
    "ReconciliationSeverity",
    "ReconciliationStatus",
    "ReconciliationStopAction",
    "ReconciliationStopScope",
    "ReconciliationStopSignal",
    "severity_rank",
]
