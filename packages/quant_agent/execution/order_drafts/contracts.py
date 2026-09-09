"""Immutable, non-executable order-draft contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import RiskCheckStatus

ORDER_DRAFT_GENERATOR_VERSION = "order-draft-generator-v1"


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


def _sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


class OrderDraftInputError(ValueError):
    """Raised when an immutable draft cannot be built safely from bound inputs."""


class DraftFundingPolicy(StrEnum):
    """Cash source permitted when reserving a draft's estimated buy requirement."""

    SNAPSHOT_AVAILABLE_CASH_ONLY = "SNAPSHOT_AVAILABLE_CASH_ONLY"


@dataclass(frozen=True, slots=True)
class OrderDraftGeneratorConfig:
    """Versioned generator behavior that deliberately excludes execution authority."""

    version: str = "order-draft-config-v1"
    validity_seconds: int = 900
    funding_policy: DraftFundingPolicy = DraftFundingPolicy.SNAPSHOT_AVAILABLE_CASH_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty(self.version, "draft config version"))
        if (
            not isinstance(self.validity_seconds, int)
            or isinstance(self.validity_seconds, bool)
            or self.validity_seconds <= 0
            or self.validity_seconds > 86_400
        ):
            raise ValueError("draft validity_seconds must be within 1..86400")
        if not isinstance(self.funding_policy, DraftFundingPolicy):
            raise ValueError("funding_policy must be a DraftFundingPolicy value")

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class OrderDraftLine:
    """One sized buy or sell estimate; it is not a broker order."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    side: Side
    current_quantity: Decimal
    target_quantity: Decimal
    quantity: Decimal
    is_full_liquidation: bool
    reference_price: Decimal
    estimated_execution_price: Decimal
    cash_reservation_price: Decimal
    price_observed_at: datetime
    price_available_at: datetime
    data_version: str
    state_revision: str
    state_hash: str
    participation_rate: Decimal
    gross_amount: Decimal
    estimated_slippage_amount: Decimal
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    other_fee: Decimal
    total_fee: Decimal
    cash_reservation_fee: Decimal
    reserved_cash: Decimal
    estimated_cash_change: Decimal
    market_rule_version: str
    market_rule_hash: str
    fee_rule_version: str
    fee_rule_hash: str
    slippage_model_version: str
    slippage_model_hash: str
    rationale: str
    line_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _non_empty(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "data_version", _non_empty(self.data_version, "data_version"))
        object.__setattr__(
            self, "state_revision", _non_empty(self.state_revision, "state_revision")
        )
        object.__setattr__(self, "rationale", _non_empty(self.rationale, "draft rationale"))
        if not isinstance(self.instrument_type, TradableInstrumentType):
            raise ValueError("instrument_type must be a TradableInstrumentType value")
        if not isinstance(self.side, Side):
            raise ValueError("side must be a Side value")
        if not isinstance(self.is_full_liquidation, bool):
            raise ValueError("is_full_liquidation must be a bool")
        for field_name in (
            "market_rule_version",
            "fee_rule_version",
            "slippage_model_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        for digest, field_name in (
            (self.state_hash, "state_hash"),
            (self.market_rule_hash, "market_rule_hash"),
            (self.fee_rule_hash, "fee_rule_hash"),
            (self.slippage_model_hash, "slippage_model_hash"),
            (self.line_hash, "line_hash"),
        ):
            _sha256(digest, field_name)
        ensure_aware(self.price_observed_at)
        ensure_aware(self.price_available_at)
        if self.price_observed_at > self.price_available_at:
            raise ValueError("price_available_at cannot precede price_observed_at")
        values = (
            self.current_quantity,
            self.target_quantity,
            self.quantity,
            self.reference_price,
            self.estimated_execution_price,
            self.cash_reservation_price,
            self.participation_rate,
            self.gross_amount,
            self.estimated_slippage_amount,
            self.commission,
            self.stamp_duty,
            self.transfer_fee,
            self.other_fee,
            self.total_fee,
            self.cash_reservation_fee,
            self.reserved_cash,
            self.estimated_cash_change,
        )
        for numeric in values:
            _exact_decimal(numeric, "order draft numeric value")
        non_negative = (
            self.current_quantity,
            self.target_quantity,
            self.participation_rate,
            self.gross_amount,
            self.estimated_slippage_amount,
            self.commission,
            self.stamp_duty,
            self.transfer_fee,
            self.other_fee,
            self.total_fee,
            self.cash_reservation_fee,
            self.reserved_cash,
        )
        if min(non_negative) < 0:
            raise ValueError("order draft values other than cash change cannot be negative")
        if self.quantity <= 0 or self.quantity != self.quantity.to_integral_value():
            raise ValueError("draft quantity must be a positive whole number")
        if (
            self.reference_price <= 0
            or self.estimated_execution_price <= 0
            or self.cash_reservation_price <= 0
        ):
            raise ValueError("draft prices must be positive")
        if self.participation_rate <= 0 or self.participation_rate > 1:
            raise ValueError("draft participation rate must be within (0, 1]")
        expected_quantity = (
            self.target_quantity - self.current_quantity
            if self.side is Side.BUY
            else self.current_quantity - self.target_quantity
        )
        if self.quantity != expected_quantity:
            raise ValueError("draft side and quantity must equal the target quantity difference")
        expected_full_liquidation = (
            self.side is Side.SELL
            and self.target_quantity == 0
            and self.quantity == self.current_quantity
        )
        if self.is_full_liquidation != expected_full_liquidation:
            raise ValueError("full-liquidation flag must match the quantity transition")
        if self.side is Side.BUY and self.estimated_execution_price < self.reference_price:
            raise ValueError("BUY slippage estimate cannot improve the reference price")
        if self.side is Side.SELL and self.estimated_execution_price > self.reference_price:
            raise ValueError("SELL slippage estimate cannot improve the reference price")
        if self.gross_amount != self.quantity * self.estimated_execution_price:
            raise ValueError("gross_amount must equal quantity times estimated execution price")
        if self.estimated_slippage_amount != self.quantity * abs(
            self.estimated_execution_price - self.reference_price
        ):
            raise ValueError("estimated slippage amount does not match prices and quantity")
        expected_fee = self.commission + self.stamp_duty + self.transfer_fee + self.other_fee
        if self.total_fee != expected_fee:
            raise ValueError("total_fee must equal the fee component sum")
        if self.side is Side.BUY:
            if self.cash_reservation_price < self.estimated_execution_price:
                raise ValueError("BUY cash reservation price cannot be below its estimate")
            if self.cash_reservation_fee < self.total_fee:
                raise ValueError("BUY cash reservation fee cannot be below its estimate")
        elif (
            self.cash_reservation_price != self.estimated_execution_price
            or self.cash_reservation_fee != self.total_fee
        ):
            raise ValueError("SELL cash reservation inputs must equal its execution estimate")
        expected_cash_change = (
            -(self.gross_amount + self.total_fee)
            if self.side is Side.BUY
            else self.gross_amount - self.total_fee
        )
        if self.estimated_cash_change != expected_cash_change:
            raise ValueError("estimated cash change does not match side, gross amount, and fees")
        expected_reserved_cash = (
            self.quantity * self.cash_reservation_price + self.cash_reservation_fee
            if self.side is Side.BUY
            else max(-self.estimated_cash_change, Decimal(0))
        )
        if self.reserved_cash != expected_reserved_cash:
            raise ValueError("reserved_cash does not match the protected cash requirement")
        if self.line_hash != stable_hash(self._content_payload()):
            raise ValueError("order draft line_hash does not match its content")

    @classmethod
    def build(
        cls,
        *,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        side: Side,
        current_quantity: Decimal,
        target_quantity: Decimal,
        quantity: Decimal,
        is_full_liquidation: bool,
        reference_price: Decimal,
        estimated_execution_price: Decimal,
        price_observed_at: datetime,
        price_available_at: datetime,
        data_version: str,
        state_revision: str,
        state_hash: str,
        participation_rate: Decimal,
        commission: Decimal,
        stamp_duty: Decimal,
        transfer_fee: Decimal,
        other_fee: Decimal,
        market_rule_version: str,
        market_rule_hash: str,
        fee_rule_version: str,
        fee_rule_hash: str,
        slippage_model_version: str,
        slippage_model_hash: str,
        rationale: str,
        cash_reservation_price: Decimal | None = None,
        cash_reservation_fee: Decimal | None = None,
    ) -> OrderDraftLine:
        gross_amount = quantity * estimated_execution_price
        slippage_amount = quantity * abs(estimated_execution_price - reference_price)
        total_fee = commission + stamp_duty + transfer_fee + other_fee
        cash_change = -(gross_amount + total_fee) if side is Side.BUY else gross_amount - total_fee
        reservation_price = (
            estimated_execution_price if cash_reservation_price is None else cash_reservation_price
        )
        reservation_fee = total_fee if cash_reservation_fee is None else cash_reservation_fee
        reserved_cash = (
            quantity * reservation_price + reservation_fee
            if side is Side.BUY
            else max(-cash_change, Decimal(0))
        )
        values: dict[str, object] = {
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "side": side,
            "current_quantity": current_quantity,
            "target_quantity": target_quantity,
            "quantity": quantity,
            "is_full_liquidation": is_full_liquidation,
            "reference_price": reference_price,
            "estimated_execution_price": estimated_execution_price,
            "cash_reservation_price": reservation_price,
            "price_observed_at": price_observed_at,
            "price_available_at": price_available_at,
            "data_version": data_version,
            "state_revision": state_revision,
            "state_hash": state_hash,
            "participation_rate": participation_rate,
            "gross_amount": gross_amount,
            "estimated_slippage_amount": slippage_amount,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "transfer_fee": transfer_fee,
            "other_fee": other_fee,
            "total_fee": total_fee,
            "cash_reservation_fee": reservation_fee,
            "reserved_cash": reserved_cash,
            "estimated_cash_change": cash_change,
            "market_rule_version": market_rule_version,
            "market_rule_hash": market_rule_hash,
            "fee_rule_version": fee_rule_version,
            "fee_rule_hash": fee_rule_hash,
            "slippage_model_version": slippage_model_version,
            "slippage_model_hash": slippage_model_hash,
            "rationale": rationale,
        }
        return cls(**values, line_hash=stable_hash(values))  # type: ignore[arg-type]

    def _content_payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "line_hash"}


@dataclass(frozen=True, slots=True)
class OrderDraftBatch:
    """A content-addressed review artifact with no approval or broker submission power."""

    schema_version: str
    decision_id: str
    draft_as_of: datetime
    trading_day: date
    expires_at: datetime
    validity_seconds: int
    data_version: str
    currency: str
    account_snapshot_id: str
    account_snapshot_hash: str
    account_snapshot_as_of: datetime
    runtime_mode: RuntimeMode
    portfolio_proposal_hash: str
    risk_request_hash: str
    risk_result_hash: str
    risk_status: RiskCheckStatus
    risk_checked_at: datetime
    risk_engine_version: str
    risk_policy_version: str
    risk_policy_hash: str
    generator_version: str
    generator_config_version: str
    generator_config_hash: str
    funding_policy: DraftFundingPolicy
    available_cash: Decimal
    reserved_cash_required: Decimal
    estimated_buy_cash_required: Decimal
    estimated_sell_cash_proceeds: Decimal
    estimated_total_fees: Decimal
    estimated_total_slippage: Decimal
    lines: tuple[OrderDraftLine, ...]
    state_hashes: tuple[str, ...]
    market_rule_hashes: tuple[str, ...]
    fee_rule_hashes: tuple[str, ...]
    slippage_model_hashes: tuple[str, ...]
    input_hash: str
    batch_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("order draft schema_version must be '1'")
        for field_name in (
            "decision_id",
            "data_version",
            "currency",
            "account_snapshot_id",
            "risk_engine_version",
            "risk_policy_version",
            "generator_version",
            "generator_config_version",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))
        ensure_aware(self.draft_as_of)
        ensure_aware(self.expires_at)
        ensure_aware(self.risk_checked_at)
        ensure_aware(self.account_snapshot_as_of)
        if not isinstance(self.runtime_mode, RuntimeMode) or self.runtime_mode not in {
            RuntimeMode.PAPER,
            RuntimeMode.LIVE_ASSISTED,
        }:
            raise ValueError("order drafts require PAPER or LIVE_ASSISTED runtime mode")
        if self.trading_day != self.draft_as_of.astimezone(SHANGHAI_TZ).date():
            raise ValueError("draft trading_day must match draft_as_of in Asia/Shanghai")
        if self.account_snapshot_as_of.astimezone(SHANGHAI_TZ).date() != self.trading_day:
            raise ValueError("account snapshot and order draft must share one trading day")
        if self.account_snapshot_as_of > self.draft_as_of:
            raise ValueError("account snapshot cannot be after draft_as_of")
        if self.risk_checked_at < self.account_snapshot_as_of:
            raise ValueError("risk result cannot precede the bound account snapshot")
        if self.risk_checked_at > self.draft_as_of:
            raise ValueError("risk result cannot be checked after draft_as_of")
        if self.expires_at != self.draft_as_of + timedelta(seconds=self.validity_seconds):
            raise ValueError("draft expiry must match the configured validity window")
        if self.generator_version != ORDER_DRAFT_GENERATOR_VERSION:
            raise ValueError("unknown order draft generator version")
        if not isinstance(self.risk_status, RiskCheckStatus) or self.risk_status not in {
            RiskCheckStatus.PASS,
            RiskCheckStatus.WARN,
        }:
            raise ValueError("only PASS or WARN risk results may create an order draft")
        if not isinstance(self.funding_policy, DraftFundingPolicy):
            raise ValueError("funding_policy must be a DraftFundingPolicy value")
        expected_config_hash = stable_hash(
            {
                "version": self.generator_config_version,
                "validity_seconds": self.validity_seconds,
                "funding_policy": self.funding_policy,
            }
        )
        if self.generator_config_hash != expected_config_hash:
            raise ValueError("generator_config_hash does not match draft configuration")
        for digest, field_name in (
            (self.account_snapshot_hash, "account_snapshot_hash"),
            (self.portfolio_proposal_hash, "portfolio_proposal_hash"),
            (self.risk_request_hash, "risk_request_hash"),
            (self.risk_result_hash, "risk_result_hash"),
            (self.risk_policy_hash, "risk_policy_hash"),
            (self.generator_config_hash, "generator_config_hash"),
            (self.input_hash, "input_hash"),
            (self.batch_hash, "batch_hash"),
        ):
            _sha256(digest, field_name)
        if not self.lines:
            raise ValueError("an order draft batch requires at least one draft line")
        line_ids = tuple(line.instrument_id for line in self.lines)
        if tuple(sorted(set(line_ids))) != line_ids:
            raise ValueError("order draft lines must be unique and sorted")
        if any(
            line.data_version != self.data_version or line.price_available_at > self.draft_as_of
            for line in self.lines
        ):
            raise ValueError("draft line data version and PIT price must align to the batch")
        required_identity_sets = (
            (self.state_hashes, {line.state_hash for line in self.lines}),
            (self.market_rule_hashes, {line.market_rule_hash for line in self.lines}),
        )
        for actual, required in required_identity_sets:
            if tuple(sorted(set(actual))) != actual or not required.issubset(actual):
                raise ValueError("draft sizing identity hashes do not cover its lines")
            for digest in actual:
                _sha256(digest, "draft input identity hash")
        exact_identity_sets = (
            (self.fee_rule_hashes, tuple(sorted({line.fee_rule_hash for line in self.lines}))),
            (
                self.slippage_model_hashes,
                tuple(sorted({line.slippage_model_hash for line in self.lines})),
            ),
        )
        for actual, expected in exact_identity_sets:
            if actual != expected:
                raise ValueError("draft pricing identity hashes do not match its lines")
            for digest in actual:
                _sha256(digest, "draft input identity hash")
        numeric_values = (
            self.available_cash,
            self.reserved_cash_required,
            self.estimated_buy_cash_required,
            self.estimated_sell_cash_proceeds,
            self.estimated_total_fees,
            self.estimated_total_slippage,
        )
        for numeric in numeric_values:
            _exact_decimal(numeric, "order draft batch numeric value")
            if numeric < 0:
                raise ValueError("order draft batch values cannot be negative")
        expected_buy_cash = -sum(
            (line.estimated_cash_change for line in self.lines if line.side is Side.BUY),
            Decimal(0),
        )
        expected_sell_cash = sum(
            (
                max(line.estimated_cash_change, Decimal(0))
                for line in self.lines
                if line.side is Side.SELL
            ),
            Decimal(0),
        )
        expected_reserved_cash = sum((line.reserved_cash for line in self.lines), Decimal(0))
        if self.reserved_cash_required != expected_reserved_cash:
            raise ValueError("reserved cash must equal protected draft-line requirements")
        if self.estimated_buy_cash_required != expected_buy_cash:
            raise ValueError("estimated buy cash must equal BUY draft-line requirements")
        if self.estimated_sell_cash_proceeds != expected_sell_cash:
            raise ValueError("estimated sell proceeds must equal SELL draft-line proceeds")
        if self.estimated_total_fees != sum((line.total_fee for line in self.lines), Decimal(0)):
            raise ValueError("estimated total fees must equal draft-line fees")
        if self.estimated_total_slippage != sum(
            (line.estimated_slippage_amount for line in self.lines), Decimal(0)
        ):
            raise ValueError("estimated total slippage must equal draft-line slippage")
        if self.reserved_cash_required > self.available_cash:
            raise ValueError("draft reserved cash requirements exceed snapshot available cash")
        if self.input_hash != stable_hash(self._input_payload()):
            raise ValueError("order draft input_hash does not match bound inputs")
        if self.batch_hash != stable_hash(self._batch_payload()):
            raise ValueError("order draft batch_hash does not match its content")

    @property
    def is_executable(self) -> bool:
        return False

    @property
    def requires_human_approval(self) -> bool:
        return True

    @property
    def batch_id(self) -> str:
        return f"draft:{self.batch_hash}"

    def _input_payload(self) -> dict[str, object]:
        return {
            "account_snapshot_hash": self.account_snapshot_hash,
            "account_snapshot_id": self.account_snapshot_id,
            "account_snapshot_as_of": self.account_snapshot_as_of,
            "data_version": self.data_version,
            "decision_id": self.decision_id,
            "draft_as_of": self.draft_as_of,
            "fee_rule_hashes": self.fee_rule_hashes,
            "funding_policy": self.funding_policy,
            "generator_config_hash": self.generator_config_hash,
            "generator_config_version": self.generator_config_version,
            "generator_version": self.generator_version,
            "market_rule_hashes": self.market_rule_hashes,
            "portfolio_proposal_hash": self.portfolio_proposal_hash,
            "risk_policy_hash": self.risk_policy_hash,
            "risk_request_hash": self.risk_request_hash,
            "risk_result_hash": self.risk_result_hash,
            "slippage_model_hashes": self.slippage_model_hashes,
            "state_hashes": self.state_hashes,
            "runtime_mode": self.runtime_mode,
            "validity_seconds": self.validity_seconds,
        }

    def _batch_payload(self) -> dict[str, object]:
        return {
            **self._input_payload(),
            "available_cash": self.available_cash,
            "batch_lines": [asdict(line) for line in self.lines],
            "currency": self.currency,
            "estimated_buy_cash_required": self.estimated_buy_cash_required,
            "estimated_sell_cash_proceeds": self.estimated_sell_cash_proceeds,
            "estimated_total_fees": self.estimated_total_fees,
            "estimated_total_slippage": self.estimated_total_slippage,
            "expires_at": self.expires_at,
            "input_hash": self.input_hash,
            "reserved_cash_required": self.reserved_cash_required,
            "risk_checked_at": self.risk_checked_at,
            "risk_engine_version": self.risk_engine_version,
            "risk_policy_version": self.risk_policy_version,
            "risk_status": self.risk_status,
            "schema_version": self.schema_version,
            "trading_day": self.trading_day,
        }

    @classmethod
    def build(
        cls,
        *,
        decision_id: str,
        draft_as_of: datetime,
        data_version: str,
        currency: str,
        account_snapshot_id: str,
        account_snapshot_hash: str,
        account_snapshot_as_of: datetime,
        runtime_mode: RuntimeMode,
        portfolio_proposal_hash: str,
        risk_request_hash: str,
        risk_result_hash: str,
        risk_status: RiskCheckStatus,
        risk_checked_at: datetime,
        risk_engine_version: str,
        risk_policy_version: str,
        risk_policy_hash: str,
        config: OrderDraftGeneratorConfig,
        available_cash: Decimal,
        lines: tuple[OrderDraftLine, ...],
        sizing_state_hashes: tuple[str, ...] | None = None,
        sizing_market_rule_hashes: tuple[str, ...] | None = None,
    ) -> OrderDraftBatch:
        ordered_lines = tuple(sorted(lines, key=lambda line: line.instrument_id))
        state_hashes = tuple(
            sorted(
                set(sizing_state_hashes)
                if sizing_state_hashes is not None
                else {line.state_hash for line in ordered_lines}
            )
        )
        market_rule_hashes = tuple(
            sorted(
                set(sizing_market_rule_hashes)
                if sizing_market_rule_hashes is not None
                else {line.market_rule_hash for line in ordered_lines}
            )
        )
        fee_rule_hashes = tuple(sorted({line.fee_rule_hash for line in ordered_lines}))
        slippage_model_hashes = tuple(sorted({line.slippage_model_hash for line in ordered_lines}))
        input_values: dict[str, object] = {
            "account_snapshot_hash": account_snapshot_hash,
            "account_snapshot_id": account_snapshot_id,
            "account_snapshot_as_of": account_snapshot_as_of,
            "data_version": data_version,
            "decision_id": decision_id,
            "draft_as_of": draft_as_of,
            "fee_rule_hashes": fee_rule_hashes,
            "funding_policy": config.funding_policy,
            "generator_config_hash": config.config_hash,
            "generator_config_version": config.version,
            "generator_version": ORDER_DRAFT_GENERATOR_VERSION,
            "market_rule_hashes": market_rule_hashes,
            "portfolio_proposal_hash": portfolio_proposal_hash,
            "risk_policy_hash": risk_policy_hash,
            "risk_request_hash": risk_request_hash,
            "risk_result_hash": risk_result_hash,
            "slippage_model_hashes": slippage_model_hashes,
            "state_hashes": state_hashes,
            "runtime_mode": runtime_mode,
            "validity_seconds": config.validity_seconds,
        }
        input_hash = stable_hash(input_values)
        buy_cash = -sum(
            (line.estimated_cash_change for line in ordered_lines if line.side is Side.BUY),
            Decimal(0),
        )
        sell_cash = sum(
            (
                max(line.estimated_cash_change, Decimal(0))
                for line in ordered_lines
                if line.side is Side.SELL
            ),
            Decimal(0),
        )
        reserved_cash = sum((line.reserved_cash for line in ordered_lines), Decimal(0))
        total_fees = sum((line.total_fee for line in ordered_lines), Decimal(0))
        total_slippage = sum((line.estimated_slippage_amount for line in ordered_lines), Decimal(0))
        expires_at = draft_as_of + timedelta(seconds=config.validity_seconds)
        batch_values = {
            **input_values,
            "available_cash": available_cash,
            "batch_lines": [asdict(line) for line in ordered_lines],
            "currency": currency,
            "estimated_buy_cash_required": buy_cash,
            "estimated_sell_cash_proceeds": sell_cash,
            "estimated_total_fees": total_fees,
            "estimated_total_slippage": total_slippage,
            "expires_at": expires_at,
            "input_hash": input_hash,
            "reserved_cash_required": reserved_cash,
            "risk_checked_at": risk_checked_at,
            "risk_engine_version": risk_engine_version,
            "risk_policy_version": risk_policy_version,
            "risk_status": risk_status,
            "schema_version": "1",
            "trading_day": draft_as_of.astimezone(SHANGHAI_TZ).date(),
        }
        return cls(
            schema_version="1",
            decision_id=decision_id,
            draft_as_of=draft_as_of,
            trading_day=draft_as_of.astimezone(SHANGHAI_TZ).date(),
            expires_at=expires_at,
            validity_seconds=config.validity_seconds,
            data_version=data_version,
            currency=currency,
            account_snapshot_id=account_snapshot_id,
            account_snapshot_hash=account_snapshot_hash,
            account_snapshot_as_of=account_snapshot_as_of,
            runtime_mode=runtime_mode,
            portfolio_proposal_hash=portfolio_proposal_hash,
            risk_request_hash=risk_request_hash,
            risk_result_hash=risk_result_hash,
            risk_status=risk_status,
            risk_checked_at=risk_checked_at,
            risk_engine_version=risk_engine_version,
            risk_policy_version=risk_policy_version,
            risk_policy_hash=risk_policy_hash,
            generator_version=ORDER_DRAFT_GENERATOR_VERSION,
            generator_config_version=config.version,
            generator_config_hash=config.config_hash,
            funding_policy=config.funding_policy,
            available_cash=available_cash,
            reserved_cash_required=reserved_cash,
            estimated_buy_cash_required=buy_cash,
            estimated_sell_cash_proceeds=sell_cash,
            estimated_total_fees=total_fees,
            estimated_total_slippage=total_slippage,
            lines=ordered_lines,
            state_hashes=state_hashes,
            market_rule_hashes=market_rule_hashes,
            fee_rule_hashes=fee_rule_hashes,
            slippage_model_hashes=slippage_model_hashes,
            input_hash=input_hash,
            batch_hash=stable_hash(batch_values),
        )


__all__ = [
    "ORDER_DRAFT_GENERATOR_VERSION",
    "DraftFundingPolicy",
    "OrderDraftBatch",
    "OrderDraftGeneratorConfig",
    "OrderDraftInputError",
    "OrderDraftLine",
]
