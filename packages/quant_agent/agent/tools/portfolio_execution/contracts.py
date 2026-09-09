"""Closed Agent-visible contracts for portfolio and paper-execution tools."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, field_validator, model_validator

from quant_agent.agent.tools.contracts import ToolArguments, ToolOutput
from quant_agent.backtest import Side, TradableInstrumentType
from quant_agent.config import RuntimeMode
from quant_agent.core.time import ensure_aware
from quant_agent.execution.order_drafts import DraftFundingPolicy
from quant_agent.execution.paper import PaperNoFillReason, PaperOrderStatus
from quant_agent.portfolio import (
    PortfolioAdjustmentCode,
    StrategySleeveKind,
    ValuationStatus,
)
from quant_agent.reconciliation import (
    ReconciliationCheckStatus,
    ReconciliationDifferenceCode,
    ReconciliationDomain,
    ReconciliationSeverity,
    ReconciliationStatus,
    ReconciliationStopAction,
    ReconciliationStopScope,
)
from quant_agent.risk import RiskCheckStatus, RuleScope, RuleSeverity


def _exact_text(value: str) -> str:
    if value != value.strip() or not value.isprintable():
        raise ValueError("text must be printable and have no surrounding whitespace")
    return value


def _aware(value: datetime) -> datetime:
    ensure_aware(value)
    return value


Identifier = Annotated[
    str,
    Field(min_length=1, max_length=256),
    AfterValidator(_exact_text),
]
ShortText = Annotated[
    str,
    Field(min_length=1, max_length=512),
    AfterValidator(_exact_text),
]
LongText = Annotated[
    str,
    Field(min_length=1, max_length=4_000),
    AfterValidator(_exact_text),
]
Version = Annotated[
    str,
    Field(min_length=1, max_length=256),
    AfterValidator(_exact_text),
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
PositiveDecimal = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
Ratio = Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, le=1_000_000)]


class _DecisionAccountOutput(ToolOutput):
    decision_id: Identifier
    as_of: datetime
    data_version: Version
    account_id: Identifier
    account_snapshot_id: Identifier
    account_snapshot_hash: Sha256

    _validate_as_of = field_validator("as_of")(_aware)


class GetPortfolioSnapshotArguments(ToolArguments):
    max_positions: Annotated[int, Field(ge=1, le=500)] = 100


class BuildTargetPortfolioArguments(ToolArguments):
    max_lines: Annotated[int, Field(ge=1, le=500)] = 100
    max_adjustments: Annotated[int, Field(ge=1, le=500)] = 100


class CheckPortfolioRiskArguments(ToolArguments):
    max_findings: Annotated[int, Field(ge=1, le=500)] = 100


class CreateOrderDraftArguments(ToolArguments):
    idempotency_key: str = Field(min_length=1, max_length=512)
    max_lines: Annotated[int, Field(ge=1, le=500)] = 100

    _validate_key = field_validator("idempotency_key")(_exact_text)


class GetOrderDraftArguments(ToolArguments):
    batch_hash: Sha256
    max_lines: Annotated[int, Field(ge=1, le=500)] = 100


class SubmitPaperOrdersArguments(ToolArguments):
    idempotency_key: str = Field(min_length=1, max_length=512)
    batch_hash: Sha256
    max_orders: Annotated[int, Field(ge=1, le=500)] = 100
    max_fills: Annotated[int, Field(ge=1, le=1_000)] = 200

    _validate_key = field_validator("idempotency_key")(_exact_text)


class ReconcileAccountArguments(ToolArguments):
    batch_hash: Sha256
    max_findings: Annotated[int, Field(ge=1, le=1_000)] = 200
    max_checks: Annotated[int, Field(ge=1, le=1_000)] = 200


class CashSnapshotOutput(ToolOutput):
    currency: Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]
    total_cash: NonNegativeDecimal
    available_cash: NonNegativeDecimal
    frozen_cash: NonNegativeDecimal

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.total_cash != self.available_cash + self.frozen_cash:
            raise ValueError("cash total must equal available plus frozen")
        return self


class PortfolioPositionOutput(ToolOutput):
    instrument_id: Identifier
    instrument_type: TradableInstrumentType
    total_quantity: PositiveDecimal
    available_quantity: NonNegativeDecimal
    frozen_quantity: NonNegativeDecimal
    unsettled_quantity: NonNegativeDecimal
    average_cost: NonNegativeDecimal
    valuation_price: PositiveDecimal
    price_observed_at: datetime
    price_available_at: datetime
    valuation_status: ValuationStatus
    market_value: NonNegativeDecimal
    cost_basis: NonNegativeDecimal
    unrealized_pnl: FiniteDecimal
    price_data_version: Version
    price_source_hash: Sha256
    mark_policy_version: Version
    mark_policy_hash: Sha256

    _validate_times = field_validator("price_observed_at", "price_available_at")(_aware)


class PortfolioSnapshotOutput(_DecisionAccountOutput):
    runtime_mode: RuntimeMode
    valuation_at: datetime
    currency: Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]
    cash: CashSnapshotOutput
    position_market_value: NonNegativeDecimal
    total_equity: PositiveDecimal
    gross_exposure: Ratio
    cash_weight: Ratio
    previous_snapshot_hash: Sha256 | None
    source_event_log_hash: Sha256 | None
    content_hash: Sha256
    total_position_count: Count
    returned_position_count: Count
    positions_truncated: bool
    positions: Annotated[tuple[PortfolioPositionOutput, ...], Field(max_length=500)]

    _validate_valuation_at = field_validator("valuation_at")(_aware)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.returned_position_count != len(self.positions):
            raise ValueError("returned position count must equal positions length")
        if self.returned_position_count > self.total_position_count:
            raise ValueError("returned positions cannot exceed total positions")
        if self.positions_truncated != (self.returned_position_count < self.total_position_count):
            raise ValueError("positions_truncated does not match position counts")
        if self.gross_exposure + self.cash_weight != 1:
            raise ValueError("gross exposure and cash weight must sum to one")
        return self


class StrategyContributionOutput(ToolOutput):
    sleeve_id: Identifier
    sleeve_hash: Sha256
    sleeve_kind: StrategySleeveKind
    source_result_hash: Sha256
    local_target_weight: Ratio
    sleeve_budget: Ratio
    proposed_portfolio_weight: Ratio
    applied_portfolio_weight: Ratio
    reason: LongText


class PortfolioAdjustmentOutput(ToolOutput):
    code: PortfolioAdjustmentCode
    scope_id: Identifier
    before_weight: Ratio
    after_weight: Ratio
    rationale: LongText


class PortfolioTargetLineOutput(ToolOutput):
    instrument_id: Identifier
    instrument_type: TradableInstrumentType
    industry_id: Identifier | None
    industry_classification_hash: Sha256 | None
    current_market_value: NonNegativeDecimal
    current_weight: Ratio
    proposed_target_weight: Ratio
    target_weight: Ratio
    target_market_value: NonNegativeDecimal
    weight_delta: FiniteDecimal
    absolute_weight_delta: Ratio
    rationale: LongText
    contributions: Annotated[tuple[StrategyContributionOutput, ...], Field(max_length=32)]


class TargetPortfolioOutput(_DecisionAccountOutput):
    builder_version: Version
    config_version: Version
    config_hash: Sha256
    total_equity: PositiveDecimal
    etf_core_budget: Ratio
    stock_enhancement_budget: Ratio
    minimum_cash_weight: Ratio
    maximum_instrument_weight: Ratio
    maximum_stock_industry_weight: Ratio
    current_gross_weight: Ratio
    current_cash_weight: Ratio
    target_gross_weight: Ratio
    target_cash_weight: Ratio
    etf_target_weight: Ratio
    stock_target_weight: Ratio
    gross_instrument_turnover: NonNegativeDecimal
    one_way_turnover: NonNegativeDecimal
    sleeve_hashes: Annotated[tuple[Sha256, ...], Field(max_length=32)]
    classification_hashes: Annotated[tuple[Sha256, ...], Field(max_length=20_000)]
    classification_input_hash: Sha256
    input_hash: Sha256
    result_hash: Sha256
    total_line_count: Count
    returned_line_count: Count
    lines_truncated: bool
    lines: Annotated[tuple[PortfolioTargetLineOutput, ...], Field(max_length=500)]
    total_adjustment_count: Count
    returned_adjustment_count: Count
    adjustments_truncated: bool
    adjustments: Annotated[tuple[PortfolioAdjustmentOutput, ...], Field(max_length=500)]

    @model_validator(mode="after")
    def validate_counts_and_weights(self) -> Self:
        if self.returned_line_count != len(self.lines):
            raise ValueError("returned line count must equal lines length")
        if self.returned_adjustment_count != len(self.adjustments):
            raise ValueError("returned adjustment count must equal adjustments length")
        if self.returned_line_count > self.total_line_count:
            raise ValueError("returned lines cannot exceed total lines")
        if self.returned_adjustment_count > self.total_adjustment_count:
            raise ValueError("returned adjustments cannot exceed total adjustments")
        if self.lines_truncated != (self.returned_line_count < self.total_line_count):
            raise ValueError("lines_truncated does not match line counts")
        if self.adjustments_truncated != (
            self.returned_adjustment_count < self.total_adjustment_count
        ):
            raise ValueError("adjustments_truncated does not match adjustment counts")
        if self.current_gross_weight + self.current_cash_weight != 1:
            raise ValueError("current portfolio weights must sum to one")
        if self.target_gross_weight + self.target_cash_weight != 1:
            raise ValueError("target portfolio weights must sum to one")
        if self.target_gross_weight != self.etf_target_weight + self.stock_target_weight:
            raise ValueError("target gross weight must equal ETF plus stock weights")
        return self


class RiskRequestIdentityOutput(ToolOutput):
    request_hash: Sha256
    portfolio_proposal_hash: Sha256
    risk_context_hash: Sha256
    valuation_version: Version
    policy_version: Version
    policy_hash: Sha256


class RiskFindingOutput(ToolOutput):
    rule_code: Identifier
    rule_version: Version
    severity: RuleSeverity
    scope: RuleScope
    scope_id: Identifier | None
    instrument_id: Identifier | None
    observed_value: FiniteDecimal | None
    limit_value: FiniteDecimal | None


class PortfolioRiskOutput(_DecisionAccountOutput):
    status: RiskCheckStatus
    evaluated_at: datetime
    risk_engine_version: Version
    error_code: Identifier | None
    passed: bool
    allows_execution: bool
    request: RiskRequestIdentityOutput
    result_hash: Sha256
    total_finding_count: Count
    returned_finding_count: Count
    findings_truncated: bool
    findings: Annotated[tuple[RiskFindingOutput, ...], Field(max_length=500)]

    _validate_evaluated_at = field_validator("evaluated_at")(_aware)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.returned_finding_count != len(self.findings):
            raise ValueError("returned finding count must equal findings length")
        if self.returned_finding_count > self.total_finding_count:
            raise ValueError("returned findings cannot exceed total findings")
        if self.findings_truncated != (self.returned_finding_count < self.total_finding_count):
            raise ValueError("findings_truncated does not match finding counts")
        if self.passed != (self.status is RiskCheckStatus.PASS):
            raise ValueError("passed flag does not match risk status")
        if self.allows_execution != (self.status in {RiskCheckStatus.PASS, RiskCheckStatus.WARN}):
            raise ValueError("allows_execution does not match risk status")
        if (self.status is RiskCheckStatus.ERROR) != (self.error_code is not None):
            raise ValueError("risk error_code must exist exactly for ERROR")
        return self


class OrderDraftLineOutput(ToolOutput):
    instrument_id: Identifier
    instrument_type: TradableInstrumentType
    side: Side
    current_quantity: NonNegativeDecimal
    target_quantity: NonNegativeDecimal
    quantity: PositiveDecimal
    is_full_liquidation: bool
    reference_price: PositiveDecimal
    estimated_execution_price: PositiveDecimal
    cash_reservation_price: PositiveDecimal
    price_observed_at: datetime
    price_available_at: datetime
    data_version: Version
    state_revision: Version
    state_hash: Sha256
    participation_rate: Annotated[Decimal, Field(gt=0, le=1, allow_inf_nan=False)]
    gross_amount: PositiveDecimal
    estimated_slippage_amount: NonNegativeDecimal
    commission: NonNegativeDecimal
    stamp_duty: NonNegativeDecimal
    transfer_fee: NonNegativeDecimal
    other_fee: NonNegativeDecimal
    total_fee: NonNegativeDecimal
    cash_reservation_fee: NonNegativeDecimal
    reserved_cash: NonNegativeDecimal
    estimated_cash_change: FiniteDecimal
    market_rule_version: Version
    market_rule_hash: Sha256
    fee_rule_version: Version
    fee_rule_hash: Sha256
    slippage_model_version: Version
    slippage_model_hash: Sha256
    rationale: LongText
    line_hash: Sha256

    _validate_times = field_validator("price_observed_at", "price_available_at")(_aware)


class OrderDraftOutput(_DecisionAccountOutput):
    draft_as_of: datetime
    trading_day: date
    expires_at: datetime
    validity_seconds: Annotated[int, Field(ge=1, le=86_400)]
    runtime_mode: RuntimeMode
    currency: Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]
    portfolio_proposal_hash: Sha256
    risk_request_hash: Sha256
    risk_result_hash: Sha256
    risk_status: RiskCheckStatus
    risk_checked_at: datetime
    risk_engine_version: Version
    risk_policy_version: Version
    risk_policy_hash: Sha256
    generator_version: Version
    generator_config_version: Version
    generator_config_hash: Sha256
    funding_policy: DraftFundingPolicy
    available_cash: NonNegativeDecimal
    reserved_cash_required: NonNegativeDecimal
    estimated_buy_cash_required: NonNegativeDecimal
    estimated_sell_cash_proceeds: NonNegativeDecimal
    estimated_total_fees: NonNegativeDecimal
    estimated_total_slippage: NonNegativeDecimal
    state_hashes: Annotated[tuple[Sha256, ...], Field(max_length=500)]
    market_rule_hashes: Annotated[tuple[Sha256, ...], Field(max_length=500)]
    fee_rule_hashes: Annotated[tuple[Sha256, ...], Field(max_length=500)]
    slippage_model_hashes: Annotated[tuple[Sha256, ...], Field(max_length=500)]
    input_hash: Sha256
    batch_hash: Sha256
    batch_id: Identifier
    is_executable: Literal[False]
    requires_human_approval: Literal[True]
    total_line_count: Count
    returned_line_count: Count
    lines_truncated: bool
    lines: Annotated[tuple[OrderDraftLineOutput, ...], Field(max_length=500)]

    _validate_times = field_validator("draft_as_of", "expires_at", "risk_checked_at")(_aware)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.returned_line_count != len(self.lines):
            raise ValueError("returned draft line count must equal lines length")
        if self.returned_line_count > self.total_line_count:
            raise ValueError("returned draft lines cannot exceed total lines")
        if self.lines_truncated != (self.returned_line_count < self.total_line_count):
            raise ValueError("draft lines_truncated does not match line counts")
        if self.batch_id != f"draft:{self.batch_hash}":
            raise ValueError("batch_id must bind batch_hash")
        if self.risk_status not in {RiskCheckStatus.PASS, RiskCheckStatus.WARN}:
            raise ValueError("order draft requires an executable risk status")
        if self.expires_at <= self.draft_as_of:
            raise ValueError("order draft expiry must follow creation")
        return self


class PaperOrderOutput(ToolOutput):
    order_id: Identifier
    line_hash: Sha256
    instrument_id: Identifier
    instrument_type: TradableInstrumentType
    side: Side
    quantity: PositiveDecimal
    filled_quantity: NonNegativeDecimal
    remaining_quantity: NonNegativeDecimal
    status: PaperOrderStatus
    is_full_liquidation: bool
    created_at: datetime
    expires_at: datetime
    cash_reserved: NonNegativeDecimal
    sell_reserved_quantity: NonNegativeDecimal
    last_attempt_id: Identifier | None
    order_hash: Sha256

    _validate_times = field_validator("created_at", "expires_at")(_aware)


class PaperAttemptOutput(ToolOutput):
    attempt_id: Identifier
    order_id: Identifier
    attempted_at: datetime
    status: PaperOrderStatus
    no_fill_reason: PaperNoFillReason | None
    requested_quantity: PositiveDecimal
    filled_quantity: NonNegativeDecimal
    unfilled_quantity: NonNegativeDecimal
    reference_price: PositiveDecimal
    execution_price: PositiveDecimal | None
    gross_amount: NonNegativeDecimal
    participation_rate: Ratio
    state_hash: Sha256
    attempt_hash: Sha256

    _validate_attempted_at = field_validator("attempted_at")(_aware)


class PaperFillOutput(ToolOutput):
    fill_id: Identifier
    order_id: Identifier
    attempt_id: Identifier
    instrument_id: Identifier
    instrument_type: TradableInstrumentType
    filled_at: datetime
    side: Side
    quantity: PositiveDecimal
    price: PositiveDecimal
    gross_amount: PositiveDecimal
    commission: NonNegativeDecimal
    stamp_duty: NonNegativeDecimal
    transfer_fee: NonNegativeDecimal
    other_fee: NonNegativeDecimal
    total_fee: NonNegativeDecimal
    cash_change: FiniteDecimal
    sellable_on: date | None
    fill_hash: Sha256

    _validate_filled_at = field_validator("filled_at")(_aware)


class PaperPositionOutput(ToolOutput):
    instrument_id: Identifier
    instrument_type: TradableInstrumentType
    total_quantity: PositiveDecimal
    available_quantity: NonNegativeDecimal
    frozen_quantity: NonNegativeDecimal
    unsettled_quantity: NonNegativeDecimal
    reserved_sell_quantity: NonNegativeDecimal
    cost_basis: NonNegativeDecimal
    average_cost: NonNegativeDecimal
    position_hash: Sha256


class PaperAccountStateOutput(ToolOutput):
    as_of: datetime
    total_cash: NonNegativeDecimal
    available_cash: NonNegativeDecimal
    frozen_cash: NonNegativeDecimal
    external_frozen_cash: NonNegativeDecimal
    state_hash: Sha256
    event_log_hash: Sha256
    total_position_count: Count
    returned_position_count: Count
    positions_truncated: bool
    positions: Annotated[tuple[PaperPositionOutput, ...], Field(max_length=500)]

    _validate_as_of = field_validator("as_of")(_aware)

    @model_validator(mode="after")
    def validate_position_count(self) -> Self:
        if self.returned_position_count != len(self.positions):
            raise ValueError("returned paper position count must equal positions length")
        if self.returned_position_count > self.total_position_count:
            raise ValueError("returned paper positions cannot exceed total positions")
        if self.positions_truncated != (self.returned_position_count < self.total_position_count):
            raise ValueError("paper positions_truncated does not match position counts")
        return self


class PaperExecutionOutput(_DecisionAccountOutput):
    receipt_id: Identifier
    request_hash: Sha256
    batch_hash: Sha256
    request_submitted_at: datetime
    processed_at: datetime
    account_before_hash: Sha256
    account_after_hash: Sha256
    event_log_before_hash: Sha256
    event_log_hash: Sha256
    receipt_hash: Sha256
    account_before: PaperAccountStateOutput
    account_after: PaperAccountStateOutput
    total_order_count: Count
    returned_order_count: Count
    orders_truncated: bool
    orders: Annotated[tuple[PaperOrderOutput, ...], Field(max_length=500)]
    total_attempt_count: Count
    returned_attempt_count: Count
    attempts_truncated: bool
    attempts: Annotated[tuple[PaperAttemptOutput, ...], Field(max_length=500)]
    total_fill_count: Count
    returned_fill_count: Count
    fills_truncated: bool
    fills: Annotated[tuple[PaperFillOutput, ...], Field(max_length=1_000)]

    _validate_times = field_validator("request_submitted_at", "processed_at")(_aware)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        triples = (
            (
                self.returned_order_count,
                self.total_order_count,
                len(self.orders),
                self.orders_truncated,
            ),
            (
                self.returned_attempt_count,
                self.total_attempt_count,
                len(self.attempts),
                self.attempts_truncated,
            ),
            (
                self.returned_fill_count,
                self.total_fill_count,
                len(self.fills),
                self.fills_truncated,
            ),
        )
        for returned, total, actual, truncated in triples:
            if returned != actual or returned > total or truncated != (returned < total):
                raise ValueError("paper execution collection counts are inconsistent")
        if self.account_before_hash != self.account_before.state_hash:
            raise ValueError("account_before_hash must bind account_before")
        if self.account_after_hash != self.account_after.state_hash:
            raise ValueError("account_after_hash must bind account_after")
        return self


class ReconciliationFindingOutput(ToolOutput):
    finding_id: Identifier
    domain: ReconciliationDomain
    code: ReconciliationDifferenceCode
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    entity_id: Identifier
    message: LongText
    line_hash: Sha256 | None
    order_id: Identifier | None
    fill_id: Identifier | None
    instrument_id: Identifier | None
    expected_value: ShortText | None
    observed_value: ShortText | None
    delta: ShortText | None
    evidence_hashes: Annotated[tuple[Sha256, ...], Field(max_length=100)]
    finding_hash: Sha256


class ReconciliationCheckOutput(ToolOutput):
    check_id: Identifier
    domain: ReconciliationDomain
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    instrument_id: Identifier | None
    order_id: Identifier | None
    check_hash: Sha256


class CashReconciliationOutput(ToolOutput):
    status: ReconciliationCheckStatus
    severity: ReconciliationSeverity
    expected_total_cash: FiniteDecimal
    observed_total_cash: FiniteDecimal
    total_delta: FiniteDecimal
    expected_available_cash: FiniteDecimal
    observed_available_cash: FiniteDecimal
    available_delta: FiniteDecimal
    expected_frozen_cash: FiniteDecimal
    observed_frozen_cash: FiniteDecimal
    frozen_delta: FiniteDecimal
    mismatched_fields: Annotated[tuple[Identifier, ...], Field(max_length=16)]
    check_hash: Sha256


class ReconciliationStopSignalOutput(ToolOutput):
    required: bool
    action: ReconciliationStopAction
    scope: ReconciliationStopScope
    severity: ReconciliationSeverity
    reason_codes: Annotated[tuple[ReconciliationDifferenceCode, ...], Field(max_length=100)]
    trigger_hashes: Annotated[tuple[Sha256, ...], Field(max_length=1_000)]
    generated_at: datetime
    signal_hash: Sha256

    _validate_generated_at = field_validator("generated_at")(_aware)


class ReconciliationOutput(_DecisionAccountOutput):
    reconciliation_id: Identifier
    engine_version: Version
    input_hash: Sha256
    batch_hash: Sha256
    draft_hash: Sha256
    receipt_id: Identifier
    receipt_hash: Sha256
    expected_account_before_hash: Sha256
    expected_account_after_hash: Sha256
    observed_snapshot_id: Identifier
    observed_snapshot_hash: Sha256
    observed_as_of: datetime
    reconciled_at: datetime
    policy_version: Version
    policy_hash: Sha256
    status: ReconciliationStatus
    max_severity: ReconciliationSeverity
    result_hash: Sha256
    total_finding_count: Count
    returned_finding_count: Count
    findings_truncated: bool
    findings: Annotated[tuple[ReconciliationFindingOutput, ...], Field(max_length=1_000)]
    total_check_count: Count
    returned_check_count: Count
    checks_truncated: bool
    checks: Annotated[tuple[ReconciliationCheckOutput, ...], Field(max_length=1_000)]
    duplicate_group_count: Count
    cash: CashReconciliationOutput
    stop_signal: ReconciliationStopSignalOutput

    _validate_times = field_validator("observed_as_of", "reconciled_at")(_aware)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if (
            self.returned_finding_count != len(self.findings)
            or self.returned_finding_count > self.total_finding_count
            or self.findings_truncated != (self.returned_finding_count < self.total_finding_count)
        ):
            raise ValueError("reconciliation finding counts are inconsistent")
        if (
            self.returned_check_count != len(self.checks)
            or self.returned_check_count > self.total_check_count
            or self.checks_truncated != (self.returned_check_count < self.total_check_count)
        ):
            raise ValueError("reconciliation check counts are inconsistent")
        return self


__all__ = [
    "BuildTargetPortfolioArguments",
    "CashReconciliationOutput",
    "CashSnapshotOutput",
    "CheckPortfolioRiskArguments",
    "CreateOrderDraftArguments",
    "GetOrderDraftArguments",
    "GetPortfolioSnapshotArguments",
    "OrderDraftLineOutput",
    "OrderDraftOutput",
    "PaperAccountStateOutput",
    "PaperAttemptOutput",
    "PaperExecutionOutput",
    "PaperFillOutput",
    "PaperOrderOutput",
    "PaperPositionOutput",
    "PortfolioAdjustmentOutput",
    "PortfolioPositionOutput",
    "PortfolioRiskOutput",
    "PortfolioSnapshotOutput",
    "PortfolioTargetLineOutput",
    "ReconcileAccountArguments",
    "ReconciliationCheckOutput",
    "ReconciliationFindingOutput",
    "ReconciliationOutput",
    "ReconciliationStopSignalOutput",
    "RiskFindingOutput",
    "RiskRequestIdentityOutput",
    "StrategyContributionOutput",
    "SubmitPaperOrdersArguments",
    "TargetPortfolioOutput",
]
