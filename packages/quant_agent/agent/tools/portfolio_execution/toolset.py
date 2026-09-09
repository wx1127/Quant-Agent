"""Strict AgentTool adapters over the deterministic P5 portfolio chain."""

from __future__ import annotations

from typing import Any, cast

from pydantic import ValidationError

from quant_agent.agent.tools.contracts import AgentTool, ToolExecutionContext, ToolOutput
from quant_agent.core.errors import ErrorCode
from quant_agent.core.responses import Provenance, ToolIssue, ToolResponse
from quant_agent.execution.order_drafts import OrderDraftBatch, OrderDraftGenerator, OrderDraftLine
from quant_agent.execution.paper import (
    PaperAccountConflict,
    PaperAccountNotFound,
    PaperAccountState,
    PaperConcurrentUpdate,
    PaperExecutionInputError,
    PaperExecutionReceipt,
    PaperFill,
    PaperIdempotencyConflict,
    PaperMatchAttempt,
    PaperOrder,
    PaperPosition,
    PaperRepositoryError,
)
from quant_agent.portfolio import (
    AccountSnapshot,
    PortfolioConstraintAdjustment,
    PortfolioPositionSnapshot,
    PortfolioTargetLine,
    StrategyWeightContribution,
    TargetPortfolio,
    TargetPortfolioBuilder,
)
from quant_agent.reconciliation import (
    RECONCILIATION_ENGINE_VERSION,
    CashReconciliation,
    FillReconciliation,
    OrderReconciliation,
    PositionReconciliation,
    ReconciliationDomain,
    ReconciliationEngine,
    ReconciliationFinding,
    ReconciliationResult,
)
from quant_agent.risk import (
    KillSwitchOrderBlocked,
    PortfolioRiskEvaluator,
    RiskCheckResult,
    RiskCheckStatus,
    RiskFinding,
)

from .contracts import (
    BuildTargetPortfolioArguments,
    CashReconciliationOutput,
    CashSnapshotOutput,
    CheckPortfolioRiskArguments,
    CreateOrderDraftArguments,
    GetOrderDraftArguments,
    GetPortfolioSnapshotArguments,
    OrderDraftLineOutput,
    OrderDraftOutput,
    PaperAccountStateOutput,
    PaperAttemptOutput,
    PaperExecutionOutput,
    PaperFillOutput,
    PaperOrderOutput,
    PaperPositionOutput,
    PortfolioAdjustmentOutput,
    PortfolioPositionOutput,
    PortfolioRiskOutput,
    PortfolioSnapshotOutput,
    PortfolioTargetLineOutput,
    ReconcileAccountArguments,
    ReconciliationCheckOutput,
    ReconciliationFindingOutput,
    ReconciliationOutput,
    ReconciliationStopSignalOutput,
    RiskFindingOutput,
    RiskRequestIdentityOutput,
    StrategyContributionOutput,
    SubmitPaperOrdersArguments,
    TargetPortfolioOutput,
)
from .gateway import PaperOrderGateway
from .inputs import (
    PortfolioExecutionInputInvalid,
    PortfolioExecutionInputSource,
    PortfolioExecutionInputUnavailable,
    PortfolioRiskBlocked,
)
from .repository import (
    DraftArtifactConflict,
    DraftArtifactNotFound,
    DraftArtifactStore,
    DraftArtifactStoreError,
)
from .service import PortfolioExecutionPipeline

PORTFOLIO_EXECUTION_TOOL_VERSION = "agent-portfolio-execution-tool-v1"
_ACCOUNT_SERVICE = "portfolio-snapshot-service"
_TARGET_SERVICE = "target-portfolio-service"
_RISK_SERVICE = "portfolio-risk-service"
_DRAFT_SERVICE = "order-draft-service"
_PAPER_SERVICE = "paper-execution-service"
_RECONCILIATION_SERVICE = "account-reconciliation-service"
_MAX_PAPER_POSITIONS = 500


def _success[OutputT: ToolOutput](
    context: ToolExecutionContext,
    data: OutputT,
    *,
    service: str,
    version: str,
) -> ToolResponse[OutputT]:
    snapshot = context.decision_snapshot
    response = ToolResponse[object](
        ok=True,
        request_id=context.request_id,
        decision_id=snapshot.decision_id,
        as_of=snapshot.as_of,
        data=data,
        warnings=(),
        errors=(),
        provenance=Provenance(
            service=service,
            version=version,
            data_version=snapshot.data_version,
        ),
    ).validate_consistency()
    return cast(ToolResponse[OutputT], response)


def _failure[OutputT: ToolOutput](
    context: ToolExecutionContext,
    *,
    code: ErrorCode,
    message: str,
    service: str,
    version: str,
    data: OutputT | None = None,
) -> ToolResponse[OutputT]:
    snapshot = context.decision_snapshot
    response = ToolResponse[object](
        ok=False,
        request_id=context.request_id,
        decision_id=snapshot.decision_id,
        as_of=snapshot.as_of,
        data=data,
        warnings=(),
        errors=(ToolIssue(code=code, message=message),),
        provenance=Provenance(
            service=service,
            version=version,
            data_version=snapshot.data_version,
        ),
    ).validate_consistency()
    return cast(ToolResponse[OutputT], response)


def _expected_failure[OutputT: ToolOutput](
    context: ToolExecutionContext,
    error: Exception,
    *,
    service: str,
    version: str,
) -> ToolResponse[OutputT]:
    if isinstance(error, PortfolioExecutionInputUnavailable):
        return _failure(
            context,
            code=ErrorCode.DATA_UNAVAILABLE,
            message="required portfolio or execution input is unavailable",
            service=service,
            version=version,
        )
    if isinstance(error, (DraftArtifactNotFound, PaperAccountNotFound)):
        return _failure(
            context,
            code=ErrorCode.NOT_FOUND,
            message="required account-scoped artifact was not found",
            service=service,
            version=version,
        )
    if isinstance(
        error,
        (
            DraftArtifactConflict,
            PaperIdempotencyConflict,
            PaperConcurrentUpdate,
            PaperAccountConflict,
        ),
    ):
        return _failure(
            context,
            code=ErrorCode.CONFLICT,
            message="account-scoped artifact or execution state conflicts with this request",
            service=service,
            version=version,
        )
    if isinstance(error, KillSwitchOrderBlocked):
        return _failure(
            context,
            code=ErrorCode.KILL_SWITCH_ACTIVE,
            message="paper submission is blocked by the independent kill switch",
            service=service,
            version=version,
        )
    if isinstance(error, (DraftArtifactStoreError, PaperRepositoryError)):
        return _failure(
            context,
            code=ErrorCode.INTERNAL_ERROR,
            message="trusted artifact persistence is unavailable",
            service=service,
            version=version,
        )
    return _failure(
        context,
        code=ErrorCode.DATA_INVALID,
        message="portfolio or execution inputs failed decision-bound validation",
        service=service,
        version=version,
    )


def _position(value: PortfolioPositionSnapshot) -> PortfolioPositionOutput:
    return PortfolioPositionOutput(
        instrument_id=value.instrument_id,
        instrument_type=value.instrument_type,
        total_quantity=value.total_quantity,
        available_quantity=value.available_quantity,
        frozen_quantity=value.frozen_quantity,
        unsettled_quantity=value.unsettled_quantity,
        average_cost=value.average_cost,
        valuation_price=value.valuation_price,
        price_observed_at=value.price_observed_at,
        price_available_at=value.price_available_at,
        valuation_status=value.valuation_status,
        market_value=value.market_value,
        cost_basis=value.cost_basis,
        unrealized_pnl=value.unrealized_pnl,
        price_data_version=value.price_data_version,
        price_source_hash=value.price_source_hash,
        mark_policy_version=value.mark_policy_version,
        mark_policy_hash=value.mark_policy_hash,
    )


def _contribution(value: StrategyWeightContribution) -> StrategyContributionOutput:
    return StrategyContributionOutput(
        sleeve_id=value.sleeve_id,
        sleeve_hash=value.sleeve_hash,
        sleeve_kind=value.sleeve_kind,
        source_result_hash=value.source_result_hash,
        local_target_weight=value.local_target_weight,
        sleeve_budget=value.sleeve_budget,
        proposed_portfolio_weight=value.proposed_portfolio_weight,
        applied_portfolio_weight=value.applied_portfolio_weight,
        reason=value.reason,
    )


def _target_line(value: PortfolioTargetLine) -> PortfolioTargetLineOutput:
    return PortfolioTargetLineOutput(
        instrument_id=value.instrument_id,
        instrument_type=value.instrument_type,
        industry_id=value.industry_id,
        industry_classification_hash=value.industry_classification_hash,
        current_market_value=value.current_market_value,
        current_weight=value.current_weight,
        proposed_target_weight=value.proposed_target_weight,
        target_weight=value.target_weight,
        target_market_value=value.target_market_value,
        weight_delta=value.weight_delta,
        absolute_weight_delta=value.absolute_weight_delta,
        contributions=tuple(_contribution(item) for item in value.contributions),
        rationale=value.rationale,
    )


def _adjustment(value: PortfolioConstraintAdjustment) -> PortfolioAdjustmentOutput:
    return PortfolioAdjustmentOutput(
        code=value.code,
        scope_id=value.scope_id,
        before_weight=value.before_weight,
        after_weight=value.after_weight,
        rationale=value.rationale,
    )


def _risk_finding(value: RiskFinding) -> RiskFindingOutput:
    return RiskFindingOutput(
        rule_code=value.rule_code,
        rule_version=value.rule_version,
        severity=value.severity,
        scope=value.scope,
        scope_id=value.scope_id,
        instrument_id=value.instrument_id,
        observed_value=value.observed_value,
        limit_value=value.limit_value,
    )


def _draft_line(value: OrderDraftLine) -> OrderDraftLineOutput:
    return OrderDraftLineOutput(
        instrument_id=value.instrument_id,
        instrument_type=value.instrument_type,
        side=value.side,
        current_quantity=value.current_quantity,
        target_quantity=value.target_quantity,
        quantity=value.quantity,
        is_full_liquidation=value.is_full_liquidation,
        reference_price=value.reference_price,
        estimated_execution_price=value.estimated_execution_price,
        cash_reservation_price=value.cash_reservation_price,
        price_observed_at=value.price_observed_at,
        price_available_at=value.price_available_at,
        data_version=value.data_version,
        state_revision=value.state_revision,
        state_hash=value.state_hash,
        participation_rate=value.participation_rate,
        gross_amount=value.gross_amount,
        estimated_slippage_amount=value.estimated_slippage_amount,
        commission=value.commission,
        stamp_duty=value.stamp_duty,
        transfer_fee=value.transfer_fee,
        other_fee=value.other_fee,
        total_fee=value.total_fee,
        cash_reservation_fee=value.cash_reservation_fee,
        reserved_cash=value.reserved_cash,
        estimated_cash_change=value.estimated_cash_change,
        market_rule_version=value.market_rule_version,
        market_rule_hash=value.market_rule_hash,
        fee_rule_version=value.fee_rule_version,
        fee_rule_hash=value.fee_rule_hash,
        slippage_model_version=value.slippage_model_version,
        slippage_model_hash=value.slippage_model_hash,
        rationale=value.rationale,
        line_hash=value.line_hash,
    )


def _paper_order(value: PaperOrder) -> PaperOrderOutput:
    return PaperOrderOutput(
        order_id=value.order_id,
        line_hash=value.line_hash,
        instrument_id=value.instrument_id,
        instrument_type=value.instrument_type,
        side=value.side,
        quantity=value.quantity,
        filled_quantity=value.filled_quantity,
        remaining_quantity=value.remaining_quantity,
        status=value.status,
        is_full_liquidation=value.is_full_liquidation,
        created_at=value.created_at,
        expires_at=value.expires_at,
        cash_reserved=value.cash_reserved,
        sell_reserved_quantity=value.sell_reserved_quantity,
        last_attempt_id=value.last_attempt_id,
        order_hash=value.order_hash,
    )


def _paper_attempt(value: PaperMatchAttempt) -> PaperAttemptOutput:
    return PaperAttemptOutput(
        attempt_id=value.attempt_id,
        order_id=value.order_id,
        attempted_at=value.attempted_at,
        status=value.status,
        no_fill_reason=value.no_fill_reason,
        requested_quantity=value.requested_quantity,
        filled_quantity=value.filled_quantity,
        unfilled_quantity=value.unfilled_quantity,
        reference_price=value.reference_price,
        execution_price=value.execution_price,
        gross_amount=value.gross_amount,
        participation_rate=value.participation_rate,
        state_hash=value.state_hash,
        attempt_hash=value.attempt_hash,
    )


def _paper_fill(value: PaperFill) -> PaperFillOutput:
    return PaperFillOutput(
        fill_id=value.fill_id,
        order_id=value.order_id,
        attempt_id=value.attempt_id,
        instrument_id=value.instrument_id,
        instrument_type=value.instrument_type,
        filled_at=value.filled_at,
        side=value.side,
        quantity=value.quantity,
        price=value.price,
        gross_amount=value.gross_amount,
        commission=value.commission,
        stamp_duty=value.stamp_duty,
        transfer_fee=value.transfer_fee,
        other_fee=value.other_fee,
        total_fee=value.total_fee,
        cash_change=value.cash_change,
        sellable_on=value.sellable_on,
        fill_hash=value.fill_hash,
    )


def _paper_position(value: PaperPosition) -> PaperPositionOutput:
    return PaperPositionOutput(
        instrument_id=value.instrument_id,
        instrument_type=value.instrument_type,
        total_quantity=value.total_quantity,
        available_quantity=value.available_quantity,
        frozen_quantity=value.frozen_quantity,
        unsettled_quantity=value.unsettled_quantity,
        reserved_sell_quantity=value.reserved_sell_quantity,
        cost_basis=value.cost_basis,
        average_cost=value.average_cost,
        position_hash=value.position_hash,
    )


def _paper_account(value: PaperAccountState) -> PaperAccountStateOutput:
    positions = tuple(_paper_position(item) for item in value.positions[:_MAX_PAPER_POSITIONS])
    return PaperAccountStateOutput(
        as_of=value.as_of,
        total_cash=value.total_cash,
        available_cash=value.available_cash,
        frozen_cash=value.frozen_cash,
        external_frozen_cash=value.external_frozen_cash,
        state_hash=value.state_hash,
        event_log_hash=value.event_log_hash,
        total_position_count=len(value.positions),
        returned_position_count=len(positions),
        positions_truncated=len(positions) < len(value.positions),
        positions=positions,
    )


def _reconciliation_finding(value: ReconciliationFinding) -> ReconciliationFindingOutput:
    return ReconciliationFindingOutput(
        finding_id=value.finding_id,
        domain=value.domain,
        code=value.code,
        status=value.status,
        severity=value.severity,
        entity_id=value.entity_id,
        message=value.message,
        line_hash=value.line_hash,
        order_id=value.order_id,
        fill_id=value.fill_id,
        instrument_id=value.instrument_id,
        expected_value=value.expected_value,
        observed_value=value.observed_value,
        delta=value.delta,
        evidence_hashes=value.evidence_hashes,
        finding_hash=value.finding_hash,
    )


def _order_check(value: OrderReconciliation) -> ReconciliationCheckOutput:
    return ReconciliationCheckOutput(
        check_id=value.check_id,
        domain=ReconciliationDomain.ORDER,
        status=value.status,
        severity=value.severity,
        instrument_id=value.instrument_id,
        order_id=value.order_id,
        check_hash=value.check_hash,
    )


def _fill_check(value: FillReconciliation) -> ReconciliationCheckOutput:
    return ReconciliationCheckOutput(
        check_id=value.check_id,
        domain=ReconciliationDomain.FILL,
        status=value.status,
        severity=value.severity,
        instrument_id=value.instrument_id,
        order_id=value.order_id,
        check_hash=value.check_hash,
    )


def _cash_check(value: CashReconciliation) -> ReconciliationCheckOutput:
    return ReconciliationCheckOutput(
        check_id=value.check_id,
        domain=ReconciliationDomain.CASH,
        status=value.status,
        severity=value.severity,
        instrument_id=None,
        order_id=None,
        check_hash=value.check_hash,
    )


def _position_check(value: PositionReconciliation) -> ReconciliationCheckOutput:
    return ReconciliationCheckOutput(
        check_id=value.check_id,
        domain=ReconciliationDomain.POSITION,
        status=value.status,
        severity=value.severity,
        instrument_id=value.instrument_id,
        order_id=None,
        check_hash=value.check_hash,
    )


def _all_findings(value: ReconciliationResult) -> tuple[ReconciliationFinding, ...]:
    return (
        value.input_findings
        + tuple(item for check in value.order_checks for item in check.findings)
        + tuple(item for check in value.fill_checks for item in check.findings)
        + value.cash_check.findings
        + tuple(item for check in value.position_checks for item in check.findings)
    )


class PortfolioExecutionToolset:
    """Account-scoped handlers with hash-linked ordering between every stage."""

    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: PortfolioExecutionPipeline) -> None:
        if type(pipeline) is not PortfolioExecutionPipeline:
            raise TypeError("toolset requires an exact PortfolioExecutionPipeline")
        self._pipeline = pipeline

    def get_portfolio_snapshot(
        self,
        context: ToolExecutionContext,
        arguments: GetPortfolioSnapshotArguments,
    ) -> ToolResponse[PortfolioSnapshotOutput]:
        try:
            value = self._pipeline.get_account_snapshot(context.decision_snapshot)
            data = self._portfolio_snapshot(context, arguments, value)
        except (
            PortfolioExecutionInputUnavailable,
            PortfolioExecutionInputInvalid,
            ValidationError,
        ) as error:
            return _expected_failure(
                context,
                error,
                service=_ACCOUNT_SERVICE,
                version="account-snapshot-v1",
            )
        return _success(
            context,
            data,
            service=_ACCOUNT_SERVICE,
            version="account-snapshot-v1",
        )

    def build_target_portfolio(
        self,
        context: ToolExecutionContext,
        arguments: BuildTargetPortfolioArguments,
    ) -> ToolResponse[TargetPortfolioOutput]:
        version = self._pipeline.builder_version
        try:
            value = self._pipeline.build_target_portfolio(context.decision_snapshot)
            data = self._target(context, arguments, value)
        except (
            PortfolioExecutionInputUnavailable,
            PortfolioExecutionInputInvalid,
            ValidationError,
        ) as error:
            return _expected_failure(
                context,
                error,
                service=_TARGET_SERVICE,
                version=version,
            )
        return _success(context, data, service=_TARGET_SERVICE, version=version)

    def check_portfolio_risk(
        self,
        context: ToolExecutionContext,
        arguments: CheckPortfolioRiskArguments,
    ) -> ToolResponse[PortfolioRiskOutput]:
        version = self._pipeline.risk_version
        try:
            value = self._pipeline.check_portfolio_risk(context.decision_snapshot)
            data = self._risk(context, arguments.max_findings, value)
        except (
            PortfolioExecutionInputUnavailable,
            PortfolioExecutionInputInvalid,
            ValidationError,
        ) as error:
            return _expected_failure(
                context,
                error,
                service=_RISK_SERVICE,
                version=version,
            )
        if value.status is RiskCheckStatus.REJECT:
            return _failure(
                context,
                code=ErrorCode.RISK_REJECTED,
                message="independent portfolio risk policy rejected the target",
                service=_RISK_SERVICE,
                version=version,
                data=data,
            )
        if value.status is RiskCheckStatus.ERROR:
            return _failure(
                context,
                code=ErrorCode.DATA_INVALID,
                message="independent portfolio risk evaluation failed closed",
                service=_RISK_SERVICE,
                version=version,
                data=data,
            )
        return _success(context, data, service=_RISK_SERVICE, version=version)

    def create_order_draft(
        self,
        context: ToolExecutionContext,
        arguments: CreateOrderDraftArguments,
    ) -> ToolResponse[OrderDraftOutput]:
        version = self._pipeline.draft_version
        try:
            value = self._pipeline.create_order_draft(
                context.decision_snapshot,
                idempotency_key=arguments.idempotency_key,
                draft_as_of=context.requested_at,
            )
            data = self._draft(context, arguments.max_lines, value)
        except PortfolioRiskBlocked as error:
            code = (
                ErrorCode.RISK_REJECTED
                if error.result.status is RiskCheckStatus.REJECT
                else ErrorCode.DATA_INVALID
            )
            return _failure(
                context,
                code=code,
                message="independent portfolio risk result blocks order-draft creation",
                service=_DRAFT_SERVICE,
                version=version,
            )
        except (
            PortfolioExecutionInputUnavailable,
            PortfolioExecutionInputInvalid,
            DraftArtifactStoreError,
            ValidationError,
        ) as error:
            return _expected_failure(
                context,
                error,
                service=_DRAFT_SERVICE,
                version=version,
            )
        return _success(context, data, service=_DRAFT_SERVICE, version=version)

    def get_order_draft(
        self,
        context: ToolExecutionContext,
        arguments: GetOrderDraftArguments,
    ) -> ToolResponse[OrderDraftOutput]:
        version = self._pipeline.draft_version
        try:
            value = self._pipeline.get_order_draft(
                context.decision_snapshot,
                batch_hash=arguments.batch_hash,
            )
            data = self._draft(context, arguments.max_lines, value)
        except (
            PortfolioExecutionInputInvalid,
            DraftArtifactStoreError,
            ValidationError,
        ) as error:
            return _expected_failure(
                context,
                error,
                service=_DRAFT_SERVICE,
                version=version,
            )
        return _success(context, data, service=_DRAFT_SERVICE, version=version)

    def submit_paper_orders(
        self,
        context: ToolExecutionContext,
        arguments: SubmitPaperOrdersArguments,
    ) -> ToolResponse[PaperExecutionOutput]:
        version = self._pipeline.paper_version or "paper-gateway-unavailable"
        try:
            value = self._pipeline.submit_paper_orders(
                context.decision_snapshot,
                request_id=context.request_id,
                idempotency_key=arguments.idempotency_key,
                batch_hash=arguments.batch_hash,
                submitted_at=context.requested_at,
            )
            data = self._paper(context, arguments, value)
        except (
            PortfolioExecutionInputUnavailable,
            PortfolioExecutionInputInvalid,
            DraftArtifactStoreError,
            PaperExecutionInputError,
            PaperRepositoryError,
            KillSwitchOrderBlocked,
            ValidationError,
        ) as error:
            return _expected_failure(
                context,
                error,
                service=_PAPER_SERVICE,
                version=version,
            )
        return _success(context, data, service=_PAPER_SERVICE, version=version)

    def reconcile_account(
        self,
        context: ToolExecutionContext,
        arguments: ReconcileAccountArguments,
    ) -> ToolResponse[ReconciliationOutput]:
        try:
            value = self._pipeline.reconcile_account(
                context.decision_snapshot,
                batch_hash=arguments.batch_hash,
                requested_at=context.requested_at,
            )
            data = self._reconciliation(context, arguments, value)
        except (
            PortfolioExecutionInputUnavailable,
            PortfolioExecutionInputInvalid,
            DraftArtifactStoreError,
            ValidationError,
        ) as error:
            return _expected_failure(
                context,
                error,
                service=_RECONCILIATION_SERVICE,
                version=RECONCILIATION_ENGINE_VERSION,
            )
        return _success(
            context,
            data,
            service=_RECONCILIATION_SERVICE,
            version=RECONCILIATION_ENGINE_VERSION,
        )

    @staticmethod
    def _portfolio_snapshot(
        context: ToolExecutionContext,
        arguments: GetPortfolioSnapshotArguments,
        value: AccountSnapshot,
    ) -> PortfolioSnapshotOutput:
        positions = tuple(_position(item) for item in value.positions[: arguments.max_positions])
        snapshot = context.decision_snapshot
        return PortfolioSnapshotOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            runtime_mode=value.runtime_mode,
            valuation_at=value.valuation_at,
            currency=value.currency,
            cash=CashSnapshotOutput(
                currency=value.cash.currency,
                total_cash=value.cash.total_cash,
                available_cash=value.cash.available_cash,
                frozen_cash=value.cash.frozen_cash,
            ),
            position_market_value=value.position_market_value,
            total_equity=value.total_equity,
            gross_exposure=value.gross_exposure,
            cash_weight=value.cash_weight,
            previous_snapshot_hash=value.previous_snapshot_hash,
            source_event_log_hash=value.source_event_log_hash,
            content_hash=value.content_hash,
            total_position_count=len(value.positions),
            returned_position_count=len(positions),
            positions_truncated=len(positions) < len(value.positions),
            positions=positions,
        )

    @staticmethod
    def _target(
        context: ToolExecutionContext,
        arguments: BuildTargetPortfolioArguments,
        value: TargetPortfolio,
    ) -> TargetPortfolioOutput:
        lines = tuple(_target_line(item) for item in value.lines[: arguments.max_lines])
        adjustments = tuple(
            _adjustment(item) for item in value.adjustments[: arguments.max_adjustments]
        )
        snapshot = context.decision_snapshot
        return TargetPortfolioOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            builder_version=value.builder_version,
            config_version=value.config_version,
            config_hash=value.config_hash,
            total_equity=value.total_equity,
            etf_core_budget=value.etf_core_budget,
            stock_enhancement_budget=value.stock_enhancement_budget,
            minimum_cash_weight=value.minimum_cash_weight,
            maximum_instrument_weight=value.maximum_instrument_weight,
            maximum_stock_industry_weight=value.maximum_stock_industry_weight,
            current_gross_weight=value.current_gross_weight,
            current_cash_weight=value.current_cash_weight,
            target_gross_weight=value.target_gross_weight,
            target_cash_weight=value.target_cash_weight,
            etf_target_weight=value.etf_target_weight,
            stock_target_weight=value.stock_target_weight,
            gross_instrument_turnover=value.gross_instrument_turnover,
            one_way_turnover=value.one_way_turnover,
            sleeve_hashes=value.sleeve_hashes,
            classification_hashes=value.classification_hashes,
            classification_input_hash=value.classification_input_hash,
            input_hash=value.input_hash,
            result_hash=value.result_hash,
            total_line_count=len(value.lines),
            returned_line_count=len(lines),
            lines_truncated=len(lines) < len(value.lines),
            lines=lines,
            total_adjustment_count=len(value.adjustments),
            returned_adjustment_count=len(adjustments),
            adjustments_truncated=len(adjustments) < len(value.adjustments),
            adjustments=adjustments,
        )

    @staticmethod
    def _risk(
        context: ToolExecutionContext,
        max_findings: int,
        value: RiskCheckResult,
    ) -> PortfolioRiskOutput:
        findings = tuple(_risk_finding(item) for item in value.findings[:max_findings])
        snapshot = context.decision_snapshot
        return PortfolioRiskOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            status=value.status,
            evaluated_at=value.evaluated_at,
            risk_engine_version=value.risk_engine_version,
            error_code=value.error_code,
            passed=value.passed,
            allows_execution=value.allows_execution,
            request=RiskRequestIdentityOutput(
                request_hash=value.request.request_hash,
                portfolio_proposal_hash=value.request.portfolio_proposal_hash,
                risk_context_hash=value.request.risk_context_hash,
                valuation_version=value.request.valuation_version,
                policy_version=value.request.policy_version,
                policy_hash=value.request.policy_hash,
            ),
            result_hash=value.result_hash,
            total_finding_count=len(value.findings),
            returned_finding_count=len(findings),
            findings_truncated=len(findings) < len(value.findings),
            findings=findings,
        )

    @staticmethod
    def _draft(
        context: ToolExecutionContext,
        max_lines: int,
        value: OrderDraftBatch,
    ) -> OrderDraftOutput:
        lines = tuple(_draft_line(item) for item in value.lines[:max_lines])
        snapshot = context.decision_snapshot
        return OrderDraftOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            draft_as_of=value.draft_as_of,
            trading_day=value.trading_day,
            expires_at=value.expires_at,
            validity_seconds=value.validity_seconds,
            runtime_mode=value.runtime_mode,
            currency=value.currency,
            portfolio_proposal_hash=value.portfolio_proposal_hash,
            risk_request_hash=value.risk_request_hash,
            risk_result_hash=value.risk_result_hash,
            risk_status=value.risk_status,
            risk_checked_at=value.risk_checked_at,
            risk_engine_version=value.risk_engine_version,
            risk_policy_version=value.risk_policy_version,
            risk_policy_hash=value.risk_policy_hash,
            generator_version=value.generator_version,
            generator_config_version=value.generator_config_version,
            generator_config_hash=value.generator_config_hash,
            funding_policy=value.funding_policy,
            available_cash=value.available_cash,
            reserved_cash_required=value.reserved_cash_required,
            estimated_buy_cash_required=value.estimated_buy_cash_required,
            estimated_sell_cash_proceeds=value.estimated_sell_cash_proceeds,
            estimated_total_fees=value.estimated_total_fees,
            estimated_total_slippage=value.estimated_total_slippage,
            state_hashes=value.state_hashes,
            market_rule_hashes=value.market_rule_hashes,
            fee_rule_hashes=value.fee_rule_hashes,
            slippage_model_hashes=value.slippage_model_hashes,
            input_hash=value.input_hash,
            batch_hash=value.batch_hash,
            batch_id=value.batch_id,
            is_executable=False,
            requires_human_approval=True,
            total_line_count=len(value.lines),
            returned_line_count=len(lines),
            lines_truncated=len(lines) < len(value.lines),
            lines=lines,
        )

    @staticmethod
    def _paper(
        context: ToolExecutionContext,
        arguments: SubmitPaperOrdersArguments,
        value: PaperExecutionReceipt,
    ) -> PaperExecutionOutput:
        orders = tuple(_paper_order(item) for item in value.orders[: arguments.max_orders])
        attempts = tuple(_paper_attempt(item) for item in value.attempts[: arguments.max_orders])
        fills = tuple(_paper_fill(item) for item in value.fills[: arguments.max_fills])
        snapshot = context.decision_snapshot
        return PaperExecutionOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            receipt_id=value.receipt_id,
            request_hash=value.request_hash,
            batch_hash=value.batch_hash,
            request_submitted_at=value.request_submitted_at,
            processed_at=value.processed_at,
            account_before_hash=value.account_before_hash,
            account_after_hash=value.account_after_hash,
            event_log_before_hash=value.event_log_before_hash,
            event_log_hash=value.event_log_hash,
            receipt_hash=value.receipt_hash,
            account_before=_paper_account(value.account_before),
            account_after=_paper_account(value.account_after),
            total_order_count=len(value.orders),
            returned_order_count=len(orders),
            orders_truncated=len(orders) < len(value.orders),
            orders=orders,
            total_attempt_count=len(value.attempts),
            returned_attempt_count=len(attempts),
            attempts_truncated=len(attempts) < len(value.attempts),
            attempts=attempts,
            total_fill_count=len(value.fills),
            returned_fill_count=len(fills),
            fills_truncated=len(fills) < len(value.fills),
            fills=fills,
        )

    @staticmethod
    def _reconciliation(
        context: ToolExecutionContext,
        arguments: ReconcileAccountArguments,
        value: ReconciliationResult,
    ) -> ReconciliationOutput:
        all_findings = _all_findings(value)
        findings = tuple(
            _reconciliation_finding(item) for item in all_findings[: arguments.max_findings]
        )
        all_checks = (
            tuple(_order_check(item) for item in value.order_checks)
            + tuple(_fill_check(item) for item in value.fill_checks)
            + (_cash_check(value.cash_check),)
            + tuple(_position_check(item) for item in value.position_checks)
        )
        checks = all_checks[: arguments.max_checks]
        cash = value.cash_check
        stop = value.stop_signal
        snapshot = context.decision_snapshot
        return ReconciliationOutput(
            decision_id=snapshot.decision_id,
            as_of=snapshot.as_of,
            data_version=snapshot.data_version,
            account_id=snapshot.account_id,
            account_snapshot_id=snapshot.account_snapshot_id,
            account_snapshot_hash=snapshot.account_snapshot_hash,
            reconciliation_id=value.reconciliation_id,
            engine_version=value.engine_version,
            input_hash=value.input_hash,
            batch_hash=value.batch_hash,
            draft_hash=value.draft_hash,
            receipt_id=value.receipt_id,
            receipt_hash=value.receipt_hash,
            expected_account_before_hash=value.expected_account_before_hash,
            expected_account_after_hash=value.expected_account_after_hash,
            observed_snapshot_id=value.observed_snapshot_id,
            observed_snapshot_hash=value.observed_snapshot_hash,
            observed_as_of=value.observed_as_of,
            reconciled_at=value.reconciled_at,
            policy_version=value.policy_version,
            policy_hash=value.policy_hash,
            status=value.status,
            max_severity=value.max_severity,
            result_hash=value.result_hash,
            total_finding_count=len(all_findings),
            returned_finding_count=len(findings),
            findings_truncated=len(findings) < len(all_findings),
            findings=findings,
            total_check_count=len(all_checks),
            returned_check_count=len(checks),
            checks_truncated=len(checks) < len(all_checks),
            checks=checks,
            duplicate_group_count=len(value.duplicate_groups),
            cash=CashReconciliationOutput(
                status=cash.status,
                severity=cash.severity,
                expected_total_cash=cash.expected_total_cash,
                observed_total_cash=cash.observed_total_cash,
                total_delta=cash.total_delta,
                expected_available_cash=cash.expected_available_cash,
                observed_available_cash=cash.observed_available_cash,
                available_delta=cash.available_delta,
                expected_frozen_cash=cash.expected_frozen_cash,
                observed_frozen_cash=cash.observed_frozen_cash,
                frozen_delta=cash.frozen_delta,
                mismatched_fields=cash.mismatched_fields,
                check_hash=cash.check_hash,
            ),
            stop_signal=ReconciliationStopSignalOutput(
                required=stop.required,
                action=stop.action,
                scope=stop.scope,
                severity=stop.severity,
                reason_codes=stop.reason_codes,
                trigger_hashes=stop.trigger_hashes,
                generated_at=stop.generated_at,
                signal_hash=stop.signal_hash,
            ),
        )


def build_portfolio_execution_tools(
    *,
    source: PortfolioExecutionInputSource,
    artifact_store: DraftArtifactStore,
    draft_generator: OrderDraftGenerator,
    builder: TargetPortfolioBuilder | None = None,
    risk_evaluator: PortfolioRiskEvaluator | None = None,
    reconciliation_engine: ReconciliationEngine | None = None,
    paper_gateway: PaperOrderGateway | None = None,
) -> tuple[AgentTool[Any, Any], ...]:
    """Build six core tools and the paper-submit tool only when its gateway exists."""

    pipeline = PortfolioExecutionPipeline(
        source=source,
        artifact_store=artifact_store,
        draft_generator=draft_generator,
        builder=builder,
        risk_evaluator=risk_evaluator,
        reconciliation_engine=reconciliation_engine,
        paper_gateway=paper_gateway,
    )
    toolset = PortfolioExecutionToolset(pipeline)
    tools: list[AgentTool[Any, Any]] = [
        AgentTool(
            name="get_portfolio_snapshot",
            version=PORTFOLIO_EXECUTION_TOOL_VERSION,
            description="Return the exact decision-bound account and position snapshot.",
            arguments_type=GetPortfolioSnapshotArguments,
            output_type=PortfolioSnapshotOutput,
            handler=toolset.get_portfolio_snapshot,
        ),
        AgentTool(
            name="build_target_portfolio",
            version=PORTFOLIO_EXECUTION_TOOL_VERSION,
            description="Build a non-executable target from frozen strategy sleeves.",
            arguments_type=BuildTargetPortfolioArguments,
            output_type=TargetPortfolioOutput,
            handler=toolset.build_target_portfolio,
        ),
        AgentTool(
            name="check_portfolio_risk",
            version=PORTFOLIO_EXECUTION_TOOL_VERSION,
            description="Run independent fail-closed portfolio risk checks.",
            arguments_type=CheckPortfolioRiskArguments,
            output_type=PortfolioRiskOutput,
            handler=toolset.check_portfolio_risk,
        ),
        AgentTool(
            name="create_order_draft",
            version=PORTFOLIO_EXECUTION_TOOL_VERSION,
            description="Persist an idempotent, non-executable order draft after risk.",
            arguments_type=CreateOrderDraftArguments,
            output_type=OrderDraftOutput,
            handler=toolset.create_order_draft,
        ),
        AgentTool(
            name="get_order_draft",
            version=PORTFOLIO_EXECUTION_TOOL_VERSION,
            description="Read one account- and decision-bound stored order draft.",
            arguments_type=GetOrderDraftArguments,
            output_type=OrderDraftOutput,
            handler=toolset.get_order_draft,
        ),
        AgentTool(
            name="reconcile_account",
            version=PORTFOLIO_EXECUTION_TOOL_VERSION,
            description="Compare a submitted draft with independent actual observations.",
            arguments_type=ReconcileAccountArguments,
            output_type=ReconciliationOutput,
            handler=toolset.reconcile_account,
        ),
    ]
    if paper_gateway is not None:
        tools.insert(
            5,
            AgentTool(
                name="submit_paper_orders",
                version=PORTFOLIO_EXECUTION_TOOL_VERSION,
                description="Submit a stored draft through guarded paper execution only.",
                arguments_type=SubmitPaperOrdersArguments,
                output_type=PaperExecutionOutput,
                handler=toolset.submit_paper_orders,
            ),
        )
    return tuple(tools)


__all__ = [
    "PORTFOLIO_EXECUTION_TOOL_VERSION",
    "PortfolioExecutionToolset",
    "build_portfolio_execution_tools",
]
