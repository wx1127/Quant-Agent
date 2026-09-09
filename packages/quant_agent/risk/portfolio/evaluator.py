"""Independent, fail-closed portfolio risk evaluation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from quant_agent.backtest import TradableInstrumentType
from quant_agent.core.time import ensure_aware
from quant_agent.portfolio import AccountSnapshot, TargetPortfolio
from quant_agent.risk.contracts import (
    RiskCheckRequest,
    RiskCheckResult,
    RiskCheckStatus,
    RiskFinding,
    RuleScope,
    RuleSeverity,
)

from .contracts import (
    PORTFOLIO_RISK_ENGINE_VERSION,
    PortfolioRiskContext,
    PortfolioRiskInputError,
    PortfolioRiskPolicy,
    PortfolioRiskRuleCode,
)

_INPUT_ERROR_CODE = "PORTFOLIO_RISK_INPUT_MISMATCH"
_EVALUATION_ERROR_CODE = "PORTFOLIO_RISK_EVALUATION_FAILED"


class PortfolioRiskEvaluator:
    """Recheck target risk independently from strategy and allocation shaping."""

    def __init__(self, policy: PortfolioRiskPolicy | None = None) -> None:
        self._policy = policy or PortfolioRiskPolicy()

    @property
    def policy(self) -> PortfolioRiskPolicy:
        """Return the immutable policy used for every emitted finding."""

        return self._policy

    def check(
        self,
        *,
        request: RiskCheckRequest,
        account: AccountSnapshot,
        proposal: TargetPortfolio,
        context: PortfolioRiskContext,
        evaluated_at: datetime,
    ) -> RiskCheckResult:
        """Return PASS/WARN/REJECT, or an explicit fail-closed ERROR result."""

        try:
            ensure_aware(evaluated_at)
            if evaluated_at < request.account_snapshot_as_of:
                raise PortfolioRiskInputError("risk evaluation cannot precede its account snapshot")
            self._validate_inputs(
                request=request,
                account=account,
                proposal=proposal,
                context=context,
            )
            findings = self._evaluate_rules(proposal=proposal, context=context)
            status = self._status(findings)
            return RiskCheckResult.build(
                request=request,
                status=status,
                evaluated_at=evaluated_at,
                risk_engine_version=PORTFOLIO_RISK_ENGINE_VERSION,
                findings=findings,
            )
        except PortfolioRiskInputError:
            return RiskCheckResult.service_error(
                request=request,
                evaluated_at=self._safe_error_time(request, evaluated_at),
                risk_engine_version=PORTFOLIO_RISK_ENGINE_VERSION,
                error_code=_INPUT_ERROR_CODE,
            )
        except Exception:
            return RiskCheckResult.service_error(
                request=request,
                evaluated_at=self._safe_error_time(request, evaluated_at),
                risk_engine_version=PORTFOLIO_RISK_ENGINE_VERSION,
                error_code=_EVALUATION_ERROR_CODE,
            )

    def _validate_inputs(
        self,
        *,
        request: RiskCheckRequest,
        account: AccountSnapshot,
        proposal: TargetPortfolio,
        context: PortfolioRiskContext,
    ) -> None:
        expected = (
            (request.decision_id, proposal.decision_id, "decision id"),
            (request.account_snapshot_id, account.snapshot_id, "account snapshot id"),
            (
                request.account_snapshot_hash,
                account.content_hash,
                "account snapshot hash",
            ),
            (
                request.account_snapshot_as_of,
                account.as_of,
                "account snapshot as_of",
            ),
            (request.portfolio_proposal_hash, proposal.result_hash, "proposal hash"),
            (request.data_version, account.data_version, "account data version"),
            (request.data_version, proposal.data_version, "proposal data version"),
            (request.policy_version, self._policy.version, "policy version"),
            (request.policy_hash, self._policy.policy_hash, "policy hash"),
            (request.risk_context_hash, context.context_hash, "risk context hash"),
            (
                request.valuation_version,
                context.valuation_version,
                "valuation version",
            ),
            (proposal.account_snapshot_id, account.snapshot_id, "proposal account id"),
            (
                proposal.account_snapshot_hash,
                account.content_hash,
                "proposal account hash",
            ),
            (proposal.as_of, account.as_of, "proposal as_of"),
            (
                context.account_snapshot_hash,
                account.content_hash,
                "context account hash",
            ),
            (context.as_of, account.as_of, "context as_of"),
            (context.current_equity, account.total_equity, "context equity"),
            (proposal.total_equity, account.total_equity, "proposal equity"),
            (
                proposal.current_gross_weight,
                account.gross_exposure,
                "proposal current gross weight",
            ),
            (
                proposal.current_cash_weight,
                account.cash_weight,
                "proposal current cash weight",
            ),
        )
        for left, right, name in expected:
            if left != right:
                raise PortfolioRiskInputError(f"{name} does not align")

        current_lines = {
            line.instrument_id: line for line in proposal.lines if line.current_market_value > 0
        }
        account_positions = {item.instrument_id: item for item in account.positions}
        if set(current_lines) != set(account_positions):
            raise PortfolioRiskInputError(
                "proposal lines do not exactly cover current account positions"
            )
        for instrument_id, position in account_positions.items():
            line = current_lines[instrument_id]
            if (
                line.instrument_type is not position.instrument_type
                or line.current_market_value != position.market_value
            ):
                raise PortfolioRiskInputError(
                    f"proposal current position {instrument_id} does not match account"
                )

    def _evaluate_rules(
        self,
        *,
        proposal: TargetPortfolio,
        context: PortfolioRiskContext,
    ) -> tuple[RiskFinding, ...]:
        findings: list[RiskFinding] = []
        self._append_limit_finding(
            findings,
            code=PortfolioRiskRuleCode.MAX_TOTAL_EXPOSURE,
            scope=RuleScope.PORTFOLIO,
            scope_id="TOTAL",
            observed=proposal.target_gross_weight,
            current=proposal.current_gross_weight,
            limit=self._policy.maximum_total_weight,
        )
        self._append_limit_finding(
            findings,
            code=PortfolioRiskRuleCode.REGIME_RISK_BUDGET,
            scope=RuleScope.PORTFOLIO,
            scope_id="REGIME",
            observed=proposal.target_gross_weight,
            current=proposal.current_gross_weight,
            limit=context.regime_risk_budget,
        )

        current_etf = self._sum_lines(
            proposal,
            instrument_type=TradableInstrumentType.ETF,
            target=False,
        )
        current_stock = self._sum_lines(
            proposal,
            instrument_type=TradableInstrumentType.STOCK,
            target=False,
        )
        self._append_limit_finding(
            findings,
            code=PortfolioRiskRuleCode.MAX_ETF_EXPOSURE,
            scope=RuleScope.PORTFOLIO,
            scope_id="ETF",
            observed=proposal.etf_target_weight,
            current=current_etf,
            limit=self._policy.maximum_etf_weight,
        )
        self._append_limit_finding(
            findings,
            code=PortfolioRiskRuleCode.MAX_STOCK_EXPOSURE,
            scope=RuleScope.PORTFOLIO,
            scope_id="STOCK",
            observed=proposal.stock_target_weight,
            current=current_stock,
            limit=self._policy.maximum_stock_weight,
        )

        for line in proposal.lines:
            if line.instrument_type is TradableInstrumentType.ETF:
                code = PortfolioRiskRuleCode.MAX_SINGLE_ETF_WEIGHT
                limit = self._policy.maximum_single_etf_weight
            else:
                code = PortfolioRiskRuleCode.MAX_SINGLE_STOCK_WEIGHT
                limit = self._policy.maximum_single_stock_weight
            self._append_limit_finding(
                findings,
                code=code,
                scope=RuleScope.INSTRUMENT,
                scope_id=None,
                instrument_id=line.instrument_id,
                observed=line.target_weight,
                current=line.current_weight,
                limit=limit,
            )

        current_industries = self._industry_weights(proposal, target=False)
        for industry_id, target_weight in self._industry_weights(
            proposal,
            target=True,
        ).items():
            self._append_limit_finding(
                findings,
                code=PortfolioRiskRuleCode.MAX_STOCK_INDUSTRY_WEIGHT,
                scope=RuleScope.INDUSTRY,
                scope_id=industry_id,
                observed=target_weight,
                current=current_industries.get(industry_id, Decimal(0)),
                limit=self._policy.maximum_stock_industry_weight,
            )

        adds_new_risk = any(line.weight_delta > 0 for line in proposal.lines)
        if proposal.one_way_turnover > self._policy.maximum_one_way_turnover:
            severity = RuleSeverity.HARD_VIOLATION if adds_new_risk else RuleSeverity.WARNING
            findings.append(
                self._finding(
                    code=PortfolioRiskRuleCode.MAX_ONE_WAY_TURNOVER,
                    severity=severity,
                    scope=RuleScope.PORTFOLIO,
                    scope_id="TURNOVER",
                    observed=proposal.one_way_turnover,
                    limit=self._policy.maximum_one_way_turnover,
                )
            )
        elif proposal.one_way_turnover > self._policy.turnover_warning_threshold:
            findings.append(
                self._finding(
                    code=PortfolioRiskRuleCode.TURNOVER_WARNING,
                    severity=RuleSeverity.WARNING,
                    scope=RuleScope.PORTFOLIO,
                    scope_id="TURNOVER",
                    observed=proposal.one_way_turnover,
                    limit=self._policy.turnover_warning_threshold,
                )
            )

        drawdown = context.current_drawdown
        if drawdown >= self._policy.drawdown_stop_new_risk_threshold and adds_new_risk:
            findings.append(
                self._finding(
                    code=PortfolioRiskRuleCode.DRAWDOWN_STOP_NEW_RISK,
                    severity=RuleSeverity.HARD_VIOLATION,
                    scope=RuleScope.ACCOUNT,
                    scope_id="DRAWDOWN",
                    observed=drawdown,
                    limit=self._policy.drawdown_stop_new_risk_threshold,
                )
            )
        elif drawdown >= self._policy.drawdown_warning_threshold:
            findings.append(
                self._finding(
                    code=PortfolioRiskRuleCode.DRAWDOWN_WARNING,
                    severity=RuleSeverity.WARNING,
                    scope=RuleScope.ACCOUNT,
                    scope_id="DRAWDOWN",
                    observed=drawdown,
                    limit=self._policy.drawdown_warning_threshold,
                )
            )
        return tuple(findings)

    def _append_limit_finding(
        self,
        findings: list[RiskFinding],
        *,
        code: PortfolioRiskRuleCode,
        scope: RuleScope,
        scope_id: str | None,
        observed: Decimal,
        current: Decimal,
        limit: Decimal,
        instrument_id: str | None = None,
    ) -> None:
        if observed <= limit:
            return
        severity = RuleSeverity.WARNING if observed < current else RuleSeverity.HARD_VIOLATION
        findings.append(
            self._finding(
                code=code,
                severity=severity,
                scope=scope,
                scope_id=scope_id,
                instrument_id=instrument_id,
                observed=observed,
                limit=limit,
            )
        )

    def _finding(
        self,
        *,
        code: PortfolioRiskRuleCode,
        severity: RuleSeverity,
        scope: RuleScope,
        scope_id: str | None,
        observed: Decimal,
        limit: Decimal,
        instrument_id: str | None = None,
    ) -> RiskFinding:
        return RiskFinding(
            rule_code=code.value,
            rule_version=self._policy.version,
            severity=severity,
            scope=scope,
            scope_id=scope_id,
            instrument_id=instrument_id,
            observed_value=observed,
            limit_value=limit,
        )

    @staticmethod
    def _sum_lines(
        proposal: TargetPortfolio,
        *,
        instrument_type: TradableInstrumentType,
        target: bool,
    ) -> Decimal:
        return sum(
            (
                line.target_weight if target else line.current_weight
                for line in proposal.lines
                if line.instrument_type is instrument_type
            ),
            Decimal(0),
        )

    @staticmethod
    def _industry_weights(
        proposal: TargetPortfolio,
        *,
        target: bool,
    ) -> dict[str, Decimal]:
        values: dict[str, Decimal] = {}
        for line in proposal.lines:
            if line.instrument_type is not TradableInstrumentType.STOCK:
                continue
            assert line.industry_id is not None
            values[line.industry_id] = values.get(line.industry_id, Decimal(0)) + (
                line.target_weight if target else line.current_weight
            )
        return values

    @staticmethod
    def _status(findings: tuple[RiskFinding, ...]) -> RiskCheckStatus:
        if any(item.severity is RuleSeverity.HARD_VIOLATION for item in findings):
            return RiskCheckStatus.REJECT
        if findings:
            return RiskCheckStatus.WARN
        return RiskCheckStatus.PASS

    @staticmethod
    def _safe_error_time(
        request: RiskCheckRequest,
        evaluated_at: datetime,
    ) -> datetime:
        try:
            ensure_aware(evaluated_at)
        except (AttributeError, ValueError):
            return request.account_snapshot_as_of
        return max(evaluated_at, request.account_snapshot_as_of)


__all__ = ["PortfolioRiskEvaluator"]
