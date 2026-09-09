"""Rule-by-rule tests for independent portfolio risk evaluation."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from tests.unit import test_portfolio_builder as fixtures

from quant_agent.portfolio import StrategySleeve, TargetPortfolio, TargetPortfolioBuilder
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import (
    PortfolioRiskContext,
    PortfolioRiskEvaluator,
    PortfolioRiskPolicy,
    PortfolioRiskRuleCode,
    RiskCheckStatus,
    RuleScope,
    build_portfolio_risk_request,
)


def _policy(**changes: object) -> PortfolioRiskPolicy:
    base = PortfolioRiskPolicy(
        version="portfolio-risk-test-v1",
        maximum_total_weight=Decimal("0.80"),
        maximum_etf_weight=Decimal("0.50"),
        maximum_stock_weight=Decimal("0.30"),
        maximum_single_etf_weight=Decimal("0.30"),
        maximum_single_stock_weight=Decimal("0.15"),
        maximum_stock_industry_weight=Decimal("0.25"),
        turnover_warning_threshold=Decimal("0.40"),
        maximum_one_way_turnover=Decimal("0.50"),
        drawdown_warning_threshold=Decimal("0.10"),
        drawdown_stop_new_risk_threshold=Decimal("0.20"),
    )
    return replace(base, **changes)


def _context(
    *,
    account_hash: str,
    equity: Decimal,
    high_watermark: Decimal | None = None,
    regime_budget: Decimal = Decimal("0.80"),
) -> PortfolioRiskContext:
    return PortfolioRiskContext.build(
        account_snapshot_hash=account_hash,
        as_of=fixtures.AS_OF,
        valuation_version="account-mark-v1",
        current_equity=equity,
        equity_high_watermark=high_watermark or equity,
        high_watermark_at=fixtures.AS_OF - timedelta(days=1),
        drawdown_source_hash=stable_hash({"account": "paper-account-1"}),
        regime_risk_budget=regime_budget,
        regime_result_hash=stable_hash({"regime": "UPTREND"}),
    )


def _check(
    *,
    policy: PortfolioRiskPolicy | None = None,
    context: PortfolioRiskContext | None = None,
    proposal: TargetPortfolio | None = None,
):  # type: ignore[no-untyped-def]
    active_policy = policy or _policy()
    account = fixtures._account()
    active_proposal = proposal or fixtures._build(account=account)
    active_context = context or _context(
        account_hash=account.content_hash,
        equity=account.total_equity,
    )
    request = build_portfolio_risk_request(
        account=account,
        proposal=active_proposal,
        context=active_context,
        policy=active_policy,
    )
    result = PortfolioRiskEvaluator(active_policy).check(
        request=request,
        account=account,
        proposal=active_proposal,
        context=active_context,
        evaluated_at=account.as_of + timedelta(seconds=1),
    )
    return account, active_proposal, active_context, request, result


def _all_cash_proposal() -> TargetPortfolio:
    account = fixtures._account()
    zero_sleeves = []
    for sleeve in fixtures._default_sleeves():
        zero_targets = tuple(
            replace(
                target,
                local_target_weight=Decimal(0),
                reason=f"exit {target.instrument_id}",
            )
            for target in sleeve.targets
        )
        zero_sleeves.append(
            StrategySleeve.build(
                sleeve_id=sleeve.sleeve_id,
                kind=sleeve.kind,
                as_of=sleeve.as_of,
                data_version=sleeve.data_version,
                strategy_name=sleeve.strategy_name,
                strategy_version=sleeve.strategy_version,
                strategy_config_hash=sleeve.strategy_config_hash,
                source_result_hash=sleeve.source_result_hash,
                targets=zero_targets,
            )
        )
    return TargetPortfolioBuilder(fixtures._config()).build(
        request=fixtures._request(account),
        account=account,
        sleeves=tuple(zero_sleeves),
        classifications=(fixtures._classification(fixtures.STOCK_A),),
    )


def test_wide_policy_passes_and_binds_every_input_identity() -> None:
    account, proposal, context, request, result = _check()

    assert result.status is RiskCheckStatus.PASS
    assert result.findings == ()
    assert result.allows_execution
    assert request.account_snapshot_hash == account.content_hash
    assert request.portfolio_proposal_hash == proposal.result_hash
    assert request.risk_context_hash == context.context_hash
    assert request.policy_hash == _policy().policy_hash


@pytest.mark.parametrize(
    ("policy", "regime_budget", "rule_code", "scope", "scope_id", "instrument_id"),
    [
        (
            _policy(maximum_total_weight=Decimal("0.62")),
            Decimal("0.80"),
            PortfolioRiskRuleCode.MAX_TOTAL_EXPOSURE,
            RuleScope.PORTFOLIO,
            "TOTAL",
            None,
        ),
        (
            _policy(),
            Decimal("0.62"),
            PortfolioRiskRuleCode.REGIME_RISK_BUDGET,
            RuleScope.PORTFOLIO,
            "REGIME",
            None,
        ),
        (
            _policy(maximum_etf_weight=Decimal("0.42")),
            Decimal("0.80"),
            PortfolioRiskRuleCode.MAX_ETF_EXPOSURE,
            RuleScope.PORTFOLIO,
            "ETF",
            None,
        ),
        (
            _policy(
                maximum_stock_weight=Decimal("0.19"),
                maximum_stock_industry_weight=Decimal("0.19"),
            ),
            Decimal("0.80"),
            PortfolioRiskRuleCode.MAX_STOCK_EXPOSURE,
            RuleScope.PORTFOLIO,
            "STOCK",
            None,
        ),
        (
            _policy(maximum_single_etf_weight=Decimal("0.24")),
            Decimal("0.80"),
            PortfolioRiskRuleCode.MAX_SINGLE_ETF_WEIGHT,
            RuleScope.INSTRUMENT,
            None,
            fixtures.ETF_A,
        ),
        (
            _policy(maximum_single_stock_weight=Decimal("0.12")),
            Decimal("0.80"),
            PortfolioRiskRuleCode.MAX_SINGLE_STOCK_WEIGHT,
            RuleScope.INSTRUMENT,
            None,
            fixtures.STOCK_A,
        ),
        (
            _policy(maximum_stock_industry_weight=Decimal("0.19")),
            Decimal("0.80"),
            PortfolioRiskRuleCode.MAX_STOCK_INDUSTRY_WEIGHT,
            RuleScope.INDUSTRY,
            fixtures.INDUSTRY,
            None,
        ),
        (
            _policy(
                turnover_warning_threshold=Decimal("0.20"),
                maximum_one_way_turnover=Decimal("0.32"),
            ),
            Decimal("0.80"),
            PortfolioRiskRuleCode.MAX_ONE_WAY_TURNOVER,
            RuleScope.PORTFOLIO,
            "TURNOVER",
            None,
        ),
    ],
)
def test_each_hard_limit_rejects_and_locates_the_exact_rule(
    policy: PortfolioRiskPolicy,
    regime_budget: Decimal,
    rule_code: PortfolioRiskRuleCode,
    scope: RuleScope,
    scope_id: str | None,
    instrument_id: str | None,
) -> None:
    account = fixtures._account()
    context = _context(
        account_hash=account.content_hash,
        equity=account.total_equity,
        regime_budget=regime_budget,
    )
    *_, result = _check(policy=policy, context=context)

    finding = next(item for item in result.findings if item.rule_code == rule_code.value)
    assert result.status is RiskCheckStatus.REJECT
    assert result.is_rejected
    assert finding.scope is scope
    assert finding.scope_id == scope_id
    assert finding.instrument_id == instrument_id
    assert finding.observed_value is not None
    assert finding.limit_value is not None
    assert finding.observed_value > finding.limit_value


def test_turnover_and_drawdown_warning_paths_do_not_claim_hard_approval() -> None:
    turnover_policy = _policy(turnover_warning_threshold=Decimal("0.30"))
    *_, turnover = _check(policy=turnover_policy)
    assert turnover.status is RiskCheckStatus.WARN
    assert turnover.allows_execution
    assert {item.rule_code for item in turnover.warnings} == {
        PortfolioRiskRuleCode.TURNOVER_WARNING.value
    }

    account = fixtures._account()
    drawdown_context = _context(
        account_hash=account.content_hash,
        equity=account.total_equity,
        high_watermark=Decimal("12500"),
    )
    drawdown_policy = _policy(
        drawdown_warning_threshold=Decimal("0.15"),
        drawdown_stop_new_risk_threshold=Decimal("0.25"),
    )
    *_, drawdown = _check(policy=drawdown_policy, context=drawdown_context)
    assert drawdown.status is RiskCheckStatus.WARN
    assert {item.rule_code for item in drawdown.warnings} == {
        PortfolioRiskRuleCode.DRAWDOWN_WARNING.value
    }


def test_drawdown_stop_rejects_any_positive_delta_but_allows_pure_deleveraging() -> None:
    account = fixtures._account()
    drawdown_context = _context(
        account_hash=account.content_hash,
        equity=account.total_equity,
        high_watermark=Decimal("12500"),
    )
    *_, increasing = _check(context=drawdown_context)
    assert increasing.status is RiskCheckStatus.REJECT
    assert PortfolioRiskRuleCode.DRAWDOWN_STOP_NEW_RISK.value in {
        item.rule_code for item in increasing.hard_violations
    }

    exit_proposal = _all_cash_proposal()
    exit_policy = _policy(
        turnover_warning_threshold=Decimal("0.10"),
        maximum_one_way_turnover=Decimal("0.25"),
    )
    *_, reducing = _check(
        policy=exit_policy,
        context=drawdown_context,
        proposal=exit_proposal,
    )
    assert reducing.status is RiskCheckStatus.WARN
    assert reducing.allows_execution
    assert not reducing.hard_violations
    assert {item.rule_code for item in reducing.warnings} == {
        PortfolioRiskRuleCode.MAX_ONE_WAY_TURNOVER.value,
        PortfolioRiskRuleCode.DRAWDOWN_WARNING.value,
    }


def test_input_policy_context_and_evaluation_failures_return_error_and_reject() -> None:
    account, proposal, context, request, _ = _check()
    evaluator = PortfolioRiskEvaluator(_policy())

    changed_context = _context(
        account_hash=account.content_hash,
        equity=account.total_equity,
        regime_budget=Decimal("0.70"),
    )
    mismatched = evaluator.check(
        request=request,
        account=account,
        proposal=proposal,
        context=changed_context,
        evaluated_at=account.as_of,
    )
    stale_time = evaluator.check(
        request=request,
        account=account,
        proposal=proposal,
        context=context,
        evaluated_at=account.as_of - timedelta(microseconds=1),
    )
    wrong_policy = PortfolioRiskEvaluator(
        replace(_policy(), version="portfolio-risk-test-v2")
    ).check(
        request=request,
        account=account,
        proposal=proposal,
        context=context,
        evaluated_at=account.as_of,
    )
    internal_failure = evaluator.check(
        request=request,
        account=account,
        proposal=None,  # type: ignore[arg-type]
        context=context,
        evaluated_at=account.as_of,
    )

    for result in (mismatched, stale_time, wrong_policy, internal_failure):
        assert result.status is RiskCheckStatus.ERROR
        assert result.is_rejected
        assert not result.allows_execution
        assert result.error_code is not None


def test_tightening_a_threshold_cannot_turn_a_rejection_into_permission() -> None:
    *_, loose = _check(policy=_policy(maximum_total_weight=Decimal("0.80")))
    *_, tight = _check(policy=_policy(maximum_total_weight=Decimal("0.62")))

    assert loose.status is RiskCheckStatus.PASS
    assert tight.status is RiskCheckStatus.REJECT
    assert loose.allows_execution and not tight.allows_execution
