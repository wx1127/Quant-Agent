"""End-to-end sizing and fail-closed tests for review-only order drafts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import NoReturn

import pytest

from quant_agent.backtest import FeeRule, FeeRuleBook, Side, TradableInstrumentType
from quant_agent.backtest.cn_market import (
    CNMarketRule,
    CNSlippageModel,
    CNSlippageModelBook,
    MarketSessionState,
)
from quant_agent.config import RuntimeMode
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.execution.order_drafts import (
    OrderDraftBatch,
    OrderDraftGenerator,
    OrderDraftInputError,
)
from quant_agent.features.tradeability import MarketTradeState
from quant_agent.portfolio import (
    AccountSnapshot,
    CashSnapshot,
    PortfolioBuilderConfig,
    PortfolioBuildRequest,
    PortfolioPositionSnapshot,
    SleeveTarget,
    StrategySleeve,
    StrategySleeveKind,
    TargetPortfolio,
    TargetPortfolioBuilder,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.risk import (
    PortfolioRiskContext,
    PortfolioRiskEvaluator,
    PortfolioRiskPolicy,
    RiskCheckRequest,
    RiskCheckResult,
    RiskCheckStatus,
    build_portfolio_risk_request,
)

TRADING_DAY = date(2026, 8, 28)
AS_OF = datetime.combine(TRADING_DAY, time(18), tzinfo=SHANGHAI_TZ)
VALUATION_AT = datetime.combine(TRADING_DAY, time(15), tzinfo=SHANGHAI_TZ)
DRAFT_AS_OF = AS_OF + timedelta(minutes=2)
STATE_OBSERVED_AT = datetime.combine(TRADING_DAY, time(14, 59), tzinfo=SHANGHAI_TZ)
STATE_AVAILABLE_AT = datetime.combine(TRADING_DAY, time(15), tzinfo=SHANGHAI_TZ)
DATA_VERSION = "cn-market-20260828-v1"

BUY_A = "CN.SSE.510100"
BUY_B = "CN.SSE.510200"
SUB_LOT_BUY = "CN.SSE.510250"
ODD_LOT = "CN.SSE.510300"
PARTIAL_SELL = "CN.SSE.510500"

DEFAULT_LOCAL_TARGETS = {
    BUY_A: Decimal("0.32"),
    BUY_B: Decimal("0.20"),
    PARTIAL_SELL: Decimal("0.155"),
}
NO_ACTION_LOCAL_TARGETS = {
    ODD_LOT: Decimal("0.01625"),
    PARTIAL_SELL: Decimal("0.3125"),
}


@dataclass(frozen=True, slots=True)
class _Chain:
    account: AccountSnapshot
    proposal: TargetPortfolio
    risk_result: RiskCheckResult


def _position(
    instrument_id: str,
    *,
    total_quantity: Decimal,
    available_quantity: Decimal,
    as_of: datetime,
    valuation_at: datetime,
    data_version: str,
) -> PortfolioPositionSnapshot:
    return PortfolioPositionSnapshot.build(
        instrument_id=instrument_id,
        instrument_type=TradableInstrumentType.ETF,
        available_quantity=available_quantity,
        frozen_quantity=total_quantity - available_quantity,
        unsettled_quantity=Decimal(0),
        average_cost=Decimal("10.00"),
        valuation_price=Decimal("10.00"),
        price_observed_at=valuation_at,
        price_available_at=valuation_at,
        position_as_of=as_of,
        price_data_version=data_version,
        price_source_hash=stable_hash({"instrument": instrument_id, "price": "10.00"}),
        mark_policy_hash=stable_hash({"policy": "position-mark-v1"}),
    )


def _account(
    *,
    runtime_mode: RuntimeMode = RuntimeMode.PAPER,
    available_cash: Decimal = Decimal(7370),
    odd_lot_available: Decimal = Decimal(13),
    partial_sell_available: Decimal = Decimal(250),
    as_of: datetime = AS_OF,
    data_version: str = DATA_VERSION,
) -> AccountSnapshot:
    valuation_at = datetime.combine(
        as_of.astimezone(SHANGHAI_TZ).date(),
        time(15),
        tzinfo=SHANGHAI_TZ,
    )
    positions = (
        _position(
            ODD_LOT,
            total_quantity=Decimal(13),
            available_quantity=odd_lot_available,
            as_of=as_of,
            valuation_at=valuation_at,
            data_version=data_version,
        ),
        _position(
            PARTIAL_SELL,
            total_quantity=Decimal(250),
            available_quantity=partial_sell_available,
            as_of=as_of,
            valuation_at=valuation_at,
            data_version=data_version,
        ),
    )
    total_cash = Decimal(7370)
    cash = CashSnapshot(
        as_of=as_of,
        currency="CNY",
        total_cash=total_cash,
        available_cash=available_cash,
        frozen_cash=total_cash - available_cash,
    )
    return AccountSnapshot.build(
        snapshot_id=f"account-{as_of.isoformat()}-{data_version}-{runtime_mode.value}",
        account_id="paper-account-1",
        runtime_mode=runtime_mode,
        as_of=as_of,
        valuation_at=valuation_at,
        data_version=data_version,
        currency="CNY",
        cash=cash,
        positions=positions,
        previous_snapshot_hash=stable_hash({"previous": "account"}),
        source_event_log_hash=stable_hash({"events": "account"}),
    )


def _builder_config() -> PortfolioBuilderConfig:
    return PortfolioBuilderConfig(
        version="order-draft-builder-test-v1",
        etf_core_budget=Decimal("0.80"),
        stock_enhancement_budget=Decimal("0.10"),
        minimum_cash_weight=Decimal("0.10"),
        maximum_instrument_weight=Decimal("0.80"),
        maximum_stock_industry_weight=Decimal("0.10"),
    )


def _sleeve(
    *,
    kind: StrategySleeveKind,
    as_of: datetime,
    data_version: str,
    local_targets: dict[str, Decimal],
) -> StrategySleeve:
    targets = (
        tuple(
            SleeveTarget(
                instrument_id=instrument_id,
                instrument_type=TradableInstrumentType.ETF,
                local_target_weight=weight,
                reason=f"target {instrument_id} for order-draft sizing",
            )
            for instrument_id, weight in sorted(local_targets.items())
        )
        if kind is StrategySleeveKind.ETF_CORE
        else ()
    )
    sleeve_id = "etf-core" if kind is StrategySleeveKind.ETF_CORE else "stock-enhancement"
    return StrategySleeve.build(
        sleeve_id=sleeve_id,
        kind=kind,
        as_of=as_of,
        data_version=data_version,
        strategy_name=sleeve_id,
        strategy_version="strategy-v1",
        strategy_config_hash=stable_hash({"strategy": sleeve_id, "config": "v1"}),
        source_result_hash=stable_hash({"strategy": sleeve_id, "result": "fixed"}),
        targets=targets,
    )


def _proposal(
    account: AccountSnapshot,
    *,
    decision_id: str = "portfolio-decision-20260828",
    local_targets: dict[str, Decimal] | None = None,
) -> TargetPortfolio:
    request = PortfolioBuildRequest(
        decision_id=decision_id,
        as_of=account.as_of,
        data_version=account.data_version,
        account_snapshot_id=account.snapshot_id,
        account_snapshot_hash=account.content_hash,
    )
    targets = DEFAULT_LOCAL_TARGETS if local_targets is None else local_targets
    sleeves = (
        _sleeve(
            kind=StrategySleeveKind.ETF_CORE,
            as_of=account.as_of,
            data_version=account.data_version,
            local_targets=targets,
        ),
        _sleeve(
            kind=StrategySleeveKind.STOCK_ENHANCEMENT,
            as_of=account.as_of,
            data_version=account.data_version,
            local_targets={},
        ),
    )
    return TargetPortfolioBuilder(_builder_config()).build(
        request=request,
        account=account,
        sleeves=sleeves,
        classifications=(),
    )


def _risk_policy(
    *,
    turnover_warning_threshold: Decimal = Decimal("0.60"),
    maximum_total_weight: Decimal = Decimal("0.95"),
    maximum_etf_weight: Decimal = Decimal("0.90"),
) -> PortfolioRiskPolicy:
    return PortfolioRiskPolicy(
        version="order-draft-risk-test-v1",
        maximum_total_weight=maximum_total_weight,
        maximum_etf_weight=maximum_etf_weight,
        maximum_stock_weight=Decimal("0.10"),
        maximum_single_etf_weight=min(Decimal("0.80"), maximum_etf_weight),
        maximum_single_stock_weight=Decimal("0.10"),
        maximum_stock_industry_weight=Decimal("0.10"),
        turnover_warning_threshold=turnover_warning_threshold,
        maximum_one_way_turnover=Decimal("0.90"),
        drawdown_warning_threshold=Decimal("0.20"),
        drawdown_stop_new_risk_threshold=Decimal("0.40"),
    )


def _risk_result(
    account: AccountSnapshot,
    proposal: TargetPortfolio,
    *,
    policy: PortfolioRiskPolicy,
) -> RiskCheckResult:
    context = PortfolioRiskContext.build(
        account_snapshot_hash=account.content_hash,
        as_of=account.as_of,
        valuation_version="account-mark-v1",
        current_equity=account.total_equity,
        equity_high_watermark=account.total_equity,
        high_watermark_at=account.as_of - timedelta(days=1),
        drawdown_source_hash=stable_hash({"drawdown": account.snapshot_id}),
        regime_risk_budget=Decimal("0.90"),
        regime_result_hash=stable_hash({"regime": "UPTREND"}),
    )
    request = build_portfolio_risk_request(
        account=account,
        proposal=proposal,
        context=context,
        policy=policy,
    )
    return PortfolioRiskEvaluator(policy).check(
        request=request,
        account=account,
        proposal=proposal,
        context=context,
        evaluated_at=account.as_of + timedelta(minutes=1),
    )


def _chain(
    *,
    account: AccountSnapshot | None = None,
    local_targets: dict[str, Decimal] | None = None,
    policy: PortfolioRiskPolicy | None = None,
    decision_id: str = "portfolio-decision-20260828",
) -> _Chain:
    bound_account = account or _account()
    proposal = _proposal(
        bound_account,
        decision_id=decision_id,
        local_targets=local_targets,
    )
    result = _risk_result(
        bound_account,
        proposal,
        policy=policy or _risk_policy(),
    )
    return _Chain(account=bound_account, proposal=proposal, risk_result=result)


def _market_states(proposal: TargetPortfolio) -> tuple[MarketSessionState, ...]:
    return tuple(
        MarketSessionState(
            instrument_id=line.instrument_id,
            instrument_type=line.instrument_type,
            trading_day=TRADING_DAY,
            observed_at=STATE_OBSERVED_AT,
            available_at=STATE_AVAILABLE_AT,
            revision=f"state-{line.instrument_id}-r1",
            data_version=proposal.data_version,
            state=MarketTradeState.NORMAL,
            buy_allowed=True,
            sell_allowed=True,
            reference_price=Decimal("10.00"),
            lower_limit_price=Decimal("9.00"),
            upper_limit_price=Decimal("11.00"),
            available_quantity=Decimal(10000),
        )
        for line in proposal.lines
    )


def _market_rule(
    states: tuple[MarketSessionState, ...],
    *,
    version: str = "cn-market-rule-v1",
    effective_from: date | None = None,
    effective_to: date | None = None,
    maximum_participation: Decimal = Decimal("0.10"),
) -> CNMarketRule:
    return CNMarketRule(
        rule_id="cn-etf-market-rule",
        version=version,
        effective_from=effective_from or TRADING_DAY - timedelta(days=30),
        effective_to=effective_to,
        instrument_type=TradableInstrumentType.ETF,
        session_states=states,
        trading_calendar=tuple(sorted({state.trading_day for state in states})),
        buy_lot_size=Decimal(100),
        sell_lot_size=Decimal(100),
        allow_odd_lot_liquidation=True,
        settlement_days=1,
        price_tick=Decimal("0.01"),
        max_participation_rate=maximum_participation,
    )


def _fee_rule_book() -> FeeRuleBook:
    return FeeRuleBook(
        (
            FeeRule(
                instrument_type=TradableInstrumentType.ETF,
                side=Side.BUY,
                effective_from=TRADING_DAY - timedelta(days=30),
                version="cn-etf-buy-fee-v1",
                commission_rate=Decimal("0.001"),
                minimum_commission=Decimal("5.00"),
                stamp_duty_rate=Decimal(0),
                transfer_fee_rate=Decimal("0.00001"),
                other_fee_rate=Decimal(0),
                rounding_increment=Decimal("0.01"),
            ),
            FeeRule(
                instrument_type=TradableInstrumentType.ETF,
                side=Side.SELL,
                effective_from=TRADING_DAY - timedelta(days=30),
                version="cn-etf-sell-fee-v1",
                commission_rate=Decimal("0.001"),
                minimum_commission=Decimal("5.00"),
                stamp_duty_rate=Decimal("0.001"),
                transfer_fee_rate=Decimal("0.00001"),
                other_fee_rate=Decimal(0),
                rounding_increment=Decimal("0.01"),
            ),
        )
    )


def _slippage_model() -> CNSlippageModel:
    return CNSlippageModel(
        model_id="cn-linear-impact",
        version="cn-linear-impact-v1",
        effective_from=TRADING_DAY - timedelta(days=30),
        base_rate=Decimal("0.001"),
        participation_impact_rate=Decimal("0.01"),
        maximum_rate=Decimal("0.05"),
    )


def _generator(
    *,
    market_rules: tuple[CNMarketRule, ...],
    fee_rule_book: FeeRuleBook | None = None,
    slippage_model: CNSlippageModel | None = None,
) -> OrderDraftGenerator:
    model = slippage_model or _slippage_model()
    return OrderDraftGenerator(
        market_rules=market_rules,
        fee_rule_book=fee_rule_book or _fee_rule_book(),
        slippage_model_book=CNSlippageModelBook((model,)),
    )


def _generate(
    chain: _Chain,
    *,
    states: tuple[MarketSessionState, ...] | None = None,
    rules: tuple[CNMarketRule, ...] | None = None,
    draft_as_of: datetime = DRAFT_AS_OF,
) -> OrderDraftBatch:
    active_states = states or _market_states(chain.proposal)
    active_rules = rules or (_market_rule(active_states),)
    return _generator(market_rules=active_rules).generate(
        account_snapshot=chain.account,
        target_portfolio=chain.proposal,
        risk_result=chain.risk_result,
        market_states=active_states,
        draft_as_of=draft_as_of,
    )


def test_sizes_buy_sell_and_odd_lot_with_protected_cash_without_calling_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    states = _market_states(chain.proposal)
    state_by_id = {state.instrument_id: state for state in states}
    rule = _market_rule(states)
    fee_book = _fee_rule_book()
    slippage = _slippage_model()

    def forbidden_match(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("order draft generation must not call the execution matcher")

    monkeypatch.setattr(CNMarketRule, "match", forbidden_match)
    batch = _generator(
        market_rules=(rule,),
        fee_rule_book=fee_book,
        slippage_model=slippage,
    ).generate(
        account_snapshot=chain.account,
        target_portfolio=chain.proposal,
        risk_result=chain.risk_result,
        market_states=states,
        draft_as_of=DRAFT_AS_OF,
    )

    lines = {line.instrument_id: line for line in batch.lines}
    assert tuple(lines) == tuple(sorted(lines))
    assert (lines[BUY_A].side, lines[BUY_A].current_quantity) == (Side.BUY, Decimal(0))
    assert (lines[BUY_A].quantity, lines[BUY_A].target_quantity) == (
        Decimal(200),
        Decimal(200),
    )
    assert (lines[BUY_B].quantity, lines[BUY_B].target_quantity) == (
        Decimal(100),
        Decimal(100),
    )
    assert (lines[PARTIAL_SELL].side, lines[PARTIAL_SELL].current_quantity) == (
        Side.SELL,
        Decimal(250),
    )
    assert (lines[PARTIAL_SELL].quantity, lines[PARTIAL_SELL].target_quantity) == (
        Decimal(100),
        Decimal(150),
    )
    assert not lines[PARTIAL_SELL].is_full_liquidation
    assert (lines[ODD_LOT].side, lines[ODD_LOT].quantity) == (Side.SELL, Decimal(13))
    assert lines[ODD_LOT].target_quantity == 0
    assert lines[ODD_LOT].is_full_liquidation

    for line in batch.lines:
        state = state_by_id[line.instrument_id]
        fee_rule = fee_book.select(
            instrument_type=line.instrument_type,
            side=line.side,
            trading_day=TRADING_DAY,
        )
        expected_price = slippage.execution_price(
            side=line.side,
            reference_price=state.reference_price,
            participation_rate=line.quantity / state.available_quantity,
            price_tick=rule.price_tick,
        )
        expected_price = min(expected_price, state.upper_limit_price or expected_price)
        expected_price = max(expected_price, state.lower_limit_price or expected_price)
        expected_fees = fee_rule.assess(line.quantity * expected_price)

        assert line.reference_price == state.reference_price
        assert line.estimated_execution_price == expected_price
        assert line.participation_rate == line.quantity / state.available_quantity
        assert line.total_fee == expected_fees.total_amount
        assert line.state_revision == state.revision
        assert line.state_hash == state.state_hash
        assert line.market_rule_version == rule.version
        assert line.market_rule_hash == rule.rule_hash
        assert line.fee_rule_version == fee_rule.version
        assert line.fee_rule_hash == stable_hash(fee_rule.model_dump(mode="python"))
        assert line.slippage_model_version == slippage.version
        assert line.slippage_model_hash == slippage.model_hash
        if line.side is Side.BUY:
            assert line.cash_reservation_price == state.upper_limit_price
            protected_fees = fee_rule.assess(line.quantity * Decimal("11.00"))
            assert line.cash_reservation_fee == protected_fees.total_amount
            assert (
                line.reserved_cash == line.quantity * Decimal("11.00") + protected_fees.total_amount
            )
        else:
            assert line.cash_reservation_price == line.estimated_execution_price
            assert line.cash_reservation_fee == line.total_fee
            assert line.reserved_cash == 0

    buy_lines = tuple(line for line in batch.lines if line.side is Side.BUY)
    sell_lines = tuple(line for line in batch.lines if line.side is Side.SELL)
    assert batch.risk_status is RiskCheckStatus.PASS
    assert batch.account_snapshot_as_of == chain.account.as_of
    assert batch.runtime_mode is RuntimeMode.PAPER
    assert batch.available_cash == chain.account.cash.available_cash
    assert batch.reserved_cash_required == sum(
        (line.reserved_cash for line in buy_lines),
        Decimal(0),
    )
    assert batch.estimated_buy_cash_required == sum(
        (-line.estimated_cash_change for line in buy_lines),
        Decimal(0),
    )
    assert batch.estimated_sell_cash_proceeds == sum(
        (line.estimated_cash_change for line in sell_lines),
        Decimal(0),
    )
    assert batch.state_hashes == tuple(sorted({state.state_hash for state in states}))
    assert batch.market_rule_hashes == (rule.rule_hash,)
    assert batch.fee_rule_hashes == tuple(
        sorted(
            {
                stable_hash(
                    fee_book.select(
                        instrument_type=line.instrument_type,
                        side=line.side,
                        trading_day=TRADING_DAY,
                    ).model_dump(mode="python")
                )
                for line in batch.lines
            }
        )
    )
    assert batch.slippage_model_hashes == (slippage.model_hash,)
    assert batch.portfolio_proposal_hash == chain.proposal.result_hash
    assert batch.risk_request_hash == chain.risk_result.request.request_hash
    assert batch.risk_result_hash == chain.risk_result.result_hash
    assert not batch.is_executable
    assert batch.requires_human_approval


def test_warn_result_and_live_assisted_account_can_create_a_review_draft() -> None:
    chain = _chain(
        account=_account(runtime_mode=RuntimeMode.LIVE_ASSISTED),
        policy=_risk_policy(turnover_warning_threshold=Decimal("0.40")),
    )
    assert chain.risk_result.status is RiskCheckStatus.WARN

    batch = _generate(chain)

    assert batch.risk_status is RiskCheckStatus.WARN
    assert batch.runtime_mode is RuntimeMode.LIVE_ASSISTED
    assert batch.requires_human_approval and not batch.is_executable


def test_reject_and_error_risk_results_fail_closed() -> None:
    rejected = _chain(
        policy=_risk_policy(
            maximum_total_weight=Decimal("0.50"),
            maximum_etf_weight=Decimal("0.50"),
        )
    )
    assert rejected.risk_result.status is RiskCheckStatus.REJECT
    with pytest.raises(OrderDraftInputError, match="risk result REJECT"):
        _generate(rejected)

    passing = _chain()
    service_error = RiskCheckResult.service_error(
        request=passing.risk_result.request,
        evaluated_at=passing.risk_result.evaluated_at,
        risk_engine_version=passing.risk_result.risk_engine_version,
        error_code="UPSTREAM_TIMEOUT",
    )
    errored = replace(passing, risk_result=service_error)
    with pytest.raises(OrderDraftInputError, match="risk result ERROR"):
        _generate(errored)


def _rebuild_risk_request(
    request: RiskCheckRequest,
    **changes: object,
) -> RiskCheckRequest:
    values: dict[str, object] = {
        "decision_id": request.decision_id,
        "account_snapshot_id": request.account_snapshot_id,
        "account_snapshot_hash": request.account_snapshot_hash,
        "account_snapshot_as_of": request.account_snapshot_as_of,
        "portfolio_proposal_hash": request.portfolio_proposal_hash,
        "data_version": request.data_version,
        "valuation_version": request.valuation_version,
        "policy_version": request.policy_version,
        "policy_hash": request.policy_hash,
        "risk_context_hash": request.risk_context_hash,
    }
    values.update(changes)
    return RiskCheckRequest.build(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "request_changes",
    [
        {"decision_id": "different-decision"},
        {"account_snapshot_id": "different-account-snapshot"},
        {"account_snapshot_hash": "a" * 64},
        {"account_snapshot_as_of": AS_OF - timedelta(seconds=1)},
        {"portfolio_proposal_hash": "b" * 64},
        {"data_version": "different-data-version"},
    ],
    ids=[
        "decision",
        "account-id",
        "account-hash",
        "account-boundary",
        "proposal-hash",
        "data-version",
    ],
)
def test_risk_request_must_bind_every_account_and_proposal_identity(
    request_changes: dict[str, object],
) -> None:
    chain = _chain()
    mismatched_request = _rebuild_risk_request(
        chain.risk_result.request,
        **request_changes,
    )
    mismatched_result = RiskCheckResult.build(
        request=mismatched_request,
        status=RiskCheckStatus.PASS,
        evaluated_at=chain.risk_result.evaluated_at,
        risk_engine_version=chain.risk_result.risk_engine_version,
    )

    with pytest.raises(OrderDraftInputError, match=r"risk .* does not match"):
        _generate(replace(chain, risk_result=mismatched_result))


def test_account_and_proposal_hash_time_data_and_decision_mismatches_fail_closed() -> None:
    chain = _chain()
    different_hash = _account(available_cash=Decimal(7000))
    different_time = _account(as_of=AS_OF + timedelta(seconds=5))
    different_data = _account(data_version="cn-market-20260828-v2")
    different_decision = _proposal(chain.account, decision_id="different-decision")

    mismatches = (
        replace(chain, account=different_hash),
        replace(chain, account=different_time),
        replace(chain, account=different_data),
        replace(chain, proposal=different_decision),
    )
    for mismatch in mismatches:
        with pytest.raises(OrderDraftInputError):
            _generate(mismatch)


@pytest.mark.parametrize(
    "runtime_mode",
    [RuntimeMode.LIVE_AUTO, RuntimeMode.RESEARCH, RuntimeMode.BACKTEST],
)
def test_non_assisted_runtime_modes_cannot_create_order_drafts(
    runtime_mode: RuntimeMode,
) -> None:
    chain = _chain(account=_account(runtime_mode=runtime_mode))

    with pytest.raises(OrderDraftInputError, match="PAPER or LIVE_ASSISTED"):
        _generate(chain)


def test_draft_risk_and_market_state_boundaries_are_same_day_and_point_in_time() -> None:
    chain = _chain()
    states = _market_states(chain.proposal)

    with pytest.raises(OrderDraftInputError, match="overnight"):
        _generate(chain, draft_as_of=DRAFT_AS_OF + timedelta(days=1))

    future_result = RiskCheckResult.build(
        request=chain.risk_result.request,
        status=RiskCheckStatus.PASS,
        evaluated_at=DRAFT_AS_OF + timedelta(seconds=1),
        risk_engine_version=chain.risk_result.risk_engine_version,
    )
    with pytest.raises(OrderDraftInputError, match="after draft_as_of"):
        _generate(replace(chain, risk_result=future_result))

    future_state = replace(
        states[0],
        observed_at=DRAFT_AS_OF + timedelta(seconds=1),
        available_at=DRAFT_AS_OF + timedelta(seconds=1),
    )
    future_states = (future_state, *states[1:])
    with pytest.raises(OrderDraftInputError, match="future market state"):
        _generate(
            chain,
            states=future_states,
            rules=(_market_rule(future_states),),
        )


def test_market_states_require_exact_identity_version_and_unique_coverage() -> None:
    chain = _chain()
    states = _market_states(chain.proposal)
    first = states[0]
    rule = _market_rule(states)

    bad_cases = (
        (replace(first, data_version="wrong-data-version"), *states[1:]),
        (
            replace(first, instrument_type=TradableInstrumentType.STOCK),
            *states[1:],
        ),
        (replace(first, instrument_id="CN.SSE.519999"), *states[1:]),
        (*states, first),
    )
    for bad_states in bad_cases:
        with pytest.raises(OrderDraftInputError):
            _generate(
                chain,
                states=bad_states,
                rules=(rule,),
            )


def test_market_state_must_be_the_latest_revision_bound_into_the_selected_rule() -> None:
    chain = _chain()
    states = _market_states(chain.proposal)
    current = states[0]
    stale = replace(
        current,
        observed_at=current.observed_at - timedelta(minutes=1),
        available_at=current.available_at - timedelta(seconds=1),
        revision="state-stale-r0",
    )
    rule = _market_rule((stale, *states))
    supplied_states = (stale, *states[1:])

    with pytest.raises(OrderDraftInputError, match="latest bound revision"):
        _generate(chain, states=supplied_states, rules=(rule,))

    unbound = replace(current, revision="state-unbound-r999")
    with pytest.raises(OrderDraftInputError, match="latest bound revision"):
        _generate(chain, states=(unbound, *states[1:]), rules=(_market_rule(states),))


@pytest.mark.parametrize(
    ("instrument_id", "scenario"),
    [
        (BUY_A, "suspended"),
        (BUY_A, "buy-blocked"),
        (PARTIAL_SELL, "sell-blocked"),
    ],
    ids=["suspended", "buy-blocked", "sell-blocked"],
)
def test_suspension_and_side_specific_market_blocks_veto_the_complete_batch(
    instrument_id: str,
    scenario: str,
) -> None:
    chain = _chain()
    states = _market_states(chain.proposal)
    blocked: list[MarketSessionState] = []
    for state in states:
        if state.instrument_id != instrument_id:
            blocked.append(state)
        elif scenario == "suspended":
            blocked.append(
                replace(
                    state,
                    state=MarketTradeState.SUSPENDED,
                    buy_allowed=False,
                    sell_allowed=False,
                )
            )
        elif scenario == "buy-blocked":
            blocked.append(replace(state, buy_allowed=False))
        else:
            blocked.append(replace(state, sell_allowed=False))
    blocked_states = tuple(blocked)

    with pytest.raises(OrderDraftInputError, match="blocks"):
        _generate(
            chain,
            states=blocked_states,
            rules=(_market_rule(blocked_states),),
        )


@pytest.mark.parametrize(
    "account",
    [
        _account(partial_sell_available=Decimal(50)),
        _account(odd_lot_available=Decimal(0)),
    ],
    ids=["partial-sell", "odd-lot-full-liquidation"],
)
def test_sell_drafts_cannot_exceed_account_available_quantity(
    account: AccountSnapshot,
) -> None:
    chain = _chain(account=account)

    with pytest.raises(OrderDraftInputError, match="available"):
        _generate(chain)


def test_snapshot_available_cash_cannot_be_augmented_with_expected_sell_proceeds() -> None:
    fully_funded = _chain()
    funded_batch = _generate(fully_funded)
    low_available_cash = Decimal(2500)
    assert funded_batch.reserved_cash_required > low_available_cash
    assert (
        low_available_cash + funded_batch.estimated_sell_cash_proceeds
        > funded_batch.reserved_cash_required
    )

    low_cash_chain = _chain(account=_account(available_cash=low_available_cash))
    with pytest.raises(OrderDraftInputError, match="reserved cash requirements exceed"):
        _generate(low_cash_chain)


def test_market_participation_limit_is_a_hard_veto_not_a_quantity_cap() -> None:
    chain = _chain()
    states = _market_states(chain.proposal)
    constrained_states = tuple(
        replace(state, available_quantity=Decimal(1000)) if state.instrument_id == BUY_A else state
        for state in states
    )

    with pytest.raises(OrderDraftInputError, match="participation limit"):
        _generate(
            chain,
            states=constrained_states,
            rules=(_market_rule(constrained_states),),
        )


def test_market_state_prices_must_align_to_the_selected_rule_tick() -> None:
    chain = _chain()
    states = _market_states(chain.proposal)
    unaligned_states = tuple(
        replace(state, reference_price=Decimal("10.005")) if state.instrument_id == BUY_A else state
        for state in states
    )

    with pytest.raises(OrderDraftInputError, match="tick aligned"):
        _generate(
            chain,
            states=unaligned_states,
            rules=(_market_rule(unaligned_states),),
        )


def test_market_state_and_market_rule_input_order_cannot_change_the_batch() -> None:
    chain = _chain()
    states = _market_states(chain.proposal)
    current_rule = _market_rule(states)
    historical_rule = _market_rule(
        (),
        version="cn-market-rule-v0",
        effective_from=TRADING_DAY - timedelta(days=90),
        effective_to=TRADING_DAY - timedelta(days=31),
    )

    ordered = _generate(
        chain,
        states=states,
        rules=(historical_rule, current_rule),
    )
    reversed_inputs = _generate(
        chain,
        states=tuple(reversed(states)),
        rules=(current_rule, historical_rule),
    )

    assert reversed_inputs == ordered
    assert reversed_inputs.input_hash == ordered.input_hash
    assert reversed_inputs.batch_hash == ordered.batch_hash


def test_sub_lot_state_is_bound_even_when_it_produces_no_draft_line() -> None:
    local_targets = {
        **DEFAULT_LOCAL_TARGETS,
        SUB_LOT_BUY: Decimal("0.01"),
    }
    chain = _chain(local_targets=local_targets)
    original_states = _market_states(chain.proposal)
    original_sub_lot_state = next(
        state for state in original_states if state.instrument_id == SUB_LOT_BUY
    )
    original = _generate(
        chain,
        states=original_states,
        rules=(_market_rule(original_states),),
    )

    changed_sub_lot_state = replace(
        original_sub_lot_state,
        reference_price=Decimal("10.50"),
        revision="state-sub-lot-r2",
    )
    changed_states = tuple(
        changed_sub_lot_state if state.instrument_id == SUB_LOT_BUY else state
        for state in original_states
    )
    changed = _generate(
        chain,
        states=changed_states,
        rules=(_market_rule(changed_states),),
    )

    assert SUB_LOT_BUY not in {line.instrument_id for line in original.lines}
    assert SUB_LOT_BUY not in {line.instrument_id for line in changed.lines}
    assert original.lines == changed.lines
    assert original_sub_lot_state.state_hash in original.state_hashes
    assert changed_sub_lot_state.state_hash in changed.state_hashes
    assert original_sub_lot_state.state_hash not in changed.state_hashes
    assert changed_sub_lot_state.state_hash not in original.state_hashes
    assert changed.input_hash != original.input_hash
    assert changed.batch_hash != original.batch_hash


def test_fractional_gaps_just_below_one_lot_never_round_up_to_an_order() -> None:
    almost_one_lot = Decimal("99.999999999999999999999999999999999")
    with localcontext() as context:
        context.prec = 100
        buy_target_weight = almost_one_lot * Decimal("10.00") / Decimal(10000)
        sell_target_quantity = Decimal(250) - almost_one_lot
        sell_target_weight = sell_target_quantity * Decimal("10.00") / Decimal(10000)
        local_targets = {
            BUY_A: Decimal("0.32"),
            BUY_B: buy_target_weight / Decimal("0.80"),
            PARTIAL_SELL: sell_target_weight / Decimal("0.80"),
        }
        chain = _chain(local_targets=local_targets)
        states = _market_states(chain.proposal)
        batch = _generate(
            chain,
            states=states,
            rules=(_market_rule(states),),
        )

    proposal_by_id = {line.instrument_id: line for line in chain.proposal.lines}
    exact_buy_target = Fraction(proposal_by_id[BUY_B].target_market_value) / Fraction(
        Decimal("10.00")
    )
    exact_sell_gap = Fraction(Decimal(250)) - Fraction(
        proposal_by_id[PARTIAL_SELL].target_market_value
    ) / Fraction(Decimal("10.00"))

    assert exact_buy_target == Fraction(almost_one_lot)
    assert exact_sell_gap == Fraction(almost_one_lot)
    assert exact_buy_target < 100
    assert exact_sell_gap < 100
    assert {line.instrument_id for line in batch.lines} == {BUY_A, ODD_LOT}
    assert next(line for line in batch.lines if line.instrument_id == BUY_A).quantity == 200
    assert next(line for line in batch.lines if line.instrument_id == ODD_LOT).quantity == 13


def test_target_changes_below_every_lot_produce_no_empty_draft_batch() -> None:
    chain = _chain(local_targets=NO_ACTION_LOCAL_TARGETS)
    assert chain.risk_result.status is RiskCheckStatus.PASS
    assert all(line.weight_delta == 0 for line in chain.proposal.lines)

    with pytest.raises(OrderDraftInputError, match="no non-zero order draft lines"):
        _generate(chain)
