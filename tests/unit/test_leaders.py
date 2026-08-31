"""Tests for deterministic mainline leader scoring and classification."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.data.domain import FundamentalPoint, InstrumentType
from quant_agent.features.fundamental_quality import FundamentalQualityAnalyzer
from quant_agent.features.mainline import (
    MainlineEvidence,
    MainlineEvidenceSide,
    MainlineIndustryResult,
    MainlineInputIdentity,
    MainlineSnapshot,
    MainlineState,
    TopKPersistence,
)
from quant_agent.features.stock_strength import (
    HorizonStrengthMetrics,
    ScoreContribution,
    StockStrengthSnapshot,
    StockStrengthStatus,
    TrendQualityMetrics,
    VolumePriceConfirmation,
)
from quant_agent.features.tradeability import (
    CapacityEstimate,
    MarketTradeState,
    TradeabilityEligibility,
    TradeabilityReason,
    TradeabilityReasonCode,
    TradeabilityRequest,
    TradeabilitySnapshot,
    TradeSide,
)
from quant_agent.leaders import (
    LeaderComponent,
    LeaderConfig,
    LeaderEngine,
    LeaderEvidenceSide,
    LeaderExclusionCode,
    LeaderInputError,
    LeaderType,
    stable_leader_hash,
)
from quant_agent.regime.contracts import MarketRegime, stable_hash

TZ = ZoneInfo("Asia/Shanghai")
SESSION = date(2026, 8, 28)
AS_OF = datetime(2026, 8, 28, 16, 0, tzinfo=TZ)
INDUSTRY_A = "SW2021:A"
INDUSTRY_B = "SW2021:B"
HASH = "a" * 64


def _persistence(rank: int = 1) -> TopKPersistence:
    return TopKPersistence(
        top_k=5,
        window_sessions=5,
        observed_sessions=5,
        top_k_hits=5,
        consecutive_top_k_sessions=5,
        recent_ranks=(rank,) * 5,
    )


def _mainline_row(
    industry_id: str,
    state: MainlineState,
    *,
    score: str = "80",
    crowding_score: str = "0",
) -> MainlineIndustryResult:
    evidence = MainlineEvidence(
        feature="industry_strength",
        value=Decimal(score),
        criterion=">= 0",
        side=MainlineEvidenceSide.SUPPORTING,
        rationale="fixed mainline evidence",
    )
    return MainlineIndustryResult(
        industry_id=industry_id,
        state=state,
        previous_state=MainlineState.EMERGING,
        current_rank=1,
        current_strength_score=Decimal(score),
        mainline_score=Decimal(score),
        crowding_score=Decimal(crowding_score),
        persistence=_persistence(),
        crowding_signals=("turnover",) if state is MainlineState.CROWDED else (),
        supporting_evidence=(evidence,),
        counter_evidence=(),
        invalidations=("industry strength fades",),
        transition_reason="fixed test transition",
    )


def _mainline(
    *,
    state_a: MainlineState = MainlineState.CONFIRMED,
    state_b: MainlineState = MainlineState.EMERGING,
    crowding_score: str = "0",
) -> MainlineSnapshot:
    identity = MainlineInputIdentity(
        industry_feature_version="industry-v1",
        industry_config_hash=HASH,
        industry_cache_key=HASH,
        classification_version="SW2021",
        industry_level=3,
        regime_model_version="regime-v1",
        regime_classifier_config_hash=HASH,
        regime_input_hash=HASH,
        regime_trend_feature_version="trend-v1",
        regime_trend_config_hash=HASH,
        regime_breadth_feature_version="breadth-v1",
        regime_breadth_config_hash=HASH,
        regime_transition_version="transition-v1",
        regime_transition_config_hash=HASH,
        regime_transition_result_hash=HASH,
    )
    rows = (
        _mainline_row(
            INDUSTRY_A,
            state_a,
            crowding_score=crowding_score,
        ),
        _mainline_row(INDUSTRY_B, state_b, score="50"),
    )
    input_hash = stable_hash({"rows": tuple(item.industry_id for item in rows)})
    return MainlineSnapshot(
        session_date=SESSION,
        as_of=AS_OF,
        data_version="snapshot-v1",
        classification_version="SW2021",
        industry_level=3,
        market_regime=MarketRegime.UPTREND,
        model_version="mainline-v1",
        config_hash=HASH,
        input_identity=identity,
        input_hash=input_hash,
        industries=rows,
        previous_result_hash=None,
        result_hash=stable_hash({"input_hash": input_hash}),
    )


def _signed(normalized: Decimal) -> Decimal:
    return normalized * 2 - 100


def _stock(
    instrument_id: str,
    *,
    industry_id: str | None = INDUSTRY_A,
    short_return: str = "0.10",
    within_theme: str = "70",
    benchmark_resilience: str = "70",
    trend: str = "70",
    volume_price: str = "70",
    pullback: str = "0.05",
    status: StockStrengthStatus = StockStrengthStatus.READY,
    as_of: datetime = AS_OF,
) -> StockStrengthSnapshot:
    within = Decimal(within_theme)
    benchmark = Decimal(benchmark_resilience)
    component_rows = (
        ScoreContribution(
            component="industry_relative",
            raw_value=Decimal("0.01"),
            normalized_score=_signed(within),
            configured_weight=Decimal("0.5"),
            effective_weight=Decimal("0.5"),
            contribution=_signed(within) * Decimal("0.5"),
        ),
        ScoreContribution(
            component="benchmark_relative",
            raw_value=Decimal("0.01"),
            normalized_score=_signed(benchmark),
            configured_weight=Decimal("0.5"),
            effective_weight=Decimal("0.5"),
            contribution=_signed(benchmark) * Decimal("0.5"),
        ),
    )
    score = sum((item.contribution for item in component_rows), Decimal(0))
    result_hash = stable_leader_hash(
        {
            "as_of": as_of,
            "id": instrument_id,
            "return": short_return,
            "trend": trend,
            "within": within_theme,
        }
    )
    return StockStrengthSnapshot(
        instrument_id=instrument_id,
        benchmark_id="000300.SH",
        session_date=SESSION,
        as_of=as_of,
        data_version="snapshot-v1",
        classification_version="SW2021",
        industry_level=3,
        current_industry_id=industry_id,
        feature_version="stock-strength-v1",
        config_hash=HASH,
        status=status,
        horizons=(
            HorizonStrengthMetrics(
                horizon=5,
                stock_return=Decimal(short_return),
                benchmark_return=Decimal("0.01"),
                benchmark_relative_return=Decimal(short_return) - Decimal("0.01"),
                industry_return=Decimal("0.02") if industry_id else None,
                industry_relative_return=(
                    Decimal(short_return) - Decimal("0.02") if industry_id else None
                ),
                historical_industry_ids=(industry_id,) if industry_id else (),
                aligned_sessions=5,
            ),
        ),
        trend_quality=TrendQualityMetrics(
            latest_close=Decimal(10),
            moving_average=Decimal(9),
            position_vs_average=Decimal("0.1"),
            moving_average_slope=Decimal("0.02"),
            breakout_level=Decimal(9),
            breakout_hold_ratio=Decimal(1),
            pullback_depth=Decimal(pullback),
            distance_to_high=Decimal(0),
            score=_signed(Decimal(trend)),
        ),
        volume_confirmation=VolumePriceConfirmation(
            current_return=Decimal(short_return),
            current_volume=Decimal(100),
            average_prior_volume=Decimal(80),
            volume_ratio=Decimal("1.25"),
            score=_signed(Decimal(volume_price)),
            baseline_observations=20,
        ),
        contributions=component_rows,
        score=score,
        reason="degraded test input" if status is StockStrengthStatus.DEGRADED else None,
        selected_dates=(SESSION,),
        input_hash=stable_leader_hash({"id": instrument_id, "kind": "stock-input"}),
        cache_key=stable_leader_hash({"id": instrument_id, "kind": "stock-cache"}),
        result_hash=result_hash,
    )


def _tradeability(
    instrument_id: str,
    *,
    liquidity: str = "70",
    eligible: bool = True,
    as_of: datetime = AS_OF,
) -> TradeabilitySnapshot:
    normalized = Decimal(liquidity)
    request = TradeabilityRequest(
        instrument_id=instrument_id,
        instrument_type=InstrumentType.STOCK,
        market="SSE",
        session_date=SESSION,
        as_of=as_of,
        data_version="snapshot-v1",
        side=TradeSide.BUY,
        account_value=Decimal("1000000"),
        target_position_weight=Decimal("0.1"),
    )
    capacity_ratio = normalized / 100 * 2
    capacity = CapacityEstimate(
        account_value=request.account_value,
        target_position_weight=request.target_position_weight,
        target_notional=request.target_notional,
        average_turnover_short=Decimal("10000000"),
        average_turnover_long=Decimal("10000000"),
        participation_rate=Decimal("0.1"),
        short_window_capacity=Decimal("1000000"),
        long_window_capacity=Decimal("1000000"),
        daily_capacity=capacity_ratio * request.target_notional,
        capacity_ratio=capacity_ratio,
        maximum_account_weight=Decimal("0.5"),
        estimated_trade_days=Decimal(1),
    )
    reasons = (
        ()
        if eligible
        else (
            TradeabilityReason(
                code=TradeabilityReasonCode.SUSPENDED,
                blocking=True,
                message="instrument is suspended",
            ),
        )
    )
    result_hash = stable_leader_hash(
        {"eligible": eligible, "id": instrument_id, "liquidity": liquidity}
    )
    return TradeabilitySnapshot(
        request=request,
        feature_version="tradeability-v1",
        config_hash=HASH,
        rule_id="cn-stock",
        rule_version="rule-v1",
        rule_hash=HASH,
        market_state=(MarketTradeState.NORMAL if eligible else MarketTradeState.SUSPENDED),
        eligibility=(
            TradeabilityEligibility.ELIGIBLE if eligible else TradeabilityEligibility.INELIGIBLE
        ),
        reasons=reasons,
        contributions=(),
        listing_days=1000,
        average_turnover_short=Decimal("10000000"),
        average_turnover_long=Decimal("10000000"),
        turnover_rate_percentile=normalized / 100,
        turnover_rate_observation_count=60,
        short_window_dates=(SESSION,),
        long_window_dates=(SESSION,),
        upper_limit_price=Decimal(11),
        lower_limit_price=Decimal(9),
        capacity=capacity,
        input_hash=stable_leader_hash({"id": instrument_id, "kind": "trade-input"}),
        result_hash=result_hash,
    )


def _point(
    instrument_id: str,
    metric: str,
    value: str,
    *,
    period: date = date(2026, 6, 30),
    revision: str = "r1",
) -> FundamentalPoint:
    return FundamentalPoint(
        instrument_id=instrument_id,
        report_period=period,
        metric_name=metric,
        metric_value=Decimal(value),
        announced_at=datetime(2026, 7, 20, 18, tzinfo=TZ),
        available_at=datetime(2026, 7, 21, 9, tzinfo=TZ),
        provider_revision=revision,
        source="test",
    )


def _fundamental(instrument_id: str, *, risky: bool = False):  # type: ignore[no-untyped-def]
    values = (
        ("-5", "-1", "90", ("100", "-100", "100"))
        if risky
        else ("12", "0.8", "46", ("20", "25", "30"))
    )
    profitability, cash_flow, leverage, growth = values
    points = (
        _point(instrument_id, "roe", profitability),
        _point(instrument_id, "ocfps", cash_flow),
        _point(instrument_id, "debt_to_assets", leverage),
        _point(
            instrument_id,
            "netprofit_yoy",
            growth[0],
            period=date(2025, 12, 31),
            revision="g1",
        ),
        _point(
            instrument_id,
            "netprofit_yoy",
            growth[1],
            period=date(2026, 3, 31),
            revision="g2",
        ),
        _point(instrument_id, "netprofit_yoy", growth[2], revision="g3"),
    )
    return FundamentalQualityAnalyzer().analyze(
        instrument_id=instrument_id,
        as_of=AS_OF,
        data_version="fund-v1",
        points=points,
    )


def _rank(
    stocks: list[StockStrengthSnapshot],
    trades: list[TradeabilitySnapshot] | None = None,
    fundamentals: list[object] | None = None,
    *,
    mainline: MainlineSnapshot | None = None,
    config: LeaderConfig | None = None,
):  # type: ignore[no-untyped-def]
    return LeaderEngine(config).rank(
        mainline=mainline or _mainline(),
        stock_strength=stocks,
        tradeability=trades or [_tradeability(item.instrument_id) for item in stocks],
        fundamentals=fundamentals or [_fundamental(item.instrument_id) for item in stocks],  # type: ignore[arg-type]
    )


def test_cross_section_ranking_and_classification_do_not_reward_return_alone() -> None:
    stocks = [
        _stock(
            "SPIKE",
            short_return="0.30",
            within_theme="95",
            trend="20",
            volume_price="100",
            benchmark_resilience="45",
            pullback="0.18",
        ),
        _stock(
            "CAPACITY",
            short_return="0.25",
            within_theme="70",
            trend="70",
            volume_price="80",
            benchmark_resilience="70",
        ),
        _stock(
            "TREND",
            short_return="0.20",
            within_theme="85",
            trend="90",
            volume_price="70",
            benchmark_resilience="85",
        ),
        _stock(
            "WATCH",
            short_return="0.05",
            within_theme="30",
            trend="30",
            volume_price="30",
            benchmark_resilience="30",
            pullback="0.15",
        ),
    ]
    trades = [
        _tradeability("SPIKE", liquidity="20"),
        _tradeability("CAPACITY", liquidity="100"),
        _tradeability("TREND", liquidity="60"),
        _tradeability("WATCH", liquidity="30"),
    ]
    result = _rank(stocks, trades)
    by_id = {item.instrument_id: item for item in result.leaders}

    assert result.leaders[0].instrument_id in {"CAPACITY", "TREND"}
    assert by_id["SPIKE"].overall_rank > 1
    assert by_id["SPIKE"].leader_type is LeaderType.ELASTICITY
    assert by_id["SPIKE"].candidate_eligible is False
    assert by_id["CAPACITY"].leader_type is LeaderType.CAPACITY
    assert by_id["TREND"].leader_type is LeaderType.TREND
    assert by_id["WATCH"].leader_type is LeaderType.OBSERVATION
    assert {item.instrument_id for item in result.candidates} == {"CAPACITY", "TREND"}


def test_single_member_and_tied_returns_do_not_create_false_theme_leadership() -> None:
    single = _rank([_stock("ONLY", short_return="0.50")])
    single_component = next(
        item
        for item in single.leaders[0].components
        if item.component is LeaderComponent.THEME_LEADERSHIP
    )
    assert single_component.score < Decimal(50)

    tied = _rank(
        [
            _stock("TIE-A", short_return="0.20"),
            _stock("TIE-B", short_return="0.20"),
            _stock("LOW", short_return="0.10"),
        ]
    )
    tied_scores = {
        leader.instrument_id: next(
            item.score
            for item in leader.components
            if item.component is LeaderComponent.THEME_LEADERSHIP
        )
        for leader in tied.leaders
    }
    assert tied_scores["TIE-A"] == tied_scores["TIE-B"]
    assert tied_scores["TIE-A"] > tied_scores["LOW"]


def test_only_confirmed_or_crowded_pit_members_are_ranked() -> None:
    active = _stock("ACTIVE", industry_id=INDUSTRY_A)
    emerging = _stock("EMERGING-MEMBER", industry_id=INDUSTRY_B)
    missing_membership = _stock("NO-MEMBERSHIP", industry_id=None)
    result = _rank([active, emerging, missing_membership])

    assert tuple(item.instrument_id for item in result.leaders) == ("ACTIVE",)
    assert {item.instrument_id for item in result.exclusions} == {
        "EMERGING-MEMBER",
        "NO-MEMBERSHIP",
    }
    assert all(
        item.code is LeaderExclusionCode.NOT_ACTIVE_MAINLINE_MEMBER for item in result.exclusions
    )

    crowded = _rank(
        [_stock("CROWDED", industry_id=INDUSTRY_A)],
        mainline=_mainline(state_a=MainlineState.CROWDED, crowding_score="80"),
    )
    assert crowded.leaders[0].mainline_state is MainlineState.CROWDED
    assert crowded.leaders[0].risk_penalty == Decimal(8)


def test_unavailable_stock_strength_is_retained_as_exclusion() -> None:
    unavailable = _stock("SUSPENDED")
    unavailable = replace(
        unavailable,
        status=StockStrengthStatus.SUSPENDED,
        horizons=(),
        trend_quality=None,
        volume_confirmation=None,
        contributions=(),
        score=None,
        reason="suspended",
    )
    result = _rank([unavailable])
    assert not result.leaders
    assert result.exclusions[0].code is LeaderExclusionCode.STOCK_STRENGTH_UNAVAILABLE


def test_ineligible_tradeability_keeps_leader_and_structured_reason() -> None:
    stock = _stock(
        "BLOCKED-CAPACITY",
        short_return="0.20",
        within_theme="80",
        trend="80",
        volume_price="80",
    )
    result = _rank(
        [stock],
        [_tradeability("BLOCKED-CAPACITY", liquidity="100", eligible=False)],
    )
    leader = result.leaders[0]

    assert leader.leader_type is LeaderType.TREND
    assert leader.candidate_eligible is False
    assert any("SUSPENDED" in reason for reason in leader.ineligibility_reasons)
    assert any(
        item.feature == "TRADEABILITY" and item.side is LeaderEvidenceSide.OPPOSING
        for item in leader.counter_evidence
    )


def test_future_upstream_snapshots_are_rejected() -> None:
    stock = _stock("FUTURE", as_of=AS_OF + timedelta(seconds=1))
    with pytest.raises(LeaderInputError, match="future"):
        _rank([stock])

    ordinary = _stock("FUTURE-TRADE")
    with pytest.raises(LeaderInputError, match="future"):
        _rank(
            [ordinary],
            [_tradeability("FUTURE-TRADE", as_of=AS_OF + timedelta(seconds=1))],
        )

    future_fundamental = replace(
        _fundamental("FUTURE-FUNDAMENTAL"),
        as_of=AS_OF + timedelta(seconds=1),
    )
    with pytest.raises(LeaderInputError, match="future"):
        _rank(
            [_stock("FUTURE-FUNDAMENTAL")],
            fundamentals=[future_fundamental],
        )


def test_input_order_and_hashes_are_deterministic_and_version_bound() -> None:
    stocks = [_stock("B"), _stock("A", short_return="0.15")]
    trades = [_tradeability("B"), _tradeability("A")]
    fundamentals = [_fundamental("B"), _fundamental("A")]
    first = _rank(stocks, trades, fundamentals)
    repeated = _rank(
        list(reversed(stocks)),
        list(reversed(trades)),
        list(reversed(fundamentals)),
    )
    changed = _rank(
        stocks,
        trades,
        fundamentals,
        config=replace(LeaderConfig(), minimum_candidate_score=Decimal(61)),
    )

    assert repeated == first
    assert first.input_hash == repeated.input_hash
    assert first.result_hash == repeated.result_hash
    assert changed.config_hash != first.config_hash
    assert changed.input_hash != first.input_hash
    assert first.identity_payload()["result_hash"] == first.result_hash
    assert all(
        item.upstream_identity.mainline_result_hash == _mainline().result_hash
        for item in first.leaders
    )


def test_score_components_evidence_risks_and_conditions_are_reconcilable() -> None:
    stock = _stock("RISKY", status=StockStrengthStatus.DEGRADED)
    result = _rank([stock], fundamentals=[_fundamental("RISKY", risky=True)])
    leader = result.leaders[0]

    assert tuple(item.component for item in leader.components) == tuple(LeaderComponent)
    assert sum((item.weight for item in leader.components), Decimal(0)) == 1
    assert sum((item.contribution for item in leader.components), Decimal(0)) == leader.gross_score
    assert sum((item.penalty for item in leader.risks), Decimal(0)) == leader.risk_penalty
    assert leader.score == max(Decimal(0), leader.gross_score - leader.risk_penalty)
    assert leader.supporting_evidence
    assert leader.counter_evidence
    assert leader.observation_conditions
    assert leader.invalidations
    assert leader.risk_penalty <= LeaderConfig().maximum_risk_penalty


def test_missing_or_duplicate_upstream_instruments_fail_closed() -> None:
    stock = _stock("A")
    with pytest.raises(LeaderInputError, match="identical IDs"):
        _rank([stock], trades=[_tradeability("OTHER")])
    with pytest.raises(LeaderInputError, match="unique"):
        LeaderEngine().rank(
            mainline=_mainline(),
            stock_strength=[stock, stock],
            tradeability=[_tradeability("A")],
            fundamentals=[_fundamental("A")],
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LeaderConfig(version=" "),
        lambda: LeaderConfig(within_theme_strength_weight=Decimal("0.20")),
        lambda: LeaderConfig(liquidity_percentile_weight=Decimal("0.40")),
        lambda: LeaderConfig(minimum_candidate_score=Decimal(101)),
        lambda: LeaderConfig(capacity_ratio_full_score=Decimal(0)),
        lambda: LeaderConfig(crowded_penalty_rate=Decimal("1.1")),
        lambda: LeaderConfig(maximum_risk_penalty=Decimal("NaN")),
    ],
)
def test_config_weight_and_numeric_boundaries_are_rejected(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_output_is_frozen_and_candidate_types_are_narrow() -> None:
    result = _rank([_stock("A")])
    leader = result.leaders[0]
    assert not leader.candidate_eligible or leader.leader_type in {
        LeaderType.TREND,
        LeaderType.CAPACITY,
    }
    with pytest.raises(FrozenInstanceError):
        leader.score = Decimal(0)  # type: ignore[misc]
