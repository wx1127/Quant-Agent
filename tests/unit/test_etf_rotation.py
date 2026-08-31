"""Fixed PIT samples for the baseline ETF rotation strategy and adapters."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.backtest import (
    BenchmarkBar,
    ResearchPriceBar,
    SignalDirection,
    TradableInstrumentType,
    VectorizedBacktestEngine,
    VectorizedBacktestRequest,
)
from quant_agent.regime import (
    EvidenceSide,
    MarketRegime,
    MarketRegimeResult,
    RegimeComponentScores,
    RegimeEvidence,
    RegimeInputIdentity,
    RegimeTransitionResult,
    TransitionAction,
    TransitionEvent,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.strategies.etf_rotation import (
    ETFRotationConfig,
    ETFRotationDecisionStatus,
    ETFRotationInputError,
    ETFRotationRequest,
    ETFRotationStrategy,
    ETFUniverseEligibility,
    ETFUniverseRevision,
    target_weight_map,
    to_event_targets,
    to_weight_signals,
)

TZ = ZoneInfo("Asia/Shanghai")
START = date(2026, 8, 3)
DAYS = tuple(START + timedelta(days=value) for value in range(7))
SIGNAL_DAY = DAYS[5]
NEXT_DAY = DAYS[6]
AS_OF = datetime.combine(SIGNAL_DAY, time(18), tzinfo=TZ)
ETF_A = "CN.SH.510300"
ETF_B = "CN.SH.510500"
ETF_C = "CN.SH.512100"
ETF_D = "CN.SH.512880"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _config(**changes: object) -> ETFRotationConfig:
    values: dict[str, object] = {
        "momentum_horizons": (2, 3, 4),
        "momentum_weights": (Decimal("0.2"), Decimal("0.3"), Decimal("0.5")),
        "volatility_window": 2,
        "volatility_annualization_sessions": 1,
        "volatility_floor": Decimal("0.0001"),
        "long_trend_window": 4,
        "rebalance_frequency_sessions": 2,
        "top_n": 2,
        "maximum_instrument_weight": Decimal("0.5"),
    }
    values.update(changes)
    return ETFRotationConfig(**values)  # type: ignore[arg-type]


def _at(day: date, hour: int = 16, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=TZ)


def _regime(
    regime: MarketRegime = MarketRegime.UPTREND,
    *,
    as_of: datetime = AS_OF,
    data_version: str = "snapshot-v1",
) -> RegimeTransitionResult:
    signal_date = as_of.date()
    score = Decimal(80) if regime is not MarketRegime.DOWNTREND else Decimal(20)
    risk_budget = Decimal("0.8") if regime is not MarketRegime.DOWNTREND else Decimal("0.15")
    components = RegimeComponentScores(
        trend=score,
        breadth=Decimal(0),
        turnover=Decimal(0),
        new_high_low=Decimal(0),
        downside_risk=Decimal(0),
        weighted_trend=score,
        weighted_breadth=Decimal(0),
        weighted_turnover=Decimal(0),
        weighted_new_high_low=Decimal(0),
        weighted_downside_risk=Decimal(0),
    )
    evidence = (
        RegimeEvidence(
            feature="fixed-regime",
            value=score,
            criterion=">= 0",
            rule_weight=Decimal(1),
            contribution=Decimal(1),
            side=EvidenceSide.SUPPORTING,
            rationale="fixed ETF rotation regime",
        ),
    )
    input_hash = stable_hash(
        {"data_version": data_version, "regime": regime, "signal_date": signal_date}
    )
    raw_hash = stable_hash({"input_hash": input_hash, "stage": "raw"})
    raw = MarketRegimeResult(
        as_of=as_of,
        session_date=signal_date,
        data_version=data_version,
        regime=regime,
        score=score,
        confidence=Decimal("0.8"),
        risk_budget_max=risk_budget,
        components=components,
        evidence=evidence,
        counter_evidence=(),
        invalidations=("fixed invalidation",),
        model_version="regime-v1",
        config_hash=HASH_A,
        input_identity=RegimeInputIdentity(
            trend_feature_version="trend-v1",
            trend_config_hash=HASH_B,
            trend_cache_key=HASH_C,
            breadth_feature_version="breadth-v1",
            breadth_config_hash=HASH_D,
            breadth_cache_key=HASH_E,
        ),
        input_hash=input_hash,
        result_hash=raw_hash,
    )
    transition_hash = stable_hash({"config": "transition-v1"})
    event_hash = stable_hash({"raw_hash": raw_hash, "stage": "event"})
    event = TransitionEvent(
        session_date=signal_date,
        as_of=as_of,
        raw_regime=regime,
        previous_regime=regime,
        final_regime=regime,
        action=TransitionAction.STABLE,
        candidate_regime=None,
        candidate_count=0,
        required_confirmation_sessions=None,
        raw_score=score,
        raw_confidence=Decimal("0.8"),
        reason="fixed stable regime",
        supporting_evidence=evidence,
        source_result_hash=raw_hash,
        transition_config_version="transition-v1",
        transition_config_hash=transition_hash,
        previous_event_hash=None,
        event_hash=event_hash,
    )
    return RegimeTransitionResult(
        raw_result=raw,
        raw_regime=regime,
        final_regime=regime,
        pending_candidate=None,
        pending_count=0,
        required_confirmation_sessions=None,
        event=event,
        transition_config_version="transition-v1",
        transition_config_hash=transition_hash,
        result_hash=stable_hash({"event_hash": event_hash, "stage": "transition"}),
    )


def _request(*, session_index: int = 0, data_version: str = "snapshot-v1") -> ETFRotationRequest:
    return ETFRotationRequest(
        signal_date=SIGNAL_DAY,
        as_of=AS_OF,
        data_version=data_version,
        session_index=session_index,
    )


def _universe_revision(
    instrument_id: str,
    *,
    enabled: bool = True,
    listed_on: date = DAYS[0],
    delisted_on: date | None = None,
    effective_from: date = DAYS[0],
    effective_to: date | None = None,
    available_at: datetime | None = None,
    revision: str = "u1",
    data_version: str = "snapshot-v1",
) -> ETFUniverseRevision:
    return ETFUniverseRevision(
        instrument_id=instrument_id,
        listed_on=listed_on,
        delisted_on=delisted_on,
        effective_from=effective_from,
        effective_to=effective_to,
        enabled=enabled,
        available_at=available_at or _at(DAYS[0], 9),
        revision=revision,
        data_version=data_version,
    )


def _universe() -> tuple[ETFUniverseRevision, ...]:
    return (
        _universe_revision(ETF_A),
        _universe_revision(ETF_B),
        _universe_revision(ETF_C),
        _universe_revision(ETF_D, enabled=False),
    )


def _bar(
    instrument_id: str,
    trading_day: date,
    close: str,
    *,
    available_at: datetime | None = None,
    data_version: str = "snapshot-v1",
    revision: str = "p1",
    instrument_type: TradableInstrumentType = TradableInstrumentType.ETF,
) -> ResearchPriceBar:
    return ResearchPriceBar(
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        trading_day=trading_day,
        open_price=Decimal(close),
        close_price=Decimal(close),
        available_at=available_at or _at(trading_day),
        data_version=data_version,
        revision=revision,
    )


def _price_history(
    instrument_id: str,
    closes: tuple[str, ...],
    *,
    days: tuple[date, ...] = DAYS[1:6],
) -> tuple[ResearchPriceBar, ...]:
    return tuple(_bar(instrument_id, day, close) for day, close in zip(days, closes, strict=True))


def _prices() -> tuple[ResearchPriceBar, ...]:
    return (
        *_price_history(ETF_A, ("10", "11", "12", "13", "14")),
        *_price_history(ETF_B, ("20", "22", "24", "26", "28")),
        *_price_history(ETF_C, ("10", "9", "8", "7", "6")),
    )


def test_fixed_sample_ranks_momentum_applies_risk_budget_and_caps_weights() -> None:
    strategy = ETFRotationStrategy(_config())
    first = strategy.decide(
        request=_request(),
        universe_revisions=_universe(),
        prices=_prices(),
        regime=_regime(),
    )
    second = strategy.decide(
        request=_request(),
        universe_revisions=reversed(_universe()),
        prices=reversed(_prices()),
        regime=_regime(),
    )

    assert first == second
    assert strategy.config == _config()
    assert first.status is ETFRotationDecisionStatus.REBALANCE
    assert first.market_regime is MarketRegime.UPTREND
    assert first.regime_risk_budget == Decimal("0.8")
    assert first.configured_regime_exposure == Decimal("0.8")
    assert first.gross_target_weight == Decimal("0.8")
    assert first.cash_target_weight == Decimal("0.2")
    assert target_weight_map(first.targets) == {
        ETF_A: Decimal("0.4"),
        ETF_B: Decimal("0.4"),
        ETF_C: Decimal(0),
        ETF_D: Decimal(0),
    }
    assert tuple(item.instrument_id for item in first.universe) == (
        ETF_A,
        ETF_B,
        ETF_C,
        ETF_D,
    )
    candidate_a = next(item for item in first.candidates if item.instrument_id == ETF_A)
    candidate_b = next(item for item in first.candidates if item.instrument_id == ETF_B)
    candidate_c = next(item for item in first.candidates if item.instrument_id == ETF_C)
    candidate_d = next(item for item in first.candidates if item.instrument_id == ETF_D)
    assert candidate_a.rank == 1
    assert candidate_b.rank == 2
    assert candidate_a.metrics is not None
    assert candidate_b.metrics is not None
    assert candidate_a.metrics.weighted_momentum == candidate_b.metrics.weighted_momentum
    assert candidate_a.metrics.risk_adjusted_momentum == candidate_b.metrics.risk_adjusted_momentum
    assert float(candidate_a.metrics.weighted_momentum) == pytest.approx(0.31515151515151515)
    assert candidate_a.metrics.above_long_trend
    assert candidate_c.metrics is not None and not candidate_c.metrics.above_long_trend
    assert not candidate_c.qualifies
    assert candidate_d.universe_eligibility is ETFUniverseEligibility.DISABLED
    assert not candidate_d.qualifies
    assert first.identity_payload()["result_hash"] == first.result_hash
    assert all(
        len(value) == 64
        for value in (
            first.config_hash,
            first.universe_input_hash,
            first.price_input_hash,
            first.input_hash,
            first.result_hash,
        )
    )

    capped = ETFRotationStrategy(_config(top_n=1)).decide(
        request=_request(),
        universe_revisions=_universe(),
        prices=_prices(),
        regime=_regime(),
    )
    assert capped.gross_target_weight == Decimal("0.5")
    assert capped.cash_target_weight == Decimal("0.5")
    assert max(target_weight_map(capped.targets).values()) == Decimal("0.5")


def test_future_price_and_universe_revision_cannot_change_old_decision() -> None:
    strategy = ETFRotationStrategy(_config())
    baseline = strategy.decide(
        request=_request(),
        universe_revisions=_universe(),
        prices=_prices(),
        regime=_regime(),
    )
    future_disable = _universe_revision(
        ETF_A,
        enabled=False,
        available_at=_at(NEXT_DAY, 9),
        revision="future-u2",
    )
    future_new_member = _universe_revision(
        "CN.SH.513900",
        available_at=_at(NEXT_DAY, 9),
        revision="future-new",
    )
    future_price = _bar(
        ETF_A,
        NEXT_DAY,
        "1",
        available_at=_at(NEXT_DAY),
        revision="future-p2",
    )

    appended = strategy.decide(
        request=_request(),
        universe_revisions=(*_universe(), future_disable, future_new_member),
        prices=(*_prices(), future_price),
        regime=_regime(),
    )

    assert appended == baseline
    assert appended.universe_input_hash == baseline.universe_input_hash
    assert appended.price_input_hash == baseline.price_input_hash
    assert appended.result_hash == baseline.result_hash


def test_downtrend_and_no_qualified_etf_emit_explicit_all_cash_targets() -> None:
    strategy = ETFRotationStrategy(_config())
    downtrend = strategy.decide(
        request=_request(session_index=1),
        universe_revisions=_universe(),
        prices=(),
        regime=_regime(MarketRegime.DOWNTREND),
    )
    weak_prices = tuple(
        bar
        for instrument_id in (ETF_A, ETF_B, ETF_C)
        for bar in _price_history(instrument_id, ("10", "9", "8", "7", "6"))
    )
    no_qualified = strategy.decide(
        request=_request(),
        universe_revisions=_universe(),
        prices=weak_prices,
        regime=_regime(),
    )

    for decision in (downtrend, no_qualified):
        assert decision.status is ETFRotationDecisionStatus.CASH
        assert decision.gross_target_weight == 0
        assert decision.cash_target_weight == 1
        assert target_weight_map(decision.targets) == {
            ETF_A: Decimal(0),
            ETF_B: Decimal(0),
            ETF_C: Decimal(0),
            ETF_D: Decimal(0),
        }
    assert "DOWNTREND" in downtrend.reason
    assert "no PIT-eligible ETF" in no_qualified.reason


def test_rebalance_frequency_emits_no_target_and_status_history_is_pit() -> None:
    strategy = ETFRotationStrategy(_config())
    future_listing = "CN.SH.513000"
    delisted = "CN.SH.513100"
    no_interval = "CN.SH.513200"
    universe = (
        *_universe(),
        _universe_revision(future_listing, listed_on=NEXT_DAY),
        _universe_revision(delisted, delisted_on=DAYS[4]),
        _universe_revision(no_interval, effective_from=NEXT_DAY),
    )
    decision = strategy.decide(
        request=_request(session_index=1),
        universe_revisions=universe,
        prices=(),
        regime=_regime(),
    )
    eligibility = {item.instrument_id: item.eligibility for item in decision.universe}

    assert decision.status is ETFRotationDecisionStatus.NO_REBALANCE
    assert not decision.has_targets
    assert decision.gross_target_weight is None
    assert decision.cash_target_weight is None
    assert not decision.targets
    assert eligibility[future_listing] is ETFUniverseEligibility.NOT_LISTED
    assert eligibility[delisted] is ETFUniverseEligibility.DELISTED
    assert eligibility[no_interval] is ETFUniverseEligibility.NO_EFFECTIVE_REVISION
    assert to_weight_signals(decision) == ()
    assert (
        to_event_targets(
            decision,
            run_id="run-no-rebalance",
            next_trading_day=NEXT_DAY,
            execution_time=_at(NEXT_DAY, 9, 30),
        ).target_events
        == ()
    )


def test_vectorized_and_event_adapters_share_targets_and_next_session_semantics() -> None:
    decision = ETFRotationStrategy(_config()).decide(
        request=_request(),
        universe_revisions=_universe(),
        prices=_prices(),
        regime=_regime(),
    )
    weight_signals = to_weight_signals(decision)
    event_batch = to_event_targets(
        decision,
        run_id="run-etf-rotation",
        next_trading_day=NEXT_DAY,
        execution_time=_at(NEXT_DAY, 9, 30),
        starting_sequence=10,
    )

    expected = target_weight_map(decision.targets)
    assert target_weight_map(weight_signals) == expected
    assert target_weight_map(event_batch.target_events) == expected
    assert event_batch.execution_semantics == "NEXT_TRADING_SESSION"
    assert all(item.signal_date == SIGNAL_DAY and item.as_of == AS_OF for item in weight_signals)
    assert all(item.trading_day == SIGNAL_DAY for item in event_batch.signal_events)
    assert all(item.trading_day == NEXT_DAY for item in event_batch.target_events)
    assert all(item.signal_time == AS_OF for item in event_batch.target_events)
    assert tuple(item.sequence for item in event_batch.signal_events) == (10, 11, 12, 13)
    assert tuple(item.sequence for item in event_batch.target_events) == (14, 15, 16, 17)
    assert tuple(item.direction for item in event_batch.signal_events) == (
        SignalDirection.LONG,
        SignalDirection.LONG,
        SignalDirection.FLAT,
        SignalDirection.FLAT,
    )
    repeated = to_event_targets(
        decision,
        run_id="run-etf-rotation",
        next_trading_day=NEXT_DAY,
        execution_time=_at(NEXT_DAY, 9, 30),
        starting_sequence=10,
    )
    assert repeated == event_batch

    vector_prices = tuple(
        _bar(instrument_id, day, "10")
        for instrument_id in (ETF_A, ETF_B, ETF_C, ETF_D)
        for day in (SIGNAL_DAY, NEXT_DAY)
    )
    benchmark = tuple(
        BenchmarkBar(
            benchmark_id="CN.SH.000300",
            trading_day=day,
            close_price=Decimal(100),
            available_at=_at(day),
            data_version="snapshot-v1",
        )
        for day in (SIGNAL_DAY, NEXT_DAY)
    )
    vector_result = VectorizedBacktestEngine().run(
        request=VectorizedBacktestRequest(
            run_id="run-etf-vector",
            data_version="snapshot-v1",
            strategy_version=decision.strategy_version,
            as_of=_at(NEXT_DAY, 18),
            trading_calendar=(SIGNAL_DAY, NEXT_DAY),
            initial_nav=Decimal(1000),
        ),
        signals=weight_signals,
        prices=vector_prices,
        benchmark=benchmark,
    )
    assert vector_result.points[0].executed_signal_date is None
    assert vector_result.points[1].executed_signal_date == SIGNAL_DAY


def test_rejects_regime_price_universe_alignment_and_revision_ambiguity() -> None:
    strategy = ETFRotationStrategy(_config())
    with pytest.raises(ETFRotationInputError, match="align exactly"):
        strategy.decide(
            request=_request(),
            universe_revisions=_universe(),
            prices=_prices(),
            regime=_regime(as_of=_at(DAYS[4], 18)),
        )
    with pytest.raises(ETFRotationInputError, match="regime data_version"):
        strategy.decide(
            request=_request(),
            universe_revisions=_universe(),
            prices=_prices(),
            regime=_regime(data_version="snapshot-v2"),
        )
    with pytest.raises(ETFRotationInputError, match="price data_version"):
        strategy.decide(
            request=_request(),
            universe_revisions=_universe(),
            prices=(*_prices(), _bar(ETF_A, SIGNAL_DAY, "14", data_version="snapshot-v2")),
            regime=_regime(),
        )
    with pytest.raises(ETFRotationInputError, match="ETF type"):
        strategy.decide(
            request=_request(),
            universe_revisions=_universe(),
            prices=(
                *_prices(),
                _bar(ETF_A, SIGNAL_DAY, "14", instrument_type=TradableInstrumentType.STOCK),
            ),
            regime=_regime(),
        )

    universe = _universe()
    ambiguous_universe = replace(
        universe[0],
        enabled=False,
        revision="ambiguous",
    )
    with pytest.raises(ETFRotationInputError, match="ambiguous ETF universe"):
        strategy.decide(
            request=_request(),
            universe_revisions=(*universe, ambiguous_universe),
            prices=_prices(),
            regime=_regime(),
        )
    overlapping = _universe_revision(
        ETF_A,
        effective_from=DAYS[1],
        revision="overlap",
    )
    with pytest.raises(ETFRotationInputError, match="overlapping"):
        strategy.decide(
            request=_request(),
            universe_revisions=(*universe, overlapping),
            prices=_prices(),
            regime=_regime(),
        )
    original_bar = _prices()[0]
    ambiguous_bar = replace(original_bar, close_price=Decimal("999"), revision="ambiguous")
    with pytest.raises(ETFRotationInputError, match="ambiguous ETF price"):
        strategy.decide(
            request=_request(),
            universe_revisions=universe,
            prices=(*_prices(), ambiguous_bar),
            regime=_regime(),
        )


def test_rejects_misaligned_rankable_histories_and_invalid_adapter_execution_time() -> None:
    strategy = ETFRotationStrategy(_config())
    misaligned_days = (DAYS[0], DAYS[2], DAYS[3], DAYS[4], DAYS[5])
    misaligned_prices = (
        *_price_history(ETF_A, ("10", "11", "12", "13", "14")),
        *_price_history(
            ETF_B,
            ("20", "22", "24", "26", "28"),
            days=misaligned_days,
        ),
        *_price_history(ETF_C, ("10", "9", "8", "7", "6")),
    )
    with pytest.raises(ETFRotationInputError, match="not aligned"):
        strategy.decide(
            request=_request(),
            universe_revisions=_universe(),
            prices=misaligned_prices,
            regime=_regime(),
        )

    decision = strategy.decide(
        request=_request(),
        universe_revisions=_universe(),
        prices=_prices(),
        regime=_regime(),
    )
    with pytest.raises(ValueError, match="execution_time"):
        to_event_targets(
            decision,
            run_id="run-bad-time",
            next_trading_day=NEXT_DAY,
            execution_time=_at(SIGNAL_DAY, 9, 30),
        )
    with pytest.raises(ValueError, match="after the signal session"):
        to_event_targets(
            decision,
            run_id="run-bad-day",
            next_trading_day=SIGNAL_DAY,
            execution_time=_at(SIGNAL_DAY, 19),
        )


def test_config_is_versioned_and_rejects_unsafe_parameter_combinations() -> None:
    baseline = _config()
    assert ETFRotationConfig().momentum_horizons == (20, 60, 120)
    assert ETFRotationConfig().required_price_observations == 121
    assert baseline.required_price_observations == 5
    assert baseline.exposure_for(MarketRegime.DOWNTREND) == 0
    assert len(baseline.config_hash) == 64
    assert replace(baseline, top_n=1).config_hash != baseline.config_hash
    with pytest.raises(ValueError, match="sum to one"):
        _config(momentum_weights=(Decimal("0.2"), Decimal("0.3"), Decimal("0.4")))
    with pytest.raises(ValueError, match="maximum_instrument_weight"):
        _config(maximum_instrument_weight=Decimal(0))
    with pytest.raises(ValueError, match="regime exposure"):
        _config(uptrend_exposure=Decimal("1.1"))
    with pytest.raises(ValueError, match="frequency"):
        _config(rebalance_frequency_sessions=0)


def test_decision_contract_rejects_candidate_target_and_rank_drift() -> None:
    decision = ETFRotationStrategy(_config()).decide(
        request=_request(),
        universe_revisions=_universe(),
        prices=_prices(),
        regime=_regime(),
    )
    first, second, third, *_rest = decision.candidates

    with pytest.raises(ValueError, match="cannot retain a rank"):
        replace(third, rank=3)
    with pytest.raises(ValueError, match="target weights must match"):
        replace(
            decision,
            candidates=(
                replace(first, selected=False, target_weight=Decimal(0)),
                *decision.candidates[1:],
            ),
        )
    with pytest.raises(ValueError, match="unique and contiguous"):
        replace(
            decision,
            candidates=(first, replace(second, rank=first.rank), *decision.candidates[2:]),
        )

    no_rebalance = ETFRotationStrategy(_config()).decide(
        request=_request(session_index=1),
        universe_revisions=_universe(),
        prices=(),
        regime=_regime(),
    )
    with pytest.raises(ValueError, match="cannot emit candidates"):
        replace(no_rebalance, candidates=(third,))
