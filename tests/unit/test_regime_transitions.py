"""Tests for chronological, deterministic market-regime transition hysteresis."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.regime import (
    EvidenceSide,
    MarketRegime,
    MarketRegimeResult,
    RegimeComponentScores,
    RegimeEvidence,
    RegimeInputIdentity,
)
from quant_agent.regime.contracts import stable_hash
from quant_agent.regime.transitions import (
    MarketRegimeTransitionEngine,
    RegimeTransitionConfig,
    RegimeTransitionEngine,
    RegimeTransitionInputMismatch,
    TransitionAction,
    apply_regime_transitions,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
START = date(2026, 8, 3)
CLASSIFIER_CONFIG_HASH = "a" * 64
TREND_CONFIG_HASH = "b" * 64
BREADTH_CONFIG_HASH = "c" * 64
CACHE_HASH = "d" * 64


def _result(
    offset: int,
    regime: MarketRegime,
    *,
    score: str = "60",
    confidence: str = "0.80",
    data_version: str = "snapshot-v1",
    model_version: str = "market-regime-v1",
    classifier_config_hash: str = CLASSIFIER_CONFIG_HASH,
    input_identity: RegimeInputIdentity | None = None,
) -> MarketRegimeResult:
    session_date = START + timedelta(days=offset)
    score_value = Decimal(score)
    confidence_value = Decimal(confidence)
    evidence = RegimeEvidence(
        feature="environment_score",
        value=score_value,
        criterion="fixed test evidence",
        rule_weight=Decimal(1),
        contribution=Decimal(100),
        side=EvidenceSide.SUPPORTING,
        rationale="fixed sequence supports the raw state",
    )
    components = RegimeComponentScores(
        trend=score_value,
        breadth=Decimal(50),
        turnover=Decimal(50),
        new_high_low=Decimal(50),
        downside_risk=Decimal(50),
        weighted_trend=score_value,
        weighted_breadth=Decimal(0),
        weighted_turnover=Decimal(0),
        weighted_new_high_low=Decimal(0),
        weighted_downside_risk=Decimal(0),
    )
    source_hash = stable_hash(
        {
            "confidence": confidence_value,
            "offset": offset,
            "regime": regime,
            "score": score_value,
        }
    )
    return MarketRegimeResult(
        as_of=datetime.combine(
            session_date,
            datetime.min.time().replace(hour=16),
            tzinfo=SHANGHAI,
        ),
        session_date=session_date,
        data_version=data_version,
        regime=regime,
        score=score_value,
        confidence=confidence_value,
        risk_budget_max=Decimal("0.50"),
        components=components,
        evidence=(evidence,),
        counter_evidence=(),
        invalidations=("raw evidence no longer holds",),
        model_version=model_version,
        config_hash=classifier_config_hash,
        input_identity=input_identity
        or RegimeInputIdentity(
            trend_feature_version="trend-v1",
            trend_config_hash=TREND_CONFIG_HASH,
            trend_cache_key=CACHE_HASH,
            breadth_feature_version="breadth-v1",
            breadth_config_hash=BREADTH_CONFIG_HASH,
            breadth_cache_key=CACHE_HASH,
        ),
        input_hash=source_hash,
        result_hash=source_hash,
    )


def test_transition_config_is_versioned_hashed_and_validated() -> None:
    baseline = RegimeTransitionConfig()

    assert baseline.config_hash == RegimeTransitionConfig().config_hash
    assert (
        replace(baseline, version="market-regime-transition-v2").config_hash != baseline.config_hash
    )

    invalid_kwargs: tuple[dict[str, object], ...] = (
        {"version": " "},
        {"confirmation_sessions": 0},
        {"confirmation_sessions": 1.5},
        {"bottom_recovery_confirmation_sessions": 0},
        {"severe_downtrend_confirmation_sessions": 4},
        {"severe_downtrend_enabled": 1},
        {"minimum_confidence": Decimal("-0.01")},
        {"minimum_confidence": Decimal("NaN")},
        {"severe_downtrend_minimum_confidence": Decimal("1.01")},
        {
            "minimum_confidence": Decimal("0.8"),
            "severe_downtrend_minimum_confidence": Decimal("0.7"),
        },
        {"severe_downtrend_score_max": Decimal("101")},
    )
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            RegimeTransitionConfig(**kwargs)  # type: ignore[arg-type]


def test_fixed_sequence_debounces_noise_confirms_and_records_every_event() -> None:
    engine = RegimeTransitionEngine(
        RegimeTransitionConfig(
            confirmation_sessions=3,
            bottom_recovery_confirmation_sessions=3,
            severe_downtrend_enabled=False,
        )
    )
    raw = (
        _result(0, MarketRegime.UPTREND, score="80"),
        _result(1, MarketRegime.DIVERGENT),
        _result(2, MarketRegime.UPTREND, score="75"),
        _result(3, MarketRegime.DIVERGENT),
        _result(4, MarketRegime.DIVERGENT),
        _result(5, MarketRegime.DIVERGENT),
        _result(6, MarketRegime.DIVERGENT),
    )

    outputs = engine.process_all(raw)

    assert tuple(item.raw_regime for item in outputs) == tuple(item.regime for item in raw)
    assert tuple(item.final_regime for item in outputs) == (
        MarketRegime.UPTREND,
        MarketRegime.UPTREND,
        MarketRegime.UPTREND,
        MarketRegime.UPTREND,
        MarketRegime.UPTREND,
        MarketRegime.DIVERGENT,
        MarketRegime.DIVERGENT,
    )
    assert tuple(item.event.action for item in outputs) == (
        TransitionAction.INITIALIZED,
        TransitionAction.CANDIDATE_STARTED,
        TransitionAction.CANDIDATE_CANCELLED,
        TransitionAction.CANDIDATE_STARTED,
        TransitionAction.CANDIDATE_ADVANCED,
        TransitionAction.TRANSITION_CONFIRMED,
        TransitionAction.STABLE,
    )
    assert outputs[1].pending_candidate is MarketRegime.DIVERGENT
    assert outputs[1].pending_count == 1
    assert outputs[4].pending_count == 2
    assert outputs[5].pending_candidate is None
    assert outputs[5].transitioned
    assert outputs[6].regime is MarketRegime.DIVERGENT
    assert outputs[6].raw_score == raw[6].score
    assert outputs[6].confidence == raw[6].confidence
    assert not outputs[6].confidence_is_probability
    assert engine.current_regime is MarketRegime.DIVERGENT
    assert engine.pending_candidate is None
    assert engine.pending_count == 0
    assert len(engine.events) == len(raw)
    assert engine.events[0].previous_event_hash is None
    assert all(
        current.previous_event_hash == previous.event_hash
        for previous, current in zip(engine.events[:-1], engine.events[1:], strict=True)
    )
    assert engine.events[5].evidence == raw[5].evidence
    assert engine.events[5].reason
    assert not engine.events[5].confidence_is_probability


def test_bottom_recovery_uses_its_own_confirmation_period() -> None:
    config = RegimeTransitionConfig(
        confirmation_sessions=2,
        bottom_recovery_confirmation_sessions=3,
        severe_downtrend_enabled=False,
    )
    outputs = apply_regime_transitions(
        (
            _result(0, MarketRegime.DOWNTREND, score="32"),
            _result(1, MarketRegime.BOTTOM_RECOVERY, score="48"),
            _result(2, MarketRegime.BOTTOM_RECOVERY, score="50"),
            _result(3, MarketRegime.BOTTOM_RECOVERY, score="52"),
        ),
        config,
    )

    assert [item.final_regime for item in outputs] == [
        MarketRegime.DOWNTREND,
        MarketRegime.DOWNTREND,
        MarketRegime.DOWNTREND,
        MarketRegime.BOTTOM_RECOVERY,
    ]
    assert outputs[2].required_confirmation_sessions == 3
    assert outputs[3].event.candidate_count == 3


def test_severe_downtrend_can_confirm_faster_without_using_future_records() -> None:
    config = RegimeTransitionConfig(
        confirmation_sessions=3,
        severe_downtrend_confirmation_sessions=1,
        severe_downtrend_score_max=Decimal(25),
        severe_downtrend_minimum_confidence=Decimal("0.85"),
    )
    severe_prefix = (
        _result(0, MarketRegime.UPTREND, score="80"),
        _result(1, MarketRegime.DOWNTREND, score="20", confidence="0.90"),
    )

    prefix_outputs = apply_regime_transitions(severe_prefix, config)
    full_outputs = apply_regime_transitions(
        (*severe_prefix, _result(2, MarketRegime.BOTTOM_RECOVERY, score="45")),
        config,
    )

    assert prefix_outputs == full_outputs[: len(prefix_outputs)]
    assert prefix_outputs[1].transitioned
    assert prefix_outputs[1].event.required_confirmation_sessions == 1
    assert "severe downtrend" in prefix_outputs[1].event.reason

    ordinary = apply_regime_transitions(
        (
            _result(0, MarketRegime.UPTREND, score="80"),
            _result(1, MarketRegime.DOWNTREND, score="20", confidence="0.80"),
        ),
        config,
    )
    assert ordinary[1].final_regime is MarketRegime.UPTREND
    assert ordinary[1].required_confirmation_sessions == 3


def test_low_confidence_rejects_and_resets_a_pending_candidate() -> None:
    engine = RegimeTransitionEngine(
        RegimeTransitionConfig(confirmation_sessions=2, severe_downtrend_enabled=False)
    )

    engine.process(_result(0, MarketRegime.UPTREND, score="80"))
    started = engine.process(_result(1, MarketRegime.RANGE_STRONG, confidence="0.70"))
    rejected = engine.process(_result(2, MarketRegime.RANGE_STRONG, confidence="0.59"))
    restarted = engine.process(_result(3, MarketRegime.DIVERGENT, confidence="0.70"))

    assert started.event.action is TransitionAction.CANDIDATE_STARTED
    assert rejected.event.action is TransitionAction.CANDIDATE_REJECTED
    assert rejected.pending_candidate is None
    assert "0.60" in rejected.event.reason
    assert restarted.event.action is TransitionAction.CANDIDATE_STARTED
    assert restarted.pending_candidate is MarketRegime.DIVERGENT
    assert restarted.pending_count == 1


def test_order_and_all_version_identities_are_strict_and_fail_without_mutation() -> None:
    baseline = _result(0, MarketRegime.UPTREND)
    identity_variants = (
        replace(_result(1, MarketRegime.UPTREND), data_version="snapshot-v2"),
        replace(_result(1, MarketRegime.UPTREND), model_version="market-regime-v2"),
        replace(_result(1, MarketRegime.UPTREND), config_hash="e" * 64),
        replace(
            _result(1, MarketRegime.UPTREND),
            input_identity=replace(
                _result(1, MarketRegime.UPTREND).input_identity,
                trend_feature_version="trend-v2",
            ),
        ),
        replace(
            _result(1, MarketRegime.UPTREND),
            input_identity=replace(
                _result(1, MarketRegime.UPTREND).input_identity,
                breadth_config_hash="f" * 64,
            ),
        ),
    )

    for variant in identity_variants:
        engine = RegimeTransitionEngine()
        first = engine.process(baseline)
        with pytest.raises(RegimeTransitionInputMismatch, match="versions aligned"):
            engine.process(variant)
        assert engine.events == (first.event,)
        assert engine.current_regime is MarketRegime.UPTREND

    engine = RegimeTransitionEngine()
    first = engine.process(baseline)
    for invalid_offset in (0, -1):
        with pytest.raises(RegimeTransitionInputMismatch, match="strictly increasing"):
            engine.process(_result(invalid_offset, MarketRegime.UPTREND))
        assert engine.events == (first.event,)


def test_fixed_sequence_is_reproducible_and_the_long_engine_alias_is_equivalent() -> None:
    inputs = (
        _result(0, MarketRegime.RANGE_STRONG),
        _result(1, MarketRegime.UPTREND, score="75"),
        _result(2, MarketRegime.UPTREND, score="76"),
        _result(3, MarketRegime.UPTREND, score="77"),
    )
    config = RegimeTransitionConfig(severe_downtrend_enabled=False)

    first = apply_regime_transitions(inputs, config)
    second = MarketRegimeTransitionEngine(config).process_all(inputs)

    assert first == second
    assert tuple(item.result_hash for item in first) == tuple(item.result_hash for item in second)
    assert apply_regime_transitions(()) == ()
