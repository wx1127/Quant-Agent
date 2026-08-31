"""Fixed synthetic tests for the P2-T08 regime/mainline evaluation framework."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.evaluation import (
    REGIME_MAINLINE_EVALUATION_CODE_VERSION,
    EvaluationPeriod,
    ForwardRelativeReturnLabel,
    RegimeMainlineEvaluationConfig,
    RegimeMainlineEvaluationError,
    evaluate_regime_mainline,
)
from quant_agent.features.mainline import (
    MainlineEvidence,
    MainlineEvidenceSide,
    MainlineIndustryResult,
    MainlineInputIdentity,
    MainlineSnapshot,
    MainlineState,
    TopKPersistence,
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

TZ = ZoneInfo("Asia/Shanghai")
START_DATE = date(2026, 1, 5)
ALL_SESSIONS = tuple(START_DATE + timedelta(days=value) for value in range(30))
INDUSTRIES = ("SW2021:A", "SW2021:B")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
TRANSITION_CONFIG_HASH = stable_hash({"kind": "transition-config"})
MAINLINE_CONFIG_HASH = stable_hash({"kind": "mainline-config"})


def _regime_evidence(score: Decimal) -> tuple[RegimeEvidence, ...]:
    return (
        RegimeEvidence(
            feature="fixed_synthetic_score",
            value=score,
            criterion=">= 0",
            rule_weight=Decimal(1),
            contribution=Decimal(1),
            side=EvidenceSide.SUPPORTING,
            rationale="fixed synthetic evidence",
        ),
    )


def _regime_result(
    offset: int,
    regime: MarketRegime,
    *,
    previous_regime: MarketRegime | None,
    previous_event_hash: str | None,
) -> RegimeTransitionResult:
    session_date = ALL_SESSIONS[offset]
    as_of = datetime.combine(session_date, datetime.min.time(), tzinfo=TZ).replace(hour=16)
    score = Decimal(80) if regime is MarketRegime.UPTREND else Decimal(20)
    evidence = _regime_evidence(score)
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
    input_hash = stable_hash({"offset": offset, "regime": regime})
    raw_result_hash = stable_hash({"input_hash": input_hash, "stage": "raw"})
    raw = MarketRegimeResult(
        as_of=as_of,
        session_date=session_date,
        data_version="pit-signals-v1",
        regime=regime,
        score=score,
        confidence=Decimal("0.8"),
        risk_budget_max=Decimal("0.5"),
        components=components,
        evidence=evidence,
        counter_evidence=(),
        invalidations=("fixed synthetic invalidation",),
        model_version="regime-model-v1",
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
        result_hash=raw_result_hash,
    )
    event_hash = stable_hash(
        {
            "offset": offset,
            "previous_event_hash": previous_event_hash,
            "regime": regime,
        }
    )
    event = TransitionEvent(
        session_date=session_date,
        as_of=as_of,
        raw_regime=regime,
        previous_regime=previous_regime,
        final_regime=regime,
        action=(
            TransitionAction.INITIALIZED
            if previous_regime is None
            else (
                TransitionAction.STABLE
                if previous_regime is regime
                else TransitionAction.TRANSITION_CONFIRMED
            )
        ),
        candidate_regime=None,
        candidate_count=0,
        required_confirmation_sessions=None,
        raw_score=score,
        raw_confidence=Decimal("0.8"),
        reason="fixed synthetic transition",
        supporting_evidence=evidence,
        source_result_hash=raw_result_hash,
        transition_config_version="transition-v1",
        transition_config_hash=TRANSITION_CONFIG_HASH,
        previous_event_hash=previous_event_hash,
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
        transition_config_hash=TRANSITION_CONFIG_HASH,
        result_hash=stable_hash({"event_hash": event_hash, "stage": "stabilized"}),
    )


def _mainline_row(
    industry_id: str,
    rank: int,
    state: MainlineState,
) -> MainlineIndustryResult:
    evidence = (
        MainlineEvidence(
            feature="fixed_synthetic_rank",
            value=Decimal(rank),
            criterion="<= 2",
            side=MainlineEvidenceSide.SUPPORTING,
            rationale="fixed synthetic Top-K evidence",
        ),
    )
    return MainlineIndustryResult(
        industry_id=industry_id,
        state=state,
        previous_state=state,
        current_rank=rank,
        current_strength_score=Decimal(50),
        mainline_score=Decimal(50),
        crowding_score=Decimal(20),
        persistence=TopKPersistence(
            top_k=2,
            window_sessions=5,
            observed_sessions=1,
            top_k_hits=1,
            consecutive_top_k_sessions=1,
            recent_ranks=(rank,),
        ),
        crowding_signals=(),
        supporting_evidence=evidence,
        counter_evidence=(),
        invalidations=("fixed synthetic invalidation",),
        transition_reason="fixed synthetic state",
    )


def _mainline_snapshot(
    offset: int,
    regime: RegimeTransitionResult,
    states: tuple[MainlineState, MainlineState],
    *,
    previous_result_hash: str | None,
) -> MainlineSnapshot:
    rows = tuple(
        _mainline_row(industry_id, index, state)
        for index, (industry_id, state) in enumerate(
            zip(INDUSTRIES, states, strict=True),
            start=1,
        )
    )
    input_hash = stable_hash({"offset": offset, "regime_hash": regime.result_hash})
    result_hash = stable_hash(
        {
            "input_hash": input_hash,
            "previous_result_hash": previous_result_hash,
            "states": states,
        }
    )
    return MainlineSnapshot(
        session_date=regime.raw_result.session_date,
        as_of=regime.raw_result.as_of,
        data_version="pit-signals-v1",
        classification_version="SW2021",
        industry_level=3,
        market_regime=regime.final_regime,
        model_version="mainline-v1",
        config_hash=MAINLINE_CONFIG_HASH,
        input_identity=MainlineInputIdentity(
            industry_feature_version="industry-strength-v1",
            industry_config_hash=HASH_E,
            industry_cache_key=stable_hash({"offset": offset, "kind": "industry"}),
            classification_version="SW2021",
            industry_level=3,
            regime_model_version="regime-model-v1",
            regime_classifier_config_hash=HASH_A,
            regime_input_hash=regime.raw_result.input_hash,
            regime_trend_feature_version="trend-v1",
            regime_trend_config_hash=HASH_B,
            regime_breadth_feature_version="breadth-v1",
            regime_breadth_config_hash=HASH_D,
            regime_transition_version="transition-v1",
            regime_transition_config_hash=TRANSITION_CONFIG_HASH,
            regime_transition_result_hash=regime.result_hash,
        ),
        input_hash=input_hash,
        industries=rows,
        previous_result_hash=previous_result_hash,
        result_hash=result_hash,
    )


def _signals() -> tuple[tuple[RegimeTransitionResult, ...], tuple[MainlineSnapshot, ...]]:
    regimes: list[RegimeTransitionResult] = []
    mainlines: list[MainlineSnapshot] = []
    regime_states = (
        MarketRegime.UPTREND,
        MarketRegime.UPTREND,
        MarketRegime.DOWNTREND,
        MarketRegime.DOWNTREND,
    )
    mainline_states = (
        (MainlineState.EMERGING, MainlineState.CONFIRMED),
        (MainlineState.CONFIRMED, MainlineState.CONFIRMED),
        (MainlineState.FADING, MainlineState.CROWDED),
        (MainlineState.FADING, MainlineState.EMERGING),
    )
    previous_regime: MarketRegime | None = None
    previous_event_hash: str | None = None
    previous_mainline_hash: str | None = None
    for offset, (regime_state, states) in enumerate(
        zip(regime_states, mainline_states, strict=True)
    ):
        regime = _regime_result(
            offset,
            regime_state,
            previous_regime=previous_regime,
            previous_event_hash=previous_event_hash,
        )
        mainline = _mainline_snapshot(
            offset,
            regime,
            states,
            previous_result_hash=previous_mainline_hash,
        )
        regimes.append(regime)
        mainlines.append(mainline)
        previous_regime = regime_state
        previous_event_hash = regime.event.event_hash
        previous_mainline_hash = mainline.result_hash
    return tuple(regimes), tuple(mainlines)


def _labels() -> tuple[ForwardRelativeReturnLabel, ...]:
    # Effective values are 2, 3, -4, 6; invalid values are -1, 5, -7, -8.
    base_values = {
        (0, INDUSTRIES[0]): Decimal(-1),
        (0, INDUSTRIES[1]): Decimal(2),
        (1, INDUSTRIES[0]): Decimal(3),
        (1, INDUSTRIES[1]): Decimal(-4),
        (2, INDUSTRIES[0]): Decimal(5),
        (2, INDUSTRIES[1]): Decimal(6),
        (3, INDUSTRIES[0]): Decimal(-7),
        (3, INDUSTRIES[1]): Decimal(-8),
    }
    labels: list[ForwardRelativeReturnLabel] = []
    for offset in range(4):
        for industry_id in INDUSTRIES:
            for horizon in (5, 10, 20):
                value = base_values[(offset, industry_id)]
                labels.append(
                    ForwardRelativeReturnLabel(
                        signal_date=ALL_SESSIONS[offset],
                        industry_id=industry_id,
                        horizon=horizon,
                        future_sessions=ALL_SESSIONS[offset + 1 : offset + 1 + horizon],
                        relative_return=value,
                        data_version="forward-label-data-v1",
                        label_version="forward-relative-return-v1",
                        source_hash=stable_hash(
                            {
                                "horizon": horizon,
                                "industry_id": industry_id,
                                "offset": offset,
                                "value": value,
                            }
                        ),
                    )
                )
    return tuple(labels)


def test_fixed_replay_computes_stability_precision_and_split_return_distributions() -> None:
    regimes, mainlines = _signals()
    labels = _labels()

    first = evaluate_regime_mainline(regimes, mainlines, labels)
    second = evaluate_regime_mainline(regimes, mainlines, labels)

    assert first == second
    assert first.result_hash == second.result_hash
    assert first.regime_stability.observation_count == 4
    assert first.regime_stability.switch_count == 1
    assert Decimal("0.33") < first.regime_stability.switch_rate < Decimal("0.34")
    assert tuple(item.duration_sessions for item in first.regime_stability.segments) == (2, 2)
    assert tuple(item.regime for item in first.regime_stability.segments) == (
        MarketRegime.UPTREND,
        MarketRegime.DOWNTREND,
    )

    assert tuple(item.horizon for item in first.horizon_metrics) == (5, 10, 20)
    five_day = first.horizon_metrics[0]
    assert five_day.overall_precision_at_k.prediction_count == 8
    assert five_day.overall_precision_at_k.precision == Decimal("0.5")
    assert five_day.effective_period_precision_at_k.period is EvaluationPeriod.EFFECTIVE
    assert five_day.effective_period_precision_at_k.precision == Decimal("0.75")
    assert five_day.invalid_period_precision_at_k.period is EvaluationPeriod.INVALID
    assert five_day.invalid_period_precision_at_k.precision == Decimal("0.25")
    assert five_day.overall_returns.values == tuple(
        Decimal(value) for value in (-8, -7, -4, -1, 2, 3, 5, 6)
    )
    assert five_day.overall_returns.mean == Decimal("-0.5")
    assert five_day.effective_period_returns.mean == Decimal("1.75")
    assert five_day.invalid_period_returns.mean == Decimal("-2.75")

    assert first.data_version == "pit-signals-v1"
    assert first.code_version == REGIME_MAINLINE_EVALUATION_CODE_VERSION
    assert first.config_version == "regime-mainline-evaluation-config-v1"
    assert first.versions.label_data_version == "forward-label-data-v1"
    assert first.versions.regime_transition_version == "transition-v1"
    assert len(first.signal_input_hash) == 64
    assert len(first.label_input_hash) == 64
    assert len(first.input_hash) == 64
    assert len(first.result_hash) == 64
    assert first.identity_payload()["result_hash"] == first.result_hash
    assert labels[0].window_start_date == ALL_SESSIONS[1]
    assert labels[0].window_end_date == ALL_SESSIONS[5]

    all_effective = RegimeMainlineEvaluationConfig(
        effective_states=tuple(MainlineState),
    )
    one_session = evaluate_regime_mainline(
        regimes[:1],
        mainlines[:1],
        labels[:6],
        all_effective,
    )
    assert one_session.regime_stability.switch_rate == Decimal(0)
    assert one_session.horizon_metrics[0].invalid_period_returns.sample_count == 0
    assert one_session.horizon_metrics[0].invalid_period_precision_at_k.precision is None


def test_future_labels_change_evaluation_but_never_the_signal_identity() -> None:
    regimes, mainlines = _signals()
    labels = list(_labels())
    baseline = evaluate_regime_mainline(regimes, mainlines, labels)
    changed_value = Decimal(1)
    labels[0] = replace(
        labels[0],
        relative_return=changed_value,
        source_hash=stable_hash({"replacement": changed_value}),
    )

    changed = evaluate_regime_mainline(regimes, mainlines, labels)

    assert changed.signal_input_hash == baseline.signal_input_hash
    assert changed.label_input_hash != baseline.label_input_hash
    assert changed.input_hash != baseline.input_hash
    assert changed.result_hash != baseline.result_hash
    assert changed.horizon_metrics[0].overall_precision_at_k.precision == Decimal("0.625")
    assert tuple(item.result_hash for item in regimes) == tuple(
        item.result_hash for item in _signals()[0]
    )
    assert tuple(item.result_hash for item in mainlines) == tuple(
        item.result_hash for item in _signals()[1]
    )


def test_session_without_top_k_prediction_still_contributes_regime_stability() -> None:
    regimes, mainlines = _signals()
    no_selection = replace(
        mainlines[-1],
        industries=tuple(replace(item, current_rank=None) for item in mainlines[-1].industries),
        result_hash=stable_hash({"kind": "no-selection"}),
    )

    result = evaluate_regime_mainline(
        regimes,
        (*mainlines[:-1], no_selection),
        _labels()[:-6],
    )

    assert result.regime_stability.observation_count == 4
    assert result.horizon_metrics[0].overall_precision_at_k.prediction_count == 6


def test_rejects_out_of_order_misaligned_duplicate_and_missing_inputs() -> None:
    regimes, mainlines = _signals()
    labels = _labels()

    with pytest.raises(RegimeMainlineEvaluationError, match="strictly increasing"):
        evaluate_regime_mainline(
            (regimes[1], regimes[0], *regimes[2:]),
            (mainlines[1], mainlines[0], *mainlines[2:]),
            labels,
        )

    misdated = replace(mainlines[0], session_date=mainlines[0].session_date - timedelta(days=1))
    with pytest.raises(RegimeMainlineEvaluationError, match="misaligned"):
        evaluate_regime_mainline(regimes, (misdated, *mainlines[1:]), labels)

    with pytest.raises(RegimeMainlineEvaluationError, match="duplicate key"):
        evaluate_regime_mainline(regimes, mainlines, (labels[0], labels[0], *labels[1:]))

    with pytest.raises(RegimeMainlineEvaluationError, match="missing future-return label"):
        evaluate_regime_mainline(regimes, mainlines, labels[1:])

    wrong_date = replace(labels[0], signal_date=START_DATE - timedelta(days=1))
    with pytest.raises(RegimeMainlineEvaluationError, match="not aligned"):
        evaluate_regime_mainline(regimes, mainlines, (wrong_date, *labels[1:]))


def test_rejects_version_mixing_and_broken_frozen_hash_chains() -> None:
    regimes, mainlines = _signals()
    labels = _labels()

    mixed_mainline = replace(mainlines[1], model_version="mainline-v2")
    with pytest.raises(RegimeMainlineEvaluationError, match="mixes data, code"):
        evaluate_regime_mainline(
            regimes,
            (mainlines[0], mixed_mainline, *mainlines[2:]),
            labels,
        )

    mixed_label = replace(labels[1], label_version="forward-relative-return-v2")
    with pytest.raises(RegimeMainlineEvaluationError, match="mix data or label-code versions"):
        evaluate_regime_mainline(
            regimes,
            mainlines,
            (labels[0], mixed_label, *labels[2:]),
        )

    broken_regime = replace(
        regimes[1],
        event=replace(regimes[1].event, previous_event_hash=HASH_A),
    )
    with pytest.raises(RegimeMainlineEvaluationError, match="regime frozen-result hash chain"):
        evaluate_regime_mainline(
            (regimes[0], broken_regime, *regimes[2:]),
            mainlines,
            labels,
        )

    broken_mainline = replace(mainlines[1], previous_result_hash=HASH_A)
    with pytest.raises(RegimeMainlineEvaluationError, match="mainline frozen-result hash chain"):
        evaluate_regime_mainline(
            regimes,
            (mainlines[0], broken_mainline, *mainlines[2:]),
            labels,
        )


def test_rejects_future_window_overlap_and_inconsistent_horizon_calendars() -> None:
    regimes, mainlines = _signals()
    labels = list(_labels())

    with pytest.raises(ValueError, match="cannot overlap the signal date"):
        ForwardRelativeReturnLabel(
            signal_date=START_DATE,
            industry_id=INDUSTRIES[0],
            horizon=2,
            future_sessions=(START_DATE, START_DATE + timedelta(days=1)),
            relative_return=Decimal(0),
            data_version="forward-label-data-v1",
            label_version="forward-relative-return-v1",
            source_hash=HASH_A,
        )

    replacement_calendar = (
        *ALL_SESSIONS[1:4],
        *ALL_SESSIONS[5:12],
    )
    for index, label in enumerate(labels):
        if label.signal_date == START_DATE and label.horizon == 10:
            labels[index] = replace(label, future_sessions=replacement_calendar)
    with pytest.raises(RegimeMainlineEvaluationError, match="nested session prefixes"):
        evaluate_regime_mainline(regimes, mainlines, labels)


def test_config_and_label_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique, positive, and increasing"):
        RegimeMainlineEvaluationConfig(horizons=(10, 5, 5))
    with pytest.raises(ValueError, match="finite"):
        RegimeMainlineEvaluationConfig(positive_return_threshold=Decimal("NaN"))
    with pytest.raises(ValueError, match="exactly horizon"):
        replace(_labels()[0], future_sessions=ALL_SESSIONS[1:5])
    with pytest.raises(ValueError, match="SHA-256"):
        replace(_labels()[0], source_hash="z" * 64)
