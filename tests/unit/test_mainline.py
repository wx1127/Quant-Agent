"""Tests for deterministic point-in-time mainline-industry lifecycle states."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_agent.features.industry_strength import (
    IndustryStrengthResult,
    IndustryStrengthSnapshot,
    IndustryStrengthStatus,
    MemberContribution,
)
from quant_agent.features.mainline import (
    MainlineConfig,
    MainlineEngine,
    MainlineEvidenceSide,
    MainlineIndustryResult,
    MainlineInputMismatch,
    MainlineSnapshot,
    MainlineState,
    apply_mainline_states,
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
START_DATE = date(2026, 8, 3)
START_TIME = datetime(2026, 8, 3, 16, 0, tzinfo=TZ)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
INDUSTRIES = tuple(f"SW2021:{value}" for value in "ABCDEF")


def _contributions(*, crowded: bool) -> tuple[MemberContribution, ...]:
    shares = (
        (Decimal("0.50"), Decimal("0.30"), Decimal("0.20"))
        if crowded
        else (Decimal("0.34"), Decimal("0.33"), Decimal("0.33"))
    )
    return tuple(
        MemberContribution(
            instrument_id=f"MEMBER-{index}",
            current_return=Decimal("0.01"),
            return_contribution=Decimal("0.01") / 3,
            turnover_share=share,
        )
        for index, share in enumerate(shares, start=1)
    )


def _industry_row(
    industry_id: str,
    rank: int,
    *,
    score: str = "50",
    crowded: bool = False,
) -> IndustryStrengthResult:
    return IndustryStrengthResult(
        industry_id=industry_id,
        status=IndustryStrengthStatus.READY,
        rank=rank,
        score=Decimal(score),
        horizon_returns=(),
        current_member_count=3,
        traded_member_count=3,
        comparable_return_count=3,
        current_member_coverage=Decimal(1),
        advancing_count=3,
        breadth_ratio=Decimal(1),
        new_high_count=3 if crowded else 0,
        new_high_denominator=3,
        new_high_ratio=Decimal("0.80") if crowded else Decimal("0.20"),
        turnover_growth=Decimal("0.80") if crowded else Decimal("0.10"),
        downside_resilience=Decimal("0.01"),
        contributions=_contributions(crowded=crowded),
        reason=None,
    )


def _industry_snapshot(
    offset: int,
    rank_a: int,
    *,
    crowded_a: bool = False,
    score_a: str = "50",
    data_version: str = "snapshot-v1",
    reverse_rows: bool = False,
    feature_version: str = "industry-strength-test-v1",
) -> IndustryStrengthSnapshot:
    remaining_ranks = iter(rank for rank in range(1, 7) if rank != rank_a)
    rows = [
        _industry_row(
            industry_id,
            rank_a if industry_id == INDUSTRIES[0] else next(remaining_ranks),
            score=score_a if industry_id == INDUSTRIES[0] else "20",
            crowded=crowded_a and industry_id == INDUSTRIES[0],
        )
        for industry_id in INDUSTRIES
    ]
    if reverse_rows:
        rows.reverse()
    session_date = START_DATE + timedelta(days=offset)
    as_of = START_TIME + timedelta(days=offset)
    return IndustryStrengthSnapshot(
        session_date=session_date,
        as_of=as_of,
        data_version=data_version,
        classification_version="SW2021",
        industry_level=3,
        feature_version=feature_version,
        config_hash=HASH_A,
        industries=tuple(rows),
        cache_key=stable_hash(
            {
                "data_version": data_version,
                "offset": offset,
                "rank_a": rank_a,
                "score_a": score_a,
            }
        ),
    )


def _regime_result(
    offset: int,
    regime: MarketRegime = MarketRegime.UPTREND,
    *,
    data_version: str = "snapshot-v1",
) -> RegimeTransitionResult:
    session_date = START_DATE + timedelta(days=offset)
    as_of = START_TIME + timedelta(days=offset)
    score = Decimal("80") if regime is not MarketRegime.DOWNTREND else Decimal("20")
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
            feature="test",
            value=score,
            criterion=">= 0",
            rule_weight=Decimal(1),
            contribution=Decimal(1),
            side=EvidenceSide.SUPPORTING,
            rationale="fixed test regime evidence",
        ),
    )
    input_hash = stable_hash({"data_version": data_version, "offset": offset, "regime": regime})
    raw_result_hash = stable_hash({"input_hash": input_hash, "kind": "raw"})
    raw = MarketRegimeResult(
        as_of=as_of,
        session_date=session_date,
        data_version=data_version,
        regime=regime,
        score=score,
        confidence=Decimal("0.90"),
        risk_budget_max=Decimal("0.80"),
        components=components,
        evidence=evidence,
        counter_evidence=(),
        invalidations=("fixed test invalidation",),
        model_version="market-regime-test-v1",
        config_hash=HASH_B,
        input_identity=RegimeInputIdentity(
            trend_feature_version="trend-test-v1",
            trend_config_hash=HASH_C,
            trend_cache_key=HASH_D,
            breadth_feature_version="breadth-test-v1",
            breadth_config_hash=HASH_E,
            breadth_cache_key=HASH_A,
        ),
        input_hash=input_hash,
        result_hash=raw_result_hash,
    )
    transition_hash = stable_hash({"kind": "transition-config"})
    event_hash = stable_hash({"input_hash": input_hash, "kind": "event"})
    event = TransitionEvent(
        session_date=session_date,
        as_of=as_of,
        raw_regime=regime,
        previous_regime=regime,
        final_regime=regime,
        action=TransitionAction.STABLE,
        candidate_regime=None,
        candidate_count=0,
        required_confirmation_sessions=None,
        raw_score=score,
        raw_confidence=Decimal("0.90"),
        reason="fixed stabilized test state",
        supporting_evidence=evidence,
        source_result_hash=raw_result_hash,
        transition_config_version="transition-test-v1",
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
        transition_config_version="transition-test-v1",
        transition_config_hash=transition_hash,
        result_hash=stable_hash({"event_hash": event_hash, "kind": "transition-result"}),
    )


def _industry(
    result: MainlineSnapshot | None,
    industry_id: str = INDUSTRIES[0],
) -> MainlineIndustryResult:
    if result is None:  # pragma: no cover - helper guard
        raise AssertionError("expected a mainline snapshot")
    return next(item for item in result.industries if item.industry_id == industry_id)


def test_three_top_five_hits_confirm_but_one_day_pulse_cannot() -> None:
    engine = MainlineEngine()
    first = engine.process(_industry_snapshot(0, 1), _regime_result(0))
    second = engine.process(_industry_snapshot(1, 2), _regime_result(1))
    third = engine.process(_industry_snapshot(2, 5), _regime_result(2))

    assert _industry(first).state is MainlineState.EMERGING
    assert _industry(first).top_k_hits == 1
    assert _industry(second).state is MainlineState.EMERGING
    assert _industry(third).state is MainlineState.CONFIRMED
    assert _industry(third).persistence.recent_ranks == (1, 2, 5)
    assert _industry(third).top_k_hits == 3
    assert third.market_regime is MarketRegime.UPTREND

    pulse = MainlineEngine()
    pulse_states = []
    for offset, rank in enumerate((1, 6, 6)):
        output = pulse.process(_industry_snapshot(offset, rank), _regime_result(offset))
        pulse_states.append(_industry(output).state)
    assert pulse_states == [
        MainlineState.EMERGING,
        MainlineState.FADING,
        MainlineState.FADING,
    ]
    assert MainlineState.CONFIRMED not in pulse_states


def test_crowding_requires_confirmation_and_exposes_reverse_evidence() -> None:
    engine = MainlineEngine()
    outputs = tuple(
        engine.process(
            _industry_snapshot(offset, 1, crowded_a=True),
            _regime_result(offset),
        )
        for offset in range(3)
    )

    assert _industry(outputs[0]).state is MainlineState.EMERGING
    crowded = _industry(outputs[-1])
    assert crowded.state is MainlineState.CROWDED
    assert crowded.crowding_score == Decimal(100)
    assert set(crowded.crowding_signals) == {
        "TURNOVER_GROWTH",
        "NEW_HIGH_HEAT",
        "TURNOVER_CONCENTRATION",
    }
    assert crowded.supporting_evidence
    assert crowded.counter_evidence
    assert all("crowding" in item.feature for item in crowded.counter_evidence)


def test_confirmation_tolerates_one_rank_loss_then_transitions_to_fading() -> None:
    engine = MainlineEngine()
    states = []
    for offset, rank in enumerate((1, 1, 1, 6, 6, 6)):
        output = engine.process(_industry_snapshot(offset, rank), _regime_result(offset))
        states.append(_industry(output).state)

    assert states == [
        MainlineState.EMERGING,
        MainlineState.EMERGING,
        MainlineState.CONFIRMED,
        MainlineState.CONFIRMED,
        MainlineState.CONFIRMED,
        MainlineState.FADING,
    ]
    assert _industry(engine.current_snapshot).top_k_hits == 2


def test_stabilized_downtrend_forces_a_confirmed_industry_to_fading() -> None:
    engine = MainlineEngine()
    for offset in range(3):
        engine.process(_industry_snapshot(offset, 1), _regime_result(offset))

    faded = engine.process(
        _industry_snapshot(3, 1),
        _regime_result(3, MarketRegime.DOWNTREND),
    )
    decision = _industry(faded)

    assert decision.previous_state is MainlineState.CONFIRMED
    assert decision.state is MainlineState.FADING
    assert any(item.feature == "market_regime_eligible" for item in decision.supporting_evidence)
    assert any(item.feature == "current_rank" for item in decision.counter_evidence)


def test_fading_rows_are_retained_for_a_versioned_number_of_sessions() -> None:
    engine = MainlineEngine()
    engine.process(_industry_snapshot(0, 1), _regime_result(0))
    first_fade = engine.process(_industry_snapshot(1, 6), _regime_result(1))
    second_fade = engine.process(_industry_snapshot(2, 6), _regime_result(2))
    expired = engine.process(_industry_snapshot(3, 6), _regime_result(3))

    assert _industry(first_fade).state is MainlineState.FADING
    assert _industry(second_fade).state is MainlineState.FADING
    assert INDUSTRIES[0] not in {item.industry_id for item in expired.industries}


def test_inputs_must_be_pit_aligned_and_failures_do_not_mutate_state() -> None:
    engine = MainlineEngine()
    first = engine.process(_industry_snapshot(0, 1), _regime_result(0))

    with pytest.raises(MainlineInputMismatch, match="data_version"):
        engine.process(
            _industry_snapshot(1, 1, data_version="snapshot-v2"),
            _regime_result(1, data_version="snapshot-v1"),
        )
    with pytest.raises(MainlineInputMismatch, match="strictly increasing"):
        engine.process(_industry_snapshot(0, 1), _regime_result(0))

    assert engine.snapshots == (first,)
    second = engine.process(_industry_snapshot(1, 1), _regime_result(1))
    assert _industry(second).top_k_hits == 2


def test_feature_version_mixing_is_rejected_but_daily_data_versions_are_bound() -> None:
    engine = MainlineEngine()
    first = engine.process(_industry_snapshot(0, 1), _regime_result(0))

    with pytest.raises(MainlineInputMismatch, match="versions aligned"):
        engine.process(
            _industry_snapshot(1, 1, feature_version="industry-strength-v2"),
            _regime_result(1),
        )

    changed_industry = _industry_snapshot(1, 1, data_version="snapshot-v2")
    changed = engine.process(
        changed_industry,
        _regime_result(1, data_version="snapshot-v2"),
    )
    assert changed.data_version == "snapshot-v2"
    assert changed.input_identity.industry_cache_key == changed_industry.cache_key
    assert changed.input_hash != first.input_hash


def test_fixed_sequence_is_deterministic_order_independent_and_hash_chained() -> None:
    inputs = tuple(
        (_industry_snapshot(offset, rank), _regime_result(offset))
        for offset, rank in enumerate((1, 2, 5, 6))
    )
    first = apply_mainline_states(inputs)
    repeated = MainlineEngine().process_all(inputs)
    reordered = MainlineEngine().process_all(
        tuple(
            (
                _industry_snapshot(offset, rank, reverse_rows=True),
                _regime_result(offset),
            )
            for offset, rank in enumerate((1, 2, 5, 6))
        )
    )

    assert repeated == first
    assert reordered == first
    assert first[0].previous_result_hash is None
    assert first[1].previous_result_hash == first[0].result_hash
    assert first[-1].identity_payload()["result_hash"] == first[-1].result_hash
    assert len(first[-1].input_hash) == len(first[-1].result_hash) == 64


def test_processing_later_sessions_never_revises_an_earlier_result() -> None:
    engine = MainlineEngine()
    first = engine.process(_industry_snapshot(0, 1), _regime_result(0))
    engine.process(_industry_snapshot(1, 1, crowded_a=True), _regime_result(1))
    engine.process(_industry_snapshot(2, 6), _regime_result(2))

    replayed_first = MainlineEngine().process(_industry_snapshot(0, 1), _regime_result(0))
    assert first == replayed_first
    assert engine.snapshots[0] == first


def test_all_policy_fields_participate_in_the_config_hash() -> None:
    baseline = MainlineConfig()

    assert len(baseline.config_hash) == 64
    assert replace(baseline, minimum_top_k_hits=4).config_hash != baseline.config_hash
    assert (
        replace(baseline, minimum_confirmed_strength_score=Decimal("0.0")).config_hash
        == baseline.config_hash
    )
    assert replace(baseline, version="mainline-v2").config_hash != baseline.config_hash


@pytest.mark.parametrize(
    "changes",
    (
        {"version": " "},
        {"top_k": 0},
        {"persistence_window_sessions": 1},
        {"minimum_top_k_hits": 1},
        {"minimum_top_k_hits": 6},
        {"confirmation_regimes": ()},
        {"confirmation_regimes": (MarketRegime.UPTREND, MarketRegime.UPTREND)},
        {"minimum_crowding_signals": 4},
        {"crowding_new_high_ratio_min": Decimal("1.1")},
        {"crowding_turnover_growth_min": Decimal("-0.1")},
        {"strength_weight": Decimal("0.5")},
        {"strength_weight": Decimal("NaN")},
    ),
)
def test_invalid_configuration_fails_early(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MainlineConfig(**changes)  # type: ignore[arg-type]


def test_contract_properties_and_validation_guards_are_explicit() -> None:
    snapshot = MainlineEngine().process(_industry_snapshot(0, 1), _regime_result(0))
    decision = _industry(snapshot)

    assert decision.status is decision.state
    assert decision.rank == decision.current_rank
    assert decision.top_k_hits == decision.persistence.top_k_hits
    assert snapshot.results == snapshot.industries
    assert decision in snapshot.top_k_industries
    assert MainlineEngine().config == MainlineConfig()

    persistence_changes = (
        {"top_k": 0},
        {"observed_sessions": 2},
        {"window_sessions": 0},
        {"recent_ranks": (0,)},
        {"top_k_hits": 0},
        {"consecutive_top_k_sessions": 0},
    )
    for changes in persistence_changes:
        with pytest.raises(ValueError):
            replace(decision.persistence, **changes)

    with pytest.raises(ValueError, match="text fields"):
        replace(decision.supporting_evidence[0], feature=" ")
    with pytest.raises(ValueError, match="finite"):
        replace(decision.supporting_evidence[0], value=Decimal("NaN"))


def test_identity_result_and_snapshot_contracts_reject_ambiguous_values() -> None:
    snapshot = MainlineEngine().process(_industry_snapshot(0, 1), _regime_result(0))
    decision = _industry(snapshot)
    supporting = decision.supporting_evidence[0]

    for changes in (
        {"industry_feature_version": " "},
        {"industry_config_hash": "short"},
        {"industry_level": 4},
    ):
        with pytest.raises(ValueError):
            replace(snapshot.input_identity, **changes)

    invalid_results = (
        {"industry_id": " "},
        {"current_rank": 0},
        {"current_strength_score": Decimal("NaN")},
        {"mainline_score": Decimal("-1")},
        {"crowding_signals": ("DUPLICATE", "DUPLICATE")},
        {"supporting_evidence": ()},
        {"supporting_evidence": (replace(supporting, side=MainlineEvidenceSide.OPPOSING),)},
        {"counter_evidence": (supporting,)},
        {"invalidations": ()},
        {"transition_reason": " "},
    )
    for changes in invalid_results:
        with pytest.raises(ValueError):
            replace(decision, **changes)

    for changes in (
        {"session_date": snapshot.as_of.date() + timedelta(days=1)},
        {"data_version": " "},
        {"industry_level": 0},
        {"config_hash": "short"},
        {"previous_result_hash": "short"},
        {"industries": (decision, decision)},
    ):
        with pytest.raises(ValueError):
            replace(snapshot, **changes)


def test_config_rejects_wrong_regime_type_out_of_range_score_and_negative_weight() -> None:
    invalid = (
        {"confirmation_regimes": ("UPTREND",)},
        {"minimum_confirmed_strength_score": Decimal("101")},
        {
            "strength_weight": Decimal("-0.1"),
            "rank_weight": Decimal("0.75"),
        },
    )
    for changes in invalid:
        with pytest.raises(ValueError):
            MainlineConfig(**changes)  # type: ignore[arg-type]


def test_input_event_identity_mismatches_are_rejected() -> None:
    industry = _industry_snapshot(0, 1)
    regime = _regime_result(0)
    mismatches = (
        (
            industry,
            replace(
                regime,
                event=replace(regime.event, session_date=regime.event.session_date + timedelta(1)),
            ),
            "session_date",
        ),
        (
            industry,
            replace(
                regime,
                event=replace(regime.event, as_of=regime.event.as_of + timedelta(minutes=1)),
            ),
            "as_of",
        ),
        (
            industry,
            replace(regime, event=replace(regime.event, source_result_hash=HASH_A)),
            "raw result hash",
        ),
        (
            industry,
            replace(
                regime,
                event=replace(regime.event, transition_config_hash=HASH_A),
            ),
            "transition versions",
        ),
    )
    for snapshot, result, message in mismatches:
        with pytest.raises(MainlineInputMismatch, match=message):
            MainlineEngine().process(snapshot, result)


def test_duplicate_and_noncontiguous_upstream_ranks_fail_closed() -> None:
    baseline = _industry_snapshot(0, 1)
    duplicate_id = replace(baseline, industries=(*baseline.industries, baseline.industries[0]))
    duplicate_rank = replace(
        baseline,
        industries=(
            baseline.industries[0],
            replace(baseline.industries[1], rank=baseline.industries[0].rank),
            *baseline.industries[2:],
        ),
    )
    noncontiguous = replace(
        baseline,
        industries=(replace(baseline.industries[0], rank=7), *baseline.industries[1:]),
    )

    for snapshot, message in (
        (duplicate_id, "duplicate industry_id"),
        (duplicate_rank, "duplicate ready ranks"),
        (noncontiguous, "contiguous"),
    ):
        with pytest.raises(MainlineInputMismatch, match=message):
            MainlineEngine().process(snapshot, _regime_result(0))


def test_missing_or_low_strength_and_ineligible_market_are_explicit_evidence() -> None:
    low_strength = MainlineEngine(
        MainlineConfig(minimum_confirmed_strength_score=Decimal("60"))
    ).process(_industry_snapshot(0, 1, score_a="50"), _regime_result(0))
    downtrend = MainlineEngine().process(
        _industry_snapshot(0, 1),
        _regime_result(0, MarketRegime.DOWNTREND),
    )

    assert _industry(low_strength).state is MainlineState.EMERGING
    assert any(
        item.feature == "strength_score" for item in _industry(low_strength).counter_evidence
    )
    assert _industry(downtrend).state is MainlineState.EMERGING
    assert any(
        item.feature == "market_regime_eligible" for item in _industry(downtrend).counter_evidence
    )

    engine = MainlineEngine()
    engine.process(_industry_snapshot(0, 1), _regime_result(0))
    without_a = _industry_snapshot(1, 6)
    retained_rows = tuple(
        replace(row, rank=rank)
        for rank, row in enumerate(
            (row for row in without_a.industries if row.industry_id != INDUSTRIES[0]),
            start=1,
        )
    )
    missing = engine.process(
        replace(without_a, industries=retained_rows),
        _regime_result(1),
    )
    missing_decision = _industry(missing)
    assert missing_decision.current_rank is None
    assert missing_decision.current_strength_score is None
    assert missing_decision.mainline_score >= Decimal(0)
    assert any(item.feature == "strength_score" for item in missing_decision.supporting_evidence)
