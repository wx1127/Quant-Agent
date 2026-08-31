"""Leakage-safe historical evaluation for frozen leader/candidate rankings."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest
from tests.unit.test_candidates import _rank

from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.evaluation.leader_candidates import (
    CandidateSliceKind,
    ForwardCandidateLabel,
    LeaderCandidateEvaluationConfig,
    LeaderCandidateEvaluationError,
    evaluate_leader_candidates,
)
from quant_agent.leaders.candidates import CandidateSnapshot, CandidateTier
from quant_agent.regime import MarketRegime
from quant_agent.regime.contracts import stable_hash


def _snapshot(session_date: date, regime: MarketRegime) -> CandidateSnapshot:
    base = _rank()
    as_of = datetime.combine(session_date, time(16), tzinfo=SHANGHAI_TZ)
    input_hash = stable_hash({"base_input_hash": base.input_hash, "session_date": session_date})
    return replace(
        base,
        session_date=session_date,
        as_of=as_of,
        regime=regime,
        input_hash=input_hash,
        result_hash=stable_hash(
            {
                "base_result_hash": base.result_hash,
                "input_hash": input_hash,
                "regime": regime,
            }
        ),
    )


def _return_values(
    signal_date: date,
    instrument_id: str,
) -> tuple[Decimal, Decimal, Decimal, Decimal, bool, str | None]:
    values = {
        (2025, "CORE"): ("0.10", "0.02", "0.01", "-0.04", True, None),
        (2025, "SECOND"): (
            "-0.01",
            "0.01",
            "0.01",
            "-0.08",
            False,
            "limit-up entry could not fill",
        ),
        (2026, "CORE"): ("0.03", "0.01", "0.03", "-0.02", True, None),
        (2026, "SECOND"): ("0.02", "0.02", "0.005", "-0.03", True, None),
        (2025, "WATCH"): ("0.01", "0.02", "0.005", "-0.05", True, None),
        (2026, "WATCH"): ("0.04", "0.02", "0.005", "-0.04", True, None),
    }[(signal_date.year, instrument_id)]
    return (
        Decimal(values[0]),
        Decimal(values[1]),
        Decimal(values[2]),
        Decimal(values[3]),
        values[4],
        values[5],
    )


def _labels(
    snapshots: tuple[CandidateSnapshot, ...],
    config: LeaderCandidateEvaluationConfig,
) -> tuple[ForwardCandidateLabel, ...]:
    result: list[ForwardCandidateLabel] = []
    for snapshot in snapshots:
        selected = sorted(
            (
                item
                for item in snapshot.candidates
                if item.eligible_rank is not None and item.eligible_rank <= config.top_k
            ),
            key=lambda item: item.instrument_id,
        )
        for candidate in selected:
            values = _return_values(snapshot.session_date, candidate.instrument_id)
            for horizon in config.horizons:
                sessions = tuple(
                    snapshot.session_date + timedelta(days=offset)
                    for offset in range(1, horizon + 1)
                )
                result.append(
                    ForwardCandidateLabel(
                        signal_date=snapshot.session_date,
                        instrument_id=candidate.instrument_id,
                        horizon=horizon,
                        future_sessions=sessions,
                        gross_return=values[0],
                        benchmark_return=values[1],
                        estimated_round_trip_cost=values[2],
                        maximum_adverse_excursion=values[3],
                        fillable=values[4],
                        unfillable_reason=values[5],
                        available_at=datetime.combine(
                            sessions[-1],
                            time(18),
                            tzinfo=SHANGHAI_TZ,
                        ),
                        data_version="forward-market-v1",
                        label_version="candidate-forward-label-v1",
                        source_hash=stable_hash(
                            {
                                "horizon": horizon,
                                "instrument_id": candidate.instrument_id,
                                "signal_date": snapshot.session_date,
                            }
                        ),
                    )
                )
    return tuple(
        sorted(
            result,
            key=lambda item: (item.signal_date, item.instrument_id, item.horizon),
        )
    )


def _history() -> tuple[CandidateSnapshot, ...]:
    return (
        _snapshot(date(2025, 8, 29), MarketRegime.RANGE_STRONG),
        _snapshot(date(2026, 8, 28), MarketRegime.UPTREND),
    )


def test_evaluation_reports_gross_net_cost_mae_fillability_and_period_slices() -> None:
    config = LeaderCandidateEvaluationConfig(top_k=2)
    snapshots = _history()
    labels = _labels(snapshots, config)

    report = evaluate_leader_candidates(snapshots, labels, config)
    repeated = evaluate_leader_candidates(snapshots, labels, config)

    assert report == repeated
    assert report.period_coverage.sufficient
    assert report.period_coverage.years == (2025, 2026)
    assert report.period_coverage.regimes == (
        MarketRegime.UPTREND,
        MarketRegime.RANGE_STRONG,
    )
    assert "no causal or guaranteed-return claim" in report.period_coverage.interpretation
    overall = next(
        item
        for item in report.metrics
        if item.kind is CandidateSliceKind.OVERALL and item.horizon == 5
    )
    assert overall.selected_count == 4
    assert overall.fillable_count == 3
    assert overall.unfillable_count == 1
    assert overall.unfillable_ratio == Decimal("0.25")
    assert overall.gross_precision_at_k.evaluated_count == 4
    assert overall.gross_precision_at_k.successful_count == 2
    assert overall.gross_precision_at_k.precision == Decimal("0.5")
    assert overall.net_precision_at_k.evaluated_count == 3
    assert overall.net_precision_at_k.successful_count == 1
    assert overall.net_precision_at_k.precision == Decimal(
        "0.33333333333333333333333333333333333333333333333333"
    )
    assert overall.gross_excess_returns.values == (
        Decimal("-0.02"),
        Decimal("0.00"),
        Decimal("0.02"),
        Decimal("0.08"),
    )
    assert overall.gross_excess_returns.median == Decimal("0.01")
    assert overall.net_excess_returns.values == (
        Decimal("-0.01"),
        Decimal("-0.005"),
        Decimal("0.07"),
    )
    assert overall.maximum_adverse_excursions.values == (
        Decimal("-0.08"),
        Decimal("-0.04"),
        Decimal("-0.03"),
        Decimal("-0.02"),
    )
    assert {item.kind for item in report.metrics} == set(CandidateSliceKind)
    assert {
        item.year
        for item in report.metrics
        if item.kind is CandidateSliceKind.YEAR and item.horizon == 5
    } == {2025, 2026}
    assert {
        item.regime
        for item in report.metrics
        if item.kind is CandidateSliceKind.REGIME and item.horizon == 5
    } == {MarketRegime.RANGE_STRONG, MarketRegime.UPTREND}
    assert len(report.input_hash) == len(report.result_hash) == 64


def test_tier_slices_include_each_observed_top_k_research_layer() -> None:
    config = LeaderCandidateEvaluationConfig(top_k=3, horizons=(5,))
    snapshots = _history()
    report = evaluate_leader_candidates(snapshots, _labels(snapshots, config), config)

    tier_slices = tuple(item for item in report.metrics if item.kind is CandidateSliceKind.TIER)
    assert {item.tier for item in tier_slices} == {CandidateTier.A, CandidateTier.B}
    assert sum(item.selected_count for item in tier_slices) == 6


def test_single_period_results_remain_descriptive_and_insufficient() -> None:
    config = LeaderCandidateEvaluationConfig(top_k=2, horizons=(5,))
    snapshots = (_history()[0],)
    report = evaluate_leader_candidates(snapshots, _labels(snapshots, config), config)

    assert not report.period_coverage.sufficient
    assert "cannot establish validity" in report.period_coverage.interpretation


def test_signal_history_requires_order_stable_identity_and_top_k_predictions() -> None:
    config = LeaderCandidateEvaluationConfig(top_k=2, horizons=(5,))
    snapshots = _history()
    labels = _labels(snapshots, config)

    with pytest.raises(LeaderCandidateEvaluationError, match="strictly increasing"):
        evaluate_leader_candidates(tuple(reversed(snapshots)), labels, config)
    changed = replace(snapshots[1], feature_version="candidate-ranking-v2")
    with pytest.raises(LeaderCandidateEvaluationError, match="mixes data, feature, or config"):
        evaluate_leader_candidates((snapshots[0], changed), labels, config)
    empty_history = tuple(replace(snapshot, candidates=(), exclusions=()) for snapshot in snapshots)
    with pytest.raises(LeaderCandidateEvaluationError, match="requires Top-K predictions"):
        evaluate_leader_candidates(empty_history, labels, config)


def test_labels_require_exact_complete_ordered_and_consistent_future_windows() -> None:
    config = LeaderCandidateEvaluationConfig(top_k=2, horizons=(5, 10, 20))
    snapshots = _history()
    labels = _labels(snapshots, config)

    with pytest.raises(LeaderCandidateEvaluationError, match="missing candidate forward label"):
        evaluate_leader_candidates(snapshots, labels[:-1], config)
    with pytest.raises(LeaderCandidateEvaluationError, match="canonical chronological key order"):
        evaluate_leader_candidates(snapshots, tuple(reversed(labels)), config)
    with pytest.raises(LeaderCandidateEvaluationError, match="duplicate key"):
        evaluate_leader_candidates(snapshots, (labels[0], labels[0], *labels[1:]), config)
    mixed = list(labels)
    mixed[-1] = replace(mixed[-1], label_version="candidate-forward-label-v2")
    with pytest.raises(LeaderCandidateEvaluationError, match="mix data or label-code versions"):
        evaluate_leader_candidates(snapshots, mixed, config)

    inconsistent = list(labels)
    target_index = next(
        index
        for index, item in enumerate(inconsistent)
        if item.signal_date.year == 2025 and item.instrument_id == "SECOND" and item.horizon == 10
    )
    target = inconsistent[target_index]
    shifted = tuple(day + timedelta(days=1) for day in target.future_sessions)
    inconsistent[target_index] = replace(
        target,
        future_sessions=shifted,
        available_at=datetime.combine(shifted[-1], time(18), tzinfo=SHANGHAI_TZ),
    )
    with pytest.raises(LeaderCandidateEvaluationError, match="different future sessions"):
        evaluate_leader_candidates(snapshots, inconsistent, config)


def test_nested_horizons_and_future_availability_fail_closed() -> None:
    config = LeaderCandidateEvaluationConfig(top_k=2, horizons=(5, 10))
    snapshots = _history()
    labels = list(_labels(snapshots, config))
    target_date = snapshots[0].session_date
    shifted = tuple(target_date + timedelta(days=offset) for offset in range(2, 12))
    for index, item in enumerate(labels):
        if item.signal_date == target_date and item.horizon == 10:
            labels[index] = replace(
                item,
                future_sessions=shifted,
                available_at=datetime.combine(shifted[-1], time(18), tzinfo=SHANGHAI_TZ),
            )
    with pytest.raises(LeaderCandidateEvaluationError, match="nested future-session prefixes"):
        evaluate_leader_candidates(snapshots, labels, config)

    valid = _labels(snapshots, config)[0]
    with pytest.raises(ValueError, match="before its future window ends"):
        replace(valid, available_at=snapshots[0].as_of)


def test_label_config_and_report_contracts_reject_invalid_or_tampered_values() -> None:
    with pytest.raises(ValueError, match="positive and increasing"):
        LeaderCandidateEvaluationConfig(horizons=(10, 5))
    with pytest.raises(ValueError, match="positive integers"):
        LeaderCandidateEvaluationConfig(top_k=0)

    config = LeaderCandidateEvaluationConfig(top_k=2, horizons=(5,))
    snapshots = _history()
    label = _labels(snapshots, config)[0]
    with pytest.raises(ValueError, match="transaction cost"):
        replace(label, estimated_round_trip_cost=Decimal("-0.01"))
    with pytest.raises(ValueError, match="non-positive"):
        replace(label, maximum_adverse_excursion=Decimal("0.01"))
    with pytest.raises(ValueError, match="present exactly"):
        replace(label, fillable=False, unfillable_reason=None)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(label, source_hash="bad")

    report = evaluate_leader_candidates(snapshots, _labels(snapshots, config), config)
    with pytest.raises(ValueError, match="result hash does not match"):
        replace(report, result_hash="f" * 64)
    with pytest.raises(ValueError, match="input hash does not match"):
        replace(report, signal_input_hash="e" * 64)
