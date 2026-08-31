"""Chronological splits, frozen parameters, OOS results, and sensitivity."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

from quant_agent.backtest.validation import (
    ParameterRegistry,
    WalkForwardSplitConfig,
    WalkForwardValidationError,
    build_out_of_sample_result,
    build_parameter_perturbation,
    build_walk_forward_plan,
    build_walk_forward_report,
    evaluate_parameter_stability,
)
from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.regime.contracts import stable_hash


def _sessions(count: int = 12) -> tuple[date, ...]:
    start = date(2025, 1, 2)
    return tuple(start + timedelta(days=index) for index in range(count))


def _config(*, expanding: bool = True) -> WalkForwardSplitConfig:
    return WalkForwardSplitConfig(
        train_sessions=4,
        validation_sessions=2,
        out_of_sample_sessions=2,
        step_sessions=2,
        expanding_train=expanding,
    )


def _plan(*, expanding: bool = True):  # type: ignore[no-untyped-def]
    return build_walk_forward_plan(_sessions(), _config(expanding=expanding))


def _registered_registry() -> ParameterRegistry:
    registry = ParameterRegistry()
    registered_at = datetime.combine(_sessions()[0], time(18), tzinfo=SHANGHAI_TZ)
    registry.register(
        strategy_name="mainline-leader",
        strategy_version="mainline-leader-v1",
        parameter_version="params-v1",
        parameters={"maximum_positions": 2, "minimum_score": Decimal("60")},
        registered_at=registered_at,
    )
    registry.register(
        strategy_name="mainline-leader",
        strategy_version="mainline-leader-v1",
        parameter_version="params-v2",
        parameters={"maximum_positions": 3, "minimum_score": Decimal("62")},
        registered_at=registered_at,
    )
    return registry


def _freezes():  # type: ignore[no-untyped-def]
    plan = _plan()
    registry = _registered_registry()
    versions = ("params-v1", "params-v2", "params-v2")
    freezes = tuple(
        registry.freeze_for_fold(
            fold=fold,
            strategy_name="mainline-leader",
            strategy_version="mainline-leader-v1",
            parameter_version=version,
            frozen_at=datetime.combine(
                fold.validation_end,
                time(18),
                tzinfo=SHANGHAI_TZ,
            ),
        )
        for fold, version in zip(plan.folds, versions, strict=True)
    )
    return plan, freezes


def _results():  # type: ignore[no-untyped-def]
    plan, freezes = _freezes()
    metrics = (
        ("0.10", "0.04", "0.08", "-0.04", "1.0", "0.01"),
        ("0.05", "0.02", "0.03", "-0.06", "2.0", "0.01"),
        ("-0.02", "0.01", "-0.03", "-0.10", "1.5", "0.02"),
    )
    results = tuple(
        build_out_of_sample_result(
            fold_id=fold.fold_id,
            fold_hash=fold.fold_hash,
            parameter_freeze_hash=freeze.freeze_hash,
            observation_count=len(fold.out_of_sample_sessions),
            gross_return=Decimal(values[0]),
            benchmark_return=Decimal(values[1]),
            net_return=Decimal(values[2]),
            maximum_drawdown=Decimal(values[3]),
            turnover=Decimal(values[4]),
            transaction_cost=Decimal(values[5]),
            data_version="snapshot-v1",
            code_version="backtest-code-v1",
            available_at=datetime.combine(
                fold.out_of_sample_end,
                time(18),
                tzinfo=SHANGHAI_TZ,
            ),
            input_hash=stable_hash(
                {"fold_hash": fold.fold_hash, "freeze_hash": freeze.freeze_hash}
            ),
        )
        for fold, freeze, values in zip(plan.folds, freezes, metrics, strict=True)
    )
    return plan, freezes, results


def _rebuild_result(result, **changes: object):  # type: ignore[no-untyped-def]
    values: dict[str, object] = {
        "fold_id": result.fold_id,
        "fold_hash": result.fold_hash,
        "parameter_freeze_hash": result.parameter_freeze_hash,
        "observation_count": result.observation_count,
        "gross_return": result.gross_return,
        "benchmark_return": result.benchmark_return,
        "net_return": result.net_return,
        "maximum_drawdown": result.maximum_drawdown,
        "turnover": result.turnover,
        "transaction_cost": result.transaction_cost,
        "data_version": result.data_version,
        "code_version": result.code_version,
        "available_at": result.available_at,
        "input_hash": result.input_hash,
    }
    values.update(changes)
    return build_out_of_sample_result(**values)  # type: ignore[arg-type]


def test_walk_forward_plan_is_chronological_expanding_and_deterministic() -> None:
    plan = _plan()
    repeated = _plan()

    assert plan == repeated
    assert len(plan.folds) == 3
    assert tuple(item.fold_id for item in plan.folds) == ("WF-001", "WF-002", "WF-003")
    first, second, third = plan.folds
    assert len(first.train_sessions) == 4
    assert len(second.train_sessions) == 6
    assert len(third.train_sessions) == 8
    assert first.train_end < first.validation_sessions[0]
    assert first.validation_end < first.out_of_sample_start
    assert first.out_of_sample_end < second.out_of_sample_start
    assert len(plan.input_hash) == len(plan.result_hash) == 64


def test_rolling_train_window_stays_fixed_without_shuffling() -> None:
    plan = _plan(expanding=False)

    assert all(len(item.train_sessions) == 4 for item in plan.folds)
    assert plan.folds[1].train_sessions == _sessions()[2:6]
    assert plan.folds[2].train_sessions == _sessions()[4:8]


def test_splitter_rejects_unsorted_duplicate_short_or_overlapping_oos_inputs() -> None:
    sessions = _sessions()
    with pytest.raises(WalkForwardValidationError, match="already be unique and strictly"):
        build_walk_forward_plan(tuple(reversed(sessions)), _config())
    with pytest.raises(WalkForwardValidationError, match="already be unique and strictly"):
        build_walk_forward_plan((sessions[0], sessions[0], *sessions[1:]), _config())
    with pytest.raises(WalkForwardValidationError, match="requires at least"):
        build_walk_forward_plan(sessions[:7], _config())
    with pytest.raises(ValueError, match="cannot overlap"):
        WalkForwardSplitConfig(
            train_sessions=4,
            validation_sessions=2,
            out_of_sample_sessions=3,
            step_sessions=2,
        )


def test_parameter_registry_is_versioned_sorted_and_idempotent() -> None:
    registry = _registered_registry()
    snapshot = registry.snapshot()
    registered_at = snapshot.entries[0].registered_at + timedelta(days=1)
    repeated = registry.register(
        strategy_name="mainline-leader",
        strategy_version="mainline-leader-v1",
        parameter_version="params-v1",
        parameters={"minimum_score": Decimal("60"), "maximum_positions": 2},
        registered_at=registered_at,
    )

    assert repeated == snapshot.entries[0]
    assert tuple(item.parameter_version for item in snapshot.entries) == (
        "params-v1",
        "params-v2",
    )
    assert len(snapshot.registry_hash) == 64
    with pytest.raises(WalkForwardValidationError, match="different values"):
        registry.register(
            strategy_name="mainline-leader",
            strategy_version="mainline-leader-v1",
            parameter_version="params-v1",
            parameters={"maximum_positions": 9},
            registered_at=registered_at,
        )
    with pytest.raises(ValueError, match="must be string"):
        registry.register(
            strategy_name="mainline-leader",
            strategy_version="mainline-leader-v1",
            parameter_version="float-params",
            parameters={"threshold": 0.5},  # type: ignore[dict-item]
            registered_at=registered_at,
        )


def test_parameters_are_frozen_before_oos_and_cannot_be_selected_inside_it() -> None:
    plan = _plan()
    registry = _registered_registry()
    fold = plan.folds[0]
    freeze = registry.freeze_for_fold(
        fold=fold,
        strategy_name="mainline-leader",
        strategy_version="mainline-leader-v1",
        parameter_version="params-v1",
        frozen_at=datetime.combine(fold.validation_end, time(18), tzinfo=SHANGHAI_TZ),
    )

    assert freeze.out_of_sample_start == fold.out_of_sample_start
    assert freeze.out_of_sample_end == fold.out_of_sample_end
    assert freeze.validation_end < freeze.out_of_sample_start
    with pytest.raises(ValueError, match="no later than validation_end"):
        registry.freeze_for_fold(
            fold=fold,
            strategy_name="mainline-leader",
            strategy_version="mainline-leader-v1",
            parameter_version="params-v1",
            frozen_at=datetime.combine(
                fold.out_of_sample_start,
                time(18),
                tzinfo=SHANGHAI_TZ,
            ),
        )
    with pytest.raises(WalkForwardValidationError, match="not registered"):
        registry.freeze_for_fold(
            fold=fold,
            strategy_name="mainline-leader",
            strategy_version="mainline-leader-v1",
            parameter_version="missing",
            frozen_at=datetime.combine(fold.validation_end, time(18), tzinfo=SHANGHAI_TZ),
        )


def test_parameter_perturbation_stability_is_computable_and_hash_bound() -> None:
    _plan_value, freezes = _freezes()
    freeze = freezes[0]
    perturbations = (
        build_parameter_perturbation(
            parameter_freeze_hash=freeze.freeze_hash,
            parameter_name="minimum_score",
            relative_change=Decimal("-0.10"),
            net_metric=Decimal("0.070"),
        ),
        build_parameter_perturbation(
            parameter_freeze_hash=freeze.freeze_hash,
            parameter_name="minimum_score",
            relative_change=Decimal("0.10"),
            net_metric=Decimal("0.075"),
        ),
    )
    summary = evaluate_parameter_stability(
        parameter_freeze_hash=freeze.freeze_hash,
        parameter_name="minimum_score",
        baseline_net_metric=Decimal("0.080"),
        perturbations=tuple(reversed(perturbations)),
    )

    assert summary.stable
    assert summary.relative_degradations == (Decimal("0.125"), Decimal("0.0625"))
    assert summary.worst_relative_degradation == Decimal("0.125")
    assert tuple(item.relative_change for item in summary.perturbations) == (
        Decimal("-0.10"),
        Decimal("0.10"),
    )
    with pytest.raises(ValueError, match="result_hash"):
        replace(summary, result_hash="f" * 64)


def test_walk_forward_report_records_versions_costs_and_frozen_oos_metrics() -> None:
    plan, freezes, results = _results()
    perturbations = tuple(
        build_parameter_perturbation(
            parameter_freeze_hash=freezes[0].freeze_hash,
            parameter_name="minimum_score",
            relative_change=change,
            net_metric=metric,
        )
        for change, metric in (
            (Decimal("-0.10"), Decimal("0.070")),
            (Decimal("0.10"), Decimal("0.075")),
        )
    )
    stability = evaluate_parameter_stability(
        parameter_freeze_hash=freezes[0].freeze_hash,
        parameter_name="minimum_score",
        baseline_net_metric=results[0].net_return,
        perturbations=perturbations,
    )

    report = build_walk_forward_report(
        plan=plan,
        freezes=freezes,
        results=results,
        stability=(stability,),
    )
    repeated = build_walk_forward_report(
        plan=plan,
        freezes=freezes,
        results=results,
        stability=(stability,),
    )

    assert report == repeated
    assert report.parameter_versions == ("params-v1", "params-v2")
    assert report.data_versions == ("snapshot-v1",)
    assert report.code_versions == ("backtest-code-v1",)
    assert report.total_turnover == Decimal("4.5")
    assert report.total_transaction_cost == Decimal("0.04")
    assert report.worst_maximum_drawdown == Decimal("-0.10")
    assert abs(report.mean_gross_return * 3 - Decimal("0.13")) < Decimal("1e-48")
    assert abs(report.mean_net_return * 3 - Decimal("0.08")) < Decimal("1e-48")
    assert "descriptive chronological out-of-sample" in report.interpretation
    assert "no guaranteed-return" in report.interpretation
    assert len(report.input_hash) == len(report.result_hash) == 64


def test_report_rejects_wrong_freeze_observation_order_and_future_availability() -> None:
    plan, freezes, results = _results()
    with pytest.raises(WalkForwardValidationError, match="exact fold, freeze"):
        changed = _rebuild_result(
            results[0],
            parameter_freeze_hash=freezes[1].freeze_hash,
        )
        build_walk_forward_report(
            plan=plan,
            freezes=freezes,
            results=(changed, *results[1:]),
        )
    with pytest.raises(WalkForwardValidationError, match="one freeze and result"):
        build_walk_forward_report(plan=plan, freezes=freezes[:-1], results=results)
    early = _rebuild_result(
        results[0],
        available_at=datetime.combine(
            plan.folds[0].validation_end,
            time(18),
            tzinfo=SHANGHAI_TZ,
        ),
    )
    with pytest.raises(WalkForwardValidationError, match="before its frozen window ends"):
        build_walk_forward_report(
            plan=plan,
            freezes=freezes,
            results=(early, *results[1:]),
        )


def test_result_and_report_contracts_reject_metric_or_hash_tampering() -> None:
    plan, freezes, results = _results()
    with pytest.raises(ValueError, match="net OOS return"):
        build_out_of_sample_result(
            fold_id=plan.folds[0].fold_id,
            fold_hash=plan.folds[0].fold_hash,
            parameter_freeze_hash=freezes[0].freeze_hash,
            observation_count=2,
            gross_return=Decimal("0.01"),
            benchmark_return=Decimal(0),
            net_return=Decimal("0.02"),
            maximum_drawdown=Decimal("-0.01"),
            turnover=Decimal(1),
            transaction_cost=Decimal("0.01"),
            data_version="snapshot-v1",
            code_version="code-v1",
            available_at=results[0].available_at,
            input_hash="a" * 64,
        )
    report = build_walk_forward_report(plan=plan, freezes=freezes, results=results)
    with pytest.raises(ValueError, match="report result_hash"):
        replace(report, result_hash="f" * 64)
    with pytest.raises(ValueError, match="report input_hash"):
        replace(report, plan_result_hash="e" * 64)
