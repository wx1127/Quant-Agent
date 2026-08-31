"""OOS fold result, parameter sensitivity, and walk-forward report builders."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal, localcontext

from quant_agent.backtest.validation.contracts import (
    FrozenParameterSet,
    OutOfSampleFoldResult,
    ParameterPerturbationResult,
    ParameterStabilitySummary,
    WalkForwardPlan,
    WalkForwardValidationError,
    WalkForwardValidationReport,
)
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core.identity import canonical_decimal
from quant_agent.regime.contracts import stable_hash


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return sum(values, Decimal(0)) / Decimal(len(values))


def build_out_of_sample_result(
    *,
    fold_id: str,
    fold_hash: str,
    parameter_freeze_hash: str,
    observation_count: int,
    gross_return: Decimal,
    benchmark_return: Decimal,
    net_return: Decimal,
    maximum_drawdown: Decimal,
    turnover: Decimal,
    transaction_cost: Decimal,
    data_version: str,
    code_version: str,
    available_at: datetime,
    input_hash: str,
) -> OutOfSampleFoldResult:
    """Build a hash-bound execution-aware result after one OOS window."""

    payload = {
        "available_at": available_at,
        "benchmark_return": benchmark_return,
        "code_version": code_version,
        "data_version": data_version,
        "fold_hash": fold_hash,
        "fold_id": fold_id,
        "gross_return": gross_return,
        "input_hash": input_hash,
        "maximum_drawdown": maximum_drawdown,
        "net_return": net_return,
        "observation_count": observation_count,
        "parameter_freeze_hash": parameter_freeze_hash,
        "transaction_cost": transaction_cost,
        "turnover": turnover,
    }
    return OutOfSampleFoldResult(
        fold_id=fold_id,
        fold_hash=fold_hash,
        parameter_freeze_hash=parameter_freeze_hash,
        observation_count=observation_count,
        gross_return=gross_return,
        benchmark_return=benchmark_return,
        net_return=net_return,
        maximum_drawdown=maximum_drawdown,
        turnover=turnover,
        transaction_cost=transaction_cost,
        data_version=data_version,
        code_version=code_version,
        available_at=available_at,
        input_hash=input_hash,
        result_hash=stable_hash(payload),
    )


def build_parameter_perturbation(
    *,
    parameter_freeze_hash: str,
    parameter_name: str,
    relative_change: Decimal,
    net_metric: Decimal,
) -> ParameterPerturbationResult:
    """Build one deterministic local parameter sensitivity observation."""

    payload = {
        "net_metric": net_metric,
        "parameter_freeze_hash": parameter_freeze_hash,
        "parameter_name": parameter_name,
        "relative_change": relative_change,
    }
    return ParameterPerturbationResult(
        parameter_freeze_hash=parameter_freeze_hash,
        parameter_name=parameter_name,
        relative_change=relative_change,
        net_metric=net_metric,
        result_hash=stable_hash(payload),
    )


def evaluate_parameter_stability(
    *,
    parameter_freeze_hash: str,
    parameter_name: str,
    baseline_net_metric: Decimal,
    perturbations: Iterable[ParameterPerturbationResult],
    maximum_allowed_degradation: Decimal = Decimal("0.20"),
    scale_floor: Decimal = Decimal("0.01"),
) -> ParameterStabilitySummary:
    """Compute normalized worst degradation for local parameter changes."""

    canonical_decimal(baseline_net_metric, field_name="baseline net metric")
    canonical_decimal(scale_floor, field_name="stability scale floor")
    canonical_decimal(
        maximum_allowed_degradation,
        field_name="maximum allowed degradation",
    )
    if scale_floor <= 0 or maximum_allowed_degradation < 0:
        raise ValueError("stability scale must be positive and threshold non-negative")
    frozen = tuple(
        sorted(
            perturbations,
            key=lambda item: (item.relative_change, item.result_hash),
        )
    )
    if len(frozen) < 2:
        raise WalkForwardValidationError(
            "parameter stability requires at least two perturbation results"
        )
    changes = tuple(item.relative_change for item in frozen)
    if len(set(changes)) != len(changes):
        raise WalkForwardValidationError("parameter perturbation changes must be unique")
    if any(
        item.parameter_freeze_hash != parameter_freeze_hash or item.parameter_name != parameter_name
        for item in frozen
    ):
        raise WalkForwardValidationError(
            "parameter perturbations must share the requested frozen identity"
        )
    denominator = max(abs(baseline_net_metric), scale_floor)
    with localcontext() as context:
        context.prec = 50
        degradations = tuple(
            (baseline_net_metric - item.net_metric) / denominator for item in frozen
        )
    worst = max(degradations)
    stable = worst <= maximum_allowed_degradation
    payload = {
        "baseline_net_metric": baseline_net_metric,
        "maximum_allowed_degradation": maximum_allowed_degradation,
        "parameter_freeze_hash": parameter_freeze_hash,
        "parameter_name": parameter_name,
        "perturbations": [item.result_hash for item in frozen],
        "relative_degradations": degradations,
        "scale_floor": scale_floor,
        "stable": stable,
        "worst_relative_degradation": worst,
    }
    return ParameterStabilitySummary(
        parameter_freeze_hash=parameter_freeze_hash,
        parameter_name=parameter_name,
        baseline_net_metric=baseline_net_metric,
        scale_floor=scale_floor,
        maximum_allowed_degradation=maximum_allowed_degradation,
        perturbations=frozen,
        relative_degradations=degradations,
        worst_relative_degradation=worst,
        stable=stable,
        result_hash=stable_hash(payload),
    )


def build_walk_forward_report(
    *,
    plan: WalkForwardPlan,
    freezes: Iterable[FrozenParameterSet],
    results: Iterable[OutOfSampleFoldResult],
    stability: Iterable[ParameterStabilitySummary] = (),
) -> WalkForwardValidationReport:
    """Aggregate exact frozen OOS folds; never choose or change parameters here."""

    frozen_parameters = tuple(freezes)
    frozen_results = tuple(results)
    frozen_stability = tuple(
        sorted(stability, key=lambda item: (item.parameter_freeze_hash, item.parameter_name))
    )
    if len(frozen_parameters) != len(plan.folds) or len(frozen_results) != len(plan.folds):
        raise WalkForwardValidationError(
            "walk-forward report requires one freeze and result for every plan fold"
        )
    for fold, parameters, result in zip(
        plan.folds,
        frozen_parameters,
        frozen_results,
        strict=True,
    ):
        if (
            parameters.fold_id != fold.fold_id
            or parameters.fold_hash != fold.fold_hash
            or parameters.validation_end != fold.validation_end
            or parameters.out_of_sample_start != fold.out_of_sample_start
            or parameters.out_of_sample_end != fold.out_of_sample_end
        ):
            raise WalkForwardValidationError(
                "frozen parameters must cover their exact validation and OOS fold"
            )
        if (
            result.fold_id != fold.fold_id
            or result.fold_hash != fold.fold_hash
            or result.parameter_freeze_hash != parameters.freeze_hash
            or result.observation_count != len(fold.out_of_sample_sessions)
        ):
            raise WalkForwardValidationError(
                "OOS result must bind its exact fold, freeze, and observation count"
            )
        ensure_aware(result.available_at)
        if result.available_at.astimezone(SHANGHAI_TZ).date() < fold.out_of_sample_end:
            raise WalkForwardValidationError(
                "OOS result cannot be available before its frozen window ends"
            )
    freeze_hashes = {item.freeze_hash for item in frozen_parameters}
    if any(item.parameter_freeze_hash not in freeze_hashes for item in frozen_stability):
        raise WalkForwardValidationError(
            "stability summaries must bind a parameter freeze in this report"
        )
    parameter_versions = tuple(sorted({item.parameter_version for item in frozen_parameters}))
    data_versions = tuple(sorted({item.data_version for item in frozen_results}))
    code_versions = tuple(sorted({item.code_version for item in frozen_results}))
    gross_returns = tuple(item.gross_return for item in frozen_results)
    net_returns = tuple(item.net_return for item in frozen_results)
    mean_gross = _mean(gross_returns)
    mean_net = _mean(net_returns)
    worst_drawdown = min(item.maximum_drawdown for item in frozen_results)
    total_turnover = sum((item.turnover for item in frozen_results), Decimal(0))
    total_cost = sum((item.transaction_cost for item in frozen_results), Decimal(0))
    input_hash = stable_hash(
        {
            "freezes": [item.freeze_hash for item in frozen_parameters],
            "plan_result_hash": plan.result_hash,
            "results": [item.result_hash for item in frozen_results],
            "stability": [item.result_hash for item in frozen_stability],
        }
    )
    interpretation = (
        "descriptive chronological out-of-sample results using parameters frozen before "
        "each fold; no guaranteed-return or causal claim"
    )
    result_payload = {
        "code_versions": code_versions,
        "data_versions": data_versions,
        "input_hash": input_hash,
        "interpretation": interpretation,
        "mean_gross_return": mean_gross,
        "mean_net_return": mean_net,
        "parameter_versions": parameter_versions,
        "total_transaction_cost": total_cost,
        "total_turnover": total_turnover,
        "worst_maximum_drawdown": worst_drawdown,
    }
    return WalkForwardValidationReport(
        plan_result_hash=plan.result_hash,
        freezes=frozen_parameters,
        results=frozen_results,
        stability=frozen_stability,
        parameter_versions=parameter_versions,
        data_versions=data_versions,
        code_versions=code_versions,
        mean_gross_return=mean_gross,
        mean_net_return=mean_net,
        worst_maximum_drawdown=worst_drawdown,
        total_turnover=total_turnover,
        total_transaction_cost=total_cost,
        input_hash=input_hash,
        result_hash=stable_hash(result_payload),
        interpretation=interpretation,
    )


__all__ = [
    "build_out_of_sample_result",
    "build_parameter_perturbation",
    "build_walk_forward_report",
    "evaluate_parameter_stability",
]
