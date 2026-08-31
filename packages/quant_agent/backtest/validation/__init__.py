"""Chronological validation, parameter freezing, and OOS reporting."""

from quant_agent.backtest.validation.contracts import (
    FrozenParameterSet,
    OutOfSampleFoldResult,
    ParameterPerturbationResult,
    ParameterRegistrySnapshot,
    ParameterScalar,
    ParameterStabilitySummary,
    ParameterValue,
    RegisteredParameterSet,
    WalkForwardFold,
    WalkForwardPlan,
    WalkForwardSplitConfig,
    WalkForwardValidationError,
    WalkForwardValidationReport,
)
from quant_agent.backtest.validation.parameters import ParameterRegistry
from quant_agent.backtest.validation.report import (
    build_out_of_sample_result,
    build_parameter_perturbation,
    build_walk_forward_report,
    evaluate_parameter_stability,
)
from quant_agent.backtest.validation.splitter import build_walk_forward_plan

__all__ = [
    "FrozenParameterSet",
    "OutOfSampleFoldResult",
    "ParameterPerturbationResult",
    "ParameterRegistry",
    "ParameterRegistrySnapshot",
    "ParameterScalar",
    "ParameterStabilitySummary",
    "ParameterValue",
    "RegisteredParameterSet",
    "WalkForwardFold",
    "WalkForwardPlan",
    "WalkForwardSplitConfig",
    "WalkForwardValidationError",
    "WalkForwardValidationReport",
    "build_out_of_sample_result",
    "build_parameter_perturbation",
    "build_walk_forward_plan",
    "build_walk_forward_report",
    "evaluate_parameter_stability",
]
