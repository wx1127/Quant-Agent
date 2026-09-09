"""Deterministic cross-strategy target portfolio construction."""

from .adapters import sleeve_from_etf_decision, sleeve_from_mainline_decision
from .builder import TargetPortfolioBuilder
from .contracts import (
    PITIndustryClassification,
    PortfolioAdjustmentCode,
    PortfolioBuilderConfig,
    PortfolioBuildInputError,
    PortfolioBuildRequest,
    PortfolioConstraintAdjustment,
    PortfolioTargetLine,
    SleeveTarget,
    StrategySleeve,
    StrategySleeveKind,
    StrategyWeightContribution,
    TargetPortfolio,
)

__all__ = [
    "PITIndustryClassification",
    "PortfolioAdjustmentCode",
    "PortfolioBuildInputError",
    "PortfolioBuildRequest",
    "PortfolioBuilderConfig",
    "PortfolioConstraintAdjustment",
    "PortfolioTargetLine",
    "SleeveTarget",
    "StrategySleeve",
    "StrategySleeveKind",
    "StrategyWeightContribution",
    "TargetPortfolio",
    "TargetPortfolioBuilder",
    "sleeve_from_etf_decision",
    "sleeve_from_mainline_decision",
]
