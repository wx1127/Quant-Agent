"""Adapters from complete P4 strategy decisions into isolated P5 sleeves."""

from __future__ import annotations

from quant_agent.strategies.etf_rotation import (
    ETFRotationDecision,
    ETFRotationDecisionStatus,
)
from quant_agent.strategies.mainline_leader import MainlineLeaderDecision

from .contracts import (
    PortfolioBuildInputError,
    SleeveTarget,
    StrategySleeve,
    StrategySleeveKind,
)


def sleeve_from_etf_decision(decision: ETFRotationDecision) -> StrategySleeve:
    """Convert a complete ETF target decision into an ETF-core sleeve."""

    if decision.status is ETFRotationDecisionStatus.NO_REBALANCE:
        raise PortfolioBuildInputError(
            "NO_REBALANCE carries no complete ETF target and cannot build a sleeve"
        )
    reasons = {item.instrument_id: item.reason for item in decision.candidates}
    targets = tuple(
        SleeveTarget(
            instrument_id=item.instrument_id,
            instrument_type=item.instrument_type,
            local_target_weight=item.target_weight,
            reason=reasons.get(item.instrument_id, decision.reason),
        )
        for item in decision.targets
    )
    return StrategySleeve.build(
        sleeve_id=f"etf-core:{decision.result_hash}",
        kind=StrategySleeveKind.ETF_CORE,
        as_of=decision.as_of,
        data_version=decision.data_version,
        strategy_name="etf-rotation",
        strategy_version=decision.strategy_version,
        strategy_config_hash=decision.config_hash,
        source_result_hash=decision.result_hash,
        targets=targets,
    )


def sleeve_from_mainline_decision(decision: MainlineLeaderDecision) -> StrategySleeve:
    """Convert a complete mainline-leader decision into a stock-enhancement sleeve."""

    reasons = {item.instrument_id: item.rationale for item in decision.selections}
    targets = tuple(
        SleeveTarget(
            instrument_id=item.instrument_id,
            instrument_type=item.instrument_type,
            local_target_weight=item.target_weight,
            reason=reasons.get(item.instrument_id, decision.reason),
        )
        for item in decision.targets
    )
    return StrategySleeve.build(
        sleeve_id=f"stock-enhancement:{decision.result_hash}",
        kind=StrategySleeveKind.STOCK_ENHANCEMENT,
        as_of=decision.as_of,
        data_version=decision.data_version,
        strategy_name="mainline-leader",
        strategy_version=decision.strategy_version,
        strategy_config_hash=decision.config_hash,
        source_result_hash=decision.result_hash,
        targets=targets,
    )


__all__ = ["sleeve_from_etf_decision", "sleeve_from_mainline_decision"]
