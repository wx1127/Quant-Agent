"""PIT-safe ETF rotation strategy and backtest adapters."""

from quant_agent.strategies.etf_rotation.adapters import (
    to_event_targets,
    to_weight_signals,
)
from quant_agent.strategies.etf_rotation.contracts import (
    ETFEventAdaptation,
    ETFHorizonMomentum,
    ETFMomentumMetrics,
    ETFRotationCandidate,
    ETFRotationConfig,
    ETFRotationDecision,
    ETFRotationDecisionStatus,
    ETFRotationInputError,
    ETFRotationRequest,
    ETFTargetWeight,
    ETFUniverseEligibility,
    ETFUniverseRevision,
    ETFUniverseSelection,
    target_weight_map,
)
from quant_agent.strategies.etf_rotation.strategy import ETFRotationStrategy

__all__ = [
    "ETFEventAdaptation",
    "ETFHorizonMomentum",
    "ETFMomentumMetrics",
    "ETFRotationCandidate",
    "ETFRotationConfig",
    "ETFRotationDecision",
    "ETFRotationDecisionStatus",
    "ETFRotationInputError",
    "ETFRotationRequest",
    "ETFRotationStrategy",
    "ETFTargetWeight",
    "ETFUniverseEligibility",
    "ETFUniverseRevision",
    "ETFUniverseSelection",
    "target_weight_map",
    "to_event_targets",
    "to_weight_signals",
]
