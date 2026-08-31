"""Deterministic multi-index market trend features."""

from quant_agent.features.market_trend.analyzer import MarketTrendAnalyzer
from quant_agent.features.market_trend.contracts import (
    IndexTrendResult,
    InsufficientMarketData,
    MarketTrendConfig,
    MarketTrendSnapshot,
    MissingIndexPolicy,
    WindowTrendMetrics,
)

__all__ = [
    "IndexTrendResult",
    "InsufficientMarketData",
    "MarketTrendAnalyzer",
    "MarketTrendConfig",
    "MarketTrendSnapshot",
    "MissingIndexPolicy",
    "WindowTrendMetrics",
]
