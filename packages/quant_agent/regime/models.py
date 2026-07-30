"""Stable market-regime output contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware


class MarketRegime(StrEnum):
    """Five market states defined by the project specification."""

    UPTREND = "UPTREND"
    RANGE_STRONG = "RANGE_STRONG"
    DIVERGENT = "DIVERGENT"
    DOWNTREND = "DOWNTREND"
    BOTTOM_RECOVERY = "BOTTOM_RECOVERY"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One transparent feature contribution."""

    feature: str
    value: float
    contribution: float
    interpretation: str


@dataclass(frozen=True, slots=True)
class MarketRegimeResult:
    """Raw deterministic regime classification.

    ``confidence`` describes classification confidence, not return probability.
    """

    as_of: datetime
    regime: MarketRegime
    score: float
    confidence: float
    max_risk_budget: float
    evidence: tuple[Evidence, ...]
    counter_evidence: tuple[Evidence, ...]
    invalidations: tuple[str, ...]
    data_version: str
    model_version: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if not 0 <= self.score <= 100:
            raise ValueError("regime score must be in [0, 100]")
        if not 0 <= self.confidence <= 1:
            raise ValueError("classification confidence must be in [0, 1]")
        if not 0 <= self.max_risk_budget <= 1:
            raise ValueError("risk budget must be in [0, 1]")
