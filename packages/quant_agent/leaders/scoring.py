"""Leader scoring restricted to confirmed mainline constituents."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.themes.scoring import ThemeState


class LeaderType(StrEnum):
    CORE = "CORE"
    TREND = "TREND"
    CAPACITY = "CAPACITY"
    ELASTICITY = "ELASTICITY"
    CATCH_UP = "CATCH_UP"
    SENTIMENT = "SENTIMENT"


@dataclass(frozen=True, slots=True)
class LeaderConfig:
    version: str = "leader_v1"
    within_theme_weight: float = 0.25
    trend_weight: float = 0.15
    liquidity_weight: float = 0.15
    theme_leadership_weight: float = 0.15
    downside_resilience_weight: float = 0.10
    logic_relevance_weight: float = 0.10
    fundamental_weight: float = 0.10

    def __post_init__(self) -> None:
        weights = (
            self.within_theme_weight,
            self.trend_weight,
            self.liquidity_weight,
            self.theme_leadership_weight,
            self.downside_resilience_weight,
            self.logic_relevance_weight,
            self.fundamental_weight,
        )
        if abs(sum(weights) - 1) > 1e-9:
            raise ValueError("leader weights must sum to one")


@dataclass(frozen=True, slots=True)
class LeaderInput:
    as_of: datetime
    instrument_id: str
    instrument_name: str
    theme_id: str
    theme_state: ThemeState
    within_theme_strength: float
    trend_quality: float
    liquidity: float
    theme_leadership: float
    downside_resilience: float
    logic_relevance: float
    fundamental_quality: float | None
    risk_penalty: float
    data_version: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        values = (
            self.within_theme_strength,
            self.trend_quality,
            self.liquidity,
            self.theme_leadership,
            self.downside_resilience,
            self.logic_relevance,
        )
        if any(not 0 <= value <= 100 for value in values):
            raise ValueError("leader components must be in [0, 100]")
        if self.fundamental_quality is not None and not 0 <= self.fundamental_quality <= 100:
            raise ValueError("fundamental quality must be in [0, 100]")
        if self.risk_penalty < 0:
            raise ValueError("risk penalty cannot be negative")


@dataclass(frozen=True, slots=True)
class LeaderResult:
    as_of: datetime
    instrument_id: str
    instrument_name: str
    theme_id: str
    leader_type: LeaderType
    score: float
    rank: int
    trade_candidate_type: bool
    support_evidence: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    data_version: str
    model_version: str


class LeaderScoringEngine:
    def __init__(self, config: LeaderConfig | None = None) -> None:
        self.config = config or LeaderConfig()

    @staticmethod
    def _type(item: LeaderInput) -> LeaderType:
        if item.trend_quality >= 70 and item.liquidity >= 70:
            return LeaderType.CORE
        if item.liquidity >= 75:
            return LeaderType.CAPACITY
        if item.trend_quality >= 65:
            return LeaderType.TREND
        if item.within_theme_strength >= 75:
            return LeaderType.ELASTICITY
        return LeaderType.CATCH_UP

    def rank(self, items: list[LeaderInput]) -> tuple[LeaderResult, ...]:
        if not items:
            return ()
        dates = {item.as_of for item in items}
        versions = {item.data_version for item in items}
        themes = {item.theme_id for item in items}
        if len(dates) != 1 or len(versions) != 1 or len(themes) != 1:
            raise ValueError("leader batch must share date, data version and theme")
        if any(
            item.theme_state not in {ThemeState.CONFIRMED, ThemeState.CROWDED} for item in items
        ):
            raise ValueError("leaders can only be ranked inside a confirmed mainline")
        scored: list[tuple[LeaderInput, float, LeaderType]] = []
        for item in items:
            fundamental = item.fundamental_quality or 0.0
            score = (
                self.config.within_theme_weight * item.within_theme_strength
                + self.config.trend_weight * item.trend_quality
                + self.config.liquidity_weight * item.liquidity
                + self.config.theme_leadership_weight * item.theme_leadership
                + self.config.downside_resilience_weight * item.downside_resilience
                + self.config.logic_relevance_weight * item.logic_relevance
                + self.config.fundamental_weight * fundamental
                - item.risk_penalty
            )
            scored.append((item, max(0.0, min(100.0, score)), self._type(item)))
        ordered = sorted(scored, key=lambda row: (-row[1], row[0].instrument_id))
        results = []
        for rank, (item, score, leader_type) in enumerate(ordered, start=1):
            support = [
                label
                for condition, label in (
                    (item.within_theme_strength >= 60, "strong within-theme return"),
                    (item.trend_quality >= 60, "trend structure is persistent"),
                    (item.liquidity >= 60, "liquidity supports execution"),
                    (item.theme_leadership >= 60, "price action leads theme peers"),
                )
                if condition
            ]
            counter = []
            if item.fundamental_quality is None:
                counter.append("fundamental evidence is incomplete")
            if item.risk_penalty > 0:
                counter.append("risk penalty applied")
            trade_type = leader_type in {LeaderType.TREND, LeaderType.CAPACITY}
            results.append(
                LeaderResult(
                    as_of=item.as_of,
                    instrument_id=item.instrument_id,
                    instrument_name=item.instrument_name,
                    theme_id=item.theme_id,
                    leader_type=leader_type,
                    score=score,
                    rank=rank,
                    trade_candidate_type=trade_type,
                    support_evidence=tuple(support),
                    counter_evidence=tuple(counter),
                    data_version=item.data_version,
                    model_version=self.config.version,
                )
            )
        return tuple(results)
