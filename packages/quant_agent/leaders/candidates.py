"""Explainable candidate ranking with explicit exclusion reasons."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.features.tradeability import TradeabilityResult
from quant_agent.leaders.scoring import LeaderResult


class CandidateTier(StrEnum):
    A = "A"
    B = "B"
    WATCH = "WATCH"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    version: str = "candidate_v1"
    market_weight: float = 0.20
    theme_weight: float = 0.20
    leader_weight: float = 0.20
    price_trend_weight: float = 0.15
    volume_price_weight: float = 0.10
    fundamental_event_weight: float = 0.10
    valuation_weight: float = 0.05
    tier_a_score: float = 75.0
    tier_b_score: float = 60.0

    def __post_init__(self) -> None:
        weights = (
            self.market_weight,
            self.theme_weight,
            self.leader_weight,
            self.price_trend_weight,
            self.volume_price_weight,
            self.fundamental_event_weight,
            self.valuation_weight,
        )
        if abs(sum(weights) - 1) > 1e-9:
            raise ValueError("candidate weights must sum to one")


@dataclass(frozen=True, slots=True)
class CandidateInput:
    as_of: datetime
    leader: LeaderResult
    tradeability: TradeabilityResult
    market_regime_score: float
    theme_score: float
    price_trend_score: float
    volume_price_score: float
    fundamental_or_event_score: float
    valuation_score: float
    risk_penalty: float
    risk_events: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.as_of != self.leader.as_of:
            raise ValueError("candidate and leader must share as_of")
        components = (
            self.market_regime_score,
            self.theme_score,
            self.price_trend_score,
            self.volume_price_score,
            self.fundamental_or_event_score,
            self.valuation_score,
        )
        if any(not 0 <= value <= 100 for value in components):
            raise ValueError("candidate components must be in [0, 100]")
        if self.risk_penalty < 0:
            raise ValueError("candidate risk penalty cannot be negative")


@dataclass(frozen=True, slots=True)
class CandidateResult:
    as_of: datetime
    instrument_id: str
    instrument_name: str
    theme_id: str
    leader_type: str
    leader_score: float
    candidate_score: float
    rank: int
    tier: CandidateTier
    tradable: bool
    exclusion_reasons: tuple[str, ...]
    support_evidence: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    risk_events: tuple[str, ...]
    observation_conditions: tuple[str, ...]
    invalidations: tuple[str, ...]
    data_version: str
    model_version: str


class CandidateEngine:
    def __init__(self, config: CandidateConfig | None = None) -> None:
        self.config = config or CandidateConfig()

    def rank(self, items: list[CandidateInput]) -> tuple[CandidateResult, ...]:
        if not items:
            return ()
        if len({item.as_of for item in items}) != 1:
            raise ValueError("candidate batch must share as_of")
        rows: list[tuple[CandidateInput, float]] = []
        for item in items:
            score = (
                self.config.market_weight * item.market_regime_score
                + self.config.theme_weight * item.theme_score
                + self.config.leader_weight * item.leader.score
                + self.config.price_trend_weight * item.price_trend_score
                + self.config.volume_price_weight * item.volume_price_score
                + self.config.fundamental_event_weight * item.fundamental_or_event_score
                + self.config.valuation_weight * item.valuation_score
                - item.risk_penalty
            )
            rows.append((item, max(0.0, min(100.0, score))))
        ordered = sorted(rows, key=lambda row: (-row[1], row[0].leader.instrument_id))
        results = []
        for rank, (item, score) in enumerate(ordered, start=1):
            exclusions = list(item.tradeability.reasons)
            if not item.leader.trade_candidate_type:
                exclusions.append("leader type is observation-only in version one")
            if item.risk_events:
                exclusions.append("unresolved material risk event")
            tradable = not exclusions
            if not tradable:
                tier = CandidateTier.EXCLUDED
            elif score >= self.config.tier_a_score:
                tier = CandidateTier.A
            elif score >= self.config.tier_b_score:
                tier = CandidateTier.B
            else:
                tier = CandidateTier.WATCH
            support = [
                label
                for condition, label in (
                    (item.market_regime_score >= 55, "market regime is supportive"),
                    (item.theme_score >= 60, "mainline score is supportive"),
                    (item.leader.score >= 60, "leader evidence is supportive"),
                    (item.price_trend_score >= 60, "price trend is supportive"),
                )
                if condition
            ]
            counter = list(exclusions)
            if item.risk_penalty > 0:
                counter.append("candidate risk penalty applied")
            results.append(
                CandidateResult(
                    as_of=item.as_of,
                    instrument_id=item.leader.instrument_id,
                    instrument_name=item.leader.instrument_name,
                    theme_id=item.leader.theme_id,
                    leader_type=item.leader.leader_type.value,
                    leader_score=item.leader.score,
                    candidate_score=score,
                    rank=rank,
                    tier=tier,
                    tradable=tradable,
                    exclusion_reasons=tuple(exclusions),
                    support_evidence=tuple(support),
                    counter_evidence=tuple(counter),
                    risk_events=item.risk_events,
                    observation_conditions=(
                        "theme remains confirmed",
                        "tradeability checks remain clear",
                    ),
                    invalidations=(
                        "theme enters FADING state",
                        "stock becomes suspended or limit-locked",
                        "material negative event is published",
                    ),
                    data_version=item.leader.data_version,
                    model_version=self.config.version,
                )
            )
        return tuple(results)
