"""Transparent weighted market-regime classifier."""

import statistics
from dataclasses import dataclass
from itertools import pairwise

from quant_agent.features.market_breadth import MarketBreadthResult
from quant_agent.features.market_trend import MarketTrendResult
from quant_agent.regime.models import Evidence, MarketRegime, MarketRegimeResult


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Versioned score weights and state thresholds."""

    version: str = "regime_v1"
    trend_weight: float = 0.25
    breadth_weight: float = 0.20
    turnover_weight: float = 0.15
    new_high_low_weight: float = 0.15
    diffusion_weight: float = 0.15
    risk_weight: float = 0.10
    uptrend_threshold: float = 70.0
    range_strong_threshold: float = 55.0
    divergent_threshold: float = 40.0
    recovery_history_days: int = 3

    def __post_init__(self) -> None:
        weights = (
            self.trend_weight,
            self.breadth_weight,
            self.turnover_weight,
            self.new_high_low_weight,
            self.diffusion_weight,
            self.risk_weight,
        )
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("regime weights must sum to one")
        if not (
            0
            < self.divergent_threshold
            < self.range_strong_threshold
            < self.uptrend_threshold
            < 100
        ):
            raise ValueError("regime thresholds must be strictly increasing")
        if self.recovery_history_days < 2:
            raise ValueError("recovery history requires at least two days")


class MarketRegimeClassifier:
    """Combine trend and breadth evidence into one raw market state."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def _is_bottom_recovery(
        self,
        trend: MarketTrendResult,
        breadth: MarketBreadthResult,
        history: list[MarketBreadthResult],
    ) -> bool:
        recent = [*history, breadth][-self.config.recovery_history_days :]
        if len(recent) < self.config.recovery_history_days:
            return False
        new_lows = [item.new_low_20_ratio for item in recent]
        breadth_scores = [item.breadth_score for item in recent]
        return (
            trend.score < self.config.range_strong_threshold
            and all(current <= previous for previous, current in pairwise(new_lows))
            and new_lows[-1] < new_lows[0]
            and all(current >= previous for previous, current in pairwise(breadth_scores))
            and breadth_scores[-1] > breadth_scores[0]
            and breadth.median_daily_return >= 0
        )

    def classify(
        self,
        trend: MarketTrendResult,
        breadth: MarketBreadthResult,
        *,
        breadth_history: list[MarketBreadthResult] | None = None,
    ) -> MarketRegimeResult:
        """Classify one date using only results available by that date."""

        if trend.as_of != breadth.as_of:
            raise ValueError("trend and breadth must share the same as_of")
        if trend.data_version != breadth.data_version:
            raise ValueError("trend and breadth must share a data version")
        components = {
            "trend": (trend.score, self.config.trend_weight),
            "breadth": (breadth.breadth_score, self.config.breadth_weight),
            "turnover": (breadth.turnover_score, self.config.turnover_weight),
            "new_high_low": (
                breadth.new_high_low_score,
                self.config.new_high_low_weight,
            ),
            "diffusion": (breadth.diffusion_score, self.config.diffusion_weight),
            "risk": (breadth.risk_score, self.config.risk_weight),
        }
        score = sum(value * weight for value, weight in components.values())
        history = [item for item in (breadth_history or []) if item.as_of < breadth.as_of]
        if self._is_bottom_recovery(trend, breadth, history):
            regime = MarketRegime.BOTTOM_RECOVERY
        elif score >= self.config.uptrend_threshold:
            regime = MarketRegime.UPTREND
        elif score >= self.config.range_strong_threshold:
            regime = MarketRegime.RANGE_STRONG
        elif score >= self.config.divergent_threshold:
            regime = MarketRegime.DIVERGENT
        else:
            regime = MarketRegime.DOWNTREND

        positive: list[Evidence] = []
        negative: list[Evidence] = []
        for name, (value, weight) in components.items():
            item = Evidence(
                feature=name,
                value=value,
                contribution=value * weight,
                interpretation=(
                    "supports stronger market state"
                    if value >= 55
                    else "opposes stronger market state"
                ),
            )
            if value >= 55:
                positive.append(item)
            elif value <= 45:
                negative.append(item)

        thresholds = [
            self.config.divergent_threshold,
            self.config.range_strong_threshold,
            self.config.uptrend_threshold,
        ]
        boundary_distance = min(abs(score - threshold) for threshold in thresholds)
        agreement = abs(statistics.fmean(value for value, _weight in components.values()) - 50) / 50
        confidence = min(1.0, 0.45 + boundary_distance / 30 + agreement * 0.25)
        if regime is MarketRegime.BOTTOM_RECOVERY:
            confidence = min(confidence, 0.75)

        risk_budgets = {
            MarketRegime.UPTREND: 0.80,
            MarketRegime.RANGE_STRONG: 0.60,
            MarketRegime.DIVERGENT: 0.40,
            MarketRegime.DOWNTREND: 0.20,
            MarketRegime.BOTTOM_RECOVERY: 0.30,
        }
        invalidations = {
            MarketRegime.UPTREND: (
                "market breadth falls below 50 for three consecutive observations",
                "major-index trend score falls below 55",
            ),
            MarketRegime.RANGE_STRONG: (
                "theme diffusion and advancing ratio deteriorate together",
                "combined regime score falls below 40",
            ),
            MarketRegime.DIVERGENT: (
                "new-low ratio expands while turnover rises",
                "combined regime score breaks either adjacent state threshold",
            ),
            MarketRegime.DOWNTREND: (
                "breadth and new-low evidence improve for the configured recovery window",
                "combined regime score recovers above 40",
            ),
            MarketRegime.BOTTOM_RECOVERY: (
                "new-low ratio rises again",
                "breadth recovery fails to persist",
            ),
        }
        return MarketRegimeResult(
            as_of=trend.as_of,
            regime=regime,
            score=score,
            confidence=confidence,
            max_risk_budget=risk_budgets[regime],
            evidence=tuple(positive),
            counter_evidence=tuple(negative),
            invalidations=invalidations[regime],
            data_version=trend.data_version,
            model_version=self.config.version,
        )
