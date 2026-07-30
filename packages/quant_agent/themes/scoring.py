"""Deterministic mainline scoring, ranking and persistence states."""

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from quant_agent.features.core import clamp_score
from quant_agent.features.themes import ThemeFeatureResult


class ThemeState(StrEnum):
    """Lifecycle of an industry mainline."""

    EMERGING = "EMERGING"
    CONFIRMED = "CONFIRMED"
    CROWDED = "CROWDED"
    FADING = "FADING"


@dataclass(frozen=True, slots=True)
class MainlineConfig:
    """Versioned mainline weights and confirmation thresholds."""

    version: str = "mainline_v1"
    relative_strength_weight: float = 0.25
    breadth_weight: float = 0.20
    turnover_weight: float = 0.15
    leader_persistence_weight: float = 0.15
    new_high_weight: float = 0.10
    event_support_weight: float = 0.10
    downside_resilience_weight: float = 0.05
    top_rank_for_confirmation: int = 3
    top_history_rank: int = 5
    history_window: int = 5
    required_top_history_days: int = 3
    minimum_breadth_score: float = 50.0
    maximum_single_stock_contribution: float = 0.60
    crowded_threshold: float = 75.0
    fading_score: float = 60.0

    def __post_init__(self) -> None:
        weights = (
            self.relative_strength_weight,
            self.breadth_weight,
            self.turnover_weight,
            self.leader_persistence_weight,
            self.new_high_weight,
            self.event_support_weight,
            self.downside_resilience_weight,
        )
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("mainline weights must sum to one")
        if not 1 <= self.required_top_history_days <= self.history_window:
            raise ValueError("required history days must fit the history window")


@dataclass(frozen=True, slots=True)
class ThemeDailyScore:
    """Ranked daily score with lifecycle state and evidence."""

    as_of: datetime
    industry_id: str
    score: float
    rank: int
    state: ThemeState
    persistence_days: int
    top5_days_in_window: int
    crowding_score: float
    crowding_penalty: float
    eligible: bool
    support_evidence: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    data_version: str
    model_version: str


class ThemeScoringEngine:
    """Apply documented weights without using lifecycle history."""

    def __init__(self, config: MainlineConfig | None = None) -> None:
        self.config = config or MainlineConfig()

    def score(
        self,
        feature: ThemeFeatureResult,
        *,
        event_support_score: float = 50.0,
    ) -> ThemeDailyScore:
        """Return an unranked daily score; rank/state are added later."""

        if not 0 <= event_support_score <= 100:
            raise ValueError("event_support_score must be in [0, 100]")
        crowding_score = 0.5 * clamp_score(
            feature.relative_return_5d, 0.03, 0.15
        ) + 0.5 * clamp_score(feature.turnover_expansion, 0.30, 1.50)
        crowding_penalty = max(0.0, crowding_score - 70.0) * 0.25
        score = (
            self.config.relative_strength_weight * feature.relative_strength_score
            + self.config.breadth_weight * feature.breadth_score
            + self.config.turnover_weight * feature.turnover_score
            + self.config.leader_persistence_weight * feature.leader_persistence_score
            + self.config.new_high_weight * feature.new_high_score
            + self.config.event_support_weight * event_support_score
            + self.config.downside_resilience_weight * feature.downside_resilience_score
            - crowding_penalty
        )
        support = []
        counter = []
        if feature.relative_strength_score >= 60:
            support.append("industry relative strength is above neutral")
        else:
            counter.append("industry relative strength is not established")
        if feature.breadth_score >= self.config.minimum_breadth_score:
            support.append("industry advance is broadly distributed")
        else:
            counter.append("industry breadth is narrow")
        if feature.single_stock_contribution > self.config.maximum_single_stock_contribution:
            counter.append("performance is dominated by one stock")
        if not feature.eligible:
            counter.append("industry does not meet history/member eligibility")
        if event_support_score == 50:
            counter.append("event evidence is neutral until P3")
        return ThemeDailyScore(
            as_of=feature.as_of,
            industry_id=feature.industry_id,
            score=max(0.0, min(100.0, score)),
            rank=0,
            state=ThemeState.EMERGING,
            persistence_days=0,
            top5_days_in_window=0,
            crowding_score=crowding_score,
            crowding_penalty=crowding_penalty,
            eligible=feature.eligible,
            support_evidence=tuple(support),
            counter_evidence=tuple(counter),
            data_version=feature.data_version,
            model_version=self.config.version,
        )


class ThemeStateMachine:
    """Rank daily themes and confirm mainlines using only past ranks."""

    def __init__(self, config: MainlineConfig | None = None) -> None:
        self.config = config or MainlineConfig()
        self._rank_history: dict[str, list[tuple[datetime, int]]] = {}
        self._last_state: dict[str, ThemeState] = {}
        self._persistence: dict[str, int] = {}
        self._last_as_of: datetime | None = None

    def apply(self, scores: list[ThemeDailyScore]) -> tuple[ThemeDailyScore, ...]:
        """Process all theme scores for one strictly increasing date."""

        if not scores:
            return ()
        dates = {item.as_of for item in scores}
        versions = {item.data_version for item in scores}
        if len(dates) != 1 or len(versions) != 1:
            raise ValueError("daily theme batch must share as_of and data version")
        as_of = scores[0].as_of
        if self._last_as_of is not None and as_of <= self._last_as_of:
            raise ValueError("theme state updates require increasing as_of")
        self._last_as_of = as_of
        ordered = sorted(scores, key=lambda item: (-item.score, item.industry_id))
        results: list[ThemeDailyScore] = []
        for rank, item in enumerate(ordered, start=1):
            history = self._rank_history.setdefault(item.industry_id, [])
            history.append((as_of, rank))
            del history[: -self.config.history_window]
            top_days = sum(
                past_rank <= self.config.top_history_rank for _date, past_rank in history
            )
            previous = self._last_state.get(item.industry_id, ThemeState.EMERGING)
            confirmed = (
                item.eligible
                and rank <= self.config.top_rank_for_confirmation
                and top_days >= self.config.required_top_history_days
                and "industry advance is broadly distributed" in item.support_evidence
                and "performance is dominated by one stock" not in item.counter_evidence
            )
            fading = rank > self.config.top_history_rank or item.score < self.config.fading_score
            if (
                previous
                in {
                    ThemeState.CONFIRMED,
                    ThemeState.CROWDED,
                    ThemeState.FADING,
                }
                and fading
            ):
                state = ThemeState.FADING
            elif confirmed and item.crowding_score >= self.config.crowded_threshold:
                state = ThemeState.CROWDED
            elif confirmed:
                state = ThemeState.CONFIRMED
            else:
                state = ThemeState.EMERGING
            persistence = self._persistence.get(item.industry_id, 0) + 1 if state is previous else 1
            self._last_state[item.industry_id] = state
            self._persistence[item.industry_id] = persistence
            results.append(
                replace(
                    item,
                    rank=rank,
                    state=state,
                    persistence_days=persistence,
                    top5_days_in_window=top_days,
                )
            )
        return tuple(results)
