"""Rule-based mainline leader target portfolio generation."""

from dataclasses import dataclass
from datetime import datetime

from quant_agent.core.time import ensure_aware
from quant_agent.leaders.candidates import CandidateResult, CandidateTier
from quant_agent.regime.models import MarketRegime
from quant_agent.strategies.contracts import TargetPortfolio, TargetWeight
from quant_agent.themes.scoring import ThemeState


@dataclass(frozen=True, slots=True)
class MainlineLeaderConfig:
    version: str = "mainline_leader_v1"
    maximum_positions: int = 5
    maximum_single_weight: float = 0.20
    maximum_total_weight: float = 0.80
    maximum_turnover: float = 0.50

    def __post_init__(self) -> None:
        if self.maximum_positions < 1:
            raise ValueError("maximum_positions must be positive")
        if not 0 < self.maximum_single_weight <= 1:
            raise ValueError("maximum single weight must be in (0, 1]")
        if not 0 <= self.maximum_total_weight <= 1:
            raise ValueError("maximum total weight must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MainlineCandidateInput:
    candidate: CandidateResult
    theme_state: ThemeState
    invalidated: bool = False


class MainlineLeaderStrategy:
    strategy_id = "MAINLINE_LEADER"

    def __init__(self, config: MainlineLeaderConfig | None = None) -> None:
        self.config = config or MainlineLeaderConfig()

    def generate(
        self,
        inputs: list[MainlineCandidateInput],
        *,
        as_of: datetime,
        market_regime: MarketRegime,
        current_weights: dict[str, float],
        parameter_version: str,
        data_version: str,
    ) -> TargetPortfolio:
        ensure_aware(as_of)
        regime_budget = {
            MarketRegime.UPTREND: self.config.maximum_total_weight,
            MarketRegime.RANGE_STRONG: min(0.60, self.config.maximum_total_weight),
            MarketRegime.DIVERGENT: min(0.40, self.config.maximum_total_weight),
            MarketRegime.BOTTOM_RECOVERY: min(0.30, self.config.maximum_total_weight),
            MarketRegime.DOWNTREND: 0.0,
        }[market_regime]
        eligible = [
            item.candidate
            for item in inputs
            if item.theme_state is ThemeState.CONFIRMED
            and not item.invalidated
            and item.candidate.tradable
            and item.candidate.tier in {CandidateTier.A, CandidateTier.B}
            and item.candidate.leader_type in {"TREND", "CAPACITY"}
        ]
        selected = sorted(eligible, key=lambda item: (-item.candidate_score, item.instrument_id))[
            : self.config.maximum_positions
        ]
        desired = (
            min(self.config.maximum_single_weight, regime_budget / len(selected))
            if selected
            else 0.0
        )
        targets = {item.instrument_id: desired for item in selected}
        turnover = sum(
            abs(targets.get(key, 0.0) - current_weights.get(key, 0.0))
            for key in set(targets) | set(current_weights)
        )
        warnings: list[str] = []
        if turnover > self.config.maximum_turnover and turnover > 0:
            scale = self.config.maximum_turnover / turnover
            targets = {
                key: current_weights.get(key, 0.0)
                + (target - current_weights.get(key, 0.0)) * scale
                for key, target in targets.items()
            }
            targets.update(
                {
                    key: current_weights[key] * (1 - scale)
                    for key in current_weights
                    if key not in targets and current_weights[key] * (1 - scale) > 0
                }
            )
            warnings.append("target transition scaled by turnover constraint")
        invested = sum(targets.values())
        return TargetPortfolio(
            as_of=as_of,
            strategy_id=self.strategy_id,
            targets=tuple(
                TargetWeight(key, value, "confirmed mainline tradable leader")
                for key, value in sorted(targets.items())
                if value > 0
            ),
            cash_weight=1.0 - invested,
            data_version=data_version,
            strategy_version=self.config.version,
            parameter_version=parameter_version,
            warnings=tuple(warnings),
        )
