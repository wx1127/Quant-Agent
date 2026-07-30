"""Versioned liquidity, capacity and tradeability checks."""

import statistics
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from quant_agent.core.time import ensure_aware
from quant_agent.features.core import RollingCalculator


class LimitStatus(StrEnum):
    NONE = "NONE"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"


@dataclass(frozen=True, slots=True)
class TradeabilityConfig:
    version: str = "tradeability_v1"
    minimum_listing_days: int = 120
    minimum_average_turnover: float = 20_000_000
    maximum_participation_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.minimum_listing_days < 0 or self.minimum_average_turnover < 0:
            raise ValueError("tradeability minimums cannot be negative")
        if not 0 < self.maximum_participation_rate <= 1:
            raise ValueError("participation rate must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class TradeabilityInput:
    instrument_id: str
    as_of: datetime
    listed_on: date
    turnovers_20d: tuple[float, ...]
    turnover_rates_20d: tuple[float, ...]
    is_suspended: bool
    limit_status: LimitStatus
    risk_flag: bool
    delisting_flag: bool
    whitelisted: bool = True

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if any(value < 0 for value in (*self.turnovers_20d, *self.turnover_rates_20d)):
            raise ValueError("turnover observations cannot be negative")


@dataclass(frozen=True, slots=True)
class TradeabilityResult:
    instrument_id: str
    tradable: bool
    reasons: tuple[str, ...]
    listing_days: int
    average_turnover_20d: float
    turnover_percentile: float
    turnover_rate_percentile: float
    estimated_capacity: float
    requested_capital: float
    capacity_sufficient: bool
    data_version: str
    config_version: str


class TradeabilityEngine:
    def __init__(self, config: TradeabilityConfig | None = None) -> None:
        self.config = config or TradeabilityConfig()

    def evaluate(
        self,
        item: TradeabilityInput,
        *,
        requested_capital: float,
        market_turnover_samples: list[float],
        market_turnover_rate_samples: list[float],
        data_version: str,
    ) -> TradeabilityResult:
        if requested_capital < 0:
            raise ValueError("requested capital cannot be negative")
        listing_days = (item.as_of.date() - item.listed_on).days
        average = statistics.fmean(item.turnovers_20d) if item.turnovers_20d else 0.0
        capacity = average * self.config.maximum_participation_rate
        reasons: list[str] = []
        if item.delisting_flag:
            reasons.append("delisting status")
        if item.risk_flag:
            reasons.append("risk flag")
        if listing_days < self.config.minimum_listing_days:
            reasons.append("insufficient listing history")
        if item.is_suspended:
            reasons.append("suspended")
        if item.limit_status is LimitStatus.LIMIT_UP:
            reasons.append("limit-up buyability uncertain")
        if item.limit_status is LimitStatus.LIMIT_DOWN:
            reasons.append("limit-down liquidity risk")
        if average < self.config.minimum_average_turnover:
            reasons.append("insufficient average turnover")
        if requested_capital > capacity:
            reasons.append("requested capital exceeds capacity")
        if not item.whitelisted:
            reasons.append("not in strategy whitelist")
        current_turnover = item.turnovers_20d[-1] if item.turnovers_20d else 0.0
        current_rate = item.turnover_rates_20d[-1] if item.turnover_rates_20d else 0.0
        return TradeabilityResult(
            instrument_id=item.instrument_id,
            tradable=not reasons,
            reasons=tuple(reasons),
            listing_days=listing_days,
            average_turnover_20d=average,
            turnover_percentile=RollingCalculator.percentile_rank(
                market_turnover_samples, current_turnover
            ),
            turnover_rate_percentile=RollingCalculator.percentile_rank(
                market_turnover_rate_samples, current_rate
            ),
            estimated_capacity=capacity,
            requested_capital=requested_capital,
            capacity_sufficient=requested_capital <= capacity,
            data_version=data_version,
            config_version=self.config.version,
        )
