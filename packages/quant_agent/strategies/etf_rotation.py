"""Versioned medium-frequency ETF rotation benchmark."""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from quant_agent.core.time import ensure_aware
from quant_agent.strategies.contracts import TargetPortfolio, TargetWeight


@dataclass(frozen=True, slots=True)
class EtfRotationConfig:
    version: str = "etf_rotation_v1"
    selected_count: int = 3
    minimum_listing_days: int = 120
    minimum_turnover: float = 50_000_000
    maximum_tracking_error: float = 0.03
    minimum_risk_adjusted_score: float = 0.0
    maximum_equity_weight: float = 0.80
    rebalance_weekday: int = 0
    drift_threshold: float = 0.10

    def __post_init__(self) -> None:
        if not 1 <= self.selected_count <= 3:
            raise ValueError("selected_count must be in [1, 3]")
        if not 0 <= self.maximum_equity_weight <= 1:
            raise ValueError("maximum equity weight must be in [0, 1]")
        if not 0 <= self.rebalance_weekday <= 6:
            raise ValueError("rebalance_weekday must be in [0, 6]")
        if not 0 <= self.drift_threshold <= 1:
            raise ValueError("drift_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EtfSignalInput:
    as_of: datetime
    instrument_id: str
    return_20d: float
    return_60d: float
    return_120d: float
    volatility_60d: float
    above_ma60: bool
    listing_days: int
    average_turnover: float
    tracking_error: float
    active: bool
    is_equity: bool
    data_version: str

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)
        if self.volatility_60d < 0:
            raise ValueError("volatility cannot be negative")


class EtfRotationStrategy:
    strategy_id = "ETF_ROTATION"

    def __init__(self, config: EtfRotationConfig | None = None) -> None:
        self.config = config or EtfRotationConfig()

    def generate(self, inputs: list[EtfSignalInput], *, parameter_version: str) -> TargetPortfolio:
        if not inputs:
            raise ValueError("ETF strategy requires a historical universe")
        dates = {item.as_of for item in inputs}
        versions = {item.data_version for item in inputs}
        if len(dates) != 1 or len(versions) != 1:
            raise ValueError("ETF universe must share date and data version")
        scored = []
        for item in inputs:
            eligible = (
                item.active
                and item.listing_days >= self.config.minimum_listing_days
                and item.average_turnover >= self.config.minimum_turnover
                and item.tracking_error <= self.config.maximum_tracking_error
                and item.above_ma60
                and item.volatility_60d > 0
            )
            if not eligible:
                continue
            momentum = 0.5 * item.return_20d + 0.3 * item.return_60d + 0.2 * item.return_120d
            risk_adjusted = momentum / item.volatility_60d
            if risk_adjusted >= self.config.minimum_risk_adjusted_score:
                scored.append((item, risk_adjusted))
        selected = sorted(scored, key=lambda row: (-row[1], row[0].instrument_id))[
            : self.config.selected_count
        ]
        if not selected:
            return TargetPortfolio(
                as_of=inputs[0].as_of,
                strategy_id=self.strategy_id,
                targets=(),
                cash_weight=1.0,
                data_version=inputs[0].data_version,
                strategy_version=self.config.version,
                parameter_version=parameter_version,
                warnings=("no eligible ETF; portfolio reduced to cash",),
            )
        equity = [row for row in selected if row[0].is_equity]
        defensive = [row for row in selected if not row[0].is_equity]
        equity_budget = self.config.maximum_equity_weight if equity else 0.0
        defensive_budget = 1.0 - equity_budget if defensive else 0.0
        if not defensive:
            defensive_budget = 0.0
        budgets: dict[str, float] = {}
        for group, budget in ((equity, equity_budget), (defensive, defensive_budget)):
            if not group or budget == 0:
                continue
            inverse_vol = [1 / row[0].volatility_60d for row in group]
            denominator = sum(inverse_vol)
            for row, value in zip(group, inverse_vol, strict=True):
                budgets[row[0].instrument_id] = budget * value / denominator
        invested = sum(budgets.values())
        targets = tuple(
            TargetWeight(
                instrument_id=instrument_id,
                weight=weight,
                reason="positive volatility-adjusted momentum and trend filter",
            )
            for instrument_id, weight in sorted(budgets.items())
        )
        warnings: tuple[str, ...] = ()
        if statistics.fmean(row[1] for row in selected) < 0.25:
            warnings = ("selected ETF scores are weak",)
        return TargetPortfolio(
            as_of=inputs[0].as_of,
            strategy_id=self.strategy_id,
            targets=targets,
            cash_weight=1.0 - invested,
            data_version=inputs[0].data_version,
            strategy_version=self.config.version,
            parameter_version=parameter_version,
            warnings=warnings,
        )

    def should_rebalance(
        self,
        target: TargetPortfolio,
        *,
        current_weights: dict[str, float],
        last_rebalanced_at: datetime | None,
    ) -> bool:
        """Apply weekly review plus threshold-triggered rebalance."""

        target_weights = {item.instrument_id: item.weight for item in target.targets}
        drift = sum(
            abs(target_weights.get(key, 0.0) - current_weights.get(key, 0.0))
            for key in set(target_weights) | set(current_weights)
        )
        threshold_triggered = drift >= self.config.drift_threshold
        weekly_due = target.as_of.weekday() == self.config.rebalance_weekday and (
            last_rebalanced_at is None or target.as_of - last_rebalanced_at >= timedelta(days=7)
        )
        return threshold_triggered or weekly_due
