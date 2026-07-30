"""Cross-sectional breadth, turnover and downside-risk features."""

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise

from quant_agent.core.time import ensure_aware
from quant_agent.features.core import InsufficientObservationsError, RollingCalculator


@dataclass(frozen=True, slots=True)
class StockBar:
    """Historical daily stock observation for breadth calculations."""

    instrument_id: str
    trade_date: date
    close: float
    turnover: float
    is_tradable: bool
    available_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.close <= 0 or self.turnover < 0:
            raise ValueError("stock close must be positive and turnover non-negative")


@dataclass(frozen=True, slots=True)
class MarketBreadthResult:
    """Market breadth and component scores used by regime classification."""

    as_of: datetime
    data_version: str
    feature_version: str
    eligible_count: int
    advancing_count: int
    declining_count: int
    unchanged_count: int
    advancing_ratio: float
    above_ma20_ratio: float
    above_ma60_ratio: float
    new_high_20_ratio: float
    new_low_20_ratio: float
    new_high_60_ratio: float
    new_low_60_ratio: float
    turnover: float
    turnover_percentile_20d: float
    median_daily_return: float
    downside_volatility_20d: float
    breadth_score: float
    turnover_score: float
    new_high_low_score: float
    diffusion_score: float
    risk_score: float
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)


class MarketBreadthEngine:
    """Compute breadth from stocks tradable at each historical date."""

    feature_version = "market_breadth_v1"

    @staticmethod
    def _daily_returns(
        by_instrument: dict[str, list[StockBar]],
        as_of: datetime,
    ) -> dict[date, list[float]]:
        daily: dict[date, list[float]] = {}
        for bars in by_instrument.values():
            eligible = [
                bar
                for bar in sorted(bars, key=lambda item: item.trade_date)
                if bar.available_at <= as_of and bar.trade_date <= as_of.date()
            ]
            for previous, current in pairwise(eligible):
                if current.is_tradable and previous.close > 0:
                    daily.setdefault(current.trade_date, []).append(
                        current.close / previous.close - 1.0
                    )
        return daily

    def calculate(
        self,
        bars: list[StockBar],
        *,
        as_of: datetime,
        data_version: str,
    ) -> MarketBreadthResult:
        """Calculate current breadth and trailing risk statistics."""

        ensure_aware(as_of)
        eligible_bars = [
            bar for bar in bars if bar.available_at <= as_of and bar.trade_date <= as_of.date()
        ]
        if not eligible_bars:
            raise InsufficientObservationsError("market breadth has no available bars")
        latest_date = max(bar.trade_date for bar in eligible_bars)
        by_instrument: dict[str, list[StockBar]] = {}
        for bar in eligible_bars:
            by_instrument.setdefault(bar.instrument_id, []).append(bar)
        current_bars: list[tuple[StockBar, list[StockBar]]] = []
        warnings: list[str] = []
        for _instrument_id, history in sorted(by_instrument.items()):
            ordered = sorted(history, key=lambda item: item.trade_date)
            current = next(
                (item for item in reversed(ordered) if item.trade_date == latest_date),
                None,
            )
            if current is None or not current.is_tradable:
                continue
            current_bars.append((current, ordered))
        if not current_bars:
            raise InsufficientObservationsError(
                "market breadth has no tradable stocks on the latest date"
            )

        advancing = declining = unchanged = 0
        above20 = above60 = high20 = low20 = high60 = low60 = 0
        denominator20 = denominator60 = 0
        current_returns: list[float] = []
        for current, history in current_bars:
            prior = [item for item in history if item.trade_date < current.trade_date]
            if prior:
                daily_return = current.close / prior[-1].close - 1.0
                current_returns.append(daily_return)
                if daily_return > 0:
                    advancing += 1
                elif daily_return < 0:
                    declining += 1
                else:
                    unchanged += 1
            else:
                unchanged += 1
                warnings.append(f"{current.instrument_id}: no previous close")
            closes = [item.close for item in history if item.trade_date <= latest_date]
            if len(closes) >= 20:
                window20 = closes[-20:]
                denominator20 += 1
                above20 += current.close > statistics.fmean(window20)
                high20 += current.close >= max(window20)
                low20 += current.close <= min(window20)
            if len(closes) >= 60:
                window60 = closes[-60:]
                denominator60 += 1
                above60 += current.close > statistics.fmean(window60)
                high60 += current.close >= max(window60)
                low60 += current.close <= min(window60)
        eligible_count = len(current_bars)
        advancing_ratio = advancing / eligible_count
        above_ma20_ratio = above20 / denominator20 if denominator20 else 0.0
        above_ma60_ratio = above60 / denominator60 if denominator60 else 0.0
        new_high_20_ratio = high20 / denominator20 if denominator20 else 0.0
        new_low_20_ratio = low20 / denominator20 if denominator20 else 0.0
        new_high_60_ratio = high60 / denominator60 if denominator60 else 0.0
        new_low_60_ratio = low60 / denominator60 if denominator60 else 0.0
        if denominator20 < eligible_count:
            warnings.append("some stocks lack 20-day breadth history")
        if denominator60 < eligible_count:
            warnings.append("some stocks lack 60-day breadth history")

        turnover_by_date: dict[date, float] = {}
        for bar in eligible_bars:
            if bar.is_tradable:
                turnover_by_date[bar.trade_date] = (
                    turnover_by_date.get(bar.trade_date, 0.0) + bar.turnover
                )
        recent_turnover_dates = sorted(turnover_by_date)[-20:]
        turnover_values = [turnover_by_date[item] for item in recent_turnover_dates]
        current_turnover = turnover_by_date.get(latest_date, 0.0)
        turnover_percentile = RollingCalculator.percentile_rank(
            turnover_values,
            current_turnover,
        )

        daily_returns = self._daily_returns(by_instrument, as_of)
        market_returns = [
            statistics.median(daily_returns[item])
            for item in sorted(daily_returns)[-20:]
            if daily_returns[item]
        ]
        downside = [min(value, 0.0) for value in market_returns]
        downside_volatility = (
            math.sqrt(statistics.fmean(value * value for value in downside)) * math.sqrt(252)
            if downside
            else 0.0
        )
        median_return = statistics.median(current_returns) if current_returns else 0.0

        breadth_score = (
            0.4 * advancing_ratio + 0.35 * above_ma20_ratio + 0.25 * above_ma60_ratio
        ) * 100
        new_high_low_score = max(
            0.0,
            min(
                100.0,
                50
                + 35 * (new_high_20_ratio - new_low_20_ratio)
                + 15 * (new_high_60_ratio - new_low_60_ratio),
            ),
        )
        diffusion_score = (
            0.5 * advancing_ratio + 0.3 * above_ma20_ratio + 0.2 * (1.0 - new_low_20_ratio)
        ) * 100
        risk_score = max(
            0.0,
            min(
                100.0,
                100
                - 300 * downside_volatility
                - 40 * new_low_20_ratio
                - 20 * (declining / eligible_count),
            ),
        )
        return MarketBreadthResult(
            as_of=as_of,
            data_version=data_version,
            feature_version=self.feature_version,
            eligible_count=eligible_count,
            advancing_count=advancing,
            declining_count=declining,
            unchanged_count=unchanged,
            advancing_ratio=advancing_ratio,
            above_ma20_ratio=above_ma20_ratio,
            above_ma60_ratio=above_ma60_ratio,
            new_high_20_ratio=new_high_20_ratio,
            new_low_20_ratio=new_low_20_ratio,
            new_high_60_ratio=new_high_60_ratio,
            new_low_60_ratio=new_low_60_ratio,
            turnover=current_turnover,
            turnover_percentile_20d=turnover_percentile,
            median_daily_return=median_return,
            downside_volatility_20d=downside_volatility,
            breadth_score=breadth_score,
            turnover_score=turnover_percentile * 100,
            new_high_low_score=new_high_low_score,
            diffusion_score=diffusion_score,
            risk_score=risk_score,
            warnings=tuple(warnings),
        )
