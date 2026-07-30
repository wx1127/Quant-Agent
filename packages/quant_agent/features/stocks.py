"""Point-in-time stock trend and relative-strength features."""

import statistics
from dataclasses import dataclass
from datetime import date, datetime

from quant_agent.core.time import ensure_aware
from quant_agent.features.core import InsufficientObservationsError, clamp_score


@dataclass(frozen=True, slots=True)
class StockPricePoint:
    instrument_id: str
    trade_date: date
    adjusted_close: float
    is_tradable: bool
    available_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.adjusted_close <= 0:
            raise ValueError("adjusted close must be positive")


@dataclass(frozen=True, slots=True)
class ReferencePricePoint:
    trade_date: date
    close: float
    available_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.close <= 0:
            raise ValueError("reference close must be positive")


@dataclass(frozen=True, slots=True)
class StockTrendFeatures:
    as_of: datetime
    instrument_id: str
    return_20d: float
    return_60d: float
    relative_industry_20d: float | None
    relative_benchmark_20d: float
    ma20_slope: float
    breakout_hold_days: int
    pullback_20d: float
    distance_from_60d_high: float
    trend_quality_score: float
    relative_strength_score: float
    data_version: str
    feature_version: str
    warnings: tuple[str, ...]


def _total_return(values: list[float], observations: int) -> float:
    if len(values) < observations:
        raise InsufficientObservationsError(f"requires {observations} observations")
    return values[-1] / values[-observations] - 1


class StockTrendEngine:
    feature_version = "stock_trend_v1"

    @staticmethod
    def _reference_returns(points: list[ReferencePricePoint], as_of: datetime) -> dict[date, float]:
        available = sorted(
            (
                point
                for point in points
                if point.available_at <= as_of and point.trade_date <= as_of.date()
            ),
            key=lambda point: point.trade_date,
        )
        closes = [point.close for point in available]
        if len(closes) < 21:
            raise InsufficientObservationsError("reference requires 21 observations")
        return {available[-1].trade_date: _total_return(closes, 21)}

    def calculate(
        self,
        prices: list[StockPricePoint],
        benchmark: list[ReferencePricePoint],
        *,
        industry: list[ReferencePricePoint] | None,
        as_of: datetime,
        data_version: str,
    ) -> StockTrendFeatures:
        ensure_aware(as_of)
        available = sorted(
            (
                point
                for point in prices
                if point.available_at <= as_of and point.trade_date <= as_of.date()
            ),
            key=lambda point: point.trade_date,
        )
        if not available:
            raise InsufficientObservationsError("stock trend has no available prices")
        instrument_ids = {point.instrument_id for point in available}
        if len(instrument_ids) != 1:
            raise ValueError("stock trend accepts one instrument at a time")
        tradable = [point for point in available if point.is_tradable]
        if len(tradable) < 61:
            raise InsufficientObservationsError("stock trend requires 61 tradable observations")
        closes = [point.adjusted_close for point in tradable]
        return20 = _total_return(closes, 21)
        return60 = _total_return(closes, 61)
        benchmark20 = next(iter(self._reference_returns(benchmark, as_of).values()))
        warnings: list[str] = []
        industry20: float | None = None
        if industry:
            industry20 = next(iter(self._reference_returns(industry, as_of).values()))
        else:
            warnings.append("industry reference missing")
        ma20_series = [
            statistics.fmean(closes[index - 19 : index + 1])
            for index in range(len(closes) - 10, len(closes))
        ]
        ma20_slope = ma20_series[-1] / ma20_series[0] - 1
        prior_high = max(closes[-61:-1])
        breakout_hold = 0
        for value in reversed(closes):
            if value < prior_high:
                break
            breakout_hold += 1
        high20 = max(closes[-20:])
        pullback = closes[-1] / high20 - 1
        distance_high = closes[-1] / max(closes[-60:]) - 1
        relative_industry = return20 - industry20 if industry20 is not None else None
        relative_benchmark = return20 - benchmark20
        relative_score = statistics.fmean(
            [
                clamp_score(relative_benchmark, -0.15, 0.15),
                clamp_score(relative_industry, -0.15, 0.15)
                if relative_industry is not None
                else 0.0,
            ]
        )
        trend_score = statistics.fmean(
            (
                clamp_score(return20, -0.15, 0.15),
                clamp_score(return60, -0.30, 0.30),
                clamp_score(ma20_slope, -0.08, 0.08),
                clamp_score(pullback, -0.20, 0.0),
                clamp_score(distance_high, -0.20, 0.0),
            )
        )
        return StockTrendFeatures(
            as_of=as_of,
            instrument_id=next(iter(instrument_ids)),
            return_20d=return20,
            return_60d=return60,
            relative_industry_20d=relative_industry,
            relative_benchmark_20d=relative_benchmark,
            ma20_slope=ma20_slope,
            breakout_hold_days=breakout_hold,
            pullback_20d=pullback,
            distance_from_60d_high=distance_high,
            trend_quality_score=trend_score,
            relative_strength_score=relative_score,
            data_version=data_version,
            feature_version=self.feature_version,
            warnings=tuple(warnings),
        )
