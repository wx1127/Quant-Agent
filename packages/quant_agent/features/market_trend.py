"""Multi-index trend, moving-average and slope features."""

import statistics
from dataclasses import dataclass
from datetime import date, datetime

from quant_agent.core.time import ensure_aware
from quant_agent.features.core import (
    FeatureMetadata,
    InsufficientObservationsError,
    RollingCalculator,
    TimeSeriesPoint,
    clamp_score,
)


@dataclass(frozen=True, slots=True)
class IndexBar:
    """Daily close for one broad-market index."""

    index_id: str
    trade_date: date
    close: float
    available_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.close <= 0:
            raise ValueError("index close must be positive")


@dataclass(frozen=True, slots=True)
class IndexTrendFeatures:
    """Trend evidence for one index."""

    index_id: str
    return_20d: float
    return_60d: float
    return_120d: float
    distance_ma20: float
    distance_ma60: float
    slope_20d: float
    score: float


@dataclass(frozen=True, slots=True)
class MarketTrendResult:
    """Aggregated broad-market trend result."""

    as_of: datetime
    data_version: str
    feature_version: str
    score: float
    per_index: tuple[IndexTrendFeatures, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)


class MarketTrendEngine:
    """Compute transparent trend scores over 20/60/120 trading-day windows."""

    feature_version = "market_trend_v1"

    def __init__(self, *, min_indices: int = 2) -> None:
        if min_indices < 1:
            raise ValueError("min_indices must be positive")
        self._min_indices = min_indices

    @staticmethod
    def _metadata(
        name: str,
        data_version: str,
        window: int,
        min_observations: int,
    ) -> FeatureMetadata:
        return FeatureMetadata(
            name=name,
            version=MarketTrendEngine.feature_version,
            dataset_version=data_version,
            window=window,
            min_observations=min_observations,
        )

    def calculate(
        self,
        bars: list[IndexBar],
        *,
        as_of: datetime,
        data_version: str,
    ) -> MarketTrendResult:
        """Calculate index-level evidence and a median aggregate."""

        ensure_aware(as_of)
        grouped: dict[str, list[TimeSeriesPoint]] = {}
        for bar in bars:
            grouped.setdefault(bar.index_id, []).append(
                TimeSeriesPoint(bar.trade_date, bar.available_at, bar.close)
            )
        results: list[IndexTrendFeatures] = []
        warnings: list[str] = []
        for index_id in sorted(grouped):
            points = grouped[index_id]
            try:
                return_20 = RollingCalculator.total_return(
                    points,
                    as_of=as_of,
                    metadata=self._metadata(
                        f"{index_id}.return_20d",
                        data_version,
                        21,
                        21,
                    ),
                ).value
                return_60 = RollingCalculator.total_return(
                    points,
                    as_of=as_of,
                    metadata=self._metadata(
                        f"{index_id}.return_60d",
                        data_version,
                        61,
                        61,
                    ),
                ).value
                return_120 = RollingCalculator.total_return(
                    points,
                    as_of=as_of,
                    metadata=self._metadata(
                        f"{index_id}.return_120d",
                        data_version,
                        121,
                        121,
                    ),
                ).value
                ma20 = RollingCalculator.mean(
                    points,
                    as_of=as_of,
                    metadata=self._metadata(
                        f"{index_id}.ma20",
                        data_version,
                        20,
                        20,
                    ),
                ).value
                ma60 = RollingCalculator.mean(
                    points,
                    as_of=as_of,
                    metadata=self._metadata(
                        f"{index_id}.ma60",
                        data_version,
                        60,
                        60,
                    ),
                ).value
                slope20 = RollingCalculator.slope(
                    points,
                    as_of=as_of,
                    metadata=self._metadata(
                        f"{index_id}.slope20",
                        data_version,
                        20,
                        20,
                    ),
                ).value
            except InsufficientObservationsError as error:
                warnings.append(f"{index_id}: {error}")
                continue
            latest = max(
                (
                    point
                    for point in points
                    if point.available_at <= as_of and point.observed_on <= as_of.date()
                ),
                key=lambda point: (point.observed_on, point.available_at),
            )
            if latest.value is None:
                raise ValueError(f"{index_id}: latest close is missing")
            latest_close = latest.value
            distance_ma20 = latest_close / ma20 - 1.0
            distance_ma60 = latest_close / ma60 - 1.0
            score = (
                0.20 * clamp_score(return_20, -0.10, 0.10)
                + 0.20 * clamp_score(return_60, -0.20, 0.20)
                + 0.20 * clamp_score(return_120, -0.30, 0.30)
                + 0.15 * clamp_score(distance_ma20, -0.08, 0.08)
                + 0.15 * clamp_score(distance_ma60, -0.15, 0.15)
                + 0.10 * clamp_score(slope20, -0.01, 0.01)
            )
            results.append(
                IndexTrendFeatures(
                    index_id=index_id,
                    return_20d=return_20,
                    return_60d=return_60,
                    return_120d=return_120,
                    distance_ma20=distance_ma20,
                    distance_ma60=distance_ma60,
                    slope_20d=slope20,
                    score=score,
                )
            )
        if len(results) < self._min_indices:
            raise InsufficientObservationsError(
                f"market trend requires {self._min_indices} complete indices; got {len(results)}"
            )
        if len(results) < len(grouped):
            warnings.append("market trend degraded because one or more indices are incomplete")
        return MarketTrendResult(
            as_of=as_of,
            data_version=data_version,
            feature_version=self.feature_version,
            score=statistics.median(item.score for item in results),
            per_index=tuple(results),
            warnings=tuple(warnings),
        )
