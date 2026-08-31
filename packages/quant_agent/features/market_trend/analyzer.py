"""Moving-average and trend aggregation for required market indices."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from decimal import Decimal

from quant_agent.core.time import ensure_aware
from quant_agent.features.core import (
    FeatureDefinition,
    FeatureObservation,
    FeatureRequest,
    FeatureStatus,
    RollingFeatureEngine,
)
from quant_agent.features.market_trend.contracts import (
    IndexTrendResult,
    InsufficientMarketData,
    MarketTrendConfig,
    MarketTrendSnapshot,
    MissingIndexPolicy,
    WindowTrendMetrics,
)


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal(0)) / len(values)


def _clamp_score(value: Decimal) -> Decimal:
    return max(Decimal(-100), min(Decimal(100), value))


class MarketTrendAnalyzer:
    """Compute 20/60/120-style point-in-time trend features without future prices."""

    def __init__(
        self,
        config: MarketTrendConfig | None = None,
        *,
        engine: RollingFeatureEngine | None = None,
    ) -> None:
        self._config = config or MarketTrendConfig()
        self._engine = engine or RollingFeatureEngine()

    def analyze(
        self,
        *,
        session_date: date,
        required_indices: tuple[str, ...],
        observations: Mapping[str, Iterable[FeatureObservation]],
        as_of: datetime,
        data_version: str,
    ) -> MarketTrendSnapshot:
        """Return an aggregate or fail/degrade according to the explicit missing policy."""

        ensure_aware(as_of)
        if session_date > as_of.date():
            raise ValueError("session_date cannot be after as_of")
        if not required_indices or any(not value.strip() for value in required_indices):
            raise ValueError("required_indices must contain non-empty identifiers")
        if len(set(required_indices)) != len(required_indices):
            raise ValueError("required_indices cannot contain duplicates")
        normalized_indices = tuple(sorted(required_indices))
        results: list[IndexTrendResult] = []
        missing: list[str] = []
        for index_id in normalized_indices:
            result = self._analyze_index(
                index_id=index_id,
                observations=observations.get(index_id, ()),
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
            )
            if result is None:
                missing.append(index_id)
            else:
                results.append(result)
        if missing and self._config.missing_index_policy is MissingIndexPolicy.FAIL:
            raise InsufficientMarketData(
                f"required index history unavailable: {', '.join(missing)}"
            )
        if not results:
            raise InsufficientMarketData("no required index has sufficient trend history")
        alignment_windows = {item.observation_dates for item in results}
        if len(alignment_windows) != 1:
            raise InsufficientMarketData("required index trend windows are not session-aligned")

        aggregate = sum((item.score for item in results), Decimal(0)) / len(results)
        cache_key = self._snapshot_cache_key(
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            indices=tuple(results),
            missing_indices=tuple(missing),
        )
        return MarketTrendSnapshot(
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            indices=tuple(results),
            missing_indices=tuple(missing),
            aggregate_score=aggregate,
            cache_key=cache_key,
        )

    def _analyze_index(
        self,
        *,
        index_id: str,
        observations: Iterable[FeatureObservation],
        session_date: date,
        as_of: datetime,
        data_version: str,
    ) -> IndexTrendResult | None:
        values = tuple(item for item in observations if item.observed_at.date() <= session_date)
        request = FeatureRequest(entity_id=index_id, as_of=as_of, data_version=data_version)
        metrics: list[WindowTrendMetrics] = []
        for window in self._config.windows:
            history_size = window + self._config.slope_lookback
            definition = FeatureDefinition.create(
                name=f"market_trend_{window}",
                version=self._config.version,
                calculator_id=f"market-trend-window-score-v1:{window}",
                window=history_size,
                min_observations=history_size,
                parameters={
                    "score_scale": str(self._config.score_scale),
                    "slope_lookback": self._config.slope_lookback,
                    "window": window,
                },
            )

            def score_calculator(selected: tuple[Decimal, ...], horizon: int = window) -> Decimal:
                return self._window_score(selected, horizon)[-1]

            feature = self._engine.evaluate(
                definition=definition,
                request=request,
                observations=values,
                calculator=score_calculator,
            )
            if feature.status is FeatureStatus.INSUFFICIENT_DATA:
                return None
            selected_observations = self._engine.select_window(
                observations=values,
                as_of=as_of,
                window=history_size,
            )
            selected = tuple(
                observation.value
                for observation in selected_observations
                if observation.value is not None
            )
            if len(selected) != history_size:  # pragma: no cover - mirrors engine status
                raise AssertionError("ready trend feature has an incomplete selected window")
            if selected_observations[-1].observed_at.date() != session_date:
                return None
            latest, average, period_return, position, slope, score = self._window_score(
                selected,
                window,
            )
            metrics.append(
                WindowTrendMetrics(
                    window=window,
                    latest_close=latest,
                    moving_average=average,
                    period_return=period_return,
                    position_vs_average=position,
                    average_slope=slope,
                    score=score,
                    input_hash=feature.input_hash,
                )
            )
        weight_total = sum(self._config.window_weights, Decimal(0))
        index_score = (
            sum(
                (
                    metric.score * weight
                    for metric, weight in zip(metrics, self._config.window_weights, strict=True)
                ),
                Decimal(0),
            )
            / weight_total
        )
        alignment_size = max(self._config.windows) + self._config.slope_lookback
        alignment_observations = self._engine.select_window(
            observations=values,
            as_of=as_of,
            window=alignment_size,
        )
        return IndexTrendResult(
            index_id=index_id,
            windows=tuple(metrics),
            observation_dates=tuple(item.observed_at.date() for item in alignment_observations),
            score=index_score,
        )

    def _window_score(
        self,
        values: tuple[Decimal, ...],
        window: int,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
        lag = self._config.slope_lookback
        current_values = values[-window:]
        prior_values = values[-(window + lag) : -lag]
        latest = current_values[-1]
        first = current_values[0]
        current_average = _mean(current_values)
        prior_average = _mean(prior_values)
        if min(latest, first, current_average, prior_average) <= 0:
            raise ValueError("market trend close prices must be positive")
        period_return = latest / first - 1
        position = latest / current_average - 1
        slope = current_average / prior_average - 1
        raw = (
            period_return * self._config.return_weight
            + position * self._config.position_weight
            + slope * self._config.slope_weight
        )
        score = _clamp_score(raw * self._config.score_scale)
        return latest, current_average, period_return, position, slope, score

    def _snapshot_cache_key(
        self,
        *,
        session_date: date,
        as_of: datetime,
        data_version: str,
        indices: tuple[IndexTrendResult, ...],
        missing_indices: tuple[str, ...],
    ) -> str:
        payload = {
            "as_of": as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self._config.config_hash,
            "data_version": data_version,
            "indices": [
                {
                    "index_id": item.index_id,
                    "input_hashes": [window.input_hash for window in item.windows],
                    "score": str(item.score),
                }
                for item in indices
            ],
            "missing_indices": list(missing_indices),
            "session_date": session_date.isoformat(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
