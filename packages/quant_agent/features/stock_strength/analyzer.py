"""Point-in-time stock trend, relative strength, and score attribution."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import pairwise

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.features.core import canonical_decimal
from quant_agent.features.core.contracts import stable_feature_hash
from quant_agent.features.stock_strength.contracts import (
    HorizonStrengthMetrics,
    IndustryMembershipError,
    InsufficientHistoryPolicy,
    InsufficientStockData,
    MissingIndustryPolicy,
    PointInTimeStockIndustryMembership,
    ReferenceBarObservation,
    ScoreContribution,
    StockBarObservation,
    StockStrengthConfig,
    StockStrengthSnapshot,
    StockStrengthStatus,
    SuspendedStockError,
    SuspensionPolicy,
    TrendQualityMetrics,
    VolumePriceConfirmation,
)


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal(0)) / len(values)


def _compound(values: tuple[Decimal, ...]) -> Decimal:
    result = Decimal(1)
    for value in values:
        result *= 1 + value
    return result - 1


def _clamp(value: Decimal, low: Decimal = Decimal(-100), high: Decimal = Decimal(100)) -> Decimal:
    return max(low, min(high, value))


class StockStrengthAnalyzer:
    """Calculate one stock snapshot without current-membership or future-data leakage."""

    def __init__(self, config: StockStrengthConfig | None = None) -> None:
        self._config = config or StockStrengthConfig()

    def analyze(
        self,
        *,
        instrument_id: str,
        benchmark_id: str,
        session_date: date,
        as_of: datetime,
        data_version: str,
        classification_version: str,
        industry_level: int,
        memberships: Iterable[PointInTimeStockIndustryMembership],
        stock_observations: Iterable[StockBarObservation],
        reference_observations: Iterable[ReferenceBarObservation],
    ) -> StockStrengthSnapshot:
        """Return an exact snapshot or follow the configured fail/degrade policy."""

        self._validate_request(
            instrument_id=instrument_id,
            benchmark_id=benchmark_id,
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
        )
        stocks = self._select_stock_revisions(
            stock_observations,
            instrument_id=instrument_id,
            session_date=session_date,
            as_of=as_of,
        )
        references = self._select_reference_revisions(
            reference_observations,
            session_date=session_date,
            as_of=as_of,
        )
        selected_memberships = self._select_membership_revisions(
            memberships,
            instrument_id=instrument_id,
            session_date=session_date,
            as_of=as_of,
            classification_version=classification_version,
            industry_level=industry_level,
        )
        provisional_hash = self._input_hash(
            stocks=stocks[-self._config.required_stock_observations :],
            references=references,
            memberships=selected_memberships,
        )
        current_membership = self._membership_on(selected_memberships, session_date)
        current_industry_id = (
            current_membership.membership.industry_id if current_membership else None
        )
        stock_by_date = {item.trade_date: item for item in stocks}
        current = stock_by_date.get(session_date)
        if current is None:
            return self._insufficient(
                reason="stock has no point-in-time bar for session_date",
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stocks),
                input_hash=provisional_hash,
            )
        if current.is_suspended:
            if self._config.suspension_policy is SuspensionPolicy.FAIL:
                raise SuspendedStockError("stock is suspended on session_date")
            return self._unavailable_snapshot(
                status=StockStrengthStatus.SUSPENDED,
                reason="stock is suspended on session_date; no score was produced",
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stocks),
                input_hash=provisional_hash,
            )

        required = self._config.required_stock_observations
        if len(stocks) < required:
            return self._insufficient(
                reason=f"stock history requires {required} observations; received {len(stocks)}",
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stocks),
                input_hash=provisional_hash,
            )
        stock_window = stocks[-required:]
        try:
            self._validate_historical_suspensions(stock_window)
        except InsufficientStockData as error:
            return self._insufficient(
                reason=str(error),
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stock_window),
                input_hash=provisional_hash,
            )

        maximum_horizon = max(self._config.horizons)
        benchmark_rows = tuple(item for item in references if item.reference_id == benchmark_id)
        if len(benchmark_rows) < maximum_horizon + 1 or not benchmark_rows:
            return self._insufficient(
                reason="benchmark history cannot cover all configured horizons",
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stock_window),
                input_hash=provisional_hash,
            )
        benchmark_used = benchmark_rows[-(maximum_horizon + 1) :]
        if benchmark_used[-1].trade_date != session_date:
            return self._insufficient(
                reason="benchmark has no point-in-time close for session_date",
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stock_window),
                input_hash=provisional_hash,
            )
        canonical_dates = tuple(item.trade_date for item in benchmark_used)
        expected_stock_dates = tuple(item.trade_date for item in stocks[-(maximum_horizon + 1) :])
        if canonical_dates != expected_stock_dates or any(
            value not in stock_by_date for value in canonical_dates
        ):
            return self._insufficient(
                reason="stock history is not aligned to every benchmark session",
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stock_window),
                input_hash=provisional_hash,
            )

        canonical_stocks = tuple(stock_by_date[value] for value in canonical_dates)
        stock_daily = tuple(
            current_row.adjusted_close / prior.adjusted_close - 1
            for prior, current_row in pairwise(canonical_stocks)
        )
        benchmark_daily = tuple(
            current_row.close / prior.close - 1 for prior, current_row in pairwise(benchmark_used)
        )
        reference_by_key = {(item.reference_id, item.trade_date): item for item in references}
        industry_daily: list[Decimal | None] = []
        industry_ids: list[str | None] = []
        used_memberships: list[PointInTimeStockIndustryMembership] = []
        used_industry_references: list[ReferenceBarObservation] = []
        degradation_reasons: list[str] = []
        for prior_date, current_date in pairwise(canonical_dates):
            assignment = self._membership_on(selected_memberships, current_date)
            if assignment is None:
                industry_daily.append(None)
                industry_ids.append(None)
                degradation_reasons.append(
                    f"industry membership missing on {current_date.isoformat()}"
                )
                continue
            industry_id = assignment.membership.industry_id
            prior_reference = reference_by_key.get((industry_id, prior_date))
            current_reference = reference_by_key.get((industry_id, current_date))
            if prior_reference is None or current_reference is None:
                industry_daily.append(None)
                industry_ids.append(industry_id)
                degradation_reasons.append(
                    f"industry reference {industry_id} is not session-aligned"
                )
                continue
            industry_daily.append(current_reference.close / prior_reference.close - 1)
            industry_ids.append(industry_id)
            used_memberships.append(assignment)
            used_industry_references.extend((prior_reference, current_reference))

        if (
            degradation_reasons
            and self._config.missing_industry_policy is MissingIndustryPolicy.FAIL
        ):
            raise IndustryMembershipError("; ".join(sorted(set(degradation_reasons))))
        horizons = self._horizon_metrics(
            stock_daily=stock_daily,
            benchmark_daily=benchmark_daily,
            industry_daily=tuple(industry_daily),
            industry_ids=tuple(industry_ids),
        )
        trend = self._trend_quality(stock_window)
        try:
            volume = self._volume_confirmation(stock_window)
        except InsufficientStockData as error:
            return self._insufficient(
                reason=str(error),
                instrument_id=instrument_id,
                benchmark_id=benchmark_id,
                session_date=session_date,
                as_of=as_of,
                data_version=data_version,
                classification_version=classification_version,
                industry_level=industry_level,
                current_industry_id=current_industry_id,
                selected_dates=tuple(item.trade_date for item in stock_window),
                input_hash=provisional_hash,
            )
        contributions = self._score_contributions(horizons, trend, volume)
        score = sum((item.contribution for item in contributions), Decimal(0))

        used_membership_tuple = self._unique_memberships(
            [*used_memberships, *([current_membership] if current_membership else [])]
        )
        used_reference_tuple = self._unique_references([*benchmark_used, *used_industry_references])
        input_hash = self._input_hash(
            stocks=stock_window,
            references=used_reference_tuple,
            memberships=used_membership_tuple,
        )
        cache_key = self._cache_key(
            instrument_id=instrument_id,
            benchmark_id=benchmark_id,
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
            input_hash=input_hash,
        )
        status = StockStrengthStatus.DEGRADED if degradation_reasons else StockStrengthStatus.READY
        reason = "; ".join(sorted(set(degradation_reasons))) if degradation_reasons else None
        result_hash = self._result_hash(
            cache_key=cache_key,
            status=status,
            horizons=horizons,
            trend=trend,
            volume=volume,
            contributions=contributions,
            score=score,
            reason=reason,
        )
        return StockStrengthSnapshot(
            instrument_id=instrument_id,
            benchmark_id=benchmark_id,
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
            current_industry_id=current_industry_id,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            status=status,
            horizons=horizons,
            trend_quality=trend,
            volume_confirmation=volume,
            contributions=contributions,
            score=score,
            reason=reason,
            selected_dates=tuple(item.trade_date for item in stock_window),
            input_hash=input_hash,
            cache_key=cache_key,
            result_hash=result_hash,
        )

    @staticmethod
    def _validate_request(
        *,
        instrument_id: str,
        benchmark_id: str,
        session_date: date,
        as_of: datetime,
        data_version: str,
        classification_version: str,
        industry_level: int,
    ) -> None:
        ensure_aware(as_of)
        if session_date > as_of.astimezone(SHANGHAI_TZ).date():
            raise ValueError("session_date cannot be after as_of")
        identifiers = (instrument_id, benchmark_id, data_version, classification_version)
        if any(not value.strip() for value in identifiers):
            raise ValueError("stock strength identifiers and versions must be non-empty")
        if industry_level not in (1, 2, 3):
            raise ValueError("industry_level must be 1, 2, or 3")

    @staticmethod
    def _select_stock_revisions(
        observations: Iterable[StockBarObservation],
        *,
        instrument_id: str,
        session_date: date,
        as_of: datetime,
    ) -> tuple[StockBarObservation, ...]:
        selected: dict[date, StockBarObservation] = {}
        for observation in observations:
            if observation.instrument_id != instrument_id:
                raise ValueError("stock observations cannot mix instrument_id values")
            if observation.trade_date > session_date:
                continue
            if observation.observed_at > as_of or observation.available_at > as_of:
                continue
            current = selected.get(observation.trade_date)
            if current is None or observation.available_at > current.available_at:
                selected[observation.trade_date] = observation
            elif observation.available_at == current.available_at and observation != current:
                raise ValueError("ambiguous stock bar revisions")
        return tuple(sorted(selected.values(), key=lambda item: item.trade_date))

    @staticmethod
    def _select_reference_revisions(
        observations: Iterable[ReferenceBarObservation],
        *,
        session_date: date,
        as_of: datetime,
    ) -> tuple[ReferenceBarObservation, ...]:
        selected: dict[tuple[str, date], ReferenceBarObservation] = {}
        for observation in observations:
            if observation.trade_date > session_date:
                continue
            if observation.observed_at > as_of or observation.available_at > as_of:
                continue
            key = (observation.reference_id, observation.trade_date)
            current = selected.get(key)
            if current is None or observation.available_at > current.available_at:
                selected[key] = observation
            elif observation.available_at == current.available_at and observation != current:
                raise ValueError("ambiguous reference bar revisions")
        return tuple(
            sorted(selected.values(), key=lambda item: (item.reference_id, item.trade_date))
        )

    @staticmethod
    def _select_membership_revisions(
        memberships: Iterable[PointInTimeStockIndustryMembership],
        *,
        instrument_id: str,
        session_date: date,
        as_of: datetime,
        classification_version: str,
        industry_level: int,
    ) -> tuple[PointInTimeStockIndustryMembership, ...]:
        eligible = tuple(
            item
            for item in memberships
            if item.membership.instrument_id == instrument_id
            and item.membership.version == classification_version
            and item.membership.effective_from <= session_date
            and item.available_at <= as_of
        )
        levels = {item.level for item in eligible}
        if levels and levels != {industry_level}:
            raise IndustryMembershipError(
                "stock memberships must contain exactly the requested industry level"
            )
        selected: dict[date, PointInTimeStockIndustryMembership] = {}
        for item in eligible:
            key = item.membership.effective_from
            current = selected.get(key)
            if current is None or item.available_at > current.available_at:
                selected[key] = item
            elif item.available_at == current.available_at and item != current:
                raise IndustryMembershipError("ambiguous industry membership revisions")
        ordered = tuple(sorted(selected.values(), key=lambda item: item.membership.effective_from))
        for prior, current in pairwise(ordered):
            prior_end = prior.membership.effective_to
            if prior_end is None or current.membership.effective_from <= prior_end:
                raise IndustryMembershipError("historical industry memberships overlap")
        return ordered

    @staticmethod
    def _membership_on(
        memberships: tuple[PointInTimeStockIndustryMembership, ...],
        trade_date: date,
    ) -> PointInTimeStockIndustryMembership | None:
        matches = tuple(
            item
            for item in memberships
            if item.membership.effective_from <= trade_date
            and (item.membership.effective_to is None or item.membership.effective_to >= trade_date)
        )
        if len(matches) > 1:
            raise IndustryMembershipError("historical industry memberships overlap")
        return matches[0] if matches else None

    def _validate_historical_suspensions(
        self, observations: tuple[StockBarObservation, ...]
    ) -> None:
        suspended = 0
        for prior, current in pairwise(observations):
            if not current.is_suspended:
                continue
            suspended += 1
            if current.adjusted_close != prior.adjusted_close:
                raise ValueError("historical suspended bars must carry the prior adjusted_close")
        if suspended > self._config.maximum_historical_suspensions:
            raise InsufficientStockData(
                "historical suspension count exceeds the configured safety limit"
            )

    def _horizon_metrics(
        self,
        *,
        stock_daily: tuple[Decimal, ...],
        benchmark_daily: tuple[Decimal, ...],
        industry_daily: tuple[Decimal | None, ...],
        industry_ids: tuple[str | None, ...],
    ) -> tuple[HorizonStrengthMetrics, ...]:
        results: list[HorizonStrengthMetrics] = []
        for horizon in self._config.horizons:
            stock_values = stock_daily[-horizon:]
            benchmark_values = benchmark_daily[-horizon:]
            stock_return = _compound(stock_values)
            benchmark_return = _compound(benchmark_values)
            industry_values = industry_daily[-horizon:]
            present = all(value is not None for value in industry_values)
            industry_return = (
                _compound(tuple(value for value in industry_values if value is not None))
                if present
                else None
            )
            historical_ids = tuple(
                sorted({value for value in industry_ids[-horizon:] if value is not None})
            )
            results.append(
                HorizonStrengthMetrics(
                    horizon=horizon,
                    stock_return=stock_return,
                    benchmark_return=benchmark_return,
                    benchmark_relative_return=stock_return - benchmark_return,
                    industry_return=industry_return,
                    industry_relative_return=(
                        stock_return - industry_return if industry_return is not None else None
                    ),
                    historical_industry_ids=historical_ids,
                    aligned_sessions=len(stock_values),
                )
            )
        return tuple(results)

    def _trend_quality(self, observations: tuple[StockBarObservation, ...]) -> TrendQualityMetrics:
        closes = tuple(item.adjusted_close for item in observations)
        window = self._config.moving_average_window
        lag = self._config.slope_lookback
        current_average = _mean(closes[-window:])
        prior_average = _mean(closes[-(window + lag) : -lag])
        latest = closes[-1]
        position = latest / current_average - 1
        slope = current_average / prior_average - 1
        hold = self._config.breakout_hold_sessions
        breakout_history = closes[-(self._config.breakout_window + hold) : -hold]
        breakout_level = max(breakout_history)
        hold_values = closes[-hold:]
        breakout_ratio = Decimal(sum(value > breakout_level for value in hold_values)) / hold
        pullback_peak = max(closes[-self._config.pullback_window :])
        pullback_depth = Decimal(1) - latest / pullback_peak
        new_high = max(closes[-self._config.new_high_window :])
        distance_to_high = latest / new_high - 1
        component_scores = {
            "position": _clamp(position * self._config.position_score_scale),
            "slope": _clamp(slope * self._config.slope_score_scale),
            "breakout": breakout_ratio * Decimal(100),
            "pullback": _clamp(
                Decimal(100) - pullback_depth / self._config.pullback_tolerance * Decimal(100)
            ),
            "new_high": _clamp(
                Decimal(100)
                + distance_to_high / self._config.new_high_distance_tolerance * Decimal(100)
            ),
        }
        score = sum(
            (
                component_scores[name] * weight
                for name, weight in self._config.trend_weights.items()
            ),
            Decimal(0),
        )
        return TrendQualityMetrics(
            latest_close=latest,
            moving_average=current_average,
            position_vs_average=position,
            moving_average_slope=slope,
            breakout_level=breakout_level,
            breakout_hold_ratio=breakout_ratio,
            pullback_depth=pullback_depth,
            distance_to_high=distance_to_high,
            score=score,
        )

    def _volume_confirmation(
        self, observations: tuple[StockBarObservation, ...]
    ) -> VolumePriceConfirmation:
        current = observations[-1]
        prior = observations[-2]
        eligible = tuple(item for item in observations[:-1] if not item.is_suspended)
        baseline = eligible[-self._config.volume_window :]
        if len(baseline) != self._config.volume_window:
            raise InsufficientStockData("non-suspended volume baseline is incomplete")
        average_volume = _mean(tuple(item.bar.volume for item in baseline))
        if average_volume <= 0:
            raise InsufficientStockData("historical average volume is zero")
        current_return = current.adjusted_close / prior.adjusted_close - 1
        volume_ratio = current.bar.volume / average_volume
        if current_return > 0:
            raw_score = (volume_ratio - 1) * self._config.volume_ratio_score_scale
        elif current_return < 0:
            raw_score = (Decimal(1) - volume_ratio) * self._config.volume_ratio_score_scale
        else:
            raw_score = Decimal(0)
        return VolumePriceConfirmation(
            current_return=current_return,
            current_volume=current.bar.volume,
            average_prior_volume=average_volume,
            volume_ratio=volume_ratio,
            score=_clamp(raw_score),
            baseline_observations=len(baseline),
        )

    def _score_contributions(
        self,
        horizons: tuple[HorizonStrengthMetrics, ...],
        trend: TrendQualityMetrics,
        volume: VolumePriceConfirmation,
    ) -> tuple[ScoreContribution, ...]:
        weight_by_horizon = dict(
            zip(self._config.horizons, self._config.horizon_weights, strict=True)
        )

        def weighted(field_name: str) -> Decimal | None:
            selected = tuple(
                (getattr(item, field_name), weight_by_horizon[item.horizon])
                for item in horizons
                if getattr(item, field_name) is not None
            )
            if not selected:
                return None
            total_weight = sum((weight for _, weight in selected), Decimal(0))
            return (
                sum(
                    (value * weight for value, weight in selected if value is not None),
                    Decimal(0),
                )
                / total_weight
            )

        raw_components: list[tuple[str, Decimal, Decimal]] = []
        absolute = weighted("stock_return")
        benchmark = weighted("benchmark_relative_return")
        industry = weighted("industry_relative_return")
        assert absolute is not None and benchmark is not None
        raw_components.extend(
            [
                (
                    "absolute_return",
                    absolute,
                    _clamp(absolute * self._config.return_score_scale),
                ),
                (
                    "benchmark_relative",
                    benchmark,
                    _clamp(benchmark * self._config.relative_score_scale),
                ),
            ]
        )
        if industry is not None:
            raw_components.append(
                (
                    "industry_relative",
                    industry,
                    _clamp(industry * self._config.relative_score_scale),
                )
            )
        raw_components.extend(
            [
                ("trend_quality", trend.score, trend.score),
                ("volume_confirmation", volume.score, volume.score),
            ]
        )
        configured = self._config.score_weights
        available_weight = sum((configured[name] for name, _, _ in raw_components), Decimal(0))
        contributions: list[ScoreContribution] = []
        assigned_weight = Decimal(0)
        for index, (name, raw_value, normalized) in enumerate(raw_components):
            effective = (
                Decimal(1) - assigned_weight
                if index == len(raw_components) - 1
                else configured[name] / available_weight
            )
            assigned_weight += effective
            contributions.append(
                ScoreContribution(
                    component=name,
                    raw_value=raw_value,
                    normalized_score=normalized,
                    configured_weight=configured[name],
                    effective_weight=effective,
                    contribution=normalized * effective,
                )
            )
        return tuple(contributions)

    def _insufficient(
        self,
        *,
        reason: str,
        instrument_id: str,
        benchmark_id: str,
        session_date: date,
        as_of: datetime,
        data_version: str,
        classification_version: str,
        industry_level: int,
        current_industry_id: str | None,
        selected_dates: tuple[date, ...],
        input_hash: str,
    ) -> StockStrengthSnapshot:
        if self._config.insufficient_history_policy is InsufficientHistoryPolicy.FAIL:
            raise InsufficientStockData(reason)
        return self._unavailable_snapshot(
            status=StockStrengthStatus.INSUFFICIENT_DATA,
            reason=reason,
            instrument_id=instrument_id,
            benchmark_id=benchmark_id,
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
            current_industry_id=current_industry_id,
            selected_dates=selected_dates,
            input_hash=input_hash,
        )

    def _unavailable_snapshot(
        self,
        *,
        status: StockStrengthStatus,
        reason: str,
        instrument_id: str,
        benchmark_id: str,
        session_date: date,
        as_of: datetime,
        data_version: str,
        classification_version: str,
        industry_level: int,
        current_industry_id: str | None,
        selected_dates: tuple[date, ...],
        input_hash: str,
    ) -> StockStrengthSnapshot:
        cache_key = self._cache_key(
            instrument_id=instrument_id,
            benchmark_id=benchmark_id,
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
            input_hash=input_hash,
        )
        result_hash = self._result_hash(
            cache_key=cache_key,
            status=status,
            horizons=(),
            trend=None,
            volume=None,
            contributions=(),
            score=None,
            reason=reason,
        )
        return StockStrengthSnapshot(
            instrument_id=instrument_id,
            benchmark_id=benchmark_id,
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
            current_industry_id=current_industry_id,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            status=status,
            horizons=(),
            trend_quality=None,
            volume_confirmation=None,
            contributions=(),
            score=None,
            reason=reason,
            selected_dates=tuple(sorted(set(selected_dates))),
            input_hash=input_hash,
            cache_key=cache_key,
            result_hash=result_hash,
        )

    @staticmethod
    def _unique_memberships(
        memberships: list[PointInTimeStockIndustryMembership],
    ) -> tuple[PointInTimeStockIndustryMembership, ...]:
        unique = {
            (
                item.membership.effective_from,
                item.membership.industry_id,
                item.available_at,
                item.revision,
            ): item
            for item in memberships
        }
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.membership.effective_from,
                    item.membership.industry_id,
                ),
            )
        )

    @staticmethod
    def _unique_references(
        references: list[ReferenceBarObservation],
    ) -> tuple[ReferenceBarObservation, ...]:
        unique = {(item.reference_id, item.trade_date): item for item in references}
        return tuple(sorted(unique.values(), key=lambda item: (item.reference_id, item.trade_date)))

    @staticmethod
    def _input_hash(
        *,
        stocks: tuple[StockBarObservation, ...],
        references: tuple[ReferenceBarObservation, ...],
        memberships: tuple[PointInTimeStockIndustryMembership, ...],
    ) -> str:
        return stable_feature_hash(
            {
                "memberships": [item.fingerprint_payload() for item in memberships],
                "references": [item.fingerprint_payload() for item in references],
                "stocks": [item.fingerprint_payload() for item in stocks],
            }
        )

    def _cache_key(
        self,
        *,
        instrument_id: str,
        benchmark_id: str,
        session_date: date,
        as_of: datetime,
        data_version: str,
        classification_version: str,
        industry_level: int,
        input_hash: str,
    ) -> str:
        return stable_feature_hash(
            {
                "as_of": as_of.astimezone(UTC).isoformat(timespec="microseconds"),
                "benchmark_id": benchmark_id,
                "classification_version": classification_version,
                "config_hash": self._config.config_hash,
                "data_version": data_version,
                "industry_level": industry_level,
                "input_hash": input_hash,
                "instrument_id": instrument_id,
                "session_date": session_date.isoformat(),
            }
        )

    @staticmethod
    def _result_hash(
        *,
        cache_key: str,
        status: StockStrengthStatus,
        horizons: tuple[HorizonStrengthMetrics, ...],
        trend: TrendQualityMetrics | None,
        volume: VolumePriceConfirmation | None,
        contributions: tuple[ScoreContribution, ...],
        score: Decimal | None,
        reason: str | None,
    ) -> str:
        return stable_feature_hash(
            {
                "cache_key": cache_key,
                "contributions": [
                    {
                        "component": item.component,
                        "configured_weight": canonical_decimal(item.configured_weight),
                        "contribution": canonical_decimal(item.contribution),
                        "effective_weight": canonical_decimal(item.effective_weight),
                        "normalized_score": canonical_decimal(item.normalized_score),
                        "raw_value": canonical_decimal(item.raw_value),
                    }
                    for item in contributions
                ],
                "horizons": [
                    {
                        "benchmark_relative_return": canonical_decimal(
                            item.benchmark_relative_return
                        ),
                        "benchmark_return": canonical_decimal(item.benchmark_return),
                        "historical_industry_ids": list(item.historical_industry_ids),
                        "horizon": item.horizon,
                        "industry_relative_return": canonical_decimal(item.industry_relative_return)
                        if item.industry_relative_return is not None
                        else None,
                        "industry_return": canonical_decimal(item.industry_return)
                        if item.industry_return is not None
                        else None,
                        "stock_return": canonical_decimal(item.stock_return),
                    }
                    for item in horizons
                ],
                "reason": reason,
                "score": canonical_decimal(score) if score is not None else None,
                "status": status.value,
                "trend_score": canonical_decimal(trend.score) if trend else None,
                "volume_score": canonical_decimal(volume.score) if volume else None,
            }
        )


__all__ = ["StockStrengthAnalyzer"]
