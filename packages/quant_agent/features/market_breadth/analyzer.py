"""Point-in-time market breadth computation over historical stock states."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from decimal import Decimal

from quant_agent.core.time import ensure_aware
from quant_agent.features.market_breadth.contracts import (
    BreadthObservation,
    InsufficientBreadthData,
    MarketBreadthConfig,
    MarketBreadthSnapshot,
)


def _ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    return Decimal(numerator) / denominator


class MarketBreadthAnalyzer:
    """Calculate breadth while excluding inactive and suspended observations by date."""

    def __init__(self, config: MarketBreadthConfig | None = None) -> None:
        self._config = config or MarketBreadthConfig()

    def analyze(
        self,
        *,
        session_date: date,
        as_of: datetime,
        data_version: str,
        expected_active_by_date: Mapping[date, tuple[str, ...]],
        observations: Iterable[BreadthObservation],
    ) -> MarketBreadthSnapshot:
        """Build a breadth snapshot with explicit coverage and metric denominators."""

        ensure_aware(as_of)
        if session_date > as_of.date():
            raise ValueError("session_date cannot be after as_of")
        if not data_version.strip():
            raise ValueError("data_version must be non-empty")
        expected_by_date = self._normalize_universes(
            expected_active_by_date,
            session_date=session_date,
        )
        if session_date not in expected_by_date:
            raise ValueError("expected historical universe must include session_date")

        selected = self._select_revisions(
            observations,
            session_date=session_date,
            as_of=as_of,
        )
        by_instrument: dict[str, list[BreadthObservation]] = defaultdict(list)
        by_date: dict[date, list[BreadthObservation]] = defaultdict(list)
        for observation in selected:
            by_instrument[observation.instrument_id].append(observation)
            by_date[observation.trade_date].append(observation)
        for history in by_instrument.values():
            history.sort(key=lambda item: item.trade_date)

        unmapped_active_dates = sorted(
            trade_date
            for trade_date, rows in by_date.items()
            if trade_date not in expected_by_date and any(item.is_active for item in rows)
        )
        if unmapped_active_dates:
            raise ValueError(
                "active observations require a historical universe for every session: "
                f"{unmapped_active_dates[0]}"
            )
        historical_coverages: list[Decimal] = []
        for trade_date, expected_ids in expected_by_date.items():
            received_ids = {
                item.instrument_id for item in by_date.get(trade_date, []) if item.is_active
            }
            unexpected = sorted(received_ids - set(expected_ids))
            if unexpected:
                unexpected_text = ", ".join(unexpected)
                raise ValueError(
                    f"active observations on {trade_date} are outside expected universe: "
                    f"{unexpected_text}"
                )
            historical_coverage = Decimal(len(received_ids & set(expected_ids))) / len(expected_ids)
            historical_coverages.append(historical_coverage)
            floor = (
                self._config.minimum_current_coverage
                if trade_date == session_date
                else self._config.minimum_historical_coverage
            )
            if historical_coverage < floor:
                raise InsufficientBreadthData(
                    f"breadth coverage {historical_coverage} on {trade_date} is below {floor}"
                )

        current_by_id = {
            item.instrument_id: item for item in by_date.get(session_date, []) if item.is_active
        }
        expected_active_instruments = expected_by_date[session_date]
        expected = set(expected_active_instruments)
        missing = tuple(sorted(expected - set(current_by_id)))
        received_count = len(expected) - len(missing)
        coverage = Decimal(received_count) / len(expected)

        advancing = declining = unchanged = return_denominator = 0
        above_average = average_denominator = 0
        new_high = new_low = high_low_denominator = 0
        suspended_count = 0
        total_turnover = Decimal(0)
        for instrument_id in expected_active_instruments:
            current = current_by_id.get(instrument_id)
            if current is None:
                continue
            if current.is_suspended:
                suspended_count += 1
                continue
            total_turnover += current.turnover
            traded_history = [
                item
                for item in by_instrument[instrument_id]
                if item.trade_date <= session_date and item.is_active and not item.is_suspended
            ]
            closes = [item.close for item in traded_history]
            if len(closes) >= 2:
                return_denominator += 1
                if closes[-1] > closes[-2]:
                    advancing += 1
                elif closes[-1] < closes[-2]:
                    declining += 1
                else:
                    unchanged += 1
            if len(closes) >= self._config.moving_average_window:
                average_denominator += 1
                window = closes[-self._config.moving_average_window :]
                moving_average = sum(window, Decimal(0)) / len(window)
                if closes[-1] > moving_average:
                    above_average += 1
            if len(closes) >= self._config.new_high_low_window:
                high_low_denominator += 1
                previous = closes[-self._config.new_high_low_window : -1]
                if closes[-1] > max(previous):
                    new_high += 1
                if closes[-1] < min(previous):
                    new_low += 1

        session_dates = tuple(sorted(expected_by_date))
        turnover_totals = self._daily_turnover_totals(
            by_date,
            session_dates=session_dates,
        )
        turnover_window = turnover_totals[-self._config.turnover_percentile_window :]
        turnover_percentile = None
        if len(turnover_window) >= self._config.minimum_turnover_observations:
            turnover_percentile = Decimal(
                sum(value <= total_turnover for value in turnover_window)
            ) / len(turnover_window)

        market_returns = self._daily_market_returns(
            by_instrument,
            session_dates=session_dates,
        )
        return_window = market_returns[-self._config.downside_volatility_window :]
        downside_volatility = None
        if len(return_window) >= self._config.min_downside_observations:
            squared_downside = [min(value, Decimal(0)) ** 2 for value in return_window]
            downside_volatility = (sum(squared_downside, Decimal(0)) / len(squared_downside)).sqrt()

        cache_key = self._cache_key(
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            expected_by_date=expected_by_date,
            selected=selected,
        )
        return MarketBreadthSnapshot(
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            expected_active_count=len(expected),
            received_active_count=received_count,
            current_coverage=coverage,
            historical_session_count=len(expected_by_date),
            minimum_historical_coverage_observed=min(historical_coverages),
            missing_instruments=missing,
            suspended_count=suspended_count,
            advancing_count=advancing,
            declining_count=declining,
            unchanged_count=unchanged,
            return_denominator=return_denominator,
            above_average_count=above_average,
            moving_average_denominator=average_denominator,
            above_average_ratio=_ratio(above_average, average_denominator),
            new_high_count=new_high,
            new_low_count=new_low,
            high_low_denominator=high_low_denominator,
            new_high_ratio=_ratio(new_high, high_low_denominator),
            new_low_ratio=_ratio(new_low, high_low_denominator),
            total_turnover=total_turnover,
            turnover_percentile=turnover_percentile,
            turnover_history_count=len(turnover_window),
            downside_volatility=downside_volatility,
            downside_observation_count=len(return_window),
            cache_key=cache_key,
        )

    @staticmethod
    def _normalize_universes(
        expected_active_by_date: Mapping[date, tuple[str, ...]],
        *,
        session_date: date,
    ) -> dict[date, tuple[str, ...]]:
        normalized: dict[date, tuple[str, ...]] = {}
        for trade_date, instrument_ids in expected_active_by_date.items():
            if trade_date > session_date:
                continue
            if not instrument_ids or any(not value.strip() for value in instrument_ids):
                raise ValueError("each historical universe must contain non-empty identifiers")
            if len(set(instrument_ids)) != len(instrument_ids):
                raise ValueError("historical universes cannot contain duplicate identifiers")
            normalized[trade_date] = tuple(sorted(instrument_ids))
        if not normalized:
            raise ValueError("expected_active_by_date must contain historical sessions")
        return dict(sorted(normalized.items()))

    @staticmethod
    def _select_revisions(
        observations: Iterable[BreadthObservation],
        *,
        session_date: date,
        as_of: datetime,
    ) -> tuple[BreadthObservation, ...]:
        selected: dict[tuple[str, date], BreadthObservation] = {}
        for observation in observations:
            if observation.trade_date > session_date:
                continue
            if observation.observed_at > as_of or observation.available_at > as_of:
                continue
            key = (observation.instrument_id, observation.trade_date)
            current = selected.get(key)
            if current is None or observation.available_at > current.available_at:
                selected[key] = observation
            elif observation.available_at == current.available_at and observation != current:
                raise ValueError("ambiguous breadth revisions")
        return tuple(
            sorted(
                selected.values(),
                key=lambda item: (item.trade_date, item.instrument_id),
            )
        )

    @staticmethod
    def _daily_turnover_totals(
        by_date: dict[date, list[BreadthObservation]],
        *,
        session_dates: tuple[date, ...],
    ) -> list[Decimal]:
        totals: list[Decimal] = []
        for trade_date in session_dates:
            eligible = [
                item for item in by_date[trade_date] if item.is_active and not item.is_suspended
            ]
            if eligible:
                totals.append(sum((item.turnover for item in eligible), Decimal(0)))
        return totals

    @staticmethod
    def _daily_market_returns(
        by_instrument: dict[str, list[BreadthObservation]],
        *,
        session_dates: tuple[date, ...],
    ) -> list[Decimal]:
        returns_by_date: dict[date, list[Decimal]] = defaultdict(list)
        previous_session = dict(zip(session_dates[1:], session_dates[:-1], strict=True))
        for history in by_instrument.values():
            by_trade_date = {item.trade_date: item for item in history}
            for trade_date, prior_date in previous_session.items():
                current = by_trade_date.get(trade_date)
                prior = by_trade_date.get(prior_date)
                if current is None or prior is None:
                    continue
                if (
                    not current.is_active
                    or not prior.is_active
                    or current.is_suspended
                    or prior.is_suspended
                ):
                    continue
                returns_by_date[trade_date].append(current.close / prior.close - 1)
        return [
            sum(returns_by_date[trade_date], Decimal(0)) / len(returns_by_date[trade_date])
            for trade_date in sorted(returns_by_date)
            if returns_by_date[trade_date]
        ]

    def _cache_key(
        self,
        *,
        session_date: date,
        as_of: datetime,
        data_version: str,
        expected_by_date: Mapping[date, tuple[str, ...]],
        selected: tuple[BreadthObservation, ...],
    ) -> str:
        payload = {
            "as_of": as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "config_hash": self._config.config_hash,
            "data_version": data_version,
            "expected_active_by_date": {
                trade_date.isoformat(): list(instrument_ids)
                for trade_date, instrument_ids in expected_by_date.items()
            },
            "observations": [item.fingerprint_payload() for item in selected],
            "session_date": session_date.isoformat(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
