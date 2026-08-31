"""Industry relative strength using historical memberships and aligned sessions."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import pairwise

from quant_agent.core.time import ensure_aware
from quant_agent.data.domain import IndustryMembership
from quant_agent.features.core import FeatureObservation, RollingFeatureEngine, canonical_decimal
from quant_agent.features.industry_strength.contracts import (
    HorizonRelativeReturn,
    IndustryStrengthConfig,
    IndustryStrengthResult,
    IndustryStrengthSnapshot,
    IndustryStrengthStatus,
    InsufficientIndustryData,
    MemberContribution,
    PointInTimeIndustryMembership,
)
from quant_agent.features.market_breadth import BreadthObservation

type BenchmarkIntervals = dict[date, tuple[date, Decimal]]
type MemberReturnsByDate = dict[date, list[tuple[str, Decimal]]]


def _compound(values: list[Decimal]) -> Decimal:
    result = Decimal(1)
    for value in values:
        result *= 1 + value
    return result - 1


def _clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return max(low, min(high, value))


class IndustryStrengthAnalyzer:
    """Rank industries without applying today's membership to historical bars."""

    def __init__(self, config: IndustryStrengthConfig | None = None) -> None:
        self._config = config or IndustryStrengthConfig()
        self._window_engine = RollingFeatureEngine()

    def analyze(
        self,
        *,
        session_date: date,
        as_of: datetime,
        data_version: str,
        classification_version: str,
        industry_level: int,
        industry_ids: tuple[str, ...],
        memberships: Iterable[PointInTimeIndustryMembership],
        stock_observations: Iterable[BreadthObservation],
        benchmark_observations: Iterable[FeatureObservation],
    ) -> IndustryStrengthSnapshot:
        """Return deterministic rankable and explicitly insufficient industry rows."""

        ensure_aware(as_of)
        if session_date > as_of.date():
            raise ValueError("session_date cannot be after as_of")
        if not data_version.strip() or not classification_version.strip():
            raise ValueError("data and classification versions must be non-empty")
        if industry_level not in (1, 2, 3):
            raise ValueError("industry_level must be 1, 2, or 3")
        if not industry_ids or any(not value.strip() for value in industry_ids):
            raise ValueError("industry_ids must contain non-empty identifiers")
        if len(set(industry_ids)) != len(industry_ids):
            raise ValueError("industry_ids cannot contain duplicates")
        normalized_industry_ids = tuple(sorted(industry_ids))
        target_ids = set(normalized_industry_ids)
        membership_revisions = self._select_membership_revisions(
            memberships,
            as_of=as_of,
            industry_level=industry_level,
            classification_version=classification_version,
            target_ids=target_ids,
        )
        selected_memberships = tuple(item.membership for item in membership_revisions)
        self._validate_memberships(selected_memberships)
        observations = self._select_stock_revisions(
            stock_observations,
            session_date=session_date,
            as_of=as_of,
        )
        benchmark_intervals = self._benchmark_returns(
            benchmark_observations,
            session_date=session_date,
            as_of=as_of,
        )
        if (
            len(benchmark_intervals) < max(self._config.horizons)
            or session_date not in benchmark_intervals
        ):
            raise InsufficientIndustryData("benchmark history cannot cover all horizons")

        daily_returns, daily_contributions, daily_turnovers, histories = self._industry_series(
            industry_ids=target_ids,
            memberships=selected_memberships,
            observations=observations,
            benchmark_intervals=benchmark_intervals,
        )
        provisional = [
            self._analyze_industry(
                industry_id=industry_id,
                session_date=session_date,
                memberships=selected_memberships,
                observations=observations,
                daily_returns=daily_returns[industry_id],
                current_contributions=daily_contributions[industry_id].get(session_date, []),
                daily_turnovers=daily_turnovers[industry_id],
                histories=histories,
                benchmark_intervals=benchmark_intervals,
            )
            for industry_id in normalized_industry_ids
        ]
        ready = [item for item in provisional if item.score is not None]

        def ranking_key(item: IndustryStrengthResult) -> tuple[Decimal, str]:
            if item.score is None:  # pragma: no cover - filtered above
                raise AssertionError("unscored industry reached ranking")
            return -item.score, item.industry_id

        ready.sort(key=ranking_key)
        ranks = {item.industry_id: rank for rank, item in enumerate(ready, start=1)}
        finalized = tuple(
            replace(item, rank=ranks[item.industry_id])
            if item.status is IndustryStrengthStatus.READY
            else item
            for item in provisional
        )
        cache_key = self._cache_key(
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
            industry_ids=normalized_industry_ids,
            membership_revisions=membership_revisions,
            observations=observations,
            benchmark_intervals=benchmark_intervals,
        )
        return IndustryStrengthSnapshot(
            session_date=session_date,
            as_of=as_of,
            data_version=data_version,
            classification_version=classification_version,
            industry_level=industry_level,
            feature_version=self._config.version,
            config_hash=self._config.config_hash,
            industries=finalized,
            cache_key=cache_key,
        )

    def _analyze_industry(
        self,
        *,
        industry_id: str,
        session_date: date,
        memberships: tuple[IndustryMembership, ...],
        observations: tuple[BreadthObservation, ...],
        daily_returns: dict[date, Decimal],
        current_contributions: list[tuple[str, Decimal]],
        daily_turnovers: dict[date, Decimal],
        histories: dict[str, list[BreadthObservation]],
        benchmark_intervals: BenchmarkIntervals,
    ) -> IndustryStrengthResult:
        current_members = {
            item.instrument_id
            for item in memberships
            if item.industry_id == industry_id
            and item.effective_from <= session_date
            and (item.effective_to is None or item.effective_to >= session_date)
        }
        current_rows = {
            item.instrument_id: item
            for item in observations
            if item.trade_date == session_date
            and item.instrument_id in current_members
            and item.is_active
        }
        traded_rows = {
            instrument_id: item
            for instrument_id, item in current_rows.items()
            if not item.is_suspended
        }
        comparable_return_count = len(current_contributions)
        current_member_coverage = (
            Decimal(comparable_return_count) / len(current_members) if current_members else None
        )
        advancing_count = sum(value > 0 for _, value in current_contributions)

        def insufficient(
            reason: str,
            *,
            horizon_returns: tuple[HorizonRelativeReturn, ...] = (),
            breadth_ratio: Decimal | None = None,
            new_high_count: int = 0,
            new_high_denominator: int = 0,
            new_high_ratio: Decimal | None = None,
            turnover_growth: Decimal | None = None,
        ) -> IndustryStrengthResult:
            return IndustryStrengthResult(
                industry_id=industry_id,
                status=IndustryStrengthStatus.INSUFFICIENT_DATA,
                rank=None,
                score=None,
                horizon_returns=horizon_returns,
                current_member_count=len(current_members),
                traded_member_count=len(traded_rows),
                comparable_return_count=comparable_return_count,
                current_member_coverage=current_member_coverage,
                advancing_count=advancing_count,
                breadth_ratio=breadth_ratio,
                new_high_count=new_high_count,
                new_high_denominator=new_high_denominator,
                new_high_ratio=new_high_ratio,
                turnover_growth=turnover_growth,
                downside_resilience=None,
                contributions=(),
                reason=reason,
            )

        if (
            comparable_return_count < self._config.minimum_members
            or current_member_coverage is None
            or current_member_coverage < self._config.minimum_member_coverage
        ):
            return insufficient("insufficient current member return coverage")

        horizon_rows: list[HorizonRelativeReturn] = []
        benchmark_dates = sorted(benchmark_intervals)
        for horizon in self._config.horizons:
            selected_dates = benchmark_dates[-horizon:]
            if len(selected_dates) != horizon or any(
                value not in daily_returns for value in selected_dates
            ):
                return insufficient(
                    f"insufficient aligned history for {horizon}-session horizon",
                    horizon_returns=tuple(horizon_rows),
                )
            industry_return = _compound([daily_returns[value] for value in selected_dates])
            benchmark_return = _compound(
                [benchmark_intervals[value][1] for value in selected_dates]
            )
            horizon_rows.append(
                HorizonRelativeReturn(
                    horizon=horizon,
                    industry_return=industry_return,
                    benchmark_return=benchmark_return,
                    relative_return=industry_return - benchmark_return,
                    aligned_sessions=len(selected_dates),
                )
            )

        breadth_ratio = Decimal(advancing_count) / len(current_contributions)
        new_high_count = 0
        new_high_denominator = 0
        for instrument_id in traded_rows:
            closes = [
                item.close
                for item in histories[instrument_id]
                if item.trade_date <= session_date and item.is_active and not item.is_suspended
            ]
            if len(closes) >= self._config.new_high_window:
                new_high_denominator += 1
                previous = closes[-self._config.new_high_window : -1]
                if closes[-1] > max(previous):
                    new_high_count += 1
        if new_high_denominator < self._config.minimum_members:
            return insufficient(
                "insufficient members with new-high history",
                horizon_returns=tuple(horizon_rows),
                breadth_ratio=breadth_ratio,
                new_high_count=new_high_count,
                new_high_denominator=new_high_denominator,
            )
        new_high_ratio = Decimal(new_high_count) / new_high_denominator

        prior_dates = [value for value in benchmark_dates if value < session_date][
            -self._config.turnover_lookback :
        ]
        current_turnover = daily_turnovers.get(session_date)
        if (
            current_turnover is None
            or len(prior_dates) != self._config.turnover_lookback
            or any(value not in daily_turnovers for value in prior_dates)
        ):
            return insufficient(
                "insufficient turnover history",
                horizon_returns=tuple(horizon_rows),
                breadth_ratio=breadth_ratio,
                new_high_count=new_high_count,
                new_high_denominator=new_high_denominator,
                new_high_ratio=new_high_ratio,
            )
        prior_turnover = sum((daily_turnovers[value] for value in prior_dates), Decimal(0)) / len(
            prior_dates
        )
        if prior_turnover <= 0:
            return insufficient(
                "historical turnover is zero and growth is undefined",
                horizon_returns=tuple(horizon_rows),
                breadth_ratio=breadth_ratio,
                new_high_count=new_high_count,
                new_high_denominator=new_high_denominator,
                new_high_ratio=new_high_ratio,
            )
        turnover_growth = current_turnover / prior_turnover - 1

        downside_dates = [
            value
            for value in benchmark_dates[-self._config.downside_window :]
            if value in daily_returns and benchmark_intervals[value][1] < 0
        ]
        downside_resilience = (
            sum(
                (daily_returns[value] - benchmark_intervals[value][1] for value in downside_dates),
                Decimal(0),
            )
            / len(downside_dates)
            if downside_dates
            else Decimal(0)
        )
        relative_weight_total = sum(self._config.horizon_weights, Decimal(0))
        weighted_relative = (
            sum(
                (
                    row.relative_return * weight
                    for row, weight in zip(
                        horizon_rows,
                        self._config.horizon_weights,
                        strict=True,
                    )
                ),
                Decimal(0),
            )
            / relative_weight_total
        )
        raw_score = (
            weighted_relative * self._config.relative_weight
            + (breadth_ratio - Decimal("0.5")) * self._config.breadth_weight
            + _clamp(turnover_growth, Decimal(-1), Decimal(1)) * self._config.turnover_weight
            + new_high_ratio * self._config.new_high_weight
            + _clamp(downside_resilience * 10, Decimal(-1), Decimal(1))
            * self._config.resilience_weight
        )
        score = _clamp(
            raw_score * self._config.score_scale,
            Decimal(-100),
            Decimal(100),
        )
        contribution_ids = {instrument_id for instrument_id, _ in current_contributions}
        total_turnover = sum(
            (traded_rows[instrument_id].turnover for instrument_id in contribution_ids),
            Decimal(0),
        )
        contribution_count = len(current_contributions)
        contributions = tuple(
            MemberContribution(
                instrument_id=instrument_id,
                current_return=current_return,
                return_contribution=current_return / contribution_count,
                turnover_share=(
                    traded_rows[instrument_id].turnover / total_turnover
                    if total_turnover > 0
                    else Decimal(0)
                ),
            )
            for instrument_id, current_return in sorted(current_contributions)
        )
        return IndustryStrengthResult(
            industry_id=industry_id,
            status=IndustryStrengthStatus.READY,
            rank=1,
            score=score,
            horizon_returns=tuple(horizon_rows),
            current_member_count=len(current_members),
            traded_member_count=len(traded_rows),
            comparable_return_count=comparable_return_count,
            current_member_coverage=current_member_coverage,
            advancing_count=advancing_count,
            breadth_ratio=breadth_ratio,
            new_high_count=new_high_count,
            new_high_denominator=new_high_denominator,
            new_high_ratio=new_high_ratio,
            turnover_growth=turnover_growth,
            downside_resilience=downside_resilience,
            contributions=contributions,
            reason=None,
        )

    def _industry_series(
        self,
        *,
        industry_ids: set[str],
        memberships: tuple[IndustryMembership, ...],
        observations: tuple[BreadthObservation, ...],
        benchmark_intervals: BenchmarkIntervals,
    ) -> tuple[
        dict[str, dict[date, Decimal]],
        dict[str, dict[date, list[tuple[str, Decimal]]]],
        dict[str, dict[date, Decimal]],
        dict[str, list[BreadthObservation]],
    ]:
        histories: dict[str, list[BreadthObservation]] = defaultdict(list)
        for observation in observations:
            histories[observation.instrument_id].append(observation)
        return_members: dict[str, dict[date, list[tuple[str, Decimal]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        turnovers: dict[str, dict[date, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        traded_members: dict[str, dict[date, set[str]]] = defaultdict(lambda: defaultdict(set))
        for instrument_id, history in histories.items():
            history.sort(key=lambda item: item.trade_date)
            previous_close: Decimal | None = None
            previous_date: date | None = None
            previous_industry: str | None = None
            for observation in history:
                if not observation.is_active:
                    previous_close = None
                    previous_date = None
                    previous_industry = None
                    continue
                if observation.is_suspended:
                    continue
                industry_id = self._membership_on(
                    memberships,
                    instrument_id=instrument_id,
                    trade_date=observation.trade_date,
                    target_ids=industry_ids,
                )
                if industry_id is None:
                    previous_close = observation.close
                    previous_date = observation.trade_date
                    previous_industry = None
                    continue
                turnovers[industry_id][observation.trade_date] += observation.turnover
                traded_members[industry_id][observation.trade_date].add(instrument_id)
                benchmark_interval = benchmark_intervals.get(observation.trade_date)
                if (
                    previous_close is not None
                    and previous_date is not None
                    and previous_industry == industry_id
                    and benchmark_interval is not None
                    and benchmark_interval[0] == previous_date
                ):
                    return_members[industry_id][observation.trade_date].append(
                        (instrument_id, observation.close / previous_close - 1)
                    )
                previous_close = observation.close
                previous_date = observation.trade_date
                previous_industry = industry_id
        daily_returns: dict[str, dict[date, Decimal]] = defaultdict(dict)
        for industry_id in industry_ids:
            relevant_dates = set(return_members[industry_id]) | set(turnovers[industry_id])
            for trade_date in relevant_dates:
                members = return_members[industry_id][trade_date]
                expected_count = len(
                    self._member_ids_on(
                        memberships,
                        industry_id=industry_id,
                        trade_date=trade_date,
                    )
                )
                return_coverage = (
                    Decimal(len(members)) / expected_count if expected_count else Decimal(0)
                )
                traded_coverage = (
                    Decimal(len(traded_members[industry_id][trade_date])) / expected_count
                    if expected_count
                    else Decimal(0)
                )
                if (
                    len(members) >= self._config.minimum_members
                    and return_coverage >= self._config.minimum_member_coverage
                ):
                    daily_returns[industry_id][trade_date] = sum(
                        (value for _, value in members), Decimal(0)
                    ) / len(members)
                if (
                    len(traded_members[industry_id][trade_date]) < self._config.minimum_members
                    or traded_coverage < self._config.minimum_member_coverage
                ):
                    turnovers[industry_id].pop(trade_date, None)
        return daily_returns, return_members, turnovers, histories

    @staticmethod
    def _member_ids_on(
        memberships: tuple[IndustryMembership, ...],
        *,
        industry_id: str,
        trade_date: date,
    ) -> set[str]:
        return {
            item.instrument_id
            for item in memberships
            if item.industry_id == industry_id
            and item.effective_from <= trade_date
            and (item.effective_to is None or item.effective_to >= trade_date)
        }

    @staticmethod
    def _select_membership_revisions(
        memberships: Iterable[PointInTimeIndustryMembership],
        *,
        as_of: datetime,
        industry_level: int,
        classification_version: str,
        target_ids: set[str],
    ) -> tuple[PointInTimeIndustryMembership, ...]:
        selected: dict[tuple[str, str, date], PointInTimeIndustryMembership] = {}
        for observation in memberships:
            membership = observation.membership
            if observation.available_at > as_of or observation.level != industry_level:
                continue
            if membership.version != classification_version:
                continue
            if membership.industry_id not in target_ids:
                continue
            key = (
                membership.instrument_id,
                membership.industry_id,
                membership.effective_from,
            )
            current = selected.get(key)
            if current is None or observation.available_at > current.available_at:
                selected[key] = observation
            elif observation.available_at == current.available_at and observation != current:
                raise ValueError("ambiguous industry membership revisions")
        return tuple(
            sorted(
                selected.values(),
                key=lambda item: (
                    item.membership.instrument_id,
                    item.membership.industry_id,
                    item.membership.effective_from,
                ),
            )
        )

    @staticmethod
    def _validate_memberships(memberships: tuple[IndustryMembership, ...]) -> None:
        by_instrument: dict[str, list[IndustryMembership]] = defaultdict(list)
        for membership in memberships:
            by_instrument[membership.instrument_id].append(membership)
        for instrument_id, values in by_instrument.items():
            ordered = sorted(values, key=lambda item: item.effective_from)
            for previous, current in pairwise(ordered):
                if previous.effective_to is None or current.effective_from <= previous.effective_to:
                    raise ValueError(
                        f"instrument {instrument_id} has overlapping target-industry memberships"
                    )

    @staticmethod
    def _membership_on(
        memberships: tuple[IndustryMembership, ...],
        *,
        instrument_id: str,
        trade_date: date,
        target_ids: set[str],
    ) -> str | None:
        matches = {
            item.industry_id
            for item in memberships
            if item.instrument_id == instrument_id
            and item.industry_id in target_ids
            and item.effective_from <= trade_date
            and (item.effective_to is None or item.effective_to >= trade_date)
        }
        if len(matches) > 1:
            raise ValueError(
                f"instrument {instrument_id} has overlapping target-industry memberships"
            )
        return next(iter(matches), None)

    @staticmethod
    def _select_stock_revisions(
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
                raise ValueError("ambiguous industry input revisions")
        return tuple(
            sorted(
                selected.values(),
                key=lambda item: (item.trade_date, item.instrument_id),
            )
        )

    def _benchmark_returns(
        self,
        observations: Iterable[FeatureObservation],
        *,
        session_date: date,
        as_of: datetime,
    ) -> BenchmarkIntervals:
        eligible = (item for item in observations if item.observed_at.date() <= session_date)
        selected = self._window_engine.select_window(
            observations=eligible,
            as_of=as_of,
            window=max(self._config.horizons) + 1,
        )
        closes = [
            (item.observed_at.date(), item.value)
            for item in selected
            if item.observed_at.date() <= session_date and item.value is not None
        ]
        close_dates = [trade_date for trade_date, _ in closes]
        if len(close_dates) != len(set(close_dates)):
            raise ValueError("benchmark contains multiple observations for one session")
        if any(value <= 0 for _, value in closes):
            raise ValueError("benchmark closes must be positive")
        return {
            current_date: (prior_date, current_close / prior_close - 1)
            for (prior_date, prior_close), (current_date, current_close) in pairwise(closes)
            if current_date > prior_date
        }

    def _cache_key(
        self,
        *,
        session_date: date,
        as_of: datetime,
        data_version: str,
        classification_version: str,
        industry_level: int,
        industry_ids: tuple[str, ...],
        membership_revisions: tuple[PointInTimeIndustryMembership, ...],
        observations: tuple[BreadthObservation, ...],
        benchmark_intervals: BenchmarkIntervals,
    ) -> str:
        payload = {
            "as_of": as_of.astimezone(UTC).isoformat(timespec="microseconds"),
            "benchmark_intervals": {
                end.isoformat(): {
                    "return": canonical_decimal(value),
                    "start": start.isoformat(),
                }
                for end, (start, value) in sorted(benchmark_intervals.items())
            },
            "classification_version": classification_version,
            "config_hash": self._config.config_hash,
            "data_version": data_version,
            "industry_ids": sorted(industry_ids),
            "industry_level": industry_level,
            "memberships": [
                {
                    "available_at": item.available_at.astimezone(UTC).isoformat(
                        timespec="microseconds"
                    ),
                    "effective_from": item.membership.effective_from.isoformat(),
                    "effective_to": (
                        item.membership.effective_to.isoformat()
                        if item.membership.effective_to
                        else None
                    ),
                    "industry_id": item.membership.industry_id,
                    "instrument_id": item.membership.instrument_id,
                    "level": item.level,
                    "revision": item.revision,
                    "source": item.membership.source,
                    "version": item.membership.version,
                }
                for item in membership_revisions
            ],
            "observations": [item.fingerprint_payload() for item in observations],
            "session_date": session_date.isoformat(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
