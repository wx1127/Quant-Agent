"""Historical industry relative-strength and diffusion features."""

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from typing import Protocol

from sqlalchemy.orm import Session

from quant_agent.core.time import ensure_aware
from quant_agent.data.industry import IndustryService
from quant_agent.features.core import clamp_score


class IndustryResolver(Protocol):
    """Resolve an instrument's industry at the observation date."""

    def industry_for(self, instrument_id: str, as_of: date) -> str | None: ...


class SqlIndustryResolver:
    """SQL-backed historical resolver using P1 membership intervals."""

    def __init__(self, session: Session, *, classification: str, level: int) -> None:
        self._service = IndustryService(session)
        self._classification = classification
        self._level = level

    def industry_for(self, instrument_id: str, as_of: date) -> str | None:
        industry = self._service.classification_for(
            instrument_id,
            self._classification,
            self._level,
            as_of,
        )
        return industry.industry_id if industry is not None else None


@dataclass(frozen=True, slots=True)
class IndustryStockBar:
    """Stock bar used for historical industry aggregation."""

    instrument_id: str
    trade_date: date
    close: float
    turnover: float
    is_tradable: bool
    available_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.close <= 0 or self.turnover < 0:
            raise ValueError("stock bar values are invalid")


@dataclass(frozen=True, slots=True)
class BenchmarkBar:
    """Benchmark close for industry relative-return calculations."""

    trade_date: date
    close: float
    available_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.available_at)
        if self.close <= 0:
            raise ValueError("benchmark close must be positive")


@dataclass(frozen=True, slots=True)
class ThemeFeatureResult:
    """Raw and normalized industry evidence."""

    as_of: datetime
    industry_id: str
    member_count: int
    eligible: bool
    relative_return_5d: float
    relative_return_20d: float
    relative_return_60d: float
    relative_strength_score: float
    breadth_ratio: float
    breadth_score: float
    turnover_expansion: float
    turnover_score: float
    new_high_20_ratio: float
    new_high_score: float
    downside_resilience: float
    downside_resilience_score: float
    leader_persistence_ratio: float
    leader_persistence_score: float
    single_stock_contribution: float
    data_version: str
    feature_version: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.as_of)


def _compound(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + value
    return result - 1.0


class ThemeFeatureEngine:
    """Aggregate stock evidence by the industry valid on each historical date."""

    feature_version = "theme_features_v1"

    def __init__(
        self,
        resolver: IndustryResolver,
        *,
        min_members: int = 3,
    ) -> None:
        if min_members < 1:
            raise ValueError("min_members must be positive")
        self._resolver = resolver
        self._min_members = min_members

    @staticmethod
    def _benchmark_returns(
        benchmark: list[BenchmarkBar],
        as_of: datetime,
    ) -> dict[date, float]:
        available = sorted(
            (
                bar
                for bar in benchmark
                if bar.available_at <= as_of and bar.trade_date <= as_of.date()
            ),
            key=lambda item: item.trade_date,
        )
        return {
            current.trade_date: current.close / previous.close - 1.0
            for previous, current in pairwise(available)
        }

    def calculate(
        self,
        bars: list[IndustryStockBar],
        benchmark: list[BenchmarkBar],
        *,
        as_of: datetime,
        data_version: str,
    ) -> tuple[ThemeFeatureResult, ...]:
        """Calculate industry evidence using only past and available records."""

        ensure_aware(as_of)
        available = [
            bar for bar in bars if bar.available_at <= as_of and bar.trade_date <= as_of.date()
        ]
        if not available:
            raise ValueError("theme features require stock bars")
        latest_date = max(bar.trade_date for bar in available)
        benchmark_returns = self._benchmark_returns(benchmark, as_of)
        by_instrument: dict[str, list[IndustryStockBar]] = {}
        for bar in available:
            by_instrument.setdefault(bar.instrument_id, []).append(bar)

        industry_daily_returns: dict[str, dict[date, list[tuple[str, float]]]] = {}
        industry_turnover: dict[str, dict[date, float]] = {}
        current_members: dict[str, list[tuple[IndustryStockBar, list[IndustryStockBar]]]] = {}
        for instrument_id, history in sorted(by_instrument.items()):
            ordered = sorted(history, key=lambda item: item.trade_date)
            for previous, current in pairwise(ordered):
                if not current.is_tradable:
                    continue
                industry_id = self._resolver.industry_for(
                    instrument_id,
                    current.trade_date,
                )
                if industry_id is None:
                    continue
                daily_return = current.close / previous.close - 1.0
                industry_daily_returns.setdefault(industry_id, {}).setdefault(
                    current.trade_date,
                    [],
                ).append((instrument_id, daily_return))
            for current in ordered:
                if not current.is_tradable:
                    continue
                industry_id = self._resolver.industry_for(
                    instrument_id,
                    current.trade_date,
                )
                if industry_id is not None:
                    industry_turnover.setdefault(industry_id, {})[current.trade_date] = (
                        industry_turnover.setdefault(industry_id, {}).get(
                            current.trade_date,
                            0.0,
                        )
                        + current.turnover
                    )
            latest = next(
                (item for item in reversed(ordered) if item.trade_date == latest_date),
                None,
            )
            if latest is not None and latest.is_tradable:
                industry_id = self._resolver.industry_for(instrument_id, latest_date)
                if industry_id is not None:
                    current_members.setdefault(industry_id, []).append((latest, ordered))

        results: list[ThemeFeatureResult] = []
        for industry_id in sorted(current_members):
            members = current_members[industry_id]
            daily_records = industry_daily_returns.get(industry_id, {})
            common_dates = sorted(set(daily_records) & set(benchmark_returns))
            industry_median_returns = {
                trade_date: statistics.median(
                    value for _instrument_id, value in daily_records[trade_date]
                )
                for trade_date in common_dates
                if daily_records[trade_date]
            }
            relative_values: dict[int, float] = {}
            warnings: list[str] = []
            for horizon in (5, 20, 60):
                dates = sorted(industry_median_returns)[-horizon:]
                if len(dates) < horizon:
                    relative_values[horizon] = 0.0
                    warnings.append(f"insufficient {horizon}-day industry history")
                else:
                    industry_return = _compound([industry_median_returns[item] for item in dates])
                    benchmark_return = _compound([benchmark_returns[item] for item in dates])
                    relative_values[horizon] = (1 + industry_return) / (1 + benchmark_return) - 1

            current_daily = daily_records.get(latest_date, [])
            breadth_ratio = (
                sum(value > 0 for _instrument_id, value in current_daily) / len(current_daily)
                if current_daily
                else 0.0
            )
            absolute_sum = sum(abs(value) for _instrument_id, value in current_daily)
            single_stock_contribution = (
                max((abs(value) for _instrument_id, value in current_daily), default=0.0)
                / absolute_sum
                if absolute_sum
                else 0.0
            )
            turnover_by_date = industry_turnover.get(industry_id, {})
            turnover_dates = sorted(turnover_by_date)
            recent5 = [turnover_by_date[item] for item in turnover_dates[-5:]]
            prior20 = [turnover_by_date[item] for item in turnover_dates[-25:-5]]
            turnover_expansion = (
                statistics.fmean(recent5) / statistics.fmean(prior20) - 1.0
                if recent5 and prior20 and statistics.fmean(prior20) > 0
                else 0.0
            )

            new_high_count = 0
            high_denominator = 0
            for latest, history in members:
                closes = [item.close for item in history if item.trade_date <= latest_date]
                if len(closes) >= 20:
                    high_denominator += 1
                    new_high_count += latest.close >= max(closes[-20:])
            new_high_ratio = new_high_count / high_denominator if high_denominator else 0.0

            down_dates = [
                item
                for item in sorted(industry_median_returns)[-20:]
                if benchmark_returns[item] < 0
            ]
            downside_resilience = (
                statistics.fmean(
                    industry_median_returns[item] - benchmark_returns[item] for item in down_dates
                )
                if down_dates
                else 0.0
            )
            recent_leaders = []
            for trade_date in sorted(daily_records)[-5:]:
                if daily_records[trade_date]:
                    recent_leaders.append(
                        max(
                            daily_records[trade_date],
                            key=lambda item: item[1],
                        )[0]
                    )
            leader_persistence = (
                max(Counter(recent_leaders).values()) / len(recent_leaders)
                if recent_leaders
                else 0.0
            )
            eligible = (
                len(members) >= self._min_members and len(sorted(industry_median_returns)) >= 60
            )
            if len(members) < self._min_members:
                warnings.append(f"member count {len(members)} below minimum {self._min_members}")
            relative_strength_score = statistics.fmean(
                (
                    clamp_score(relative_values[5], -0.08, 0.08),
                    clamp_score(relative_values[20], -0.15, 0.15),
                    clamp_score(relative_values[60], -0.30, 0.30),
                )
            )
            results.append(
                ThemeFeatureResult(
                    as_of=as_of,
                    industry_id=industry_id,
                    member_count=len(members),
                    eligible=eligible,
                    relative_return_5d=relative_values[5],
                    relative_return_20d=relative_values[20],
                    relative_return_60d=relative_values[60],
                    relative_strength_score=relative_strength_score,
                    breadth_ratio=breadth_ratio,
                    breadth_score=breadth_ratio * 100,
                    turnover_expansion=turnover_expansion,
                    turnover_score=clamp_score(turnover_expansion, -0.30, 0.50),
                    new_high_20_ratio=new_high_ratio,
                    new_high_score=new_high_ratio * 100,
                    downside_resilience=downside_resilience,
                    downside_resilience_score=clamp_score(
                        downside_resilience,
                        -0.03,
                        0.03,
                    ),
                    leader_persistence_ratio=leader_persistence,
                    leader_persistence_score=leader_persistence * 100,
                    single_stock_contribution=single_stock_contribution,
                    data_version=data_version,
                    feature_version=self.feature_version,
                    warnings=tuple(warnings),
                )
            )
        if any(not math.isfinite(item.relative_strength_score) for item in results):
            raise ValueError("theme features produced a non-finite score")
        return tuple(results)
