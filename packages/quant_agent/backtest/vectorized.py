"""Deterministic, point-in-time-safe vectorized research backtests.

The engine deliberately models a small, explicit research contract:

* signals are complete, long-only target-weight snapshots;
* a signal can only execute at the next session's open;
* positions are marked at every session's close;
* transaction fees are selected from the shared :class:`FeeRuleBook`;
* benchmark observations must align exactly to the requested calendar.

It is not an order simulator.  Exchange constraints, partial fills, and settlement
belong to the event-driven engine.  This module is the faster, fail-closed path for
factor layering and daily strategy research.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from enum import Enum
from typing import Any

from quant_agent.backtest.contracts import Side, TradableInstrumentType
from quant_agent.backtest.rules import FeeRuleBook
from quant_agent.core.time import SHANGHAI_TZ, ensure_aware

_ZERO = Decimal(0)
_ONE = Decimal(1)
_HASH_LENGTH = 64


class VectorizedBacktestError(ValueError):
    """Raised when research inputs cannot be aligned without an unsafe assumption."""


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _decimal(value: Decimal | int | str, field_name: str) -> Decimal:
    if isinstance(value, bool | float):
        raise ValueError(f"{field_name} must be an exact decimal input")
    try:
        normalized = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f"{field_name} must be a valid decimal") from error
    if not normalized.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != _HASH_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("cannot hash a non-finite decimal")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        ensure_aware(value)
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class VectorizedBacktestRequest:
    """Immutable identity, calendar, and metric conventions for one research run."""

    run_id: str
    data_version: str
    strategy_version: str
    as_of: datetime
    trading_calendar: tuple[date, ...]
    initial_nav: Decimal = Decimal("1000000")
    annualization_periods: int = 252
    annual_risk_free_rate: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _non_empty(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        object.__setattr__(
            self,
            "strategy_version",
            _non_empty(self.strategy_version, "strategy_version"),
        )
        ensure_aware(self.as_of)
        calendar = tuple(self.trading_calendar)
        object.__setattr__(self, "trading_calendar", calendar)
        if not calendar:
            raise ValueError("trading_calendar must not be empty")
        if tuple(sorted(set(calendar))) != calendar:
            raise ValueError("trading_calendar must contain unique ascending dates")
        if calendar[-1] > self.as_of.astimezone(SHANGHAI_TZ).date():
            raise ValueError("trading_calendar cannot extend beyond request as_of")
        initial_nav = _decimal(self.initial_nav, "initial_nav")
        risk_free_rate = _decimal(self.annual_risk_free_rate, "annual_risk_free_rate")
        object.__setattr__(self, "initial_nav", initial_nav)
        object.__setattr__(self, "annual_risk_free_rate", risk_free_rate)
        if initial_nav <= 0:
            raise ValueError("initial_nav must be positive")
        if self.annualization_periods <= 0:
            raise ValueError("annualization_periods must be positive")
        if risk_free_rate <= -1:
            raise ValueError("annual_risk_free_rate must be greater than -1")


@dataclass(frozen=True, slots=True)
class WeightSignal:
    """A complete target-weight contribution known at a decision-time boundary."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    signal_date: date
    as_of: datetime
    target_weight: Decimal
    data_version: str
    strategy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            _non_empty(self.instrument_id, "instrument_id"),
        )
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        object.__setattr__(
            self,
            "strategy_version",
            _non_empty(self.strategy_version, "strategy_version"),
        )
        ensure_aware(self.as_of)
        if self.as_of.astimezone(SHANGHAI_TZ).date() != self.signal_date:
            raise ValueError("signal_date must match signal as_of in Asia/Shanghai")
        target_weight = _decimal(self.target_weight, "target_weight")
        object.__setattr__(self, "target_weight", target_weight)
        if target_weight < 0 or target_weight > 1:
            raise ValueError("target_weight must be between zero and one; shorting is unsupported")


@dataclass(frozen=True, slots=True)
class FactorSignal:
    """A point-in-time factor score that can be converted into a ranked layer."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    signal_date: date
    as_of: datetime
    score: Decimal
    data_version: str
    strategy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            _non_empty(self.instrument_id, "instrument_id"),
        )
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        object.__setattr__(
            self,
            "strategy_version",
            _non_empty(self.strategy_version, "strategy_version"),
        )
        ensure_aware(self.as_of)
        if self.as_of.astimezone(SHANGHAI_TZ).date() != self.signal_date:
            raise ValueError("signal_date must match signal as_of in Asia/Shanghai")
        object.__setattr__(self, "score", _decimal(self.score, "score"))


@dataclass(frozen=True, slots=True)
class ResearchPriceBar:
    """One immutable open/close observation from a versioned research snapshot."""

    instrument_id: str
    instrument_type: TradableInstrumentType
    trading_day: date
    open_price: Decimal
    close_price: Decimal
    available_at: datetime
    data_version: str
    revision: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instrument_id",
            _non_empty(self.instrument_id, "instrument_id"),
        )
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        ensure_aware(self.available_at)
        open_price = _decimal(self.open_price, "open_price")
        close_price = _decimal(self.close_price, "close_price")
        object.__setattr__(self, "open_price", open_price)
        object.__setattr__(self, "close_price", close_price)
        if open_price <= 0 or close_price <= 0:
            raise ValueError("research prices must be positive")
        if self.available_at.astimezone(SHANGHAI_TZ).date() < self.trading_day:
            raise ValueError("price available_at cannot precede its trading_day")


@dataclass(frozen=True, slots=True)
class BenchmarkBar:
    """One immutable benchmark close observation used only after exact alignment."""

    benchmark_id: str
    trading_day: date
    close_price: Decimal
    available_at: datetime
    data_version: str
    revision: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "benchmark_id",
            _non_empty(self.benchmark_id, "benchmark_id"),
        )
        object.__setattr__(
            self,
            "data_version",
            _non_empty(self.data_version, "data_version"),
        )
        object.__setattr__(self, "revision", _non_empty(self.revision, "revision"))
        ensure_aware(self.available_at)
        close_price = _decimal(self.close_price, "close_price")
        object.__setattr__(self, "close_price", close_price)
        if close_price <= 0:
            raise ValueError("benchmark close_price must be positive")
        if self.available_at.astimezone(SHANGHAI_TZ).date() < self.trading_day:
            raise ValueError("benchmark available_at cannot precede its trading_day")


@dataclass(frozen=True, slots=True)
class PortfolioWeight:
    """End-of-session marked portfolio weight for one held instrument."""

    instrument_id: str
    weight: Decimal

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if not self.weight.is_finite() or self.weight < 0:
            raise ValueError("portfolio weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class VectorTrade:
    """Auditable open-auction rebalance trade and its exact fee components."""

    signal_date: date
    execution_date: date
    instrument_id: str
    instrument_type: TradableInstrumentType
    side: Side
    execution_price: Decimal
    notional: Decimal
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    other_fee: Decimal
    total_fee: Decimal
    fee_rule_version: str | None

    def __post_init__(self) -> None:
        if self.execution_date <= self.signal_date:
            raise ValueError("trade execution_date must follow signal_date")
        if self.execution_price <= 0 or self.notional <= 0:
            raise ValueError("trade execution price and notional must be positive")
        components = (self.commission, self.stamp_duty, self.transfer_fee, self.other_fee)
        if any(value < 0 or not value.is_finite() for value in components):
            raise ValueError("trade fee components must be finite and non-negative")
        if self.total_fee != sum(components, _ZERO):
            raise ValueError("trade total_fee must equal its components")
        if self.fee_rule_version is not None and not self.fee_rule_version.strip():
            raise ValueError("fee_rule_version must be non-empty when present")


@dataclass(frozen=True, slots=True)
class DailyBacktestPoint:
    """End-of-session portfolio, benchmark, cost, and risk state."""

    trading_day: date
    nav: Decimal
    benchmark_nav: Decimal
    daily_return: Decimal
    benchmark_return: Decimal
    excess_return: Decimal
    drawdown: Decimal
    turnover: Decimal
    buy_fees: Decimal
    sell_fees: Decimal
    total_fees: Decimal
    cash: Decimal
    cash_weight: Decimal
    weights: tuple[PortfolioWeight, ...]
    executed_signal_date: date | None

    def __post_init__(self) -> None:
        finite_values = (
            self.nav,
            self.benchmark_nav,
            self.daily_return,
            self.benchmark_return,
            self.excess_return,
            self.drawdown,
            self.turnover,
            self.buy_fees,
            self.sell_fees,
            self.total_fees,
            self.cash,
            self.cash_weight,
        )
        if any(not value.is_finite() for value in finite_values):
            raise ValueError("daily backtest values must be finite")
        if self.nav <= 0 or self.benchmark_nav <= 0:
            raise ValueError("daily NAV values must be positive")
        if self.drawdown < 0 or self.drawdown >= 1:
            raise ValueError("drawdown must be between zero inclusive and one exclusive")
        if self.turnover < 0:
            raise ValueError("turnover cannot be negative")
        if self.cash < 0 or self.cash_weight < 0 or self.cash_weight > 1:
            raise ValueError("long-only research cash and cash_weight must be non-negative")
        if min(self.buy_fees, self.sell_fees, self.total_fees) < 0:
            raise ValueError("daily fees cannot be negative")
        if self.buy_fees + self.sell_fees != self.total_fees:
            raise ValueError("daily total_fees must equal buy_fees plus sell_fees")
        if tuple(sorted(self.weights, key=lambda item: item.instrument_id)) != self.weights:
            raise ValueError("daily weights must be sorted by instrument_id")
        if self.executed_signal_date is not None and self.executed_signal_date >= self.trading_day:
            raise ValueError("executed signal must precede the trading day")


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Standard daily research metrics with explicit zero-volatility Sharpe semantics."""

    total_return: Decimal
    benchmark_total_return: Decimal
    excess_return: Decimal
    annualized_return: Decimal
    annualized_volatility: Decimal
    sharpe_ratio: Decimal | None
    max_drawdown: Decimal
    total_turnover: Decimal
    average_daily_turnover: Decimal
    total_fees: Decimal
    observation_count: int

    def __post_init__(self) -> None:
        decimals = (
            self.total_return,
            self.benchmark_total_return,
            self.excess_return,
            self.annualized_return,
            self.annualized_volatility,
            self.max_drawdown,
            self.total_turnover,
            self.average_daily_turnover,
            self.total_fees,
        )
        if any(not value.is_finite() for value in decimals):
            raise ValueError("backtest metrics must be finite")
        if self.sharpe_ratio is not None and not self.sharpe_ratio.is_finite():
            raise ValueError("sharpe_ratio must be finite when defined")
        if self.annualized_volatility < 0 or self.max_drawdown < 0 or self.max_drawdown >= 1:
            raise ValueError("volatility must be non-negative and drawdown must be below one")
        if self.total_turnover < 0 or self.average_daily_turnover < 0 or self.total_fees < 0:
            raise ValueError("turnover and fees cannot be negative")
        if self.observation_count <= 0:
            raise ValueError("observation_count must be positive")
        if self.annualized_volatility == 0 and self.sharpe_ratio is not None:
            raise ValueError("zero-volatility Sharpe ratio is undefined and must be None")


@dataclass(frozen=True, slots=True)
class VectorizedBacktestResult:
    """Fully reproducible vectorized run output bound to canonical input and result hashes."""

    request: VectorizedBacktestRequest
    benchmark_id: str
    points: tuple[DailyBacktestPoint, ...]
    trades: tuple[VectorTrade, ...]
    metrics: BacktestMetrics
    input_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        if not self.points:
            raise ValueError("backtest result requires at least one daily point")
        if tuple(point.trading_day for point in self.points) != self.request.trading_calendar:
            raise ValueError("result points must align exactly to the request calendar")
        _validate_sha256(self.input_hash, "input_hash")
        _validate_sha256(self.result_hash, "result_hash")


@dataclass(frozen=True, slots=True)
class _Holding:
    instrument_type: TradableInstrumentType
    quantity: Decimal


def build_equal_weight_layer(
    signals: Iterable[FactorSignal],
    *,
    layer: int,
    layer_count: int,
    gross_weight: Decimal = Decimal(1),
) -> tuple[WeightSignal, ...]:
    """Rank each signal date deterministically and equal-weight one zero-based layer.

    Scores are sorted descending; ``instrument_id`` is the deterministic tie-breaker.
    Bucket boundaries use ``floor(rank * layer_count / item_count)`` so layer sizes
    differ by at most one when the universe is not evenly divisible.
    """

    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if layer < 0 or layer >= layer_count:
        raise ValueError("layer must be between zero and layer_count minus one")
    normalized_gross = _decimal(gross_weight, "gross_weight")
    if normalized_gross < 0 or normalized_gross > 1:
        raise ValueError("gross_weight must be between zero and one")

    grouped: dict[date, list[FactorSignal]] = {}
    for signal in tuple(signals):
        grouped.setdefault(signal.signal_date, []).append(signal)

    weights: list[WeightSignal] = []
    for signal_date in sorted(grouped):
        batch = grouped[signal_date]
        _validate_factor_batch(batch)
        ranked = sorted(batch, key=lambda item: (-item.score, item.instrument_id))
        if len(ranked) < layer_count:
            raise VectorizedBacktestError(
                f"signal date {signal_date.isoformat()} has fewer instruments than layers"
            )
        selected = [
            item for rank, item in enumerate(ranked) if rank * layer_count // len(ranked) == layer
        ]
        weight = normalized_gross / len(selected)
        weights.extend(
            WeightSignal(
                instrument_id=item.instrument_id,
                instrument_type=item.instrument_type,
                signal_date=item.signal_date,
                as_of=item.as_of,
                target_weight=weight,
                data_version=item.data_version,
                strategy_version=item.strategy_version,
            )
            for item in selected
        )
    return tuple(sorted(weights, key=lambda item: (item.signal_date, item.instrument_id)))


def _validate_factor_batch(batch: list[FactorSignal]) -> None:
    identities: set[str] = set()
    first = batch[0]
    for item in batch:
        if item.instrument_id in identities:
            raise VectorizedBacktestError(
                f"duplicate factor signal for {item.instrument_id} on {item.signal_date}"
            )
        identities.add(item.instrument_id)
        if (
            item.as_of != first.as_of
            or item.data_version != first.data_version
            or item.strategy_version != first.strategy_version
        ):
            raise VectorizedBacktestError(
                f"factor signals on {item.signal_date} must share as_of and versions"
            )


class VectorizedBacktestEngine:
    """Run a long-only daily research portfolio with next-session-open execution."""

    def __init__(self, *, fee_rule_book: FeeRuleBook | None = None) -> None:
        self._fee_rule_book = fee_rule_book

    def run(
        self,
        *,
        request: VectorizedBacktestRequest,
        signals: Iterable[WeightSignal],
        prices: Iterable[ResearchPriceBar],
        benchmark: Iterable[BenchmarkBar],
    ) -> VectorizedBacktestResult:
        """Validate, align, and evaluate one immutable research run."""

        signal_rows = self._select_signals(request, tuple(signals))
        price_rows = self._select_price_revisions(request, tuple(prices))
        benchmark_rows = self._select_benchmark_revisions(request, tuple(benchmark))
        signal_batches = self._validate_and_group_signals(request, signal_rows)
        price_map, instrument_types = self._validate_prices(request, price_rows, signal_rows)
        benchmark_id, benchmark_map = self._validate_benchmark(request, benchmark_rows)
        input_hash = self._input_hash(
            request=request,
            signals=signal_rows,
            prices=price_rows,
            benchmark=benchmark_rows,
        )

        points, trades = self._simulate(
            request=request,
            signal_batches=signal_batches,
            prices=price_map,
            instrument_types=instrument_types,
            benchmark=benchmark_map,
        )
        metrics = _metrics(request, points)
        result_hash = self._result_hash(
            request=request,
            benchmark_id=benchmark_id,
            points=points,
            trades=trades,
            metrics=metrics,
            input_hash=input_hash,
        )
        return VectorizedBacktestResult(
            request=request,
            benchmark_id=benchmark_id,
            points=points,
            trades=trades,
            metrics=metrics,
            input_hash=input_hash,
            result_hash=result_hash,
        )

    @staticmethod
    def _select_signals(
        request: VectorizedBacktestRequest,
        signals: tuple[WeightSignal, ...],
    ) -> tuple[WeightSignal, ...]:
        selected = tuple(item for item in signals if item.as_of <= request.as_of)
        return tuple(sorted(selected, key=lambda item: (item.signal_date, item.instrument_id)))

    @staticmethod
    def _select_price_revisions(
        request: VectorizedBacktestRequest,
        prices: tuple[ResearchPriceBar, ...],
    ) -> tuple[ResearchPriceBar, ...]:
        calendar = set(request.trading_calendar)
        selected: dict[tuple[str, date], ResearchPriceBar] = {}
        for item in prices:
            if item.trading_day not in calendar or item.available_at > request.as_of:
                continue
            key = (item.instrument_id, item.trading_day)
            current = selected.get(key)
            if current is None or item.available_at > current.available_at:
                selected[key] = item
            elif item.available_at == current.available_at and item != current:
                raise VectorizedBacktestError(
                    "ambiguous price revisions share instrument, date, and available_at"
                )
        return tuple(
            sorted(selected.values(), key=lambda item: (item.trading_day, item.instrument_id))
        )

    @staticmethod
    def _select_benchmark_revisions(
        request: VectorizedBacktestRequest,
        benchmark: tuple[BenchmarkBar, ...],
    ) -> tuple[BenchmarkBar, ...]:
        calendar = set(request.trading_calendar)
        selected: dict[tuple[str, date], BenchmarkBar] = {}
        for item in benchmark:
            if item.trading_day not in calendar or item.available_at > request.as_of:
                continue
            key = (item.benchmark_id, item.trading_day)
            current = selected.get(key)
            if current is None or item.available_at > current.available_at:
                selected[key] = item
            elif item.available_at == current.available_at and item != current:
                raise VectorizedBacktestError(
                    "ambiguous benchmark revisions share identifier, date, and available_at"
                )
        return tuple(
            sorted(selected.values(), key=lambda item: (item.trading_day, item.benchmark_id))
        )

    @staticmethod
    def _validate_and_group_signals(
        request: VectorizedBacktestRequest,
        signals: tuple[WeightSignal, ...],
    ) -> dict[date, tuple[WeightSignal, ...]]:
        calendar_index = {day: index for index, day in enumerate(request.trading_calendar)}
        grouped: dict[date, list[WeightSignal]] = {}
        for signal in signals:
            if signal.data_version != request.data_version:
                raise VectorizedBacktestError("signal data_version does not match request")
            if signal.strategy_version != request.strategy_version:
                raise VectorizedBacktestError("signal strategy_version does not match request")
            index = calendar_index.get(signal.signal_date)
            if index is None:
                raise VectorizedBacktestError("signal_date is absent from the trading calendar")
            if index == len(request.trading_calendar) - 1:
                raise VectorizedBacktestError(
                    "last-calendar-date signal has no next session to execute"
                )
            grouped.setdefault(signal.signal_date, []).append(signal)

        frozen: dict[date, tuple[WeightSignal, ...]] = {}
        for signal_date in sorted(grouped):
            batch = grouped[signal_date]
            first = batch[0]
            seen: set[str] = set()
            total_weight = _ZERO
            for signal in batch:
                if signal.instrument_id in seen:
                    raise VectorizedBacktestError(
                        f"duplicate weight signal for {signal.instrument_id} on {signal_date}"
                    )
                seen.add(signal.instrument_id)
                if signal.as_of != first.as_of:
                    raise VectorizedBacktestError(
                        f"weight signals on {signal_date} must share one as_of instant"
                    )
                total_weight += signal.target_weight
            if total_weight > 1:
                raise VectorizedBacktestError(f"target weights on {signal_date} exceed one")
            frozen[signal_date] = tuple(sorted(batch, key=lambda item: item.instrument_id))
        return frozen

    @staticmethod
    def _validate_prices(
        request: VectorizedBacktestRequest,
        prices: tuple[ResearchPriceBar, ...],
        signals: tuple[WeightSignal, ...],
    ) -> tuple[
        dict[tuple[str, date], ResearchPriceBar],
        dict[str, TradableInstrumentType],
    ]:
        price_map: dict[tuple[str, date], ResearchPriceBar] = {}
        instrument_types: dict[str, TradableInstrumentType] = {}
        for item in prices:
            if item.data_version != request.data_version:
                raise VectorizedBacktestError("price data_version does not match request")
            price_map[(item.instrument_id, item.trading_day)] = item
            known_type = instrument_types.setdefault(item.instrument_id, item.instrument_type)
            if known_type is not item.instrument_type:
                raise VectorizedBacktestError("instrument type changes across price history")
        for signal in signals:
            signal_type = instrument_types.get(signal.instrument_id)
            if signal_type is not None and signal_type is not signal.instrument_type:
                raise VectorizedBacktestError("signal and price instrument types do not match")
            instrument_types.setdefault(signal.instrument_id, signal.instrument_type)
        return price_map, instrument_types

    @staticmethod
    def _validate_benchmark(
        request: VectorizedBacktestRequest,
        benchmark: tuple[BenchmarkBar, ...],
    ) -> tuple[str, dict[date, BenchmarkBar]]:
        identifiers = {item.benchmark_id for item in benchmark}
        if len(identifiers) != 1:
            raise VectorizedBacktestError("benchmark must contain exactly one identifier")
        benchmark_id = next(iter(identifiers))
        by_date = {item.trading_day: item for item in benchmark}
        if tuple(sorted(by_date)) != request.trading_calendar:
            raise VectorizedBacktestError(
                "benchmark dates must align exactly to the trading calendar; filling is forbidden"
            )
        if any(item.data_version != request.data_version for item in benchmark):
            raise VectorizedBacktestError("benchmark data_version does not match request")
        return benchmark_id, by_date

    def _simulate(
        self,
        *,
        request: VectorizedBacktestRequest,
        signal_batches: dict[date, tuple[WeightSignal, ...]],
        prices: dict[tuple[str, date], ResearchPriceBar],
        instrument_types: dict[str, TradableInstrumentType],
        benchmark: dict[date, BenchmarkBar],
    ) -> tuple[tuple[DailyBacktestPoint, ...], tuple[VectorTrade, ...]]:
        holdings: dict[str, _Holding] = {}
        cash = request.initial_nav
        nav = request.initial_nav
        peak_nav = nav
        benchmark_nav = request.initial_nav
        previous_benchmark_close: Decimal | None = None
        points: list[DailyBacktestPoint] = []
        trades: list[VectorTrade] = []
        calendar = request.trading_calendar

        for index, trading_day in enumerate(calendar):
            previous_nav = nav
            benchmark_bar = benchmark[trading_day]
            if previous_benchmark_close is None:
                benchmark_return = _ZERO
            else:
                benchmark_return = benchmark_bar.close_price / previous_benchmark_close - 1
                benchmark_nav *= _ONE + benchmark_return
            previous_benchmark_close = benchmark_bar.close_price

            executed_signal_date = calendar[index - 1] if index > 0 else None
            batch = signal_batches.get(executed_signal_date) if executed_signal_date else None
            day_trades: list[VectorTrade] = []
            turnover = _ZERO
            if batch is not None:
                if executed_signal_date is None:  # pragma: no cover - guarded by calendar index
                    raise AssertionError("execution batch has no prior signal date")
                pretrade_nav = self._portfolio_value(
                    trading_day=trading_day,
                    cash=cash,
                    holdings=holdings,
                    prices=prices,
                    at_open=True,
                )
                if pretrade_nav <= 0:
                    raise VectorizedBacktestError("portfolio NAV is non-positive before rebalance")
                targets = {item.instrument_id: item.target_weight for item in batch}
                union = sorted(set(holdings) | set(targets))
                target_quantities = {
                    instrument_id: holding.quantity for instrument_id, holding in holdings.items()
                }
                sells: list[tuple[str, TradableInstrumentType, Decimal, Decimal, Decimal]] = []
                buys: list[tuple[str, TradableInstrumentType, Decimal, Decimal, Decimal]] = []
                for instrument_id in union:
                    bar = self._required_price(prices, instrument_id, trading_day)
                    current = holdings.get(instrument_id)
                    current_quantity = current.quantity if current is not None else _ZERO
                    current_notional = current_quantity * bar.open_price
                    desired_notional = pretrade_nav * targets.get(instrument_id, _ZERO)
                    desired_quantity = (
                        desired_notional / bar.open_price if desired_notional > 0 else _ZERO
                    )
                    delta_notional = desired_notional - current_notional
                    if delta_notional < 0:
                        sells.append(
                            (
                                instrument_id,
                                instrument_types[instrument_id],
                                bar.open_price,
                                -delta_notional,
                                desired_quantity,
                            )
                        )
                    elif delta_notional > 0:
                        buys.append(
                            (
                                instrument_id,
                                instrument_types[instrument_id],
                                bar.open_price,
                                delta_notional,
                                current_quantity,
                            )
                        )

                # Sales settle into research cash before purchases. This prevents
                # ordering by instrument ID from creating accidental intraday leverage.
                for instrument_id, instrument_type, price, notional, desired_quantity in sells:
                    trade = self._trade(
                        signal_date=executed_signal_date,
                        execution_date=trading_day,
                        instrument_id=instrument_id,
                        instrument_type=instrument_type,
                        side=Side.SELL,
                        execution_price=price,
                        notional=notional,
                    )
                    day_trades.append(trade)
                    cash += trade.notional - trade.total_fee
                    target_quantities[instrument_id] = desired_quantity
                if cash < 0:
                    raise VectorizedBacktestError(
                        "sell proceeds and existing cash cannot cover transaction fees"
                    )

                buy_scale = self._affordable_buy_scale(
                    signal_date=executed_signal_date,
                    execution_date=trading_day,
                    buys=buys,
                    available_cash=cash,
                )
                if buys and buy_scale == 0:
                    raise VectorizedBacktestError("available cash cannot cover the minimum buy fee")
                for (
                    instrument_id,
                    instrument_type,
                    price,
                    desired_notional,
                    current_quantity,
                ) in buys:
                    actual_notional = desired_notional * buy_scale
                    if actual_notional == 0:
                        continue
                    trade = self._trade(
                        signal_date=executed_signal_date,
                        execution_date=trading_day,
                        instrument_id=instrument_id,
                        instrument_type=instrument_type,
                        side=Side.BUY,
                        execution_price=price,
                        notional=actual_notional,
                    )
                    maximum_notional = cash - trade.total_fee
                    if trade.notional > maximum_notional:
                        if maximum_notional <= 0:
                            raise VectorizedBacktestError(
                                "available cash cannot cover the minimum buy fee"
                            )
                        trade = self._trade(
                            signal_date=executed_signal_date,
                            execution_date=trading_day,
                            instrument_id=instrument_id,
                            instrument_type=instrument_type,
                            side=Side.BUY,
                            execution_price=price,
                            notional=maximum_notional,
                        )
                    day_trades.append(trade)
                    cash -= trade.notional + trade.total_fee
                    target_quantities[instrument_id] = current_quantity + (trade.notional / price)
                if cash < 0:
                    raise AssertionError("fee-aware buy scaling produced negative cash")
                holdings = {
                    instrument_id: _Holding(
                        instrument_type=instrument_types[instrument_id],
                        quantity=quantity,
                    )
                    for instrument_id, quantity in target_quantities.items()
                }
                total_fees = sum((item.total_fee for item in day_trades), _ZERO)
                posttrade_value = cash + sum(
                    (
                        holding.quantity
                        * self._required_price(prices, instrument_id, trading_day).open_price
                        for instrument_id, holding in holdings.items()
                    ),
                    _ZERO,
                )
                if posttrade_value != pretrade_nav - total_fees:
                    raise AssertionError("rebalance cash flows do not conserve NAV")
                turnover = sum((item.notional for item in day_trades), _ZERO) / pretrade_nav
                trades.extend(day_trades)

            nav = self._portfolio_value(
                trading_day=trading_day,
                cash=cash,
                holdings=holdings,
                prices=prices,
                at_open=False,
            )
            if nav <= 0:
                raise VectorizedBacktestError("portfolio NAV became non-positive")
            daily_return = _ZERO if index == 0 else nav / previous_nav - 1
            peak_nav = max(peak_nav, nav)
            drawdown = _ONE - nav / peak_nav
            marked_weights = tuple(
                PortfolioWeight(
                    instrument_id=instrument_id,
                    weight=(
                        holding.quantity
                        * self._required_price(prices, instrument_id, trading_day).close_price
                        / nav
                    ),
                )
                for instrument_id, holding in sorted(holdings.items())
            )
            buy_fees = sum(
                (item.total_fee for item in day_trades if item.side is Side.BUY),
                _ZERO,
            )
            sell_fees = sum(
                (item.total_fee for item in day_trades if item.side is Side.SELL),
                _ZERO,
            )
            total_fees = buy_fees + sell_fees
            points.append(
                DailyBacktestPoint(
                    trading_day=trading_day,
                    nav=nav,
                    benchmark_nav=benchmark_nav,
                    daily_return=daily_return,
                    benchmark_return=benchmark_return,
                    excess_return=daily_return - benchmark_return,
                    drawdown=drawdown,
                    turnover=turnover,
                    buy_fees=buy_fees,
                    sell_fees=sell_fees,
                    total_fees=total_fees,
                    cash=cash,
                    cash_weight=cash / nav,
                    weights=marked_weights,
                    executed_signal_date=executed_signal_date if batch is not None else None,
                )
            )
        return tuple(points), tuple(trades)

    def _affordable_buy_scale(
        self,
        *,
        signal_date: date,
        execution_date: date,
        buys: list[tuple[str, TradableInstrumentType, Decimal, Decimal, Decimal]],
        available_cash: Decimal,
    ) -> Decimal:
        """Scale all purchases proportionally so notional plus fees never borrow cash."""

        if not buys:
            return _ONE
        if available_cash <= 0:
            return _ZERO

        def cost(scale: Decimal) -> Decimal:
            result = _ZERO
            for instrument_id, instrument_type, price, desired_notional, _ in buys:
                notional = desired_notional * scale
                if notional == 0:
                    continue
                trade = self._trade(
                    signal_date=signal_date,
                    execution_date=execution_date,
                    instrument_id=instrument_id,
                    instrument_type=instrument_type,
                    side=Side.BUY,
                    execution_price=price,
                    notional=notional,
                )
                result += trade.notional + trade.total_fee
            return result

        if cost(_ONE) <= available_cash:
            return _ONE
        low = _ZERO
        high = _ONE
        # Decimal's active precision makes this deterministic; stop when another
        # midpoint is no longer representable rather than using a float tolerance.
        for _ in range(128):
            midpoint = (low + high) / 2
            if midpoint in {low, high}:
                break
            if cost(midpoint) <= available_cash:
                low = midpoint
            else:
                high = midpoint
        desired_total = sum((item[3] for item in buys), _ZERO)
        fees_at_low = cost(low) - desired_total * low
        exact_candidate = (available_cash - fees_at_low) / desired_total
        if low <= exact_candidate <= _ONE and cost(exact_candidate) <= available_cash:
            return exact_candidate
        return low

    @staticmethod
    def _required_price(
        prices: dict[tuple[str, date], ResearchPriceBar],
        instrument_id: str,
        trading_day: date,
    ) -> ResearchPriceBar:
        try:
            return prices[(instrument_id, trading_day)]
        except KeyError as error:
            raise VectorizedBacktestError(
                f"missing price for held or targeted instrument {instrument_id} on {trading_day}"
            ) from error

    @classmethod
    def _portfolio_value(
        cls,
        *,
        trading_day: date,
        cash: Decimal,
        holdings: dict[str, _Holding],
        prices: dict[tuple[str, date], ResearchPriceBar],
        at_open: bool,
    ) -> Decimal:
        value = cash
        for instrument_id, holding in holdings.items():
            bar = cls._required_price(prices, instrument_id, trading_day)
            price = bar.open_price if at_open else bar.close_price
            value += holding.quantity * price
        return value

    def _trade(
        self,
        *,
        signal_date: date,
        execution_date: date,
        instrument_id: str,
        instrument_type: TradableInstrumentType,
        side: Side,
        execution_price: Decimal,
        notional: Decimal,
    ) -> VectorTrade:
        if self._fee_rule_book is None:
            commission = stamp_duty = transfer_fee = other_fee = total_fee = _ZERO
            fee_rule_version = None
        else:
            rule = self._fee_rule_book.select(
                instrument_type=instrument_type,
                side=side,
                trading_day=execution_date,
            )
            fee = rule.assess(notional)
            commission = fee.commission
            stamp_duty = fee.stamp_duty
            transfer_fee = fee.transfer_fee
            other_fee = fee.other_fee
            total_fee = fee.total_amount
            fee_rule_version = rule.version
        return VectorTrade(
            signal_date=signal_date,
            execution_date=execution_date,
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            side=side,
            execution_price=execution_price,
            notional=notional,
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            other_fee=other_fee,
            total_fee=total_fee,
            fee_rule_version=fee_rule_version,
        )

    @staticmethod
    def _input_hash(
        *,
        request: VectorizedBacktestRequest,
        signals: tuple[WeightSignal, ...],
        prices: tuple[ResearchPriceBar, ...],
        benchmark: tuple[BenchmarkBar, ...],
    ) -> str:
        return _stable_hash(
            {
                "request": _request_payload(request),
                "signals": [_signal_payload(item) for item in signals],
                "prices": [_price_payload(item) for item in prices],
                "benchmark": [_benchmark_payload(item) for item in benchmark],
            }
        )

    @staticmethod
    def _result_hash(
        *,
        request: VectorizedBacktestRequest,
        benchmark_id: str,
        points: tuple[DailyBacktestPoint, ...],
        trades: tuple[VectorTrade, ...],
        metrics: BacktestMetrics,
        input_hash: str,
    ) -> str:
        return _stable_hash(
            {
                "benchmark_id": benchmark_id,
                "input_hash": input_hash,
                "metrics": _metrics_payload(metrics),
                "points": [_point_payload(item) for item in points],
                "request": _request_payload(request),
                "trades": [_trade_payload(item) for item in trades],
            }
        )


def _metrics(
    request: VectorizedBacktestRequest,
    points: tuple[DailyBacktestPoint, ...],
) -> BacktestMetrics:
    total_return = points[-1].nav / request.initial_nav - 1
    benchmark_total_return = points[-1].benchmark_nav / request.initial_nav - 1
    period_returns = tuple(point.daily_return for point in points[1:])
    period_count = len(period_returns)
    annualized_return = _annualized_return(
        total_return=total_return,
        period_count=period_count,
        annualization_periods=request.annualization_periods,
    )
    annualized_volatility = _annualized_volatility(
        period_returns,
        request.annualization_periods,
    )
    sharpe_ratio = _sharpe_ratio(
        period_returns=period_returns,
        annualized_volatility=annualized_volatility,
        annual_risk_free_rate=request.annual_risk_free_rate,
        annualization_periods=request.annualization_periods,
    )
    total_turnover = sum((point.turnover for point in points), _ZERO)
    average_daily_turnover = total_turnover / len(points)
    return BacktestMetrics(
        total_return=total_return,
        benchmark_total_return=benchmark_total_return,
        excess_return=total_return - benchmark_total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max(point.drawdown for point in points),
        total_turnover=total_turnover,
        average_daily_turnover=average_daily_turnover,
        total_fees=sum((point.total_fees for point in points), _ZERO),
        observation_count=len(points),
    )


def _annualized_return(
    *,
    total_return: Decimal,
    period_count: int,
    annualization_periods: int,
) -> Decimal:
    if period_count == 0 or total_return == 0:
        return _ZERO
    growth = _ONE + total_return
    if growth <= 0:
        raise VectorizedBacktestError("annualized return requires positive terminal growth")
    with localcontext() as context:
        context.prec = 34
        context.rounding = ROUND_HALF_EVEN
        exponent = Decimal(annualization_periods) / period_count
        return (growth.ln() * exponent).exp() - 1


def _annualized_volatility(
    returns: tuple[Decimal, ...],
    annualization_periods: int,
) -> Decimal:
    if len(returns) < 2:
        return _ZERO
    with localcontext() as context:
        context.prec = 34
        context.rounding = ROUND_HALF_EVEN
        mean = sum(returns, _ZERO) / len(returns)
        variance = sum(((value - mean) ** 2 for value in returns), _ZERO) / (len(returns) - 1)
        return (variance * annualization_periods).sqrt()


def _sharpe_ratio(
    *,
    period_returns: tuple[Decimal, ...],
    annualized_volatility: Decimal,
    annual_risk_free_rate: Decimal,
    annualization_periods: int,
) -> Decimal | None:
    if not period_returns or annualized_volatility == 0:
        return None
    with localcontext() as context:
        context.prec = 34
        context.rounding = ROUND_HALF_EVEN
        mean_return = sum(period_returns, _ZERO) / len(period_returns)
        periodic_risk_free = annual_risk_free_rate / annualization_periods
        annualized_excess = (mean_return - periodic_risk_free) * annualization_periods
        return annualized_excess / annualized_volatility


def _request_payload(request: VectorizedBacktestRequest) -> dict[str, Any]:
    return {
        "annual_risk_free_rate": request.annual_risk_free_rate,
        "annualization_periods": request.annualization_periods,
        "as_of": request.as_of,
        "data_version": request.data_version,
        "initial_nav": request.initial_nav,
        "run_id": request.run_id,
        "strategy_version": request.strategy_version,
        "trading_calendar": request.trading_calendar,
    }


def _signal_payload(signal: WeightSignal) -> dict[str, Any]:
    return {
        "as_of": signal.as_of,
        "data_version": signal.data_version,
        "instrument_id": signal.instrument_id,
        "instrument_type": signal.instrument_type,
        "signal_date": signal.signal_date,
        "strategy_version": signal.strategy_version,
        "target_weight": signal.target_weight,
    }


def _price_payload(bar: ResearchPriceBar) -> dict[str, Any]:
    return {
        "available_at": bar.available_at,
        "close_price": bar.close_price,
        "data_version": bar.data_version,
        "instrument_id": bar.instrument_id,
        "instrument_type": bar.instrument_type,
        "open_price": bar.open_price,
        "revision": bar.revision,
        "trading_day": bar.trading_day,
    }


def _benchmark_payload(bar: BenchmarkBar) -> dict[str, Any]:
    return {
        "available_at": bar.available_at,
        "benchmark_id": bar.benchmark_id,
        "close_price": bar.close_price,
        "data_version": bar.data_version,
        "revision": bar.revision,
        "trading_day": bar.trading_day,
    }


def _trade_payload(trade: VectorTrade) -> dict[str, Any]:
    return {
        "commission": trade.commission,
        "execution_date": trade.execution_date,
        "execution_price": trade.execution_price,
        "fee_rule_version": trade.fee_rule_version,
        "instrument_id": trade.instrument_id,
        "instrument_type": trade.instrument_type,
        "notional": trade.notional,
        "other_fee": trade.other_fee,
        "side": trade.side,
        "signal_date": trade.signal_date,
        "stamp_duty": trade.stamp_duty,
        "total_fee": trade.total_fee,
        "transfer_fee": trade.transfer_fee,
    }


def _point_payload(point: DailyBacktestPoint) -> dict[str, Any]:
    return {
        "benchmark_nav": point.benchmark_nav,
        "benchmark_return": point.benchmark_return,
        "buy_fees": point.buy_fees,
        "cash": point.cash,
        "cash_weight": point.cash_weight,
        "daily_return": point.daily_return,
        "drawdown": point.drawdown,
        "excess_return": point.excess_return,
        "executed_signal_date": point.executed_signal_date,
        "nav": point.nav,
        "sell_fees": point.sell_fees,
        "total_fees": point.total_fees,
        "trading_day": point.trading_day,
        "turnover": point.turnover,
        "weights": [
            {"instrument_id": item.instrument_id, "weight": item.weight} for item in point.weights
        ],
    }


def _metrics_payload(metrics: BacktestMetrics) -> dict[str, Any]:
    return {
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "average_daily_turnover": metrics.average_daily_turnover,
        "benchmark_total_return": metrics.benchmark_total_return,
        "excess_return": metrics.excess_return,
        "max_drawdown": metrics.max_drawdown,
        "observation_count": metrics.observation_count,
        "sharpe_ratio": metrics.sharpe_ratio,
        "total_fees": metrics.total_fees,
        "total_return": metrics.total_return,
        "total_turnover": metrics.total_turnover,
    }


__all__ = [
    "BacktestMetrics",
    "BenchmarkBar",
    "DailyBacktestPoint",
    "FactorSignal",
    "PortfolioWeight",
    "ResearchPriceBar",
    "VectorTrade",
    "VectorizedBacktestEngine",
    "VectorizedBacktestError",
    "VectorizedBacktestRequest",
    "VectorizedBacktestResult",
    "WeightSignal",
    "build_equal_weight_layer",
]
