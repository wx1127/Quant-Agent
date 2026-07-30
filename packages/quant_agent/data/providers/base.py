"""Provider contracts that isolate third-party APIs from domain services."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol, TypeVar

from quant_agent.data.domain import (
    AdjustmentFactor,
    DailyBar,
    FundamentalPoint,
    IndustryMembership,
    Instrument,
    TradingDay,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProviderRequestPolicy:
    """Common timeout and bounded retry convention."""

    timeout_seconds: float = 15.0
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds cannot be negative")


class ProviderError(RuntimeError):
    """Typed provider failure that cannot be mistaken for an empty dataset."""

    def __init__(
        self,
        provider: str,
        endpoint: str,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.endpoint = endpoint
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class ProviderBatch[T]:
    """Records plus immutable raw response metadata."""

    provider: str
    endpoint: str
    request_params: dict[str, Any]
    raw_payload: dict[str, Any]
    available_at: datetime
    records: tuple[T, ...]


class MarketDataProvider(Protocol):
    """Provider-neutral P1 market-data interface."""

    def fetch_instruments(self, as_of: date) -> ProviderBatch[Instrument]: ...

    def fetch_trading_calendar(
        self,
        market: str,
        start: date,
        end: date,
    ) -> ProviderBatch[TradingDay]: ...

    def fetch_daily_bars(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
    ) -> ProviderBatch[DailyBar]: ...

    def fetch_adjustment_factors(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
    ) -> ProviderBatch[AdjustmentFactor]: ...

    def fetch_fundamentals(
        self,
        instrument_ids: Sequence[str],
        as_of: datetime,
    ) -> ProviderBatch[FundamentalPoint]: ...

    def fetch_industry_memberships(
        self,
        as_of: date,
    ) -> ProviderBatch[IndustryMembership]: ...
