"""Deterministic in-memory provider for tests and local development."""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, TypeVar

from quant_agent.data.domain import (
    AdjustmentFactor,
    DailyBar,
    FundamentalPoint,
    IndustryMembership,
    Instrument,
    TradingDay,
)
from quant_agent.data.providers.base import ProviderBatch

T = TypeVar("T")


class FakeMarketDataProvider:
    """Return preloaded records without network access."""

    def __init__(
        self,
        *,
        available_at: datetime,
        instruments: Sequence[Instrument] = (),
        trading_days: Sequence[TradingDay] = (),
        daily_bars: Sequence[DailyBar] = (),
        adjustment_factors: Sequence[AdjustmentFactor] = (),
        fundamentals: Sequence[FundamentalPoint] = (),
        industry_memberships: Sequence[IndustryMembership] = (),
    ) -> None:
        self.available_at = available_at
        self.instruments = tuple(instruments)
        self.trading_days = tuple(trading_days)
        self.daily_bars = tuple(daily_bars)
        self.adjustment_factors = tuple(adjustment_factors)
        self.fundamentals = tuple(fundamentals)
        self.industry_memberships = tuple(industry_memberships)

    def _batch(
        self,
        endpoint: str,
        params: dict[str, Any],
        records: Sequence[T],
    ) -> ProviderBatch[T]:
        return ProviderBatch(
            provider="fake",
            endpoint=endpoint,
            request_params=params,
            raw_payload={"endpoint": endpoint, "count": len(records)},
            available_at=self.available_at,
            records=tuple(records),
        )

    def fetch_instruments(self, as_of: date) -> ProviderBatch[Instrument]:
        return self._batch("instruments", {"as_of": as_of.isoformat()}, self.instruments)

    def fetch_trading_calendar(
        self,
        market: str,
        start: date,
        end: date,
    ) -> ProviderBatch[TradingDay]:
        records = [
            item
            for item in self.trading_days
            if item.market == market and start <= item.trade_date <= end
        ]
        return self._batch(
            "trading_calendar",
            {"market": market, "start": start.isoformat(), "end": end.isoformat()},
            records,
        )

    def fetch_daily_bars(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
    ) -> ProviderBatch[DailyBar]:
        allowed = set(instrument_ids) if instrument_ids is not None else None
        records = [
            item
            for item in self.daily_bars
            if item.trade_date == trade_date and (allowed is None or item.instrument_id in allowed)
        ]
        return self._batch("daily_bars", {"trade_date": trade_date.isoformat()}, records)

    def fetch_adjustment_factors(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
    ) -> ProviderBatch[AdjustmentFactor]:
        allowed = set(instrument_ids) if instrument_ids is not None else None
        records = [
            item
            for item in self.adjustment_factors
            if item.trade_date == trade_date and (allowed is None or item.instrument_id in allowed)
        ]
        return self._batch(
            "adjustment_factors",
            {"trade_date": trade_date.isoformat()},
            records,
        )

    def fetch_fundamentals(
        self,
        instrument_ids: Sequence[str],
        as_of: datetime,
    ) -> ProviderBatch[FundamentalPoint]:
        allowed = set(instrument_ids)
        records = [
            item
            for item in self.fundamentals
            if item.instrument_id in allowed and item.available_at <= as_of
        ]
        return self._batch(
            "fundamentals",
            {"instrument_ids": sorted(allowed), "as_of": as_of.isoformat()},
            records,
        )

    def fetch_industry_memberships(
        self,
        as_of: date,
    ) -> ProviderBatch[IndustryMembership]:
        records = [
            item
            for item in self.industry_memberships
            if item.effective_from <= as_of
            and (item.effective_to is None or item.effective_to >= as_of)
        ]
        return self._batch(
            "industry_memberships",
            {"as_of": as_of.isoformat()},
            records,
        )
