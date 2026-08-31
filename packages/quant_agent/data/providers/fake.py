"""Deterministic in-memory provider for tests and local development."""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, TypeVar

from quant_agent.data.domain import (
    AdjustmentFactor,
    DailyBar,
    FundamentalPoint,
    IndexConstituentWeight,
    Industry,
    IndustryMembership,
    Instrument,
    InstrumentType,
    TradingDay,
)
from quant_agent.data.providers.base import ProviderBatch, ProviderError

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
        index_constituent_weights: Sequence[IndexConstituentWeight] = (),
        fundamentals: Sequence[FundamentalPoint] = (),
        industries: Sequence[Industry] = (),
        industry_memberships: Sequence[IndustryMembership] = (),
    ) -> None:
        self.available_at = available_at
        self.instruments = tuple(instruments)
        self.trading_days = tuple(trading_days)
        self.daily_bars = tuple(daily_bars)
        self.adjustment_factors = tuple(adjustment_factors)
        self.index_constituent_weights = tuple(index_constituent_weights)
        self.fundamentals = tuple(fundamentals)
        self.industries = tuple(industries)
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

    def fetch_instruments(
        self,
        as_of: date,
        *,
        instrument_type: InstrumentType = InstrumentType.STOCK,
        market: str | None = None,
    ) -> ProviderBatch[Instrument]:
        records = [
            item
            for item in self.instruments
            if item.instrument_type is instrument_type
            and (market is None or item.exchange == market)
        ]
        return self._batch(
            "instruments",
            {
                "as_of": as_of.isoformat(),
                "instrument_type": instrument_type.value,
                "market": market,
            },
            records,
        )

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
        *,
        instrument_type: InstrumentType = InstrumentType.STOCK,
    ) -> ProviderBatch[DailyBar]:
        allowed = set(instrument_ids) if instrument_ids is not None else None
        instrument_types = {item.instrument_id: item.instrument_type for item in self.instruments}
        records = [
            item
            for item in self.daily_bars
            if item.trade_date == trade_date
            and (allowed is None or item.instrument_id in allowed)
            and instrument_types.get(item.instrument_id, InstrumentType.STOCK) is instrument_type
        ]
        return self._batch(
            "daily_bars",
            {
                "trade_date": trade_date.isoformat(),
                "instrument_type": instrument_type.value,
            },
            records,
        )

    def fetch_adjustment_factors(
        self,
        trade_date: date,
        instrument_ids: Sequence[str] | None = None,
        *,
        instrument_type: InstrumentType = InstrumentType.STOCK,
    ) -> ProviderBatch[AdjustmentFactor]:
        if instrument_type is not InstrumentType.STOCK:
            raise ProviderError(
                "fake",
                "adjustment_factors",
                "adjustment factors support STOCK instruments only",
                retryable=False,
            )
        allowed = set(instrument_ids) if instrument_ids is not None else None
        records = [
            item
            for item in self.adjustment_factors
            if item.trade_date == trade_date and (allowed is None or item.instrument_id in allowed)
        ]
        return self._batch(
            "adjustment_factors",
            {
                "trade_date": trade_date.isoformat(),
                "instrument_type": instrument_type.value,
            },
            records,
        )

    def fetch_fundamentals(
        self,
        instrument_ids: Sequence[str],
        as_of: datetime,
        *,
        start_date: date,
    ) -> ProviderBatch[FundamentalPoint]:
        allowed = set(instrument_ids)
        records = [
            item
            for item in self.fundamentals
            if item.instrument_id in allowed
            and start_date <= item.report_period <= as_of.date()
            and item.available_at <= as_of
        ]
        return self._batch(
            "fundamentals",
            {
                "instrument_ids": sorted(allowed),
                "as_of": as_of.isoformat(),
                "start_date": start_date.isoformat(),
            },
            records,
        )

    def fetch_index_constituent_weights(
        self,
        index_id: str,
        start: date,
        end: date,
    ) -> ProviderBatch[IndexConstituentWeight]:
        records = [
            item
            for item in self.index_constituent_weights
            if item.index_instrument_id == index_id and start <= item.trade_date <= end
        ]
        return self._batch(
            "index_constituent_weights",
            {"index_id": index_id, "start": start.isoformat(), "end": end.isoformat()},
            records,
        )

    def fetch_industries(
        self,
        as_of: date,
        classification: str,
        level: int | None = None,
    ) -> ProviderBatch[Industry]:
        records = [
            item
            for item in self.industries
            if item.classification == classification and (level is None or item.level == level)
        ]
        return self._batch(
            "industries",
            {
                "as_of": as_of.isoformat(),
                "classification": classification,
                "level": level,
            },
            records,
        )

    def fetch_industry_memberships(
        self,
        as_of: date,
        *,
        classification: str,
        l3_codes: Sequence[str],
    ) -> ProviderBatch[IndustryMembership]:
        selected_codes = set(l3_codes)
        selected_industry_ids = {
            item.industry_id
            for item in self.industries
            if item.classification == classification
            and item.level == 3
            and (
                item.code in selected_codes
                or item.industry_id in selected_codes
                or item.industry_id.removeprefix(f"{classification}:") in selected_codes
            )
        }
        parents = {item.industry_id: item.parent_id for item in self.industries}
        pending = list(selected_industry_ids)
        while pending:
            parent_id = parents.get(pending.pop())
            if parent_id is not None and parent_id not in selected_industry_ids:
                selected_industry_ids.add(parent_id)
                pending.append(parent_id)
        if not selected_industry_ids:
            selected_industry_ids = selected_codes
        records = [
            item
            for item in self.industry_memberships
            if item.industry_id in selected_industry_ids and item.effective_from <= as_of
        ]
        return self._batch(
            "industry_memberships",
            {
                "as_of": as_of.isoformat(),
                "classification": classification,
                "l3_codes": sorted(selected_codes),
            },
            records,
        )
