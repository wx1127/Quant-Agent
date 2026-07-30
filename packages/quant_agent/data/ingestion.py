"""Idempotent raw payload and curated daily-bar ingestion."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_agent.data.domain import DailyBar
from quant_agent.data.master import InstrumentMasterService, TradingCalendarService
from quant_agent.data.models import DailyBarRow, RawPayloadRow
from quant_agent.data.providers.base import ProviderBatch


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Counts and raw-payload identity from one ingestion."""

    inserted: int
    skipped: int
    raw_payload_hash: str


class RawPayloadArchive:
    """Store immutable provider responses once by request fingerprint."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def archive(self, batch: ProviderBatch[Any]) -> str:
        payload_bytes = _canonical_json(batch.raw_payload)
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        request_fingerprint = hashlib.sha256(
            _canonical_json(
                {
                    "provider": batch.provider,
                    "endpoint": batch.endpoint,
                    "params": batch.request_params,
                    "payload_hash": payload_hash,
                }
            )
        ).hexdigest()
        existing = self._session.scalar(
            select(RawPayloadRow).where(RawPayloadRow.request_fingerprint == request_fingerprint)
        )
        if existing is None:
            self._session.add(
                RawPayloadRow(
                    provider=batch.provider,
                    endpoint=batch.endpoint,
                    request_fingerprint=request_fingerprint,
                    payload_hash=payload_hash,
                    payload=batch.raw_payload,
                    available_at=batch.available_at,
                )
            )
            self._session.flush()
        elif existing.payload_hash != payload_hash:
            raise ValueError("raw payload fingerprint collision")
        return payload_hash


class DailyBarIngestionService:
    """Validate master/calendar state and persist unadjusted daily bars."""

    def __init__(self, session: Session, *, market: str = "SSE") -> None:
        self._session = session
        self._market = market
        self._archive = RawPayloadArchive(session)
        self._master = InstrumentMasterService(session)
        self._calendar = TradingCalendarService(session)

    def ingest(self, batch: ProviderBatch[DailyBar]) -> IngestionResult:
        payload_hash = self._archive.archive(batch)
        inserted = 0
        skipped = 0
        for bar in batch.records:
            if not self._calendar.is_trading_day(self._market, bar.trade_date):
                raise ValueError(f"{bar.trade_date} is not an open {self._market} day")
            if not self._master.is_active(bar.instrument_id, bar.trade_date):
                raise ValueError(f"{bar.instrument_id} is not active on {bar.trade_date}")
            existing = self._session.scalar(
                select(DailyBarRow.id).where(
                    DailyBarRow.instrument_id == bar.instrument_id,
                    DailyBarRow.trade_date == bar.trade_date,
                    DailyBarRow.source == bar.source,
                    DailyBarRow.version == bar.version,
                )
            )
            if existing is not None:
                skipped += 1
                continue
            self._session.add(
                DailyBarRow(
                    instrument_id=bar.instrument_id,
                    trade_date=bar.trade_date,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    source=bar.source,
                    available_at=bar.available_at,
                    version=bar.version,
                )
            )
            inserted += 1
        self._session.flush()
        return IngestionResult(
            inserted=inserted,
            skipped=skipped,
            raw_payload_hash=payload_hash,
        )
