"""Idempotent raw payload and curated daily-bar ingestion."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_agent.core.time import SHANGHAI_TZ
from quant_agent.data.domain import DailyBar
from quant_agent.data.master import InstrumentMasterService, TradingCalendarService
from quant_agent.data.models import DailyBarRow, RawPayloadRow
from quant_agent.data.providers.base import ProviderBatch
from quant_agent.data.sync.contracts import RawPage
from quant_agent.data.sync.hashing import (
    JsonValue,
    canonical_request_hash,
    sanitized_request_params,
)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Counts and raw-payload identity from one ingestion."""

    inserted: int
    skipped: int
    raw_payload_hash: str


@dataclass(frozen=True, slots=True)
class RawArchiveResult:
    """Database identity and content identities for one archived raw response."""

    raw_payload_id: int
    payload_hash: str
    request_hash: str


class RawPayloadArchive:
    """Store immutable provider responses once by request fingerprint."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _store(
        self,
        *,
        provider: str,
        endpoint: str,
        request_fingerprint: str,
        request_hash: str,
        request_params: dict[str, Any],
        payload: dict[str, Any],
        payload_bytes: bytes,
        fetched_at: datetime,
        available_at: datetime,
        schema_version: str,
    ) -> RawArchiveResult:
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        existing = self._session.scalar(
            select(RawPayloadRow).where(RawPayloadRow.request_fingerprint == request_fingerprint)
        )
        if existing is None:
            existing = RawPayloadRow(
                provider=provider,
                endpoint=endpoint,
                request_fingerprint=request_fingerprint,
                request_hash=request_hash,
                request_params=request_params,
                payload_hash=payload_hash,
                payload=payload,
                fetched_at=fetched_at.astimezone(SHANGHAI_TZ),
                available_at=available_at.astimezone(SHANGHAI_TZ),
                schema_version=schema_version,
                size_bytes=len(payload_bytes),
            )
            self._session.add(existing)
            self._session.flush()
        elif (
            existing.payload_hash != payload_hash
            or existing.provider != provider
            or existing.endpoint != endpoint
        ):
            raise ValueError("raw payload fingerprint collision")
        return RawArchiveResult(
            raw_payload_id=existing.id,
            payload_hash=payload_hash,
            request_hash=request_hash,
        )

    def archive_raw(self, raw_page: RawPage) -> RawArchiveResult:
        """Archive an undecoded page with sanitized request metadata."""

        payload = cast(dict[str, Any], raw_page.payload)
        payload_bytes = _canonical_json(payload)
        request_params = sanitized_request_params(raw_page.request_params)
        request_hash = canonical_request_hash(
            provider=raw_page.provider,
            endpoint=raw_page.endpoint,
            params=request_params,
        )
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        request_fingerprint = hashlib.sha256(
            _canonical_json(
                {
                    "provider": raw_page.provider,
                    "endpoint": raw_page.endpoint,
                    "request_hash": request_hash,
                    "payload_hash": payload_hash,
                }
            )
        ).hexdigest()
        return self._store(
            provider=raw_page.provider,
            endpoint=raw_page.endpoint,
            request_fingerprint=request_fingerprint,
            request_hash=request_hash,
            request_params=cast(dict[str, Any], request_params),
            payload=payload,
            payload_bytes=payload_bytes,
            fetched_at=raw_page.fetched_at,
            available_at=raw_page.available_at,
            schema_version=raw_page.schema_version,
        )

    def archive(self, batch: ProviderBatch[Any]) -> str:
        """Retain the original ProviderBatch archive identity for compatibility."""

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
        typed_params = cast(dict[str, JsonValue], batch.request_params)
        request_params = sanitized_request_params(typed_params)
        request_hash = canonical_request_hash(
            provider=batch.provider,
            endpoint=batch.endpoint,
            params=request_params,
        )
        result = self._store(
            provider=batch.provider,
            endpoint=batch.endpoint,
            request_fingerprint=request_fingerprint,
            request_hash=request_hash,
            request_params=cast(dict[str, Any], request_params),
            payload=batch.raw_payload,
            payload_bytes=payload_bytes,
            fetched_at=batch.available_at,
            available_at=batch.available_at,
            schema_version="provider-batch-v1",
        )
        return result.payload_hash


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
