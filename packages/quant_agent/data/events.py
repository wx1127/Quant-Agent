"""Point-in-time contracts and persistence for untrusted external event evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware
from quant_agent.data.models import (
    CuratedRecordLineageRow,
    EventEvidenceRow,
    EventInstrumentLinkRow,
    InstrumentRow,
    RawPayloadRow,
)
from quant_agent.data.sync.contracts import IngestionSession
from quant_agent.data.sync.hashing import JsonValue, canonical_hash, canonical_json

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class EventType(StrEnum):
    """Coarse, deterministic categories; interpretation belongs downstream."""

    EARNINGS = "EARNINGS"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    REGULATORY = "REGULATORY"
    MANAGEMENT = "MANAGEMENT"
    PRODUCT = "PRODUCT"
    OTHER = "OTHER"


class EventTrustLevel(StrEnum):
    """External text is never treated as an instruction at the data boundary."""

    UNTRUSTED_SOURCE = "UNTRUSTED_SOURCE"


def event_identity(*, source: str, source_record_id: str, version: str) -> str:
    """Return the stable identity of one provider revision."""

    return canonical_hash(
        {
            "source": source.strip(),
            "source_record_id": source_record_id.strip(),
            "version": version.strip(),
        }
    )


class EventEvidence(BaseModel):
    """One immutable event revision with explicit first-observed availability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    event_type: EventType
    headline: str = Field(max_length=512)
    published_at: datetime
    observed_at: datetime
    available_at: datetime
    source: str = Field(max_length=32)
    source_record_id: str = Field(max_length=128)
    source_url: str | None = None
    source_path: str = Field(default="", max_length=512)
    content_hash: str
    version: str = Field(max_length=64)
    instrument_ids: tuple[str, ...]
    trust_level: EventTrustLevel = EventTrustLevel.UNTRUSTED_SOURCE

    @field_validator("event_id", "content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("event_id and content_hash must be lowercase SHA-256 values")
        return normalized

    @field_validator("headline", "source", "source_record_id", "version")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("event text and identity fields must be non-empty")
        return normalized

    @field_validator("published_at", "observed_at", "available_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return normalized

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        if value and not value.startswith("/"):
            raise ValueError("source_path must be an RFC 6901 JSON pointer")
        for token in value.split("/")[1:]:
            if re.search(r"~(?![01])", token):
                raise ValueError("source_path contains an invalid JSON pointer escape")
        return value

    @field_validator("instrument_ids", mode="before")
    @classmethod
    def normalize_instruments(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("instrument_ids must be a sequence")
        normalized = tuple(sorted({str(item).strip() for item in value}))
        if not normalized or any(not item for item in normalized):
            raise ValueError("instrument_ids must contain non-empty stable IDs")
        return normalized

    @model_validator(mode="after")
    def validate_identity_and_availability(self) -> EventEvidence:
        if self.observed_at < self.published_at:
            raise ValueError("observed_at cannot precede published_at")
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot precede first observation")
        expected = event_identity(
            source=self.source,
            source_record_id=self.source_record_id,
            version=self.version,
        )
        if self.event_id != expected:
            raise ValueError("event_id does not match its source revision identity")
        return self


def build_event_evidence(
    *,
    event_type: EventType,
    headline: str,
    published_at: datetime,
    observed_at: datetime,
    source: str,
    source_record_id: str,
    version: str,
    instrument_ids: Sequence[str],
    source_content: JsonValue,
    source_url: str | None = None,
    source_path: str = "",
    available_at: datetime | None = None,
) -> EventEvidence:
    """Build a verified event while keeping source content outside curated storage."""

    effective_available_at = available_at or observed_at
    return EventEvidence(
        event_id=event_identity(
            source=source,
            source_record_id=source_record_id,
            version=version,
        ),
        event_type=event_type,
        headline=headline,
        published_at=published_at,
        observed_at=observed_at,
        available_at=effective_available_at,
        source=source,
        source_record_id=source_record_id,
        source_url=source_url,
        source_path=source_path,
        content_hash=canonical_hash(source_content),
        version=version,
        instrument_ids=tuple(instrument_ids),
    )


@dataclass(frozen=True, slots=True)
class EventIngestionResult:
    inserted: int
    skipped: int
    lineage_inserted: int


def _stored_time(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=SHANGHAI_TZ)
    return ensure_aware(value)


class EventEvidenceService:
    """Persist immutable revisions and expose only evidence known by a decision time."""

    def __init__(self, session: IngestionSession) -> None:
        self._session = session

    def ingest(
        self,
        events: Sequence[EventEvidence],
        *,
        raw_payload_id: int,
    ) -> EventIngestionResult:
        raw = self._session.get(RawPayloadRow, raw_payload_id)
        if raw is None:
            raise ValueError("event evidence requires an existing raw payload")
        inserted = skipped = lineage_inserted = 0
        for event in events:
            self._validate_source(event, raw)
            self._validate_instruments(event.instrument_ids)
            existing = self._session.get(EventEvidenceRow, event.event_id)
            if existing is None:
                self._session.add(self._to_row(event))
                self._session.flush()
                for instrument_id in event.instrument_ids:
                    self._session.add(
                        EventInstrumentLinkRow(
                            event_id=event.event_id,
                            instrument_id=instrument_id,
                        )
                    )
                inserted += 1
            else:
                existing_ids = self._instrument_ids(event.event_id)
                if existing_ids != event.instrument_ids:
                    raise ValueError("event entity mapping conflicts with its stored revision")
                self._validate_existing(existing, event)
                skipped += 1
            lineage_inserted += self._add_lineage(event, raw_payload_id)
        self._session.flush()
        return EventIngestionResult(
            inserted=inserted,
            skipped=skipped,
            lineage_inserted=lineage_inserted,
        )

    def as_of(
        self,
        decision_time: datetime,
        *,
        instrument_id: str | None = None,
        event_types: Sequence[EventType] | None = None,
    ) -> tuple[EventEvidence, ...]:
        """Return an ordered PIT view without substituting future revisions."""

        decision = ensure_aware(decision_time)
        statement = select(EventEvidenceRow).where(EventEvidenceRow.available_at <= decision)
        if instrument_id is not None:
            statement = statement.join(
                EventInstrumentLinkRow,
                EventInstrumentLinkRow.event_id == EventEvidenceRow.event_id,
            ).where(EventInstrumentLinkRow.instrument_id == instrument_id)
        if event_types is not None:
            normalized_types = tuple(dict.fromkeys(item.value for item in event_types))
            if not normalized_types:
                return ()
            statement = statement.where(EventEvidenceRow.event_type.in_(normalized_types))
        statement = statement.order_by(
            EventEvidenceRow.available_at,
            EventEvidenceRow.published_at,
            EventEvidenceRow.event_id,
        )
        return tuple(self._from_row(row) for row in self._session.scalars(statement))

    def _validate_source(self, event: EventEvidence, raw: RawPayloadRow) -> None:
        if event.source != raw.provider:
            raise ValueError("event source does not match its raw payload provider")
        if event.observed_at < _stored_time(raw.fetched_at):
            raise ValueError("event observed_at cannot precede raw payload fetch time")
        if event.available_at < _stored_time(raw.available_at):
            raise ValueError("event available_at cannot precede raw payload availability")
        source_content = _resolve_json_pointer(
            cast(JsonValue, raw.payload),
            event.source_path,
        )
        if canonical_hash(source_content) != event.content_hash:
            raise ValueError("event content hash does not match its raw source path")

    def _validate_instruments(self, instrument_ids: Sequence[str]) -> None:
        known = set(
            self._session.scalars(
                select(InstrumentRow.instrument_id).where(
                    InstrumentRow.instrument_id.in_(instrument_ids)
                )
            )
        )
        missing = sorted(set(instrument_ids) - known)
        if missing:
            raise ValueError(f"event references unknown instrument IDs: {missing}")

    @staticmethod
    def _to_row(event: EventEvidence) -> EventEvidenceRow:
        return EventEvidenceRow(
            event_id=event.event_id,
            event_type=event.event_type.value,
            headline=event.headline,
            published_at=event.published_at,
            observed_at=event.observed_at,
            available_at=event.available_at,
            source=event.source,
            source_record_id=event.source_record_id,
            source_url=event.source_url,
            source_path=event.source_path,
            content_hash=event.content_hash,
            version=event.version,
            trust_level=event.trust_level.value,
        )

    def _validate_existing(self, row: EventEvidenceRow, event: EventEvidence) -> None:
        stored = self._from_row(row)
        if stored != event:
            raise ValueError("event identity conflicts with an existing immutable revision")

    def _instrument_ids(self, event_id: str) -> tuple[str, ...]:
        return tuple(
            self._session.scalars(
                select(EventInstrumentLinkRow.instrument_id)
                .where(EventInstrumentLinkRow.event_id == event_id)
                .order_by(EventInstrumentLinkRow.instrument_id)
            )
        )

    def _from_row(self, row: EventEvidenceRow) -> EventEvidence:
        return EventEvidence(
            event_id=row.event_id,
            event_type=EventType(row.event_type),
            headline=row.headline,
            published_at=_stored_time(row.published_at),
            observed_at=_stored_time(row.observed_at),
            available_at=_stored_time(row.available_at),
            source=row.source,
            source_record_id=row.source_record_id,
            source_url=row.source_url,
            source_path=row.source_path,
            content_hash=row.content_hash,
            version=row.version,
            instrument_ids=self._instrument_ids(row.event_id),
            trust_level=EventTrustLevel(row.trust_level),
        )

    def _add_lineage(self, event: EventEvidence, raw_payload_id: int) -> int:
        entity_key = canonical_json({"event_id": event.event_id})
        existing = self._session.scalar(
            select(CuratedRecordLineageRow.id).where(
                CuratedRecordLineageRow.raw_payload_id == raw_payload_id,
                CuratedRecordLineageRow.dataset == "event_evidence",
                CuratedRecordLineageRow.entity_type == "EventEvidence",
                CuratedRecordLineageRow.entity_key == entity_key,
                CuratedRecordLineageRow.content_hash == event.content_hash,
            )
        )
        if existing is not None:
            return 0
        self._session.add(
            CuratedRecordLineageRow(
                raw_payload_id=raw_payload_id,
                dataset="event_evidence",
                entity_type="EventEvidence",
                entity_key=entity_key,
                content_hash=event.content_hash,
            )
        )
        return 1


def _resolve_json_pointer(value: JsonValue, pointer: str) -> JsonValue:
    """Resolve a validated RFC 6901 pointer without accepting Python object access."""

    current = value
    if pointer == "":
        return current
    for encoded in pointer.split("/")[1:]:
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ValueError("event source_path does not exist in raw payload")
            current = current[token]
            continue
        if isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ValueError("event source_path contains an invalid array index")
            index = int(token)
            if index >= len(current):
                raise ValueError("event source_path array index is out of range")
            current = current[index]
            continue
        raise ValueError("event source_path traverses a scalar value")
    return current


__all__ = [
    "EventEvidence",
    "EventEvidenceService",
    "EventIngestionResult",
    "EventTrustLevel",
    "EventType",
    "build_event_evidence",
    "event_identity",
]
