"""Tests for point-in-time, untrusted external event evidence."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from quant_agent.data.events import (
    EventEvidence,
    EventEvidenceService,
    EventTrustLevel,
    EventType,
    build_event_evidence,
)
from quant_agent.data.models import (
    CuratedRecordLineageRow,
    EventEvidenceRow,
    EventInstrumentLinkRow,
    RawPayloadRow,
)
from quant_agent.data.sync.hashing import JsonObject, JsonValue

TZ = ZoneInfo("Asia/Shanghai")
PUBLISHED = datetime(2026, 7, 30, 8, 0, tzinfo=TZ)
OBSERVED = datetime(2026, 7, 30, 9, 0, tzinfo=TZ)


def _raw(
    session: Session,
    *,
    suffix: str = "a",
    provider: str = "test",
    fetched_at: datetime = OBSERVED,
    available_at: datetime | None = None,
    headline: str = "业绩公告",
    version: str = "v1",
    payload: JsonObject | None = None,
) -> int:
    digest = suffix * 64
    row = RawPayloadRow(
        provider=provider,
        endpoint="announcement",
        request_fingerprint=digest,
        request_hash=digest,
        request_params={"page": suffix},
        payload_hash=digest,
        payload=({"headline": headline, "revision": version} if payload is None else payload),
        fetched_at=fetched_at,
        available_at=available_at or fetched_at,
        schema_version="announcement-v1",
        size_bytes=48,
    )
    session.add(row)
    session.flush()
    return row.id


def _event(
    *,
    version: str = "v1",
    headline: str = "业绩公告",
    observed_at: datetime = OBSERVED,
    source_content: JsonValue | None = None,
    source_path: str = "",
) -> EventEvidence:
    return build_event_evidence(
        event_type=EventType.EARNINGS,
        headline=headline,
        published_at=PUBLISHED,
        observed_at=observed_at,
        source="test",
        source_record_id="notice-1",
        version=version,
        instrument_ids=(" CN.SZ.000001 ", "CN.SZ.000001"),
        source_content=(
            {"headline": headline, "revision": version}
            if source_content is None
            else source_content
        ),
        source_url="https://example.test/notices/1",
        source_path=source_path,
    )


def test_builder_normalizes_identity_and_marks_external_text_untrusted() -> None:
    first = _event()
    second = _event()

    assert first == second
    assert first.instrument_ids == ("CN.SZ.000001",)
    assert first.available_at == OBSERVED
    assert first.trust_level is EventTrustLevel.UNTRUSTED_SOURCE
    assert len(first.event_id) == len(first.content_hash) == 64


def test_event_contract_rejects_bad_time_identity_and_url() -> None:
    with pytest.raises(ValueError, match="observed_at cannot precede"):
        _event(observed_at=PUBLISHED - timedelta(minutes=1))

    payload = _event().model_dump()
    payload["event_id"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        EventEvidence.model_validate(payload)

    payload = _event().model_dump()
    payload["source_url"] = "file:///private/notice.txt"
    with pytest.raises(ValueError, match="HTTP"):
        EventEvidence.model_validate(payload)

    payload = _event().model_dump()
    payload["source_path"] = "items/0"
    with pytest.raises(ValueError, match="JSON pointer"):
        EventEvidence.model_validate(payload)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("content_hash", "not-a-hash", "SHA-256"),
        ("headline", " ", "non-empty"),
        ("instrument_ids", "CN.SZ.000001", "sequence"),
        ("instrument_ids", (), "non-empty stable IDs"),
        ("source_path", "/bad~2escape", "invalid JSON pointer escape"),
    ],
)
def test_event_contract_rejects_malformed_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _event().model_dump()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        EventEvidence.model_validate(payload)


def test_event_contract_rejects_availability_before_observation() -> None:
    payload = _event().model_dump()
    payload["available_at"] = OBSERVED - timedelta(microseconds=1)

    with pytest.raises(ValueError, match="available_at cannot precede"):
        EventEvidence.model_validate(payload)


def test_builder_accepts_absent_source_url() -> None:
    event = build_event_evidence(
        event_type=EventType.OTHER,
        headline="market notice",
        published_at=PUBLISHED,
        observed_at=OBSERVED,
        source="test",
        source_record_id="market-1",
        version="v1",
        instrument_ids=("CN.SZ.000001",),
        source_content={"notice": 1},
    )

    assert event.source_url is None


def test_ingestion_is_idempotent_lineage_aware_and_point_in_time(
    seeded_session: Session,
) -> None:
    raw_id = _raw(seeded_session)
    service = EventEvidenceService(seeded_session)
    event = _event()

    first = service.ingest([event], raw_payload_id=raw_id)
    second = service.ingest([event], raw_payload_id=raw_id)

    assert (first.inserted, first.skipped, first.lineage_inserted) == (1, 0, 1)
    assert (second.inserted, second.skipped, second.lineage_inserted) == (0, 1, 0)
    assert service.as_of(OBSERVED - timedelta(microseconds=1)) == ()
    assert service.as_of(OBSERVED) == (event,)
    assert service.as_of(OBSERVED, instrument_id="CN.SZ.000002") == ()
    assert service.as_of(OBSERVED, event_types=[]) == ()
    assert service.as_of(OBSERVED, event_types=[EventType.REGULATORY]) == ()
    assert len(list(seeded_session.scalars(select(EventEvidenceRow)))) == 1
    assert len(list(seeded_session.scalars(select(EventInstrumentLinkRow)))) == 1
    assert len(list(seeded_session.scalars(select(CuratedRecordLineageRow)))) == 1


def test_future_revision_does_not_change_an_earlier_view(seeded_session: Session) -> None:
    first_raw = _raw(seeded_session, suffix="a")
    second_observed = OBSERVED + timedelta(days=1)
    second_raw = _raw(
        seeded_session,
        suffix="b",
        fetched_at=second_observed,
        headline="业绩公告(更正)",
        version="v2",
    )
    service = EventEvidenceService(seeded_session)
    first = _event()
    second = _event(version="v2", headline="业绩公告(更正)", observed_at=second_observed)

    service.ingest([first], raw_payload_id=first_raw)
    before = service.as_of(OBSERVED)
    service.ingest([second], raw_payload_id=second_raw)

    assert service.as_of(OBSERVED) == before == (first,)
    assert service.as_of(second_observed) == (first, second)


def test_ingestion_fails_closed_for_conflicts_and_unverifiable_sources(
    seeded_session: Session,
) -> None:
    raw_id = _raw(seeded_session)
    service = EventEvidenceService(seeded_session)
    event = _event()
    service.ingest([event], raw_payload_id=raw_id)

    conflicting_raw = _raw(seeded_session, suffix="e", headline="changed")
    with pytest.raises(ValueError, match="immutable revision"):
        service.ingest([_event(headline="changed")], raw_payload_id=conflicting_raw)

    unknown_payload = event.model_dump()
    unknown_payload["instrument_ids"] = ("CN.SZ.999999",)
    unknown = EventEvidence.model_validate(unknown_payload)
    with pytest.raises(ValueError, match="unknown instrument"):
        service.ingest([unknown], raw_payload_id=raw_id)

    wrong_raw = _raw(seeded_session, suffix="c", provider="other")
    with pytest.raises(ValueError, match="does not match"):
        service.ingest([event], raw_payload_id=wrong_raw)

    later_raw = _raw(
        seeded_session,
        suffix="d",
        fetched_at=OBSERVED + timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="cannot precede raw payload fetch"):
        service.ingest([event], raw_payload_id=later_raw)

    mismatched_raw = _raw(seeded_session, suffix="f", headline="other content")
    with pytest.raises(ValueError, match="content hash does not match"):
        service.ingest([event], raw_payload_id=mismatched_raw)

    with pytest.raises(ValueError, match="existing raw payload"):
        service.ingest([event], raw_payload_id=999999)


def test_ingestion_verifies_json_pointer_and_raw_availability(
    seeded_session: Session,
) -> None:
    nested_content: JsonObject = {"headline": "nested", "revision": "pointer"}
    payload: JsonObject = {
        "items": [nested_content],
        "a/b": {"~key": {"value": 7}},
    }
    raw_id = _raw(seeded_session, suffix="g", payload=payload)
    service = EventEvidenceService(seeded_session)
    nested = _event(
        version="pointer",
        headline="nested",
        source_content=nested_content,
        source_path="/items/0",
    )

    result = service.ingest([nested], raw_payload_id=raw_id)
    assert result.inserted == 1

    escaped = _event(
        version="escaped",
        headline="escaped",
        source_content={"value": 7},
        source_path="/a~1b/~0key",
    )
    service.ingest([escaped], raw_payload_id=raw_id)

    for version, path, message in (
        ("missing", "/missing", "does not exist"),
        ("leading-zero", "/items/01", "invalid array index"),
        ("out-of-range", "/items/2", "out of range"),
        ("scalar", "/items/0/headline/child", "traverses a scalar"),
    ):
        candidate = _event(
            version=version,
            source_content={"unused": True},
            source_path=path,
        )
        with pytest.raises(ValueError, match=message):
            service.ingest([candidate], raw_payload_id=raw_id)

    delayed_raw = _raw(
        seeded_session,
        suffix="h",
        available_at=OBSERVED + timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="cannot precede raw payload availability"):
        service.ingest([_event()], raw_payload_id=delayed_raw)


def test_ingestion_detects_stored_entity_mapping_corruption(seeded_session: Session) -> None:
    raw_id = _raw(seeded_session, suffix="i")
    service = EventEvidenceService(seeded_session)
    event = _event()
    service.ingest([event], raw_payload_id=raw_id)
    seeded_session.execute(
        delete(EventInstrumentLinkRow).where(EventInstrumentLinkRow.event_id == event.event_id)
    )
    seeded_session.flush()

    with pytest.raises(ValueError, match="entity mapping conflicts"):
        service.ingest([event], raw_payload_id=raw_id)
