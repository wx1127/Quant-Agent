"""Tests for raw-page contracts and immutable sanitized archival."""

from dataclasses import replace
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from quant_agent.data.ingestion import RawPayloadArchive
from quant_agent.data.models import RawPayloadRow
from quant_agent.data.sync import DecodedPage, PageIngestionResult, RawPage

TZ = ZoneInfo("Asia/Shanghai")
FETCHED_AT = datetime(2026, 8, 28, 16, 1, tzinfo=TZ)
AVAILABLE_AT = datetime(2026, 8, 28, 16, 0, tzinfo=TZ)


def _raw_page(*, token: str = "secret") -> RawPage:
    return RawPage(
        provider="fake",
        endpoint="daily",
        request_params={
            "token": token,
            "offset": 0,
            "headers": {"Authorization": f"Bearer {token}", "X-Trace": "trace"},
        },
        payload={"records": [{"id": "one"}]},
        partition={"trade_date": "2026-08-28"},
        cursor_in=None,
        cursor_out={"offset": 1},
        has_more=True,
        record_count=1,
        fetched_at=FETCHED_AT,
        available_at=AVAILABLE_AT,
        schema_version="fake-v1",
    )


def test_raw_page_and_result_contracts_reject_invalid_counts_and_cursors() -> None:
    raw = _raw_page()
    assert DecodedPage(raw_page=raw, records=({"id": "one"},)).records
    assert PageIngestionResult(inserted=1).inserted == 1

    with pytest.raises(ValueError, match="has_more"):
        replace(raw, cursor_out=None)
    with pytest.raises(ValueError, match="non-empty"):
        replace(raw, provider="")
    with pytest.raises(ValueError, match="record_count"):
        replace(raw, record_count=-1)
    with pytest.raises(ValueError, match="rejected_count"):
        DecodedPage(raw_page=raw, records=(), rejected_count=-1)
    with pytest.raises(ValueError, match="must equal"):
        DecodedPage(raw_page=raw, records=({"id": "one"}, {"id": "two"}))
    with pytest.raises(ValueError, match="counts"):
        PageIngestionResult(skipped=-1)


def test_archive_raw_deduplicates_without_persisting_credentials(db_session: Session) -> None:
    archive = RawPayloadArchive(db_session)

    first = archive.archive_raw(_raw_page(token="first-secret"))
    second = archive.archive_raw(_raw_page(token="rotated-secret"))

    assert first == second
    row = db_session.get(RawPayloadRow, first.raw_payload_id)
    assert row is not None
    assert row.request_hash == first.request_hash
    assert row.request_params == {
        "offset": 0,
        "headers": {"X-Trace": "trace"},
    }
    assert "secret" not in str(row.request_params)
    assert row.schema_version == "fake-v1"
    assert row.size_bytes > 0
    assert row.fetched_at.replace(tzinfo=TZ) == FETCHED_AT


def test_archive_normalizes_utc_before_sqlite_drops_timezone(db_session: Session) -> None:
    utc_fetched = datetime(2026, 8, 28, 8, 1, tzinfo=UTC)
    utc_available = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
    result = RawPayloadArchive(db_session).archive_raw(
        replace(
            _raw_page(),
            fetched_at=utc_fetched,
            available_at=utc_available,
        )
    )
    row = db_session.get(RawPayloadRow, result.raw_payload_id)
    assert row is not None
    db_session.expire(row)

    assert row.fetched_at.replace(tzinfo=TZ) == utc_fetched.astimezone(TZ)
    assert row.available_at.replace(tzinfo=TZ) == utc_available.astimezone(TZ)
