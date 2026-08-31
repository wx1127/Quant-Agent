"""End-to-end raw archive, event evidence, and replay coverage for ``anns_d``."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import select

from quant_agent.data.database import Database
from quant_agent.data.events import EventEvidenceService, EventType
from quant_agent.data.models import (
    CuratedRecordLineageRow,
    EventEvidenceRow,
    EventInstrumentLinkRow,
    InstrumentRow,
    RawPayloadRow,
)
from quant_agent.data.providers import TushareHttpProvider
from quant_agent.data.sync import SyncRunState
from quant_agent.data.sync.runner import IncrementalSyncRunner
from quant_agent.data.tushare_sync import (
    TushareAnnouncementIngestor,
    TushareAnnouncementSource,
)

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 30, 18, 1, tzinfo=TZ)
INSTRUMENT_ID = "CN.SZ.000001"
FIELDS = ["ann_date", "ts_code", "name", "title", "url", "rec_time"]


def _provider(handler: httpx.MockTransport) -> TushareHttpProvider:
    return TushareHttpProvider("secret", client=httpx.Client(transport=handler))


def test_announcement_pipeline_archives_exact_row_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'announcement.db').as_posix()}")
    database.create_schema()
    with database.session() as session:
        session.add(
            InstrumentRow(
                instrument_id=INSTRUMENT_ID,
                symbol="000001",
                exchange="SZ",
                instrument_type="STOCK",
                name="平安银行",
                listed_on=date(1991, 4, 3),
                status="LISTED",
                source="tushare",
                version="master-v1",
            )
        )

    calls = 0
    raw_item: list[object] = [
        "20260730",
        "000001.SZ",
        "平安银行",
        "2026年半年度业绩公告",
        "https://example.test/earnings.pdf",
        "2026-07-30 17:30:00",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["api_name"] == "anns_d"
        assert body["params"] == {"ann_date": "20260730", "ts_code": "000001.SZ"}
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {"fields": FIELDS, "items": [raw_item]},
            },
        )

    source = TushareAnnouncementSource(
        _provider(httpx.MockTransport(handler)),
        instrument_ids=(INSTRUMENT_ID,),
        clock=lambda: NOW,
    )
    runner = IncrementalSyncRunner(
        session_factory=database.session,
        source=source,
        ingestor=TushareAnnouncementIngestor(),
        lease_owner="announcement-test",
        clock=lambda: NOW,
    )
    kwargs = {
        "scope": {"instrument_ids": [INSTRUMENT_ID], "instrument_type": "STOCK"},
        "initial_watermark": {"ann_date": "2026-07-29"},
        "target_watermark": {"ann_date": "2026-07-30"},
        "idempotency_key": "announcement-sync-v1",
    }
    first = runner.run(**kwargs)
    second = runner.run(**kwargs)

    assert first.state is SyncRunState.SUCCEEDED
    assert first.inserted == 1
    assert second.run_id == first.run_id
    assert second.fetched_pages == 0
    assert calls == 1

    with database.session() as session:
        raw = session.scalar(select(RawPayloadRow).where(RawPayloadRow.endpoint == "anns_d"))
        event_row = session.scalar(select(EventEvidenceRow))
        link = session.scalar(select(EventInstrumentLinkRow))
        lineage = session.scalar(
            select(CuratedRecordLineageRow).where(
                CuratedRecordLineageRow.dataset == "event_evidence"
            )
        )
        assert raw is not None and raw.payload["data"]["items"][0] == raw_item
        assert "secret" not in json.dumps(raw.request_params, sort_keys=True)
        assert event_row is not None and event_row.source_path == "/data/items/0"
        assert event_row.event_type == EventType.EARNINGS.value
        assert link is not None and link.instrument_id == INSTRUMENT_ID
        assert lineage is not None and lineage.raw_payload_id == raw.id

        service = EventEvidenceService(session)
        assert service.as_of(NOW - timedelta(seconds=1), instrument_id=INSTRUMENT_ID) == ()
        events = service.as_of(NOW, instrument_id=INSTRUMENT_ID)
        assert len(events) == 1
        assert events[0].headline == "2026年半年度业绩公告"
        assert events[0].content_hash == lineage.content_hash

    offline = TushareAnnouncementSource(
        _provider(
            httpx.MockTransport(
                lambda _request: (_ for _ in ()).throw(
                    AssertionError("idempotent replay must not call the provider")
                )
            )
        ),
        instrument_ids=(INSTRUMENT_ID,),
        clock=lambda: NOW,
    )
    replay = IncrementalSyncRunner(
        session_factory=database.session,
        source=offline,
        ingestor=TushareAnnouncementIngestor(),
        lease_owner="announcement-replay",
        clock=lambda: NOW,
    ).run(**kwargs)
    assert replay.run_id == first.run_id
    assert replay.fetched_pages == 0
