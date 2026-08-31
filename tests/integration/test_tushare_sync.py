"""Integration coverage for the raw-to-curated Tushare synchronization chain."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import select

from quant_agent.data.database import Database
from quant_agent.data.domain import InstrumentType
from quant_agent.data.models import (
    AdjustmentFactorRow,
    CuratedRecordLineageRow,
    DailyBarRow,
    DataSyncPageRow,
    DataSyncRunRow,
    InstrumentRow,
    RawPayloadRow,
    TradingDayRow,
)
from quant_agent.data.providers import TushareHttpProvider
from quant_agent.data.sync import SyncPageState, SyncRunState
from quant_agent.data.sync.runner import IncrementalSyncRunner
from quant_agent.data.tushare_sync import (
    TushareAdjustmentFactorIngestor,
    TushareAdjustmentFactorSource,
    TushareDailyBarIngestor,
    TushareDailyBarSource,
    TushareInstrumentIngestor,
    TushareInstrumentSource,
    TushareTradingCalendarIngestor,
    TushareTradingCalendarSource,
)

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 30, 18, 1, tzinfo=TZ)
INITIAL_DATE = "2026-07-29"
TARGET_DATE = "2026-07-30"
INSTRUMENT_ID = "CN.SZ.000001"


def _payload(fields: list[str], items: list[list[object]]) -> dict[str, object]:
    return {
        "code": 0,
        "msg": None,
        "data": {"fields": fields, "items": items},
    }


def _provider(handler: httpx.MockTransport) -> TushareHttpProvider:
    return TushareHttpProvider("secret", client=httpx.Client(transport=handler))


def _runner[T](
    database: Database,
    *,
    source: object,
    ingestor: object,
    owner: str,
) -> IncrementalSyncRunner[T]:
    return IncrementalSyncRunner(
        session_factory=database.session,
        source=source,  # type: ignore[arg-type]
        ingestor=ingestor,  # type: ignore[arg-type]
        lease_owner=owner,
        clock=lambda: NOW,
    )


def test_failed_daily_raw_replays_offline_then_remains_idempotent_with_lineage(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'tushare-sync.db').as_posix()}")
    database.create_schema()
    daily_calls = 0

    def daily_handler(request: httpx.Request) -> httpx.Response:
        nonlocal daily_calls
        daily_calls += 1
        body = json.loads(request.content)
        assert body["api_name"] == "daily"
        assert body["token"] == "secret"
        return httpx.Response(
            200,
            json=_payload(
                [
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                ],
                [["000001.SZ", "20260730", 10, 11, 9, 10.5, 12, 34]],
            ),
        )

    failed_source = TushareDailyBarSource(
        _provider(httpx.MockTransport(daily_handler)),
        instrument_type=InstrumentType.STOCK,
        instrument_ids=[INSTRUMENT_ID],
        clock=lambda: NOW,
    )
    daily_key = "tushare-daily-replay".ljust(64, "0")
    daily_scope = {"instrument_type": "STOCK", "instrument_ids": [INSTRUMENT_ID]}

    with pytest.raises(ValueError, match="not an open SSE day"):
        _runner(
            database,
            source=failed_source,
            ingestor=TushareDailyBarIngestor(),
            owner="daily-first",
        ).run(
            scope=daily_scope,
            initial_watermark={"trade_date": INITIAL_DATE},
            target_watermark={"trade_date": TARGET_DATE},
            idempotency_key=daily_key,
        )

    assert daily_calls == 1
    with database.session() as session:
        raw = session.scalar(select(RawPayloadRow).where(RawPayloadRow.endpoint == "daily"))
        page = session.scalar(
            select(DataSyncPageRow).where(DataSyncPageRow.raw_payload_id.is_not(None))
        )
        run = session.scalar(select(DataSyncRunRow).where(DataSyncRunRow.dataset == "daily_bar"))
        assert raw is not None and raw.request_params == {
            "trade_date": "20260730",
            "ts_code": "000001.SZ",
        }
        assert "secret" not in str(raw.request_params)
        assert page is not None and page.state == SyncPageState.FETCHED.value
        assert run is not None and run.state == SyncRunState.FAILED.value
        assert session.scalar(select(DailyBarRow)) is None

    def calendar_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "trade_cal"
        return httpx.Response(
            200,
            json=_payload(
                ["exchange", "cal_date", "is_open", "pretrade_date"],
                [["SSE", "20260730", "1", "20260729"]],
            ),
        )

    calendar_source = TushareTradingCalendarSource(
        _provider(httpx.MockTransport(calendar_handler)),
        market="SSE",
        clock=lambda: NOW,
    )
    calendar_result = _runner(
        database,
        source=calendar_source,
        ingestor=TushareTradingCalendarIngestor(),
        owner="calendar",
    ).run(
        scope={"market": "SSE"},
        initial_watermark={"cal_date": INITIAL_DATE},
        target_watermark={"cal_date": TARGET_DATE},
        idempotency_key="tushare-calendar".ljust(64, "0"),
    )
    assert calendar_result.inserted == 1

    master_calls: list[str] = []

    def master_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "stock_basic"
        status = body["params"]["list_status"]
        master_calls.append(status)
        items: list[list[object]] = []
        if status == "L":
            items = [
                [
                    "000001.SZ",
                    "000001",
                    "平安银行",
                    "深圳",
                    "银行",
                    "主板",
                    "19910403",
                    None,
                    "L",
                ]
            ]
        return httpx.Response(
            200,
            json=_payload(
                [
                    "ts_code",
                    "symbol",
                    "name",
                    "area",
                    "industry",
                    "market",
                    "list_date",
                    "delist_date",
                    "list_status",
                ],
                items,
            ),
        )

    master_source = TushareInstrumentSource(
        _provider(httpx.MockTransport(master_handler)),
        instrument_type=InstrumentType.STOCK,
        clock=lambda: NOW,
    )
    master_result = _runner(
        database,
        source=master_source,
        ingestor=TushareInstrumentIngestor(),
        owner="master",
    ).run(
        scope={"instrument_type": "STOCK"},
        initial_watermark={"as_of": INITIAL_DATE},
        target_watermark={"as_of": TARGET_DATE},
        idempotency_key="tushare-master".ljust(64, "0"),
    )
    assert master_calls == ["L", "D", "P"]
    assert master_result.inserted == 1

    replay_source = TushareDailyBarSource(
        TushareHttpProvider(
            "rotated-secret",
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _request: pytest.fail("archived raw replay must stay offline")
                )
            ),
        ),
        instrument_type=InstrumentType.STOCK,
        instrument_ids=[INSTRUMENT_ID],
        clock=lambda: NOW,
    )
    replay_result = _runner(
        database,
        source=replay_source,
        ingestor=TushareDailyBarIngestor(),
        owner="daily-replay",
    ).run(
        scope=daily_scope,
        initial_watermark={"trade_date": INITIAL_DATE},
        target_watermark={"trade_date": TARGET_DATE},
        idempotency_key=daily_key,
    )

    assert replay_result.state is SyncRunState.SUCCEEDED
    assert replay_result.fetched_pages == 0
    assert replay_result.reused_raw_pages == 1
    assert replay_result.inserted == 1
    assert daily_calls == 1

    duplicate_source = TushareDailyBarSource(
        _provider(httpx.MockTransport(daily_handler)),
        instrument_type=InstrumentType.STOCK,
        instrument_ids=[INSTRUMENT_ID],
        clock=lambda: NOW,
    )
    duplicate_result = _runner(
        database,
        source=duplicate_source,
        ingestor=TushareDailyBarIngestor(),
        owner="daily-duplicate",
    ).run(
        scope={**daily_scope, "independent_checkpoint": "duplicate-proof"},
        initial_watermark={"trade_date": INITIAL_DATE},
        target_watermark={"trade_date": TARGET_DATE},
        idempotency_key="tushare-daily-duplicate".ljust(64, "0"),
    )
    assert duplicate_result.skipped == 1
    assert daily_calls == 2

    with database.session() as session:
        bar = session.scalar(select(DailyBarRow))
        assert bar is not None
        assert bar.volume == Decimal("1200")
        assert bar.turnover == Decimal("34000")
        assert bar.available_at.replace(tzinfo=TZ).hour == 16
        assert session.get(InstrumentRow, INSTRUMENT_ID) is not None
        assert session.get(TradingDayRow, ("SSE", datetime(2026, 7, 30).date())) is not None
        daily_raw_rows = list(
            session.scalars(select(RawPayloadRow).where(RawPayloadRow.endpoint == "daily"))
        )
        daily_lineage = list(
            session.scalars(
                select(CuratedRecordLineageRow).where(
                    CuratedRecordLineageRow.dataset == "daily_bar"
                )
            )
        )
        assert len(daily_raw_rows) == 1
        assert len(daily_lineage) == 1
        assert daily_lineage[0].raw_payload_id == daily_raw_rows[0].id
        assert len(daily_lineage[0].content_hash) == 64
        trace = session.execute(
            select(
                DataSyncRunRow.run_id,
                DataSyncPageRow.page_id,
                RawPayloadRow.id,
                CuratedRecordLineageRow.id,
            )
            .join(DataSyncPageRow, DataSyncPageRow.run_id == DataSyncRunRow.run_id)
            .join(RawPayloadRow, RawPayloadRow.id == DataSyncPageRow.raw_payload_id)
            .join(
                CuratedRecordLineageRow,
                CuratedRecordLineageRow.raw_payload_id == RawPayloadRow.id,
            )
            .where(DataSyncRunRow.dataset == "daily_bar")
        ).all()
        assert trace
        assert all(all(identity is not None for identity in row) for row in trace)


def test_adjustment_factor_runner_uses_same_master_calendar_and_lineage(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'adjustment-sync.db').as_posix()}")
    database.create_schema()
    with database.session() as session:
        session.add(
            InstrumentRow(
                instrument_id=INSTRUMENT_ID,
                symbol="000001",
                exchange="SZ",
                instrument_type="STOCK",
                name="平安银行",
                listed_on=datetime(1991, 4, 3).date(),
                status="LISTED",
                source="tushare",
                version="2026-07-30",
            )
        )
        session.add(
            TradingDayRow(
                market="SSE",
                trade_date=datetime(2026, 7, 30).date(),
                is_open=True,
                source="tushare",
                version="2026-07-30",
            )
        )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["api_name"] == "adj_factor"
        return httpx.Response(
            200,
            json=_payload(
                ["ts_code", "trade_date", "adj_factor"],
                [["000001.SZ", "20260730", 123.456]],
            ),
        )

    source = TushareAdjustmentFactorSource(
        _provider(httpx.MockTransport(handler)),
        instrument_ids=[INSTRUMENT_ID],
        clock=lambda: NOW,
    )
    result = _runner(
        database,
        source=source,
        ingestor=TushareAdjustmentFactorIngestor(),
        owner="adjustment",
    ).run(
        scope={"instrument_type": "STOCK", "instrument_ids": [INSTRUMENT_ID]},
        initial_watermark={"trade_date": INITIAL_DATE},
        target_watermark={"trade_date": TARGET_DATE},
        idempotency_key="tushare-adjustment".ljust(64, "0"),
    )

    assert result.inserted == 1
    with database.session() as session:
        factor = session.scalar(select(AdjustmentFactorRow))
        lineage = session.scalar(
            select(CuratedRecordLineageRow).where(
                CuratedRecordLineageRow.dataset == "adjustment_factor"
            )
        )
        assert factor is not None and factor.factor == Decimal("123.456")
        assert factor.available_at.replace(tzinfo=TZ) == datetime(2026, 7, 30, 9, 30, tzinfo=TZ)
        assert lineage is not None
