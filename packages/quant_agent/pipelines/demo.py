"""Deterministic end-to-end data pipeline for local verification and onboarding."""

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
from sqlalchemy import select
from sqlalchemy.engine import make_url

from quant_agent.data.database import Database
from quant_agent.data.domain import (
    DailyBar,
    Instrument,
    InstrumentStatus,
    InstrumentType,
    TradingDay,
)
from quant_agent.data.ingestion import DailyBarIngestionService
from quant_agent.data.master import InstrumentMasterService, TradingCalendarService
from quant_agent.data.models import DailyBarRow
from quant_agent.data.providers.fake import FakeMarketDataProvider
from quant_agent.data.quality import BarQualityRecord, QualityContext, QualityEngine
from quant_agent.data.snapshots import SnapshotStore, validate_data_version

_TZ = ZoneInfo("Asia/Shanghai")
_MARKET = "SSE"
_SOURCE = "fake"
_RECORD_VERSION = "demo-v1"
_OPEN_DATES = (date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30))
_AS_OF = _OPEN_DATES[-1]
_AVAILABLE_AT = datetime.combine(_AS_OF, time(16), tzinfo=_TZ)
DEFAULT_DATA_VERSION = "demo_20260730_v1"


@dataclass(frozen=True, slots=True)
class DemoPipelineResult:
    """Stable, non-secret summary emitted by the demo pipeline."""

    data_version: str
    content_hash: str
    snapshot_path: str
    instrument_count: int
    open_day_count: int
    row_count: int
    inserted_bars: int
    skipped_bars: int
    quality_issue_count: int
    qualified: bool
    snapshot_verified: bool
    snapshot_reused: bool
    query_row_count: int

    def to_dict(self) -> dict[str, str | int | bool]:
        """Return a JSON-compatible summary without database credentials."""

        return asdict(self)


def _demo_instruments() -> tuple[Instrument, ...]:
    return (
        Instrument(
            instrument_id="CN.SH.600000",
            symbol="600000",
            exchange="SH",
            instrument_type=InstrumentType.STOCK,
            name="Demo Stock",
            listed_on=date(1999, 11, 10),
            status=InstrumentStatus.LISTED,
            source=_SOURCE,
            version=_RECORD_VERSION,
        ),
        Instrument(
            instrument_id="CN.SH.510300",
            symbol="510300",
            exchange="SH",
            instrument_type=InstrumentType.ETF,
            name="Demo ETF",
            listed_on=date(2012, 5, 28),
            status=InstrumentStatus.LISTED,
            source=_SOURCE,
            version=_RECORD_VERSION,
        ),
    )


def _bar(
    instrument_id: str,
    trade_date: date,
    *,
    open_price: str,
    high: str,
    low: str,
    close: str,
    volume: str,
) -> DailyBar:
    return DailyBar(
        instrument_id=instrument_id,
        trade_date=trade_date,
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        turnover=Decimal(volume) * Decimal(close),
        source=_SOURCE,
        available_at=datetime.combine(trade_date, time(16), tzinfo=_TZ),
        version=_RECORD_VERSION,
    )


def _demo_bars() -> tuple[DailyBar, ...]:
    return (
        _bar(
            "CN.SH.600000",
            _OPEN_DATES[0],
            open_price="10.00",
            high="10.30",
            low="9.90",
            close="10.20",
            volume="1000000",
        ),
        _bar(
            "CN.SH.510300",
            _OPEN_DATES[0],
            open_price="4.00",
            high="4.08",
            low="3.98",
            close="4.05",
            volume="2000000",
        ),
        _bar(
            "CN.SH.600000",
            _OPEN_DATES[1],
            open_price="10.25",
            high="10.45",
            low="10.10",
            close="10.35",
            volume="1100000",
        ),
        _bar(
            "CN.SH.510300",
            _OPEN_DATES[1],
            open_price="4.06",
            high="4.12",
            low="4.01",
            close="4.10",
            volume="2100000",
        ),
        _bar(
            "CN.SH.600000",
            _OPEN_DATES[2],
            open_price="10.30",
            high="10.60",
            low="10.20",
            close="10.50",
            volume="1200000",
        ),
        _bar(
            "CN.SH.510300",
            _OPEN_DATES[2],
            open_price="4.11",
            high="4.20",
            low="4.08",
            close="4.18",
            volume="2200000",
        ),
    )


def _demo_provider() -> FakeMarketDataProvider:
    return FakeMarketDataProvider(
        available_at=_AVAILABLE_AT,
        instruments=_demo_instruments(),
        trading_days=tuple(
            TradingDay(
                market=_MARKET,
                trade_date=trade_date,
                is_open=True,
                source=_SOURCE,
                version=_RECORD_VERSION,
            )
            for trade_date in _OPEN_DATES
        ),
        daily_bars=_demo_bars(),
    )


def _snapshot_table(rows: list[DailyBarRow]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("instrument_id", pa.string(), nullable=False),
            pa.field("trade_date", pa.date32(), nullable=False),
            pa.field("open", pa.decimal128(20, 6), nullable=False),
            pa.field("high", pa.decimal128(20, 6), nullable=False),
            pa.field("low", pa.decimal128(20, 6), nullable=False),
            pa.field("close", pa.decimal128(20, 6), nullable=False),
            pa.field("volume", pa.decimal128(28, 4), nullable=False),
            pa.field("turnover", pa.decimal128(28, 4), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("version", pa.string(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(
        [
            {
                "instrument_id": row.instrument_id,
                "trade_date": row.trade_date,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "turnover": row.turnover,
                "source": row.source,
                "version": row.version,
            }
            for row in rows
        ],
        schema=schema,
    )


def run_demo_pipeline(
    *,
    database_url: str = "sqlite:///./artifacts/quant_agent.db",
    storage_path: str | Path = "./artifacts/snapshots",
    data_version: str = DEFAULT_DATA_VERSION,
) -> DemoPipelineResult:
    """Run provider-to-snapshot flow and return a deterministic verification summary."""

    validate_data_version(data_version)
    if make_url(database_url).get_backend_name() != "sqlite":
        raise ValueError("the deterministic demo pipeline only supports SQLite")
    database = Database(database_url)
    database.create_schema()
    provider = _demo_provider()
    snapshot_root = Path(storage_path)
    snapshot_target = snapshot_root / data_version
    snapshot_reused = snapshot_target.exists()
    inserted_bars = 0
    skipped_bars = 0

    with database.session() as session:
        instrument_batches = (
            provider.fetch_instruments(_AS_OF, instrument_type=InstrumentType.STOCK),
            provider.fetch_instruments(_AS_OF, instrument_type=InstrumentType.ETF),
        )
        instruments = tuple(
            instrument for batch in instrument_batches for instrument in batch.records
        )
        calendar_batch = provider.fetch_trading_calendar(
            _MARKET,
            _OPEN_DATES[0],
            _OPEN_DATES[-1],
        )
        InstrumentMasterService(session).upsert(list(instruments))
        TradingCalendarService(session).upsert(list(calendar_batch.records))
        ingestion = DailyBarIngestionService(session, market=_MARKET)
        for trade_date in _OPEN_DATES:
            for instrument_type in (InstrumentType.STOCK, InstrumentType.ETF):
                result = ingestion.ingest(
                    provider.fetch_daily_bars(
                        trade_date,
                        instrument_type=instrument_type,
                    )
                )
                inserted_bars += result.inserted
                skipped_bars += result.skipped

        instrument_ids = tuple(item.instrument_id for item in instruments)
        rows = list(
            session.scalars(
                select(DailyBarRow)
                .where(
                    DailyBarRow.instrument_id.in_(instrument_ids),
                    DailyBarRow.trade_date.in_(_OPEN_DATES),
                    DailyBarRow.source == _SOURCE,
                    DailyBarRow.version == _RECORD_VERSION,
                )
                .order_by(DailyBarRow.trade_date, DailyBarRow.instrument_id)
            )
        )
        quality_context = QualityContext(
            bars=tuple(
                BarQualityRecord(
                    instrument_id=row.instrument_id,
                    trade_date=row.trade_date,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    volume=row.volume,
                    turnover=row.turnover,
                )
                for row in rows
            ),
            expected_instrument_ids=instrument_ids,
            expected_open_dates=_OPEN_DATES,
        )
        quality_report = QualityEngine().run(
            data_version,
            quality_context,
            session=session,
            observed_at=_AVAILABLE_AT,
        )
        table = _snapshot_table(rows)
        store = SnapshotStore(snapshot_root)
        manifest = store.create(
            data_version,
            {"daily_bars": table},
            quality_report=quality_report,
            metadata={
                "as_of": _AVAILABLE_AT.isoformat(),
                "market": _MARKET,
                "provider": _SOURCE,
                "record_version": _RECORD_VERSION,
            },
            session=session,
        )
        query_result = store.query(
            data_version,
            "SELECT instrument_id, trade_date, close "
            "FROM daily_bars ORDER BY trade_date, instrument_id",
        )
        snapshot_verified = store.verify(data_version)
        if not snapshot_verified:
            raise ValueError("snapshot integrity verification failed")

    return DemoPipelineResult(
        data_version=data_version,
        content_hash=manifest.content_hash,
        snapshot_path=str(snapshot_target.resolve()),
        instrument_count=len(instruments),
        open_day_count=len(calendar_batch.records),
        row_count=table.num_rows,
        inserted_bars=inserted_bars,
        skipped_bars=skipped_bars,
        quality_issue_count=len(quality_report.issues),
        qualified=quality_report.qualified,
        snapshot_verified=snapshot_verified,
        snapshot_reused=snapshot_reused,
        query_row_count=query_result.num_rows,
    )
