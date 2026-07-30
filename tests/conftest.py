"""Shared pytest configuration for Quant Agent."""

from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy.orm import Session

from quant_agent.data.database import Database
from quant_agent.data.domain import (
    Instrument,
    InstrumentStatus,
    InstrumentType,
    TradingDay,
)
from quant_agent.data.master import InstrumentMasterService, TradingCalendarService


@pytest.fixture
def database() -> Database:
    """Return a fresh in-memory P1 database."""

    value = Database("sqlite+pysqlite:///:memory:")
    value.create_schema()
    return value


@pytest.fixture
def db_session(database: Database) -> Iterator[Session]:
    """Return a transactionally managed test session."""

    with database.session() as session:
        yield session


@pytest.fixture
def seeded_session(db_session: Session) -> Session:
    """Seed one stock and a compact Shanghai trading calendar."""

    InstrumentMasterService(db_session).upsert(
        [
            Instrument(
                instrument_id="CN.SZ.000001",
                symbol="000001",
                exchange="SZ",
                instrument_type=InstrumentType.STOCK,
                name="平安银行",
                listed_on=date(1991, 4, 3),
                status=InstrumentStatus.LISTED,
                source="test",
                version="v1",
            )
        ]
    )
    TradingCalendarService(db_session).upsert(
        [
            TradingDay(
                market="SSE",
                trade_date=date(2026, 7, 29),
                is_open=True,
                source="test",
                version="v1",
            ),
            TradingDay(
                market="SSE",
                trade_date=date(2026, 7, 30),
                is_open=True,
                source="test",
                version="v1",
            ),
            TradingDay(
                market="SSE",
                trade_date=date(2026, 8, 1),
                is_open=False,
                source="test",
                version="v1",
            ),
        ]
    )
    return db_session
