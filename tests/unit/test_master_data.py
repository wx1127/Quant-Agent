"""Tests for calendar and historical code resolution."""

from datetime import date

import pytest
from sqlalchemy.orm import Session

from quant_agent.data.domain import (
    Instrument,
    InstrumentStatus,
    InstrumentType,
    TradingDay,
)
from quant_agent.data.master import InstrumentMasterService, TradingCalendarService


def test_calendar_is_idempotent_and_queries_open_days(db_session: Session) -> None:
    service = TradingCalendarService(db_session)
    days = [
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
            is_open=False,
            source="test",
            version="v1",
        ),
    ]

    assert service.upsert(days) == (2, 0)
    assert service.upsert(days) == (0, 0)
    changed = [days[1].model_copy(update={"is_open": True, "version": "v2"})]
    assert service.upsert(changed) == (0, 1)
    assert service.open_days("SSE", date(2026, 7, 29), date(2026, 7, 30)) == [
        date(2026, 7, 29),
        date(2026, 7, 30),
    ]
    assert service.previous_open_day("SSE", date(2026, 7, 30)) == date(2026, 7, 29)


def test_instrument_alias_resolves_only_inside_effective_range(
    db_session: Session,
) -> None:
    service = InstrumentMasterService(db_session)
    instrument = Instrument(
        instrument_id="CN.SZ.STABLE1",
        symbol="000001",
        exchange="SZ",
        instrument_type=InstrumentType.STOCK,
        name="test",
        listed_on=date(2000, 1, 1),
        delisted_on=date(2020, 12, 31),
        status=InstrumentStatus.DELISTED,
        source="test",
        version="v1",
    )
    assert service.upsert([instrument]) == (1, 0)
    assert service.upsert([instrument]) == (0, 0)

    assert service.resolve("000001", "SZ", date(2010, 1, 1)) is not None
    assert service.resolve("000001", "SZ", date(2021, 1, 1)) is None
    assert service.is_active("CN.SZ.STABLE1", date(2010, 1, 1)) is True
    assert service.is_active("CN.SZ.STABLE1", date(2020, 12, 31)) is False
    assert service.status_on("CN.SZ.STABLE1", date(2010, 1, 1)) == "LISTED"
    assert service.status_on("CN.SZ.STABLE1", date(2021, 1, 1)) == "DELISTED"


def test_instrument_alias_rejects_overlap(seeded_session: Session) -> None:
    service = InstrumentMasterService(seeded_session)
    with pytest.raises(ValueError, match="overlaps"):
        service.add_alias(
            instrument_id="CN.SZ.000001",
            symbol="000001",
            exchange="SZ",
            effective_from=date(2000, 1, 1),
            effective_to=None,
            source="test",
        )
