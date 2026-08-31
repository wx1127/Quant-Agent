"""Migration and constraint tests for curated data lineage."""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.exc import IntegrityError

from quant_agent.data.database import Database
from quant_agent.data.models import (
    CuratedRecordLineageRow,
    IndexConstituentWeightRow,
    InstrumentRow,
    RawPayloadRow,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 28, 10, tzinfo=UTC)
P1_TABLES = {
    "adjustment_factor",
    "corporate_action",
    "data_quality_result",
    "dataset_version",
    "financial_statement_point_in_time",
    "industry_classification",
    "industry_membership_history",
    "instrument",
    "instrument_alias",
    "instrument_status",
    "market_bar_daily",
    "raw_payload",
    "trading_calendar",
}


def _alembic_config(database_path: Path) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    return config


def test_frozen_0001_and_clean_head_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "round-trip.db"
    config = _alembic_config(database_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)

    command.upgrade(config, "0001")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    assert set(inspect(engine).get_table_names()) == P1_TABLES | {"alembic_version"}

    command.upgrade(config, "head")
    assert {
        "data_sync_run",
        "data_sync_checkpoint",
        "data_sync_page",
        "index_constituent_weight",
        "curated_record_lineage",
    } <= set(inspect(engine).get_table_names())

    command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}


def test_0004_explicit_schema_upgrades_and_downgrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "0004.db"
    config = _alembic_config(database_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)
    command.upgrade(config, "0003")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    assert not {
        "index_constituent_weight",
        "curated_record_lineage",
    } & set(inspect(engine).get_table_names())

    command.upgrade(config, "0004")
    inspector = inspect(engine)
    assert {"index_constituent_weight", "curated_record_lineage"} <= set(
        inspector.get_table_names()
    )
    assert {
        item["name"] for item in inspector.get_unique_constraints("index_constituent_weight")
    } >= {"uq_index_constituent_weight_version"}
    assert {
        item["name"] for item in inspector.get_unique_constraints("curated_record_lineage")
    } >= {"uq_curated_record_lineage_association"}
    assert {
        item["referred_table"] for item in inspector.get_foreign_keys("index_constituent_weight")
    } == {"instrument"}
    assert {
        item["referred_table"] for item in inspector.get_foreign_keys("curated_record_lineage")
    } == {"raw_payload"}

    command.downgrade(config, "0003")
    assert not {
        "index_constituent_weight",
        "curated_record_lineage",
    } & set(inspect(engine).get_table_names())


def test_0004_accepts_coherent_tables_from_historical_live_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "historical.db"
    config = _alembic_config(database_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)
    command.upgrade(config, "0003")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    IndexConstituentWeightRow.__table__.create(engine)
    CuratedRecordLineageRow.__table__.create(engine)

    command.upgrade(config, "0004")

    assert {"index_constituent_weight", "curated_record_lineage"} <= set(
        inspect(engine).get_table_names()
    )


def test_curated_lineage_and_index_weight_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "constraints.db"
    config = _alembic_config(database_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)
    command.upgrade(config, "head")
    database = Database(f"sqlite:///{database_path.as_posix()}")

    with database.session() as session:
        session.add_all(
            [
                _instrument("CN.SH.000300", "000300", "INDEX"),
                _instrument("CN.SZ.000001", "000001", "STOCK"),
            ]
        )
        first_raw = _raw_payload("a")
        second_raw = _raw_payload("b")
        session.add_all([first_raw, second_raw])
        session.flush()
        first_raw_id = first_raw.id
        second_raw_id = second_raw.id
        session.add(
            IndexConstituentWeightRow(
                index_instrument_id="CN.SH.000300",
                constituent_instrument_id="CN.SZ.000001",
                trade_date=date(2026, 8, 28),
                weight_percent=Decimal("1.25"),
                available_at=NOW,
                source="test",
                version="v1",
            )
        )
        session.add_all(
            [
                _lineage(first_raw_id),
                _lineage(second_raw_id),
            ]
        )

    with database.session() as session:
        assert len(list(session.scalars(select(CuratedRecordLineageRow)))) == 2

    with (
        pytest.raises(IntegrityError, match="UNIQUE constraint failed"),
        database.session() as session,
    ):
        session.add(_lineage(first_raw_id))

    with (
        pytest.raises(IntegrityError, match="CHECK constraint failed"),
        database.session() as session,
    ):
        session.add(
            IndexConstituentWeightRow(
                index_instrument_id="CN.SH.000300",
                constituent_instrument_id="CN.SZ.000001",
                trade_date=date(2026, 8, 29),
                weight_percent=Decimal("100.01"),
                available_at=NOW,
                source="test",
                version="v1",
            )
        )


def _instrument(instrument_id: str, symbol: str, instrument_type: str) -> InstrumentRow:
    return InstrumentRow(
        instrument_id=instrument_id,
        symbol=symbol,
        exchange="SH" if ".SH." in instrument_id else "SZ",
        instrument_type=instrument_type,
        name=symbol,
        listed_on=date(2005, 1, 1),
        status="LISTED",
        source="test",
        version="v1",
    )


def _raw_payload(suffix: str) -> RawPayloadRow:
    digest = suffix * 64
    return RawPayloadRow(
        provider="test",
        endpoint="daily",
        request_fingerprint=digest,
        request_hash=digest,
        request_params={"page": suffix},
        payload_hash=digest,
        payload={"records": []},
        fetched_at=NOW,
        available_at=NOW,
        schema_version="v1",
        size_bytes=2,
    )


def _lineage(raw_payload_id: int) -> CuratedRecordLineageRow:
    return CuratedRecordLineageRow(
        raw_payload_id=raw_payload_id,
        dataset="daily_bar",
        entity_type="DailyBar",
        entity_key="CN.SZ.000001|2026-08-28|test|v1",
        content_hash="c" * 64,
        created_at=NOW,
    )
