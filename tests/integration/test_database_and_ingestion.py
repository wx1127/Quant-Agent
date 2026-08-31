"""Integration tests for migrations and idempotent daily ingestion."""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from quant_agent.config import QuantAgentSettings
from quant_agent.data.domain import DailyBar
from quant_agent.data.ingestion import DailyBarIngestionService
from quant_agent.data.models import DailyBarRow, RawPayloadRow
from quant_agent.data.providers.base import ProviderBatch

TZ = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[2]
MIGRATED_TABLES = {
    "instrument",
    "instrument_alias",
    "instrument_status",
    "market_bar_daily",
    "financial_statement_point_in_time",
    "dataset_version",
}


def _alembic_config() -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    return config


def _bar(trade_date: date = date(2026, 7, 30)) -> DailyBar:
    return DailyBar(
        instrument_id="CN.SZ.000001",
        trade_date=trade_date,
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=Decimal("100"),
        turnover=Decimal("1000"),
        source="test",
        available_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=TZ),
        version="v1",
    )


def _batch(bar: DailyBar) -> ProviderBatch[DailyBar]:
    return ProviderBatch(
        provider="test",
        endpoint="daily",
        request_params={"trade_date": bar.trade_date.isoformat()},
        raw_payload={"records": [bar.instrument_id, bar.trade_date.isoformat()]},
        available_at=bar.available_at,
        records=(bar,),
    )


def test_local_configuration_uses_zero_config_sqlite() -> None:
    settings = QuantAgentSettings.from_toml(ROOT / "configs/environments/local.toml")

    assert settings.storage.database_url == "sqlite:///./artifacts/quant_agent.db"


def test_initial_alembic_migration_creates_parent_and_drops_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "clean" / "nested" / "migration.db"
    config = _alembic_config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    tables = set(inspect(engine).get_table_names())
    assert database_path.is_file()
    assert tables >= MIGRATED_TABLES

    command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}


def test_default_alembic_url_works_from_clean_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)

    command.upgrade(_alembic_config(), "head")

    database_path = tmp_path / "artifacts" / "quant_agent.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    assert database_path.is_file()
    assert set(inspect(engine).get_table_names()) >= MIGRATED_TABLES


def test_database_url_environment_variable_overrides_alembic_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_path = tmp_path / "configured" / "database.db"
    override_path = tmp_path / "override%20target" / "database.db"
    config = _alembic_config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{configured_path.as_posix()}")
    monkeypatch.setenv(
        "QUANT_AGENT_DATABASE_URL",
        f"sqlite:///{override_path.as_posix()}",
    )

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{override_path.as_posix()}")
    assert override_path.is_file()
    assert not configured_path.exists()
    assert set(inspect(engine).get_table_names()) >= MIGRATED_TABLES


def test_daily_ingestion_is_idempotent_and_archives_raw_payload(
    seeded_session: Session,
) -> None:
    service = DailyBarIngestionService(seeded_session)
    batch = _batch(_bar())

    first = service.ingest(batch)
    second = service.ingest(batch)

    assert (first.inserted, first.skipped) == (1, 0)
    assert (second.inserted, second.skipped) == (0, 1)
    assert len(list(seeded_session.scalars(select(DailyBarRow)))) == 1
    assert len(list(seeded_session.scalars(select(RawPayloadRow)))) == 1
    assert first.raw_payload_hash == second.raw_payload_hash


def test_daily_ingestion_rejects_closed_date(seeded_session: Session) -> None:
    with pytest.raises(ValueError, match="not an open"):
        DailyBarIngestionService(seeded_session).ingest(_batch(_bar(date(2026, 8, 1))))
