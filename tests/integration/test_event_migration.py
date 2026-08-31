"""Migration coverage for point-in-time event evidence."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

ROOT = Path(__file__).resolve().parents[2]


def _config(database_path: Path) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    return config


def test_event_schema_upgrades_and_downgrades_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "event.db"
    config = _config(database_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)
    command.upgrade(config, "0004")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "0005")
    inspector = inspect(engine)
    assert {"event_evidence", "event_instrument_link"} <= set(inspector.get_table_names())
    assert {item["name"] for item in inspector.get_unique_constraints("event_evidence")} >= {
        "uq_event_evidence_source_revision"
    }
    assert {
        item["referred_table"] for item in inspector.get_foreign_keys("event_instrument_link")
    } == {"event_evidence", "instrument"}

    command.downgrade(config, "0004")
    assert not {"event_evidence", "event_instrument_link"} & set(inspect(engine).get_table_names())
