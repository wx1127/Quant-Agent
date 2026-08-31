"""Tests for the database bootstrap boundary."""

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from quant_agent.data.database import Database
from quant_agent.data.models import InstrumentAliasRow


def test_file_sqlite_database_creates_missing_parent(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "quant-agent.db"

    database = Database(f"sqlite:///{database_path.as_posix()}")
    database.create_schema()

    assert database_path.is_file()


def test_sqlite_connections_enforce_foreign_keys() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()

    with database.session() as session:
        assert session.scalar(text("PRAGMA foreign_keys")) == 1

    with (
        pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"),
        database.session() as session,
    ):
        session.add(
            InstrumentAliasRow(
                instrument_id="CN.SZ.MISSING",
                symbol="MISSING",
                exchange="SZ",
                effective_from=date(2026, 8, 28),
                source="test",
            )
        )


@pytest.mark.parametrize("reference", ["env://DATABASE_URL", "vault://database/url"])
def test_database_rejects_unresolved_reference(reference: str) -> None:
    with pytest.raises(ValueError, match="must be resolved"):
        Database(reference)
