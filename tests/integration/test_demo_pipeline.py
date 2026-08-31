"""End-to-end tests for the deterministic local data pipeline."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from quant_agent.data.database import Database
from quant_agent.data.models import DailyBarRow, DatasetVersionRow, RawPayloadRow
from quant_agent.data.snapshots import SnapshotStore
from quant_agent.pipelines import run_demo_pipeline


def test_demo_pipeline_is_complete_verifiable_and_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "database" / "demo.db"
    snapshot_path = tmp_path / "snapshots"
    database_url = f"sqlite:///{database_path.as_posix()}"

    first = run_demo_pipeline(database_url=database_url, storage_path=snapshot_path)
    second = run_demo_pipeline(database_url=database_url, storage_path=snapshot_path)

    assert first.qualified is True
    assert first.snapshot_verified is True
    assert first.snapshot_reused is False
    assert (first.instrument_count, first.open_day_count, first.row_count) == (2, 3, 6)
    assert (first.inserted_bars, first.skipped_bars) == (6, 0)
    assert (second.inserted_bars, second.skipped_bars) == (0, 6)
    assert second.snapshot_reused is True
    assert second.content_hash == first.content_hash
    assert second.query_row_count == 6
    assert SnapshotStore(snapshot_path).verify(first.data_version) is True

    database = Database(database_url)
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(DailyBarRow)) == 6
        assert session.scalar(select(func.count()).select_from(RawPayloadRow)) == 6
        assert session.scalar(select(func.count()).select_from(DatasetVersionRow)) == 1


def test_demo_pipeline_rejects_non_sqlite_database() -> None:
    with pytest.raises(ValueError, match="only supports SQLite"):
        run_demo_pipeline(database_url="postgresql+psycopg://user:secret@localhost/demo")
