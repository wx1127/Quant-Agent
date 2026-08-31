"""Integration tests for raw-first paginated synchronization and recovery."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from quant_agent.data.database import Base, Database
from quant_agent.data.models import (
    DataQualityResultRow,
    DataSyncCheckpointRow,
    DataSyncPageRow,
    DataSyncRunRow,
    RawPayloadRow,
)
from quant_agent.data.sync import (
    DecodedPage,
    FakeDatasetSource,
    FakePageSpec,
    FakePartitionSpec,
    FakeSourceError,
    IngestionSession,
    JsonObject,
    PageIngestionResult,
    SyncMode,
    SyncPageState,
    SyncRepository,
    SyncRepositoryError,
    SyncRunState,
)
from quant_agent.data.sync.runner import IncrementalSyncRunner

ROOT = Path(__file__).resolve().parents[2]
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 28, 18, 0, tzinfo=TZ)
SCOPE = {"market": "CN", "instrument_type": "STOCK"}
INITIAL_WATERMARK = {"trade_date": "2026-08-25"}
TARGET_WATERMARK = {"trade_date": "2026-08-26"}
PARTITION = {"trade_date": "2026-08-26"}


class JsonRecordIngestor:
    """Idempotently persist fake records in the existing quality-result table."""

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[JsonObject],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        inserted = 0
        skipped = 0
        for record in page.records:
            record_id = record.get("id")
            if not isinstance(record_id, str):
                raise ValueError("fake record id must be a string")
            existing = session.scalar(
                select(DataQualityResultRow.id).where(
                    DataQualityResultRow.data_version == "fake-sync",
                    DataQualityResultRow.rule_id == "FAKE_INGEST",
                    DataQualityResultRow.entity_key == record_id,
                )
            )
            if existing is not None:
                skipped += 1
                continue
            session.add(
                DataQualityResultRow(
                    data_version="fake-sync",
                    rule_id="FAKE_INGEST",
                    severity="INFO",
                    passed=True,
                    entity_key=record_id,
                    message=f"raw_payload_id={raw_payload_id}",
                    observed_at=NOW,
                )
            )
            inserted += 1
        session.flush()
        return PageIngestionResult(inserted=inserted, skipped=skipped)


class CommitAttemptIngestor:
    """Attempt forbidden transaction control after staging a curated row."""

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[JsonObject],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        del page, raw_payload_id
        session.add(
            DataQualityResultRow(
                data_version="fake-sync",
                rule_id="FORBIDDEN_COMMIT",
                severity="INFO",
                passed=True,
                entity_key="forbidden",
                message="must roll back",
                observed_at=NOW,
            )
        )
        session.commit()  # type: ignore[attr-defined]
        return PageIngestionResult(inserted=1)  # pragma: no cover


class UnderCountingIngestor:
    """Return an invalid count summary without accounting for decoded records."""

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[JsonObject],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult:
        del session, page, raw_payload_id
        return PageIngestionResult()


class EmptyPlanSource(FakeDatasetSource):
    """Misbehaving source that claims no work before reaching its target."""

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        del committed_watermark, target_watermark
        return ()


def _alembic_config(database_path: Path) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    return config


def _database(tmp_path: Path, name: str) -> tuple[object, sessionmaker[Session]]:
    engine = create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _two_page_source() -> FakeDatasetSource:
    return FakeDatasetSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out={"offset": 1},
                        records=({"id": "one"},),
                        request_params={"token": "must-not-persist", "offset": 0},
                    ),
                    FakePageSpec(
                        page_ordinal=1,
                        cursor_in={"offset": 1},
                        cursor_out=None,
                        records=({"id": "two"},),
                        fail_fetch_once=True,
                        request_params={"token": "must-not-persist", "offset": 1},
                    ),
                ),
            ),
        ),
    )


def _runner(
    factory: sessionmaker[Session],
    source: FakeDatasetSource,
) -> IncrementalSyncRunner[JsonObject]:
    return IncrementalSyncRunner(
        session_factory=factory,
        source=source,
        ingestor=JsonRecordIngestor(),
        lease_owner="test-worker",
        lease_ttl=timedelta(minutes=5),
        clock=lambda: NOW,
    )


def _run(runner: IncrementalSyncRunner[JsonObject], key: str) -> object:
    return runner.run(
        scope=SCOPE,
        initial_watermark=INITIAL_WATERMARK,
        target_watermark=TARGET_WATERMARK,
        idempotency_key=key,
        code_version="test",
    )


def test_second_page_fetch_failure_resumes_without_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    engine, factory = _database(tmp_path, "fetch-recovery.db")
    source = _two_page_source()
    runner = _runner(factory, source)

    with pytest.raises(FakeSourceError, match="fetch failure"):
        _run(runner, "fetch-recovery".ljust(64, "0"))

    with Session(engine) as session:
        checkpoint = session.scalar(select(DataSyncCheckpointRow))
        run = session.scalar(select(DataSyncRunRow))
        assert checkpoint is not None
        assert checkpoint.committed_watermark == INITIAL_WATERMARK
        assert checkpoint.active_partition == PARTITION
        assert checkpoint.resume_cursor == {"offset": 1}
        assert checkpoint.lease_owner is None
        assert run is not None and run.state == SyncRunState.FAILED.value
        assert len(list(session.scalars(select(RawPayloadRow)))) == 1
        assert len(list(session.scalars(select(DataQualityResultRow)))) == 1
        assert {page.state for page in session.scalars(select(DataSyncPageRow))} == {
            SyncPageState.COMMITTED.value,
            SyncPageState.FAILED.value,
        }

    result = _run(runner, "fetch-recovery".ljust(64, "0"))

    assert result.state is SyncRunState.SUCCEEDED
    assert result.committed_watermark == TARGET_WATERMARK
    assert source.fetch_attempts(0) == 1
    assert source.fetch_attempts(1) == 2
    with Session(engine) as session:
        assert len(list(session.scalars(select(RawPayloadRow)))) == 2
        assert len(list(session.scalars(select(DataQualityResultRow)))) == 2
        assert {page.state for page in session.scalars(select(DataSyncPageRow))} == {
            SyncPageState.COMMITTED.value
        }
        checkpoint = session.scalar(select(DataSyncCheckpointRow))
        assert checkpoint is not None
        assert checkpoint.committed_watermark == TARGET_WATERMARK
        assert checkpoint.active_partition is None
        assert checkpoint.resume_cursor is None
        assert checkpoint.last_success_run_id == result.run_id
        assert all(
            "token" not in row.request_params for row in session.scalars(select(RawPayloadRow))
        )

    with Session(engine) as session, session.begin():
        blocked_checkpoint = SyncRepository(session).acquire_checkpoint(
            provider="fake",
            dataset="fake_records",
            scope=SCOPE,
            initial_watermark=INITIAL_WATERMARK,
            lease_owner="unrelated-worker",
            lease_ttl=timedelta(minutes=5),
            now=NOW,
        )
        blocked_checkpoint_id = blocked_checkpoint.id
    repeated = _run(runner, "fetch-recovery".ljust(64, "0"))
    assert repeated.run_id == result.run_id
    assert repeated.committed_watermark == result.committed_watermark
    assert source.fetch_attempts(0) == 1
    assert source.fetch_attempts(1) == 2
    with Session(engine) as session:
        still_blocked = session.get(DataSyncCheckpointRow, blocked_checkpoint_id)
        assert still_blocked is not None
        assert still_blocked.lease_owner == "unrelated-worker"


def test_decode_failure_keeps_raw_auditable_and_replays_without_network(
    tmp_path: Path,
) -> None:
    engine, factory = _database(tmp_path, "decode-recovery.db")
    source = FakeDatasetSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out=None,
                        records=({"id": "one"},),
                        fail_decode_once=True,
                        request_params={"token": "must-not-persist"},
                    ),
                ),
            ),
        ),
    )
    runner = _runner(factory, source)
    key = "decode-recovery".ljust(64, "0")

    with pytest.raises(FakeSourceError, match="decode failure"):
        _run(runner, key)

    with Session(engine) as session:
        raw_rows = list(session.scalars(select(RawPayloadRow)))
        pages = list(session.scalars(select(DataSyncPageRow)))
        checkpoint = session.scalar(select(DataSyncCheckpointRow))
        assert len(raw_rows) == 1
        assert raw_rows[0].request_params == {}
        assert len(pages) == 1
        assert pages[0].state == SyncPageState.FETCHED.value
        assert pages[0].error_code == "FakeSourceError"
        assert pages[0].raw_payload_id == raw_rows[0].id
        assert checkpoint is not None
        assert checkpoint.committed_watermark == INITIAL_WATERMARK
        assert len(list(session.scalars(select(DataQualityResultRow)))) == 0

    replay_source = FakeDatasetSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out=None,
                        records=({"id": "one"},),
                        fail_fetch_once=True,
                        request_params={"token": "must-not-persist"},
                    ),
                ),
            ),
        ),
    )
    result = _run(_runner(factory, replay_source), key)

    assert result.state is SyncRunState.SUCCEEDED
    assert result.fetched_pages == 0
    assert result.reused_raw_pages == 1
    assert source.fetch_attempts(0) == 1
    assert source.decode_attempts(0) == 1
    assert replay_source.fetch_attempts(0) == 0
    assert replay_source.decode_attempts(0) == 1
    with Session(engine) as session:
        assert len(list(session.scalars(select(RawPayloadRow)))) == 1
        assert len(list(session.scalars(select(DataQualityResultRow)))) == 1
        page = session.scalar(select(DataSyncPageRow))
        assert page is not None
        assert page.state == SyncPageState.COMMITTED.value
        assert page.error_code == "FakeSourceError"


def test_ingestor_cannot_commit_and_curated_write_rolls_back_with_checkpoint(
    tmp_path: Path,
) -> None:
    engine, factory = _database(tmp_path, "forbidden-commit.db")
    source = FakeDatasetSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out=None,
                        records=({"id": "one"},),
                    ),
                ),
            ),
        ),
    )
    runner = IncrementalSyncRunner(
        session_factory=factory,
        source=source,
        ingestor=CommitAttemptIngestor(),
        lease_owner="test-worker",
        clock=lambda: NOW,
    )
    key = "forbidden-commit".ljust(64, "0")

    with pytest.raises(AttributeError, match="commit"):
        _run(runner, key)

    with Session(engine) as session:
        assert len(list(session.scalars(select(RawPayloadRow)))) == 1
        assert len(list(session.scalars(select(DataQualityResultRow)))) == 0
        page = session.scalar(select(DataSyncPageRow))
        checkpoint = session.scalar(select(DataSyncCheckpointRow))
        assert page is not None and page.state == SyncPageState.FETCHED.value
        assert checkpoint is not None
        assert checkpoint.committed_watermark == INITIAL_WATERMARK

    recovered = _run(_runner(factory, source), key)
    assert recovered.reused_raw_pages == 1
    assert source.fetch_attempts(0) == 1


def test_runner_rejects_ingestion_counts_that_drop_decoded_records(tmp_path: Path) -> None:
    engine, factory = _database(tmp_path, "count-conservation.db")
    source = FakeDatasetSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out=None,
                        records=({"id": "one"},),
                    ),
                ),
            ),
        ),
    )
    runner = IncrementalSyncRunner(
        session_factory=factory,
        source=source,
        ingestor=UnderCountingIngestor(),
        lease_owner="test-worker",
        clock=lambda: NOW,
    )

    with pytest.raises(SyncRepositoryError, match="account for every"):
        _run(runner, "count-conservation".ljust(64, "0"))

    with Session(engine) as session:
        page = session.scalar(select(DataSyncPageRow))
        checkpoint = session.scalar(select(DataSyncCheckpointRow))
        assert page is not None and page.state == SyncPageState.FETCHED.value
        assert checkpoint is not None
        assert checkpoint.committed_watermark == INITIAL_WATERMARK


def test_failed_run_cannot_resume_after_a_new_run_advances_checkpoint(
    tmp_path: Path,
) -> None:
    _engine, factory = _database(tmp_path, "stale-run.db")
    old_source = FakeDatasetSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out=None,
                        records=({"id": "one"},),
                        fail_decode_once=True,
                    ),
                ),
            ),
        ),
    )
    old_runner = _runner(factory, old_source)
    old_key = "old-failed-run".ljust(64, "0")
    with pytest.raises(FakeSourceError):
        _run(old_runner, old_key)

    new_source = FakeDatasetSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out=None,
                        records=({"id": "one"},),
                    ),
                ),
            ),
        ),
    )
    _run(_runner(factory, new_source), "new-success-run".ljust(64, "0"))

    with pytest.raises(SyncRepositoryError, match="another run advanced"):
        _run(old_runner, old_key)
    assert old_source.fetch_attempts(0) == 1


def test_runner_supports_database_session_factory_and_rejects_false_completion(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'database-session.db').as_posix()}")
    database.create_schema()
    source = EmptyPlanSource(
        dataset="fake_records",
        available_at=NOW,
        clock=lambda: NOW,
        partitions=(
            FakePartitionSpec(
                partition=PARTITION,
                watermark=TARGET_WATERMARK,
                pages=(
                    FakePageSpec(
                        page_ordinal=0,
                        cursor_in=None,
                        cursor_out=None,
                        records=({"id": "one"},),
                    ),
                ),
            ),
        ),
    )
    runner = IncrementalSyncRunner(
        session_factory=database.session,
        source=source,
        ingestor=JsonRecordIngestor(),
        lease_owner="database-worker",
        clock=lambda: NOW,
    )

    with pytest.raises(SyncRepositoryError, match="target watermark"):
        _run(runner, "empty-plan".ljust(64, "0"))

    with database.session() as session:
        checkpoint = session.scalar(select(DataSyncCheckpointRow))
        run = session.scalar(select(DataSyncRunRow))
        assert checkpoint is not None
        assert checkpoint.committed_watermark == INITIAL_WATERMARK
        assert checkpoint.lease_owner is None
        assert run is not None and run.state == SyncRunState.FAILED.value


def test_expired_worker_cannot_fail_run_after_unique_owner_takes_over(
    tmp_path: Path,
) -> None:
    engine, factory = _database(tmp_path, "lease-fencing.db")
    source = _two_page_source()
    old_runner = IncrementalSyncRunner(
        session_factory=factory,
        source=source,
        ingestor=JsonRecordIngestor(),
        lease_owner="same-worker-label",
        lease_ttl=timedelta(seconds=1),
        clock=lambda: NOW,
    )
    another_runner = IncrementalSyncRunner(
        session_factory=factory,
        source=source,
        ingestor=JsonRecordIngestor(),
        lease_owner="same-worker-label",
        lease_ttl=timedelta(seconds=1),
        clock=lambda: NOW,
    )
    assert old_runner._lease_owner != another_runner._lease_owner
    context = old_runner._setup(
        scope=SCOPE,
        initial_watermark=INITIAL_WATERMARK,
        target_watermark=TARGET_WATERMARK,
        idempotency_key="lease-fencing".ljust(64, "0"),
        mode=SyncMode.INCREMENTAL,
        requested_from=None,
        config_hash=None,
        code_version="test",
    )

    with Session(engine) as session, session.begin():
        takeover = SyncRepository(session).acquire_checkpoint(
            provider="fake",
            dataset="fake_records",
            scope=SCOPE,
            initial_watermark=INITIAL_WATERMARK,
            lease_owner="new-owner",
            lease_ttl=timedelta(minutes=5),
            now=NOW + timedelta(seconds=2),
        )
        takeover_id = takeover.id

    old_runner._record_failure(context, RuntimeError("late worker"))

    with Session(engine) as session:
        run = session.get(DataSyncRunRow, context.run_id)
        checkpoint = session.get(DataSyncCheckpointRow, takeover_id)
        assert run is not None and run.state == SyncRunState.RUNNING.value
        assert checkpoint is not None and checkpoint.lease_owner == "new-owner"


def test_0003_migrates_legacy_raw_rows_and_downgrades_to_0002(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration.db"
    config = _alembic_config(database_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)
    command.upgrade(config, "0002")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO raw_payload "
                "(provider, endpoint, request_fingerprint, payload_hash, payload, "
                "available_at, ingested_at) VALUES "
                "(:provider, :endpoint, :fingerprint, :payload_hash, :payload, "
                ":available_at, :ingested_at)"
            ),
            {
                "provider": "legacy",
                "endpoint": "daily",
                "fingerprint": "a" * 64,
                "payload_hash": "b" * 64,
                "payload": "{}",
                "available_at": "2026-08-28 16:00:00",
                "ingested_at": "2026-08-28 16:01:00",
            },
        )

    command.upgrade(config, "0003")
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("raw_payload")}
    assert set(columns) >= _LINEAGE_COLUMN_NAMES
    assert all(columns[name]["nullable"] is False for name in _LINEAGE_COLUMN_NAMES)
    assert "ix_raw_payload_request_hash" in {
        index["name"] for index in inspector.get_indexes("raw_payload")
    }
    with engine.connect() as connection:
        migrated = connection.execute(
            text("SELECT request_hash, request_params, schema_version, size_bytes FROM raw_payload")
        ).one()
    assert migrated.request_hash == "a" * 64
    assert migrated.request_params == "{}"
    assert migrated.schema_version == "legacy-v1"
    assert migrated.size_bytes == 0
    assert "progress_run_id" in {
        column["name"] for column in inspect(engine).get_columns("data_sync_checkpoint")
    }

    command.downgrade(config, "0002")
    assert not _LINEAGE_COLUMN_NAMES.intersection(
        {column["name"] for column in inspect(engine).get_columns("raw_payload")}
    )
    assert "progress_run_id" not in {
        column["name"] for column in inspect(engine).get_columns("data_sync_checkpoint")
    }


_LINEAGE_COLUMN_NAMES = {
    "request_hash",
    "request_params",
    "fetched_at",
    "schema_version",
    "size_bytes",
}
