"""Integration tests for migration and resumable synchronization persistence."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from quant_agent.data.database import Base
from quant_agent.data.models import DataSyncCheckpointRow, DataSyncPageRow
from quant_agent.data.sync import (
    CheckpointConflict,
    CheckpointLeaseUnavailable,
    IdempotencyConflict,
    InvalidStateTransition,
    JsonObject,
    SyncPageState,
    SyncRepository,
    SyncRepositoryError,
    SyncRowNotFound,
    SyncRunState,
    canonical_request_hash,
)

ROOT = Path(__file__).resolve().parents[2]
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 28, 10, 0, tzinfo=TZ)
SCOPE = {"market": "CN", "instrument_type": "STOCK"}
INITIAL_WATERMARK = {"trade_date": "2026-08-25"}
TARGET_WATERMARK = {"trade_date": "2026-08-28"}


def _alembic_config(database_path: Path) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    return config


def _running_run(repository: SyncRepository) -> str:
    row = repository.create_run(
        provider="tushare",
        dataset="daily_bar",
        scope=SCOPE,
        target_watermark=TARGET_WATERMARK,
        checkpoint_before=INITIAL_WATERMARK,
        idempotency_key="a" * 64,
        code_version="test",
        created_at=NOW,
    )
    return repository.transition_run(row.run_id, SyncRunState.RUNNING, at=NOW).run_id


def _decoded_page(
    repository: SyncRepository,
    run_id: str,
    ordinal: int,
    *,
    cursor_out: JsonObject | None = None,
) -> DataSyncPageRow:
    request_hash = canonical_request_hash(
        provider="tushare",
        endpoint="daily",
        params={"trade_date": "20260826", "offset": ordinal * 1000},
    )
    page = repository.create_page(
        run_id=run_id,
        partition={"trade_date": "2026-08-26"},
        page_ordinal=ordinal,
        request_hash=request_hash,
        cursor_in={"offset": ordinal * 1000},
        created_at=NOW,
    )
    repository.transition_page(page.page_id, SyncPageState.FETCHING, at=NOW)
    repository.transition_page(
        page.page_id,
        SyncPageState.FETCHED,
        at=NOW,
        cursor_out=cursor_out,
        received_count=1000,
    )
    return repository.transition_page(page.page_id, SyncPageState.DECODED, at=NOW)


def test_migration_upgrades_legacy_0001_and_downgrades_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration.db"
    config = _alembic_config(database_path)
    monkeypatch.delenv("QUANT_AGENT_DATABASE_URL", raising=False)

    command.upgrade(config, "0001")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    # 0001 historically used live metadata. Drop any newly visible sync tables to
    # reproduce an already-deployed 0001 database before applying fixed 0002 DDL.
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE IF EXISTS data_sync_page")
        connection.exec_driver_sql("DROP TABLE IF EXISTS data_sync_checkpoint")
        connection.exec_driver_sql("DROP TABLE IF EXISTS data_sync_run")

    command.upgrade(config, "0002")
    inspector = inspect(engine)
    assert {"data_sync_run", "data_sync_checkpoint", "data_sync_page"} <= set(
        inspector.get_table_names()
    )
    assert {item["name"] for item in inspector.get_unique_constraints("data_sync_checkpoint")} >= {
        "uq_data_sync_checkpoint_scope"
    }
    assert {item["name"] for item in inspector.get_unique_constraints("data_sync_page")} >= {
        "uq_data_sync_page_ordinal",
        "uq_data_sync_page_request",
    }

    command.downgrade(config, "0001")
    assert not {
        "data_sync_run",
        "data_sync_checkpoint",
        "data_sync_page",
    } & set(inspect(engine).get_table_names())
    assert "instrument" in inspect(engine).get_table_names()


def test_checkpoint_lease_and_revision_reject_stale_or_competing_workers(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'leases.db').as_posix()}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        checkpoint = SyncRepository(session).acquire_checkpoint(
            provider="tushare",
            dataset="daily_bar",
            scope=SCOPE,
            initial_watermark=INITIAL_WATERMARK,
            lease_owner="worker-1",
            lease_ttl=timedelta(minutes=5),
            now=NOW,
        )
        checkpoint_id = checkpoint.id
        stale_revision = checkpoint.revision
        session.commit()

    with Session(engine) as session:
        with pytest.raises(CheckpointLeaseUnavailable):
            SyncRepository(session).acquire_checkpoint(
                provider="tushare",
                dataset="daily_bar",
                scope=SCOPE,
                initial_watermark=INITIAL_WATERMARK,
                lease_owner="worker-2",
                lease_ttl=timedelta(minutes=5),
                now=NOW + timedelta(minutes=1),
            )
        session.rollback()

    with Session(engine) as session:
        renewed = SyncRepository(session).renew_checkpoint(
            checkpoint_id,
            lease_owner="worker-1",
            expected_revision=stale_revision,
            lease_ttl=timedelta(minutes=5),
            now=NOW + timedelta(minutes=1),
        )
        assert renewed.revision == stale_revision + 1
        session.commit()

    with Session(engine) as session:
        with pytest.raises(CheckpointConflict):
            SyncRepository(session).renew_checkpoint(
                checkpoint_id,
                lease_owner="worker-1",
                expected_revision=stale_revision,
                lease_ttl=timedelta(minutes=5),
                now=NOW + timedelta(minutes=2),
            )
        session.rollback()

    with Session(engine) as session:
        taken_over = SyncRepository(session).acquire_checkpoint(
            provider="tushare",
            dataset="daily_bar",
            scope=SCOPE,
            initial_watermark={"trade_date": "1900-01-01"},
            lease_owner="worker-2",
            lease_ttl=timedelta(minutes=5),
            now=NOW + timedelta(minutes=7),
        )
        assert taken_over.lease_owner == "worker-2"
        assert taken_over.committed_watermark == INITIAL_WATERMARK
        released = SyncRepository(session).release_checkpoint(
            checkpoint_id,
            lease_owner="worker-2",
            expected_revision=taken_over.revision,
            now=NOW + timedelta(minutes=8),
        )
        assert released.lease_owner is None
        assert released.lease_expires_at is None
        assert released.committed_watermark == INITIAL_WATERMARK
        with pytest.raises(CheckpointConflict):
            SyncRepository(session).release_checkpoint(
                checkpoint_id,
                lease_owner="worker-2",
                expected_revision=taken_over.revision,
                now=NOW + timedelta(minutes=9),
            )


def test_create_run_is_idempotent_only_for_the_same_fixed_target(db_session: Session) -> None:
    repository = SyncRepository(db_session)
    arguments = {
        "provider": "tushare",
        "dataset": "daily_bar",
        "scope": SCOPE,
        "target_watermark": TARGET_WATERMARK,
        "checkpoint_before": INITIAL_WATERMARK,
        "idempotency_key": "b" * 64,
        "code_version": "test",
        "created_at": NOW,
    }

    first = repository.create_run(**arguments)
    repeated = repository.create_run(**arguments)

    assert repeated.run_id == first.run_id
    with pytest.raises(IdempotencyConflict):
        repository.create_run(
            **{
                **arguments,
                "target_watermark": {"trade_date": "2026-08-29"},
            }
        )


def test_page_commit_and_checkpoint_cursor_are_atomic_and_keep_double_watermark(
    db_session: Session,
) -> None:
    repository = SyncRepository(db_session)
    checkpoint = repository.acquire_checkpoint(
        provider="tushare",
        dataset="daily_bar",
        scope=SCOPE,
        initial_watermark=INITIAL_WATERMARK,
        lease_owner="worker-1",
        lease_ttl=timedelta(minutes=10),
        now=NOW,
    )
    run_id = _running_run(repository)
    first_page = _decoded_page(repository, run_id, 0, cursor_out={"offset": 1000})

    with pytest.raises(CheckpointConflict):
        repository.commit_page_and_checkpoint(
            page_id=first_page.page_id,
            checkpoint_id=checkpoint.id,
            lease_owner="worker-1",
            expected_revision=checkpoint.revision - 1,
            expected_active_partition=None,
            expected_resume_cursor=None,
            active_partition={"trade_date": "2026-08-26"},
            cursor_out={"offset": 1000},
            now=NOW + timedelta(seconds=1),
        )
    unchanged_page = db_session.get(DataSyncPageRow, first_page.page_id, populate_existing=True)
    unchanged_checkpoint = db_session.get(
        DataSyncCheckpointRow,
        checkpoint.id,
        populate_existing=True,
    )
    assert unchanged_page is not None and unchanged_page.state == SyncPageState.DECODED.value
    assert unchanged_checkpoint is not None
    assert unchanged_checkpoint.resume_cursor is None
    assert unchanged_checkpoint.committed_watermark == INITIAL_WATERMARK

    committed_page, progressed = repository.commit_page_and_checkpoint(
        page_id=first_page.page_id,
        checkpoint_id=checkpoint.id,
        lease_owner="worker-1",
        expected_revision=checkpoint.revision,
        expected_active_partition=None,
        expected_resume_cursor=None,
        active_partition={"trade_date": "2026-08-26"},
        cursor_out={"offset": 1000},
        now=NOW + timedelta(seconds=2),
    )
    assert committed_page.state == SyncPageState.COMMITTED.value
    assert progressed.active_partition == {"trade_date": "2026-08-26"}
    assert progressed.resume_cursor == {"offset": 1000}
    assert progressed.committed_watermark == INITIAL_WATERMARK

    second_page = _decoded_page(repository, run_id, 1)
    with pytest.raises(CheckpointConflict, match="progress changed"):
        repository.commit_page_and_checkpoint(
            page_id=second_page.page_id,
            checkpoint_id=checkpoint.id,
            lease_owner="worker-1",
            expected_revision=progressed.revision,
            expected_active_partition=None,
            expected_resume_cursor=None,
            active_partition=None,
            cursor_out=None,
            advance_committed_watermark={"trade_date": "2026-08-26"},
            now=NOW + timedelta(seconds=3),
        )
    _, completed_partition = repository.commit_page_and_checkpoint(
        page_id=second_page.page_id,
        checkpoint_id=checkpoint.id,
        lease_owner="worker-1",
        expected_revision=progressed.revision,
        expected_active_partition={"trade_date": "2026-08-26"},
        expected_resume_cursor={"offset": 1000},
        active_partition=None,
        cursor_out=None,
        advance_committed_watermark={"trade_date": "2026-08-26"},
        now=NOW + timedelta(seconds=3),
    )
    assert completed_partition.active_partition is None
    assert completed_partition.resume_cursor is None
    assert completed_partition.committed_watermark == {"trade_date": "2026-08-26"}


def test_repository_rejects_skipped_page_state(db_session: Session) -> None:
    repository = SyncRepository(db_session)
    run_id = _running_run(repository)
    page = repository.create_page(
        run_id=run_id,
        partition={"trade_date": "2026-08-26"},
        page_ordinal=0,
        request_hash="f" * 64,
        created_at=NOW,
    )

    with pytest.raises(InvalidStateTransition):
        repository.transition_page(page.page_id, SyncPageState.COMMITTED, at=NOW)


def test_run_failure_resume_and_page_retry_are_auditable(db_session: Session) -> None:
    repository = SyncRepository(db_session)
    with pytest.raises(SyncRowNotFound):
        repository.transition_run("sync_missing", SyncRunState.RUNNING, at=NOW)

    run = repository.create_run(
        provider="tushare",
        dataset="daily_bar",
        scope=SCOPE,
        target_watermark=TARGET_WATERMARK,
        checkpoint_before=INITIAL_WATERMARK,
        requested_from={"trade_date": "2026-08-01"},
        code_version="test",
        created_at=NOW,
    )
    running = repository.transition_run(run.run_id, SyncRunState.RUNNING, at=NOW)
    failed = repository.transition_run(
        running.run_id,
        SyncRunState.FAILED,
        at=NOW + timedelta(seconds=1),
        error_code="UPSTREAM_TIMEOUT",
        error_message="request failed safely",
    )
    assert failed.error_code == "UPSTREAM_TIMEOUT"
    resumed = repository.transition_run(
        failed.run_id,
        SyncRunState.RUNNING,
        at=NOW + timedelta(seconds=2),
    )
    assert resumed.error_code is None
    assert resumed.started_at == running.started_at

    page = repository.create_page(
        run_id=run.run_id,
        partition={"trade_date": "2026-08-26"},
        page_ordinal=0,
        request_hash="e" * 64,
        created_at=NOW,
    )
    first_attempt = repository.transition_page(
        page.page_id,
        SyncPageState.FETCHING,
        at=NOW,
    )
    assert first_attempt.attempt_count == 1
    repository.transition_page(page.page_id, SyncPageState.RETRY_WAIT, at=NOW)
    second_attempt = repository.transition_page(
        page.page_id,
        SyncPageState.FETCHING,
        at=NOW,
    )
    assert second_attempt.attempt_count == 2
    with pytest.raises(ValueError, match="received_count"):
        repository.transition_page(
            page.page_id,
            SyncPageState.FETCHED,
            at=NOW,
            received_count=-1,
        )
    fetched = repository.transition_page(
        page.page_id,
        SyncPageState.FETCHED,
        at=NOW,
        received_count=10,
        accepted_count=9,
        rejected_count=1,
    )
    assert (fetched.received_count, fetched.accepted_count, fetched.rejected_count) == (10, 9, 1)
    failed_page = repository.transition_page(
        page.page_id,
        SyncPageState.FAILED,
        at=NOW,
        error_code="SCHEMA_ERROR",
        error_message="invalid field",
    )
    assert failed_page.error_code == "SCHEMA_ERROR"
    assert (
        repository.transition_page(
            page.page_id,
            SyncPageState.FETCHING,
            at=NOW,
        ).attempt_count
        == 3
    )


def test_repository_validates_page_scope_and_double_watermark_invariants(
    db_session: Session,
) -> None:
    repository = SyncRepository(db_session)
    pending = repository.create_run(
        provider="tushare",
        dataset="daily_bar",
        scope=SCOPE,
        target_watermark=TARGET_WATERMARK,
        checkpoint_before=INITIAL_WATERMARK,
        code_version="test",
        created_at=NOW,
    )
    with pytest.raises(SyncRepositoryError, match="running"):
        repository.create_page(
            run_id=pending.run_id,
            partition={"trade_date": "2026-08-26"},
            page_ordinal=0,
            request_hash="d" * 64,
            created_at=NOW,
        )
    with pytest.raises(SyncRowNotFound):
        repository.create_page(
            run_id="sync_missing",
            partition={},
            page_ordinal=0,
            request_hash="d" * 64,
            created_at=NOW,
        )

    run_id = repository.transition_run(
        pending.run_id,
        SyncRunState.RUNNING,
        at=NOW,
    ).run_id
    page = _decoded_page(repository, run_id, 0, cursor_out={"offset": 1000})
    checkpoint = repository.acquire_checkpoint(
        provider="tushare",
        dataset="daily_bar",
        scope=SCOPE,
        initial_watermark=INITIAL_WATERMARK,
        lease_owner="worker-1",
        lease_ttl=timedelta(minutes=5),
        now=NOW,
    )
    with pytest.raises(ValueError, match="active partition"):
        repository.commit_page_and_checkpoint(
            page_id=page.page_id,
            checkpoint_id=checkpoint.id,
            lease_owner="worker-1",
            expected_revision=checkpoint.revision,
            expected_active_partition=None,
            expected_resume_cursor=None,
            active_partition=None,
            cursor_out={"offset": 1000},
            now=NOW,
        )
    with pytest.raises(ValueError, match="must clear"):
        repository.commit_page_and_checkpoint(
            page_id=page.page_id,
            checkpoint_id=checkpoint.id,
            lease_owner="worker-1",
            expected_revision=checkpoint.revision,
            expected_active_partition=None,
            expected_resume_cursor=None,
            active_partition={"trade_date": "2026-08-26"},
            cursor_out={"offset": 1000},
            advance_committed_watermark={"trade_date": "2026-08-26"},
            now=NOW,
        )

    other_checkpoint = repository.acquire_checkpoint(
        provider="tushare",
        dataset="daily_bar",
        scope={"market": "CN", "instrument_type": "ETF"},
        initial_watermark=INITIAL_WATERMARK,
        lease_owner="worker-1",
        lease_ttl=timedelta(minutes=5),
        now=NOW,
    )
    with pytest.raises(SyncRepositoryError, match="scopes do not match"):
        repository.commit_page_and_checkpoint(
            page_id=page.page_id,
            checkpoint_id=other_checkpoint.id,
            lease_owner="worker-1",
            expected_revision=other_checkpoint.revision,
            expected_active_partition=None,
            expected_resume_cursor=None,
            active_partition={"trade_date": "2026-08-26"},
            cursor_out=None,
            now=NOW,
        )
