"""Transactional repository for resumable synchronization state."""

from datetime import datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from quant_agent.core.time import ensure_aware, shanghai_now
from quant_agent.data.models import (
    DataSyncCheckpointRow,
    DataSyncPageRow,
    DataSyncRunRow,
)
from quant_agent.data.sync.contracts import (
    SyncMode,
    SyncPageState,
    SyncRunState,
    ensure_page_transition,
    ensure_run_transition,
)
from quant_agent.data.sync.hashing import (
    JsonObject,
    JsonValue,
    canonical_hash,
    canonical_scope_hash,
    clone_json_object,
)


class SyncRepositoryError(RuntimeError):
    """Base class for synchronization persistence failures."""


class CheckpointLeaseUnavailable(SyncRepositoryError):
    """Raised when another worker owns an unexpired checkpoint lease."""


class CheckpointConflict(SyncRepositoryError):
    """Raised when a checkpoint revision, owner, or lease changed concurrently."""


class IdempotencyConflict(SyncRepositoryError):
    """Raised when an idempotency key is reused for a different run specification."""


class SyncRowNotFound(SyncRepositoryError):
    """Raised when a requested synchronization row does not exist."""


def _non_empty(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _timestamp(value: datetime | None) -> datetime:
    return ensure_aware(value or shanghai_now())


def _positive_ttl(value: timedelta) -> timedelta:
    if value <= timedelta(0):
        raise ValueError("lease_ttl must be positive")
    return value


def _row_count(result: Any) -> int:
    return int(cast(CursorResult[Any], result).rowcount or 0)


class SyncRepository:
    """Persist sync progress using short caller-controlled transactions.

    Checkpoint mutation uses optimistic revisions even on SQLite. SQLite deployments
    intentionally support one worker; PostgreSQL deployments may use distinct scopes
    concurrently because every mutation is scoped and conditionally updated.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_run(
        self,
        *,
        provider: str,
        dataset: str,
        scope: JsonObject,
        target_watermark: JsonObject,
        checkpoint_before: JsonObject,
        mode: SyncMode = SyncMode.INCREMENTAL,
        requested_from: JsonObject | None = None,
        idempotency_key: str | None = None,
        config_hash: str | None = None,
        code_version: str = "unknown",
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> DataSyncRunRow:
        """Create a fixed-target run or return its exact idempotent predecessor."""

        provider_name = _non_empty(provider, "provider")
        dataset_name = _non_empty(dataset, "dataset")
        normalized_scope = clone_json_object(scope)
        scope_hash = canonical_scope_hash(normalized_scope)
        normalized_target = clone_json_object(target_watermark)
        normalized_before = clone_json_object(checkpoint_before)
        normalized_requested = (
            clone_json_object(requested_from) if requested_from is not None else None
        )
        normalized_idempotency_key = (
            _non_empty(idempotency_key, "idempotency_key") if idempotency_key is not None else None
        )
        if normalized_idempotency_key is not None:
            existing = self._session.scalar(
                select(DataSyncRunRow).where(
                    DataSyncRunRow.idempotency_key == normalized_idempotency_key
                )
            )
            if existing is not None:
                same_specification = (
                    existing.provider == provider_name
                    and existing.dataset == dataset_name
                    and existing.scope_hash == scope_hash
                    and existing.mode == mode.value
                    and existing.scope == normalized_scope
                    and existing.target_watermark == normalized_target
                    and existing.checkpoint_before == normalized_before
                    and existing.requested_from == normalized_requested
                )
                if not same_specification:
                    raise IdempotencyConflict(
                        "idempotency key belongs to a different synchronization run"
                    )
                return existing

        actual_config_hash = config_hash or canonical_hash({})
        if len(actual_config_hash) != 64:
            raise ValueError("config_hash must be a SHA-256 hexadecimal digest")
        actual_run_id = run_id or f"sync_{uuid4().hex}"
        row = DataSyncRunRow(
            run_id=_non_empty(actual_run_id, "run_id"),
            provider=provider_name,
            dataset=dataset_name,
            scope=normalized_scope,
            scope_hash=scope_hash,
            mode=mode.value,
            state=SyncRunState.PENDING.value,
            idempotency_key=normalized_idempotency_key,
            requested_from=normalized_requested,
            target_watermark=normalized_target,
            checkpoint_before=normalized_before,
            checkpoint_after=None,
            config_hash=actual_config_hash,
            code_version=_non_empty(code_version, "code_version"),
            created_at=_timestamp(created_at),
            started_at=None,
            heartbeat_at=None,
            finished_at=None,
            error_code=None,
            error_message=None,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def transition_run(
        self,
        run_id: str,
        target: SyncRunState,
        *,
        at: datetime | None = None,
        checkpoint_after: JsonObject | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> DataSyncRunRow:
        """Apply one legal run transition with an optimistic current-state predicate."""

        row = self._session.get(DataSyncRunRow, run_id)
        if row is None:
            raise SyncRowNotFound(f"unknown sync run: {run_id}")
        current = SyncRunState(row.state)
        ensure_run_transition(current, target)
        timestamp = _timestamp(at)
        values: dict[str, Any] = {"state": target.value}
        if target is SyncRunState.RUNNING:
            values.update(
                started_at=row.started_at or timestamp,
                heartbeat_at=timestamp,
                finished_at=None,
                error_code=None,
                error_message=None,
            )
        elif target in {
            SyncRunState.SUCCEEDED,
            SyncRunState.FAILED,
            SyncRunState.CANCELLED,
        }:
            values["finished_at"] = timestamp
            values["error_code"] = error_code
            values["error_message"] = error_message
        if checkpoint_after is not None:
            values["checkpoint_after"] = clone_json_object(checkpoint_after)
        result = self._session.execute(
            update(DataSyncRunRow)
            .where(DataSyncRunRow.run_id == run_id, DataSyncRunRow.state == current.value)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if _row_count(result) != 1:
            raise SyncRepositoryError("sync run changed concurrently")
        self._session.flush()
        refreshed = self._session.get(DataSyncRunRow, run_id, populate_existing=True)
        if refreshed is None:  # pragma: no cover - guarded by the update predicate
            raise SyncRowNotFound(f"unknown sync run: {run_id}")
        return refreshed

    def get_checkpoint(
        self,
        *,
        provider: str,
        dataset: str,
        scope: JsonObject,
    ) -> DataSyncCheckpointRow | None:
        """Return the checkpoint for an exact canonical scope."""

        scope_hash = canonical_scope_hash(scope)
        row = self._session.scalar(
            select(DataSyncCheckpointRow).where(
                DataSyncCheckpointRow.provider == provider,
                DataSyncCheckpointRow.dataset == dataset,
                DataSyncCheckpointRow.scope_hash == scope_hash,
            )
        )
        if row is not None and row.scope != clone_json_object(scope):
            raise SyncRepositoryError("checkpoint scope hash collision")
        return row

    def acquire_checkpoint(
        self,
        *,
        provider: str,
        dataset: str,
        scope: JsonObject,
        initial_watermark: JsonObject,
        lease_owner: str,
        lease_ttl: timedelta,
        now: datetime | None = None,
    ) -> DataSyncCheckpointRow:
        """Create/get a checkpoint and conditionally acquire its expiring lease."""

        provider_name = _non_empty(provider, "provider")
        dataset_name = _non_empty(dataset, "dataset")
        owner = _non_empty(lease_owner, "lease_owner")
        ttl = _positive_ttl(lease_ttl)
        timestamp = _timestamp(now)
        normalized_scope = clone_json_object(scope)
        row = self.get_checkpoint(
            provider=provider_name,
            dataset=dataset_name,
            scope=normalized_scope,
        )
        if row is None:
            row = DataSyncCheckpointRow(
                provider=provider_name,
                dataset=dataset_name,
                scope=normalized_scope,
                scope_hash=canonical_scope_hash(normalized_scope),
                committed_watermark=clone_json_object(initial_watermark),
                active_partition=None,
                resume_cursor=None,
                last_success_run_id=None,
                progress_run_id=None,
                lease_owner=None,
                lease_expires_at=None,
                revision=0,
                updated_at=timestamp,
            )
            self._session.add(row)
            self._session.flush()

        result = self._session.execute(
            update(DataSyncCheckpointRow)
            .where(
                DataSyncCheckpointRow.id == row.id,
                or_(
                    DataSyncCheckpointRow.lease_owner.is_(None),
                    DataSyncCheckpointRow.lease_owner == owner,
                    DataSyncCheckpointRow.lease_expires_at <= timestamp,
                ),
            )
            .values(
                lease_owner=owner,
                lease_expires_at=timestamp + ttl,
                revision=DataSyncCheckpointRow.revision + 1,
                updated_at=timestamp,
            )
            .execution_options(synchronize_session=False)
        )
        if _row_count(result) != 1:
            raise CheckpointLeaseUnavailable(
                f"checkpoint lease is held by another worker: {provider_name}/{dataset_name}"
            )
        self._session.flush()
        return self._checkpoint_by_id(row.id)

    def renew_checkpoint(
        self,
        checkpoint_id: int,
        *,
        lease_owner: str,
        expected_revision: int,
        lease_ttl: timedelta,
        now: datetime | None = None,
    ) -> DataSyncCheckpointRow:
        """Extend an unexpired lease using its optimistic revision."""

        owner = _non_empty(lease_owner, "lease_owner")
        ttl = _positive_ttl(lease_ttl)
        timestamp = _timestamp(now)
        result = self._session.execute(
            update(DataSyncCheckpointRow)
            .where(
                DataSyncCheckpointRow.id == checkpoint_id,
                DataSyncCheckpointRow.lease_owner == owner,
                DataSyncCheckpointRow.lease_expires_at > timestamp,
                DataSyncCheckpointRow.revision == expected_revision,
            )
            .values(
                lease_expires_at=timestamp + ttl,
                revision=DataSyncCheckpointRow.revision + 1,
                updated_at=timestamp,
            )
            .execution_options(synchronize_session=False)
        )
        if _row_count(result) != 1:
            raise CheckpointConflict("checkpoint lease could not be renewed")
        self._session.flush()
        return self._checkpoint_by_id(checkpoint_id)

    def release_checkpoint(
        self,
        checkpoint_id: int,
        *,
        lease_owner: str,
        expected_revision: int,
        last_success_run_id: str | None = None,
        now: datetime | None = None,
    ) -> DataSyncCheckpointRow:
        """Release an owned lease without changing either synchronization watermark."""

        owner = _non_empty(lease_owner, "lease_owner")
        timestamp = _timestamp(now)
        values: dict[str, Any] = {
            "lease_owner": None,
            "lease_expires_at": None,
            "revision": DataSyncCheckpointRow.revision + 1,
            "updated_at": timestamp,
        }
        if last_success_run_id is not None:
            values["last_success_run_id"] = last_success_run_id
        result = self._session.execute(
            update(DataSyncCheckpointRow)
            .where(
                DataSyncCheckpointRow.id == checkpoint_id,
                DataSyncCheckpointRow.lease_owner == owner,
                DataSyncCheckpointRow.revision == expected_revision,
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if _row_count(result) != 1:
            raise CheckpointConflict("checkpoint lease could not be released")
        self._session.flush()
        return self._checkpoint_by_id(checkpoint_id)

    def create_page(
        self,
        *,
        run_id: str,
        partition: JsonObject,
        page_ordinal: int,
        request_hash: str,
        cursor_in: JsonObject | None = None,
        page_id: str | None = None,
        created_at: datetime | None = None,
    ) -> DataSyncPageRow:
        """Plan a unique logical page under a running synchronization run."""

        if page_ordinal < 0:
            raise ValueError("page_ordinal cannot be negative")
        if len(request_hash) != 64:
            raise ValueError("request_hash must be a SHA-256 hexadecimal digest")
        run = self._session.get(DataSyncRunRow, run_id)
        if run is None:
            raise SyncRowNotFound(f"unknown sync run: {run_id}")
        if SyncRunState(run.state) is not SyncRunState.RUNNING:
            raise SyncRepositoryError("pages can only be planned for a running sync run")
        normalized_partition = clone_json_object(partition)
        row = DataSyncPageRow(
            page_id=page_id or f"page_{uuid4().hex}",
            run_id=run_id,
            partition=normalized_partition,
            partition_hash=canonical_scope_hash(normalized_partition),
            page_ordinal=page_ordinal,
            cursor_in=clone_json_object(cursor_in) if cursor_in is not None else None,
            cursor_out=None,
            request_hash=request_hash,
            state=SyncPageState.PLANNED.value,
            attempt_count=0,
            raw_payload_id=None,
            received_count=0,
            accepted_count=0,
            rejected_count=0,
            created_at=_timestamp(created_at),
            fetched_at=None,
            committed_at=None,
            error_code=None,
            error_message=None,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def next_page_ordinal(self, run_id: str, partition: JsonObject) -> int:
        """Return the next stable page ordinal for one run partition."""

        partition_hash = canonical_scope_hash(partition)
        latest = self._session.scalar(
            select(func.max(DataSyncPageRow.page_ordinal)).where(
                DataSyncPageRow.run_id == run_id,
                DataSyncPageRow.partition_hash == partition_hash,
            )
        )
        return 0 if latest is None else int(latest) + 1

    def find_resumable_page(
        self,
        *,
        run_id: str,
        partition: JsonObject,
        cursor_in: JsonObject | None,
    ) -> DataSyncPageRow | None:
        """Find the earliest archived page that has not reached curated commit."""

        row = self.find_page_for_cursor(
            run_id=run_id,
            partition=partition,
            cursor_in=cursor_in,
        )
        if (
            row is not None
            and row.state == SyncPageState.FETCHED.value
            and row.raw_payload_id is not None
        ):
            return row
        return None

    def find_page_for_cursor(
        self,
        *,
        run_id: str,
        partition: JsonObject,
        cursor_in: JsonObject | None,
    ) -> DataSyncPageRow | None:
        """Find an exact logical page regardless of its lifecycle state."""

        normalized_partition = clone_json_object(partition)
        normalized_cursor = clone_json_object(cursor_in) if cursor_in is not None else None
        rows = self._session.scalars(
            select(DataSyncPageRow)
            .where(
                DataSyncPageRow.run_id == run_id,
                DataSyncPageRow.partition_hash == canonical_scope_hash(normalized_partition),
            )
            .order_by(DataSyncPageRow.page_ordinal)
        )
        for row in rows:
            if row.partition != normalized_partition:
                raise SyncRepositoryError("sync page partition hash collision")
            if row.cursor_in == normalized_cursor:
                return row
        return None

    def transition_page(
        self,
        page_id: str,
        target: SyncPageState,
        *,
        at: datetime | None = None,
        cursor_out: JsonObject | None = None,
        raw_payload_id: int | None = None,
        received_count: int | None = None,
        accepted_count: int | None = None,
        rejected_count: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> DataSyncPageRow:
        """Apply one legal page transition without advancing checkpoint progress."""

        row = self._session.get(DataSyncPageRow, page_id)
        if row is None:
            raise SyncRowNotFound(f"unknown sync page: {page_id}")
        current = SyncPageState(row.state)
        ensure_page_transition(current, target)
        timestamp = _timestamp(at)
        values: dict[str, Any] = {"state": target.value}
        if target is SyncPageState.FETCHING:
            values["attempt_count"] = DataSyncPageRow.attempt_count + 1
            values["error_code"] = None
            values["error_message"] = None
        elif target is SyncPageState.FETCHED:
            values["fetched_at"] = timestamp
            values["cursor_out"] = clone_json_object(cursor_out) if cursor_out is not None else None
        elif target is SyncPageState.FAILED:
            values["error_code"] = error_code
            values["error_message"] = error_message
        for name, value in (
            ("raw_payload_id", raw_payload_id),
            ("received_count", received_count),
            ("accepted_count", accepted_count),
            ("rejected_count", rejected_count),
        ):
            if value is not None:
                if name.endswith("_count") and value < 0:
                    raise ValueError(f"{name} cannot be negative")
                values[name] = value
        result = self._session.execute(
            update(DataSyncPageRow)
            .where(DataSyncPageRow.page_id == page_id, DataSyncPageRow.state == current.value)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if _row_count(result) != 1:
            raise SyncRepositoryError("sync page changed concurrently")
        self._session.flush()
        return self._page_by_id(page_id)

    def record_page_error(
        self,
        page_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> DataSyncPageRow:
        """Record a decode failure while retaining the truthful FETCHED state."""

        row = self._session.get(DataSyncPageRow, page_id)
        if row is None:
            raise SyncRowNotFound(f"unknown sync page: {page_id}")
        if SyncPageState(row.state) is not SyncPageState.FETCHED:
            raise SyncRepositoryError("only a fetched page can retain a decode error")
        result = self._session.execute(
            update(DataSyncPageRow)
            .where(
                DataSyncPageRow.page_id == page_id,
                DataSyncPageRow.state == SyncPageState.FETCHED.value,
            )
            .values(error_code=error_code, error_message=error_message)
            .execution_options(synchronize_session=False)
        )
        if _row_count(result) != 1:
            raise SyncRepositoryError("sync page changed while recording its error")
        self._session.flush()
        return self._page_by_id(page_id)

    def commit_page_and_checkpoint(
        self,
        *,
        page_id: str,
        checkpoint_id: int,
        lease_owner: str,
        expected_revision: int,
        expected_active_partition: JsonObject | None,
        expected_resume_cursor: JsonObject | None,
        active_partition: JsonObject | None,
        cursor_out: JsonObject | None,
        advance_committed_watermark: JsonObject | None = None,
        now: datetime | None = None,
    ) -> tuple[DataSyncPageRow, DataSyncCheckpointRow]:
        """Atomically commit a decoded page and CAS-update both checkpoint watermarks."""

        owner = _non_empty(lease_owner, "lease_owner")
        timestamp = _timestamp(now)
        page = self._session.get(DataSyncPageRow, page_id)
        checkpoint = self._session.get(DataSyncCheckpointRow, checkpoint_id)
        if page is None:
            raise SyncRowNotFound(f"unknown sync page: {page_id}")
        if checkpoint is None:
            raise SyncRowNotFound(f"unknown sync checkpoint: {checkpoint_id}")
        normalized_expected_active = (
            clone_json_object(expected_active_partition)
            if expected_active_partition is not None
            else None
        )
        normalized_expected_cursor = (
            clone_json_object(expected_resume_cursor)
            if expected_resume_cursor is not None
            else None
        )
        if (
            checkpoint.active_partition != normalized_expected_active
            or checkpoint.resume_cursor != normalized_expected_cursor
        ):
            raise CheckpointConflict("checkpoint progress changed before page commit")
        ensure_page_transition(SyncPageState(page.state), SyncPageState.COMMITTED)
        run = self._session.get(DataSyncRunRow, page.run_id)
        if run is None:  # pragma: no cover - protected by the foreign key
            raise SyncRowNotFound(f"unknown sync run: {page.run_id}")
        if (
            run.provider != checkpoint.provider
            or run.dataset != checkpoint.dataset
            or run.scope_hash != checkpoint.scope_hash
        ):
            raise SyncRepositoryError("page run and checkpoint scopes do not match")

        normalized_active = (
            clone_json_object(active_partition) if active_partition is not None else None
        )
        normalized_cursor = clone_json_object(cursor_out) if cursor_out is not None else None
        stored_cursor = (
            clone_json_object(cast(dict[str, JsonValue], page.cursor_out))
            if page.cursor_out is not None
            else None
        )
        if normalized_cursor != stored_cursor:
            raise ValueError("checkpoint cursor must match the fetched page cursor")
        if normalized_cursor is not None and normalized_active is None:
            raise ValueError("a resume cursor requires an active partition")
        if normalized_active is not None and normalized_active != page.partition:
            raise ValueError("active partition must match the fetched page partition")
        if advance_committed_watermark is not None and (
            normalized_active is not None or normalized_cursor is not None
        ):
            raise ValueError(
                "advancing the committed watermark must clear active partition progress"
            )
        checkpoint_values: dict[str, Any] = {
            "active_partition": normalized_active,
            "resume_cursor": normalized_cursor,
            "progress_run_id": page.run_id,
            "revision": DataSyncCheckpointRow.revision + 1,
            "updated_at": timestamp,
        }
        if advance_committed_watermark is not None:
            checkpoint_values["committed_watermark"] = clone_json_object(
                advance_committed_watermark
            )

        # A SAVEPOINT makes the method atomic even when a caller catches a conflict
        # and continues using the surrounding transaction.
        with self._session.begin_nested():
            checkpoint_result = self._session.execute(
                update(DataSyncCheckpointRow)
                .where(
                    DataSyncCheckpointRow.id == checkpoint_id,
                    DataSyncCheckpointRow.lease_owner == owner,
                    DataSyncCheckpointRow.lease_expires_at > timestamp,
                    DataSyncCheckpointRow.revision == expected_revision,
                )
                .values(**checkpoint_values)
                .execution_options(synchronize_session=False)
            )
            if _row_count(checkpoint_result) != 1:
                raise CheckpointConflict("checkpoint cursor changed or its lease expired")
            page_result = self._session.execute(
                update(DataSyncPageRow)
                .where(
                    DataSyncPageRow.page_id == page_id,
                    DataSyncPageRow.state == SyncPageState.DECODED.value,
                )
                .values(
                    state=SyncPageState.COMMITTED.value,
                    cursor_out=normalized_cursor,
                    committed_at=timestamp,
                )
                .execution_options(synchronize_session=False)
            )
            if _row_count(page_result) != 1:
                raise SyncRepositoryError("sync page changed before checkpoint commit")

        self._session.flush()
        return self._page_by_id(page_id), self._checkpoint_by_id(checkpoint_id)

    def _checkpoint_by_id(self, checkpoint_id: int) -> DataSyncCheckpointRow:
        row = self._session.get(
            DataSyncCheckpointRow,
            checkpoint_id,
            populate_existing=True,
        )
        if row is None:
            raise SyncRowNotFound(f"unknown sync checkpoint: {checkpoint_id}")
        return row

    def checkpoint_by_id(self, checkpoint_id: int) -> DataSyncCheckpointRow:
        """Return a current checkpoint by primary key."""

        return self._checkpoint_by_id(checkpoint_id)

    def run_by_id(self, run_id: str) -> DataSyncRunRow:
        """Return a current run by primary key."""

        row = self._session.get(DataSyncRunRow, run_id, populate_existing=True)
        if row is None:
            raise SyncRowNotFound(f"unknown sync run: {run_id}")
        return row

    def run_by_idempotency_key(self, idempotency_key: str) -> DataSyncRunRow | None:
        """Return the run previously created for an external delivery identity."""

        return self._session.scalar(
            select(DataSyncRunRow).where(DataSyncRunRow.idempotency_key == idempotency_key)
        )

    def page_by_id(self, page_id: str) -> DataSyncPageRow:
        """Return a current page by primary key."""

        return self._page_by_id(page_id)

    def _page_by_id(self, page_id: str) -> DataSyncPageRow:
        row = self._session.get(DataSyncPageRow, page_id, populate_existing=True)
        if row is None:
            raise SyncRowNotFound(f"unknown sync page: {page_id}")
        return row
