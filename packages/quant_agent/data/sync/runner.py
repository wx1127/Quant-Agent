"""Two-phase incremental runner with raw-first durability and page recovery."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy.orm import Session

from quant_agent.core.time import SHANGHAI_TZ, ensure_aware, shanghai_now
from quant_agent.data.ingestion import RawPayloadArchive
from quant_agent.data.models import RawPayloadRow
from quant_agent.data.sync.contracts import (
    DatasetIngestor,
    DatasetSource,
    DecodedPage,
    IngestionSession,
    PageIngestionResult,
    PageRequest,
    RawPage,
    SyncMode,
    SyncPageState,
    SyncRunState,
)
from quant_agent.data.sync.hashing import (
    JsonObject,
    JsonValue,
    canonical_request_hash,
    canonical_scope_hash,
    clone_json_object,
)
from quant_agent.data.sync.repository import SyncRepository, SyncRepositoryError

type SessionFactory = Callable[[], AbstractContextManager[Session]]
type Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class SyncExecutionResult:
    """Stable summary for one completed or idempotently reused synchronization run."""

    run_id: str
    state: SyncRunState
    committed_watermark: JsonObject
    checkpoint_revision: int
    fetched_pages: int
    reused_raw_pages: int
    inserted: int
    updated: int
    skipped: int


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    run_id: str
    checkpoint_id: int
    committed_watermark: JsonObject
    checkpoint_revision: int
    already_succeeded: bool


@dataclass(frozen=True, slots=True)
class _ArchivedPage:
    page_id: str
    raw_payload_id: int
    raw_page: RawPage


@dataclass(frozen=True, slots=True)
class _PreparedPage:
    page_id: str
    request: PageRequest


class _IngestionSessionView(IngestionSession):
    """Delegate curated operations while making transaction control unavailable."""

    __slots__ = ("__session",)

    def __init__(self, session: Session) -> None:
        self.__session = session

    def add(self, instance: object) -> None:
        self.__session.add(instance)

    def flush(self, objects: Sequence[object] | None = None) -> None:
        self.__session.flush(objects)

    def scalar(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        return self.__session.scalar(statement, *args, **kwargs)

    def scalars(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        return self.__session.scalars(statement, *args, **kwargs)

    def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        return self.__session.execute(statement, *args, **kwargs)

    def get(self, entity: Any, ident: Any, **kwargs: Any) -> Any:
        return self.__session.get(entity, ident, **kwargs)


def _stored_time(value: datetime) -> datetime:
    """Restore SQLite timestamps, whose driver drops timezone offsets."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=SHANGHAI_TZ)
    return ensure_aware(value)


class IncrementalSyncRunner[T]:
    """Execute source pages without holding database transactions across external code."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        source: DatasetSource[T],
        ingestor: DatasetIngestor[T],
        lease_owner: str,
        lease_ttl: timedelta = timedelta(minutes=5),
        clock: Clock = shanghai_now,
    ) -> None:
        if not lease_owner.strip():
            raise ValueError("lease_owner must be non-empty")
        if lease_ttl <= timedelta(0):
            raise ValueError("lease_ttl must be positive")
        self._session_factory = session_factory
        self._source = source
        self._ingestor = ingestor
        self._lease_owner = f"{lease_owner.strip()[:90]}:{uuid4().hex}"
        self._lease_ttl = lease_ttl
        self._clock = clock

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        resource = self._session_factory()
        with resource as session, session.begin():
            yield session

    def run(
        self,
        *,
        scope: JsonObject,
        initial_watermark: JsonObject,
        target_watermark: JsonObject,
        idempotency_key: str | None = None,
        mode: SyncMode = SyncMode.INCREMENTAL,
        requested_from: JsonObject | None = None,
        config_hash: str | None = None,
        code_version: str = "unknown",
    ) -> SyncExecutionResult:
        """Synchronize every planned partition, reusing raw pages after safe failures."""

        context: _ExecutionContext | None = None
        fetched_pages = 0
        reused_pages = 0
        inserted = 0
        updated = 0
        skipped = 0
        try:
            context = self._setup(
                scope=scope,
                initial_watermark=initial_watermark,
                target_watermark=target_watermark,
                idempotency_key=idempotency_key,
                mode=mode,
                requested_from=requested_from,
                config_hash=config_hash,
                code_version=code_version,
            )
            if context.already_succeeded:
                return SyncExecutionResult(
                    run_id=context.run_id,
                    state=SyncRunState.SUCCEEDED,
                    committed_watermark=context.committed_watermark,
                    checkpoint_revision=context.checkpoint_revision,
                    fetched_pages=0,
                    reused_raw_pages=0,
                    inserted=0,
                    updated=0,
                    skipped=0,
                )

            partitions = self._planned_partitions(
                context.committed_watermark,
                target_watermark,
            )
            for partition in partitions:
                cursor = self._starting_cursor(context.checkpoint_id, partition)
                while True:
                    request = self._source.page_request(partition, cursor)
                    self._validate_page_request(request, partition, cursor)
                    archived = self._load_archived_page(
                        context.run_id,
                        partition,
                        cursor,
                    )
                    if archived is None:
                        prepared = self._prepare_fetch_page(context.run_id, request)
                        # No Session exists while provider/network code executes.
                        try:
                            raw_page = self._source.fetch_page(partition, cursor)
                        except Exception as error:
                            self._record_fetch_failure(prepared.page_id, error)
                            raise
                        self._validate_raw_page(raw_page, request)
                        archived = self._archive_page(prepared.page_id, raw_page)
                        fetched_pages += 1
                    else:
                        self._validate_raw_page(archived.raw_page, request)
                        reused_pages += 1

                    try:
                        # Decode is intentionally outside both archive and curated DB
                        # transactions. A bad schema cannot roll back the raw evidence.
                        decoded = self._source.decode(archived.raw_page)
                    except Exception as error:
                        self._record_decode_failure(archived.page_id, error)
                        raise
                    if decoded.raw_page != archived.raw_page:
                        raise SyncRepositoryError("decoded page does not reference its raw page")

                    result = self._ingest_and_commit(
                        context=context,
                        partition=partition,
                        archived=archived,
                        decoded=decoded,
                    )
                    inserted += result.inserted
                    updated += result.updated
                    skipped += result.skipped
                    if not archived.raw_page.has_more:
                        break
                    cursor = archived.raw_page.cursor_out

            return self._finish(
                context,
                target_watermark=target_watermark,
                fetched_pages=fetched_pages,
                reused_pages=reused_pages,
                inserted=inserted,
                updated=updated,
                skipped=skipped,
            )
        except Exception as primary_error:
            if context is not None and not context.already_succeeded:
                try:
                    self._record_failure(context, primary_error)
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "synchronization and failure persistence both failed",
                        [primary_error, cleanup_error],
                    ) from primary_error
            raise

    def _setup(
        self,
        *,
        scope: JsonObject,
        initial_watermark: JsonObject,
        target_watermark: JsonObject,
        idempotency_key: str | None,
        mode: SyncMode,
        requested_from: JsonObject | None,
        config_hash: str | None,
        code_version: str,
    ) -> _ExecutionContext:
        completed = self._completed_idempotent_context(
            scope=scope,
            target_watermark=target_watermark,
            idempotency_key=idempotency_key,
            mode=mode,
            requested_from=requested_from,
        )
        if completed is not None:
            return completed
        timestamp = ensure_aware(self._clock())
        with self._transaction() as session:
            repository = SyncRepository(session)
            checkpoint = repository.acquire_checkpoint(
                provider=self._source.provider,
                dataset=self._source.dataset,
                scope=scope,
                initial_watermark=initial_watermark,
                lease_owner=self._lease_owner,
                lease_ttl=self._lease_ttl,
                now=timestamp,
            )
            previous = (
                repository.run_by_idempotency_key(idempotency_key)
                if idempotency_key is not None
                else None
            )
            checkpoint_before = (
                cast(dict[str, JsonValue], previous.checkpoint_before)
                if previous is not None
                else cast(dict[str, JsonValue], checkpoint.committed_watermark)
            )
            if (
                previous is not None
                and SyncRunState(previous.state) is SyncRunState.FAILED
                and checkpoint.progress_run_id not in {None, previous.run_id}
                and (
                    checkpoint.active_partition is not None
                    or checkpoint.committed_watermark != previous.checkpoint_before
                )
            ):
                raise SyncRepositoryError(
                    "failed run cannot resume after another run advanced checkpoint progress"
                )
            run = repository.create_run(
                provider=self._source.provider,
                dataset=self._source.dataset,
                scope=scope,
                target_watermark=target_watermark,
                checkpoint_before=clone_json_object(checkpoint_before),
                mode=mode,
                requested_from=requested_from,
                idempotency_key=idempotency_key,
                config_hash=config_hash,
                code_version=code_version,
                created_at=timestamp,
            )
            state = SyncRunState(run.state)
            if state is SyncRunState.SUCCEEDED:
                released = repository.release_checkpoint(
                    checkpoint.id,
                    lease_owner=self._lease_owner,
                    expected_revision=checkpoint.revision,
                    last_success_run_id=run.run_id,
                    now=timestamp,
                )
                return _ExecutionContext(
                    run_id=run.run_id,
                    checkpoint_id=released.id,
                    committed_watermark=clone_json_object(
                        cast(dict[str, JsonValue], released.committed_watermark)
                    ),
                    checkpoint_revision=released.revision,
                    already_succeeded=True,
                )
            if state in {SyncRunState.PENDING, SyncRunState.FAILED}:
                repository.transition_run(run.run_id, SyncRunState.RUNNING, at=timestamp)
            elif state is not SyncRunState.RUNNING:
                raise SyncRepositoryError(f"sync run cannot be resumed from {state}")
            return _ExecutionContext(
                run_id=run.run_id,
                checkpoint_id=checkpoint.id,
                committed_watermark=clone_json_object(
                    cast(dict[str, JsonValue], checkpoint.committed_watermark)
                ),
                checkpoint_revision=checkpoint.revision,
                already_succeeded=False,
            )

    def _completed_idempotent_context(
        self,
        *,
        scope: JsonObject,
        target_watermark: JsonObject,
        idempotency_key: str | None,
        mode: SyncMode,
        requested_from: JsonObject | None,
    ) -> _ExecutionContext | None:
        if idempotency_key is None:
            return None
        with self._transaction() as session:
            repository = SyncRepository(session)
            run = repository.run_by_idempotency_key(idempotency_key)
            if run is None or SyncRunState(run.state) is not SyncRunState.SUCCEEDED:
                return None
            normalized_requested = (
                clone_json_object(requested_from) if requested_from is not None else None
            )
            if (
                run.provider != self._source.provider
                or run.dataset != self._source.dataset
                or run.scope_hash != canonical_scope_hash(scope)
                or run.scope != scope
                or run.target_watermark != target_watermark
                or run.mode != mode.value
                or run.requested_from != normalized_requested
            ):
                raise SyncRepositoryError(
                    "idempotency key belongs to a different synchronization request"
                )
            if run.checkpoint_after is None:
                raise SyncRepositoryError("succeeded run is missing its checkpoint result")
            checkpoint = repository.get_checkpoint(
                provider=self._source.provider,
                dataset=self._source.dataset,
                scope=scope,
            )
            if checkpoint is None:
                raise SyncRepositoryError("succeeded run checkpoint is missing")
            return _ExecutionContext(
                run_id=run.run_id,
                checkpoint_id=checkpoint.id,
                committed_watermark=clone_json_object(
                    cast(dict[str, JsonValue], run.checkpoint_after)
                ),
                checkpoint_revision=checkpoint.revision,
                already_succeeded=True,
            )

    def _planned_partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> tuple[JsonObject, ...]:
        planned: Sequence[JsonObject] = self._source.partitions(
            clone_json_object(committed_watermark),
            clone_json_object(target_watermark),
        )
        result = tuple(clone_json_object(partition) for partition in planned)
        identities = [canonical_scope_hash(partition) for partition in result]
        if len(identities) != len(set(identities)):
            raise ValueError("source planned a duplicate partition")
        return result

    def _starting_cursor(
        self,
        checkpoint_id: int,
        partition: JsonObject,
    ) -> JsonObject | None:
        with self._transaction() as session:
            checkpoint = SyncRepository(session).checkpoint_by_id(checkpoint_id)
            active = (
                clone_json_object(cast(dict[str, JsonValue], checkpoint.active_partition))
                if checkpoint.active_partition is not None
                else None
            )
            if active is None:
                return None
            if active != partition:
                raise SyncRepositoryError(
                    "planned partitions do not begin with the checkpoint's active partition"
                )
            return (
                clone_json_object(cast(dict[str, JsonValue], checkpoint.resume_cursor))
                if checkpoint.resume_cursor is not None
                else None
            )

    def _load_archived_page(
        self,
        run_id: str,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> _ArchivedPage | None:
        with self._transaction() as session:
            repository = SyncRepository(session)
            page = repository.find_resumable_page(
                run_id=run_id,
                partition=partition,
                cursor_in=cursor,
            )
            if page is None or page.raw_payload_id is None:
                return None
            raw = session.get(RawPayloadRow, page.raw_payload_id)
            if raw is None:
                raise SyncRepositoryError("fetched page references a missing raw payload")
            if raw.request_hash != page.request_hash:
                raise SyncRepositoryError("page request hash does not match its raw payload")
            raw_page = RawPage(
                provider=raw.provider,
                endpoint=raw.endpoint,
                request_params=clone_json_object(cast(dict[str, JsonValue], raw.request_params)),
                payload=clone_json_object(cast(dict[str, JsonValue], raw.payload)),
                partition=clone_json_object(cast(dict[str, JsonValue], page.partition)),
                cursor_in=(
                    clone_json_object(cast(dict[str, JsonValue], page.cursor_in))
                    if page.cursor_in is not None
                    else None
                ),
                cursor_out=(
                    clone_json_object(cast(dict[str, JsonValue], page.cursor_out))
                    if page.cursor_out is not None
                    else None
                ),
                has_more=page.cursor_out is not None,
                record_count=page.received_count,
                fetched_at=_stored_time(raw.fetched_at),
                available_at=_stored_time(raw.available_at),
                schema_version=raw.schema_version,
            )
            return _ArchivedPage(
                page_id=page.page_id,
                raw_payload_id=raw.id,
                raw_page=raw_page,
            )

    def _validate_raw_page(
        self,
        raw_page: RawPage,
        request: PageRequest,
    ) -> None:
        if raw_page.provider != request.provider or raw_page.endpoint != request.endpoint:
            raise ValueError("raw page provider or endpoint does not match its request")
        if raw_page.partition != request.partition or raw_page.cursor_in != request.cursor_in:
            raise ValueError("raw page partition or input cursor does not match the request")
        expected_hash = canonical_request_hash(
            provider=request.provider,
            endpoint=request.endpoint,
            params=request.request_params,
        )
        actual_hash = canonical_request_hash(
            provider=raw_page.provider,
            endpoint=raw_page.endpoint,
            params=raw_page.request_params,
        )
        if actual_hash != expected_hash:
            raise ValueError("raw page request metadata changed during fetch")

    def _validate_page_request(
        self,
        request: PageRequest,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> None:
        if request.provider != self._source.provider:
            raise ValueError("page request provider does not match the source")
        if request.partition != partition or request.cursor_in != cursor:
            raise ValueError("page request partition or cursor does not match the plan")

    def _prepare_fetch_page(self, run_id: str, request: PageRequest) -> _PreparedPage:
        request_hash = canonical_request_hash(
            provider=request.provider,
            endpoint=request.endpoint,
            params=request.request_params,
        )
        with self._transaction() as session:
            repository = SyncRepository(session)
            if SyncRunState(repository.run_by_id(run_id).state) is not SyncRunState.RUNNING:
                raise SyncRepositoryError("fetch planning requires a running sync run")
            page = repository.find_page_for_cursor(
                run_id=run_id,
                partition=request.partition,
                cursor_in=request.cursor_in,
            )
            if page is None:
                page = repository.create_page(
                    run_id=run_id,
                    partition=request.partition,
                    page_ordinal=repository.next_page_ordinal(run_id, request.partition),
                    request_hash=request_hash,
                    cursor_in=request.cursor_in,
                    created_at=ensure_aware(self._clock()),
                )
            elif page.request_hash != request_hash:
                raise SyncRepositoryError("resumed page request identity changed")
            state = SyncPageState(page.state)
            if state is SyncPageState.FETCHING:
                repository.transition_page(
                    page.page_id,
                    SyncPageState.FAILED,
                    at=ensure_aware(self._clock()),
                    error_code="STALE_FETCH",
                    error_message="stale fetch attempt recovered safely",
                )
                state = SyncPageState.FAILED
            if state in {SyncPageState.PLANNED, SyncPageState.FAILED}:
                repository.transition_page(
                    page.page_id,
                    SyncPageState.FETCHING,
                    at=ensure_aware(self._clock()),
                )
            else:
                raise SyncRepositoryError(f"page cannot fetch from state {state}")
            return _PreparedPage(page_id=page.page_id, request=request)

    def _record_fetch_failure(self, page_id: str, error: Exception) -> None:
        with self._transaction() as session:
            SyncRepository(session).transition_page(
                page_id,
                SyncPageState.FAILED,
                at=ensure_aware(self._clock()),
                error_code=type(error).__name__[:64],
                error_message="provider page fetch failed safely",
            )

    def _archive_page(self, page_id: str, raw_page: RawPage) -> _ArchivedPage:
        with self._transaction() as session:
            repository = SyncRepository(session)
            archive = RawPayloadArchive(session).archive_raw(raw_page)
            fetched = repository.transition_page(
                page_id,
                SyncPageState.FETCHED,
                at=raw_page.fetched_at,
                cursor_out=raw_page.cursor_out,
                raw_payload_id=archive.raw_payload_id,
                received_count=raw_page.record_count,
            )
            return _ArchivedPage(
                page_id=fetched.page_id,
                raw_payload_id=archive.raw_payload_id,
                raw_page=raw_page,
            )

    def _record_decode_failure(self, page_id: str, error: Exception) -> None:
        with self._transaction() as session:
            SyncRepository(session).record_page_error(
                page_id,
                error_code=type(error).__name__[:64],
                error_message="raw page decoding failed safely",
            )

    def _ingest_and_commit(
        self,
        *,
        context: _ExecutionContext,
        partition: JsonObject,
        archived: _ArchivedPage,
        decoded: DecodedPage[T],
    ) -> PageIngestionResult:
        timestamp = ensure_aware(self._clock())
        final_page = not archived.raw_page.has_more
        partition_watermark = (
            clone_json_object(self._source.partition_watermark(partition)) if final_page else None
        )
        with self._transaction() as session:
            repository = SyncRepository(session)
            checkpoint = repository.checkpoint_by_id(context.checkpoint_id)
            expected_active = (
                clone_json_object(cast(dict[str, JsonValue], checkpoint.active_partition))
                if checkpoint.active_partition is not None
                else None
            )
            expected_cursor = (
                clone_json_object(cast(dict[str, JsonValue], checkpoint.resume_cursor))
                if checkpoint.resume_cursor is not None
                else None
            )
            repository.transition_page(
                archived.page_id,
                SyncPageState.DECODED,
                at=timestamp,
                accepted_count=len(decoded.records),
                rejected_count=decoded.rejected_count,
            )
            result = self._ingestor.ingest(
                _IngestionSessionView(session),
                decoded,
                raw_payload_id=archived.raw_payload_id,
            )
            if result.inserted + result.updated + result.skipped != len(decoded.records):
                raise SyncRepositoryError("ingestion counts must account for every decoded record")
            repository.commit_page_and_checkpoint(
                page_id=archived.page_id,
                checkpoint_id=checkpoint.id,
                lease_owner=self._lease_owner,
                expected_revision=checkpoint.revision,
                expected_active_partition=expected_active,
                expected_resume_cursor=expected_cursor,
                active_partition=None if final_page else partition,
                cursor_out=archived.raw_page.cursor_out,
                advance_committed_watermark=partition_watermark,
                now=timestamp,
            )
            return result

    def _finish(
        self,
        context: _ExecutionContext,
        *,
        target_watermark: JsonObject,
        fetched_pages: int,
        reused_pages: int,
        inserted: int,
        updated: int,
        skipped: int,
    ) -> SyncExecutionResult:
        timestamp = ensure_aware(self._clock())
        with self._transaction() as session:
            repository = SyncRepository(session)
            checkpoint = repository.checkpoint_by_id(context.checkpoint_id)
            watermark = clone_json_object(
                cast(dict[str, JsonValue], checkpoint.committed_watermark)
            )
            if watermark != target_watermark:
                raise SyncRepositoryError(
                    "source partitions completed without reaching the target watermark"
                )
            repository.transition_run(
                context.run_id,
                SyncRunState.SUCCEEDED,
                at=timestamp,
                checkpoint_after=watermark,
            )
            released = repository.release_checkpoint(
                checkpoint.id,
                lease_owner=self._lease_owner,
                expected_revision=checkpoint.revision,
                last_success_run_id=context.run_id,
                now=timestamp,
            )
            return SyncExecutionResult(
                run_id=context.run_id,
                state=SyncRunState.SUCCEEDED,
                committed_watermark=watermark,
                checkpoint_revision=released.revision,
                fetched_pages=fetched_pages,
                reused_raw_pages=reused_pages,
                inserted=inserted,
                updated=updated,
                skipped=skipped,
            )

    def _record_failure(
        self,
        context: _ExecutionContext,
        error: Exception,
    ) -> None:
        timestamp = ensure_aware(self._clock())
        with self._transaction() as session:
            repository = SyncRepository(session)
            checkpoint = repository.checkpoint_by_id(context.checkpoint_id)
            if checkpoint.lease_owner != self._lease_owner:
                return
            run = repository.run_by_id(context.run_id)
            if SyncRunState(run.state) is SyncRunState.RUNNING:
                repository.transition_run(
                    context.run_id,
                    SyncRunState.FAILED,
                    at=timestamp,
                    error_code=type(error).__name__[:64],
                    error_message="synchronization failed safely",
                )
            repository.release_checkpoint(
                checkpoint.id,
                lease_owner=self._lease_owner,
                expected_revision=checkpoint.revision,
                now=timestamp,
            )
