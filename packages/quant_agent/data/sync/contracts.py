"""Typed contracts and legal state transitions for resumable data synchronization."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Protocol

from quant_agent.core.time import ensure_aware
from quant_agent.data.sync.hashing import JsonObject


class SyncMode(StrEnum):
    """Supported synchronization execution modes."""

    INCREMENTAL = "INCREMENTAL"
    BACKFILL = "BACKFILL"
    REPLAY = "REPLAY"


class SyncRunState(StrEnum):
    """Persistent lifecycle for one synchronization run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SyncPageState(StrEnum):
    """Persistent lifecycle for one logical provider page."""

    PLANNED = "PLANNED"
    FETCHING = "FETCHING"
    RETRY_WAIT = "RETRY_WAIT"
    FETCHED = "FETCHED"
    DECODED = "DECODED"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


class InvalidStateTransition(ValueError):
    """Raised when persisted synchronization state would skip a required stage."""


@dataclass(frozen=True, slots=True)
class RawPage:
    """One parsed JSON response captured before any domain decoding occurs.

    Transport-level non-JSON/error-body archival belongs below this contract and is
    intentionally deferred until a raw-byte transport envelope is introduced.
    """

    provider: str
    endpoint: str
    request_params: JsonObject
    payload: JsonObject
    partition: JsonObject
    cursor_in: JsonObject | None
    cursor_out: JsonObject | None
    has_more: bool
    record_count: int
    fetched_at: datetime
    available_at: datetime
    schema_version: str

    def __post_init__(self) -> None:
        if not self.provider or not self.endpoint or not self.schema_version:
            raise ValueError("provider, endpoint and schema_version must be non-empty")
        if self.record_count < 0:
            raise ValueError("record_count cannot be negative")
        ensure_aware(self.fetched_at)
        ensure_aware(self.available_at)
        if self.has_more != (self.cursor_out is not None):
            raise ValueError("has_more must agree with the presence of cursor_out")


@dataclass(frozen=True, slots=True)
class DecodedPage[T]:
    """Validated records decoded from one durably archived raw page."""

    raw_page: RawPage
    records: tuple[T, ...]
    rejected_count: int = 0

    def __post_init__(self) -> None:
        if self.rejected_count < 0:
            raise ValueError("rejected_count cannot be negative")
        if len(self.records) + self.rejected_count != self.raw_page.record_count:
            raise ValueError("decoded and rejected records must equal raw record_count")


@dataclass(frozen=True, slots=True)
class PageIngestionResult:
    """Deterministic curated-write counts for one decoded page."""

    inserted: int = 0
    updated: int = 0
    skipped: int = 0

    def __post_init__(self) -> None:
        if min(self.inserted, self.updated, self.skipped) < 0:
            raise ValueError("ingestion counts cannot be negative")


@dataclass(frozen=True, slots=True)
class PageRequest:
    """Stable, credential-bearing request plan produced before external I/O."""

    provider: str
    endpoint: str
    request_params: JsonObject
    partition: JsonObject
    cursor_in: JsonObject | None

    def __post_init__(self) -> None:
        if not self.provider or not self.endpoint:
            raise ValueError("provider and endpoint must be non-empty")


class IngestionSession(Protocol):
    """Restricted Session view that deliberately omits commit, rollback, and close."""

    def add(self, instance: object) -> None: ...

    def flush(self, objects: Sequence[object] | None = None) -> None: ...

    def scalar(self, statement: Any, *args: Any, **kwargs: Any) -> Any: ...

    def scalars(self, statement: Any, *args: Any, **kwargs: Any) -> Any: ...

    def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any: ...

    def get(self, entity: Any, ident: Any, **kwargs: Any) -> Any: ...


class DatasetSource[T](Protocol):
    """Provider-neutral partition planning, fetching, and decoding contract."""

    provider: str
    dataset: str

    def partitions(
        self,
        committed_watermark: JsonObject,
        target_watermark: JsonObject,
    ) -> Sequence[JsonObject]: ...

    def partition_watermark(self, partition: JsonObject) -> JsonObject: ...

    def page_request(
        self,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> PageRequest: ...

    def fetch_page(
        self,
        partition: JsonObject,
        cursor: JsonObject | None,
    ) -> RawPage: ...

    def decode(self, raw_page: RawPage) -> DecodedPage[T]: ...


class DatasetIngestor[T](Protocol):
    """Curated write contract executed inside the runner's short transaction."""

    def ingest(
        self,
        session: IngestionSession,
        page: DecodedPage[T],
        *,
        raw_payload_id: int,
    ) -> PageIngestionResult: ...


_RUN_TRANSITIONS: Final[dict[SyncRunState, frozenset[SyncRunState]]] = {
    SyncRunState.PENDING: frozenset({SyncRunState.RUNNING, SyncRunState.CANCELLED}),
    SyncRunState.RUNNING: frozenset(
        {SyncRunState.SUCCEEDED, SyncRunState.FAILED, SyncRunState.CANCELLED}
    ),
    SyncRunState.SUCCEEDED: frozenset(),
    SyncRunState.FAILED: frozenset({SyncRunState.RUNNING, SyncRunState.CANCELLED}),
    SyncRunState.CANCELLED: frozenset(),
}

_PAGE_TRANSITIONS: Final[dict[SyncPageState, frozenset[SyncPageState]]] = {
    SyncPageState.PLANNED: frozenset({SyncPageState.FETCHING, SyncPageState.FAILED}),
    SyncPageState.FETCHING: frozenset(
        {SyncPageState.RETRY_WAIT, SyncPageState.FETCHED, SyncPageState.FAILED}
    ),
    SyncPageState.RETRY_WAIT: frozenset({SyncPageState.FETCHING, SyncPageState.FAILED}),
    SyncPageState.FETCHED: frozenset({SyncPageState.DECODED, SyncPageState.FAILED}),
    SyncPageState.DECODED: frozenset({SyncPageState.COMMITTED, SyncPageState.FAILED}),
    SyncPageState.COMMITTED: frozenset(),
    SyncPageState.FAILED: frozenset({SyncPageState.FETCHING}),
}


def ensure_run_transition(current: SyncRunState, target: SyncRunState) -> None:
    """Reject a run transition that is absent from the explicit state graph."""

    if target not in _RUN_TRANSITIONS[current]:
        raise InvalidStateTransition(f"illegal run transition: {current} -> {target}")


def ensure_page_transition(current: SyncPageState, target: SyncPageState) -> None:
    """Reject a page transition that is absent from the explicit state graph."""

    if target not in _PAGE_TRANSITIONS[current]:
        raise InvalidStateTransition(f"illegal page transition: {current} -> {target}")
